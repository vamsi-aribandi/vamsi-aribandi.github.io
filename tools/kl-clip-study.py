#!/usr/bin/env python3
"""KL placement and clipping, on a softmax bandit.

A 1,000-action bandit with logits theta, policy p = softmax(theta), fixed
rewards r and fixed reference q. The objective
    J(p) = E_p[r] - beta KL(p || q)
is maximized by p* proportional to q exp(r / beta), and J* - J = beta KL(p || p*)
exactly, so every run is scored by its exact objective gap.

Two experiments, both written to static/klviz/clip.json:
  identity: four KL placements around a PPO clipped surrogate, checked by autograd.
  training: exact-expectation and sampled-batch training with a frozen rollout
            policy over several inner updates.

  pip install torch numpy
  python tools/kl-clip-study.py
"""
import json
from pathlib import Path
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
K, BETA, EPS = 1000, 0.5, 0.2
PLACEMENTS = {
    'k1_reward': 'C(rho, A - beta l)',
    'k2_loss': 'C(rho, A) + beta sg(rho) k2',
    'ratio_k3_loss': 'C(rho, A) + beta rho k3',
    'k3_split': 'C(rho, A - beta k3) + beta sg(rho) k3',
}


def bandit(seed=0):
    g = np.random.default_rng(seed)
    zq = g.normal(0, 1.5, K)
    log_q = zq - np.logaddexp.reduce(zq)
    r = g.normal(0, 1, K)
    log_star = log_q + r / BETA
    log_star -= np.logaddexp.reduce(log_star)
    return torch.tensor(r), torch.tensor(log_q), torch.tensor(log_star)


R, LOG_Q, LOG_STAR = bandit()


def kl(log_a, log_b):
    return float((log_a.exp() * (log_a - log_b)).sum())


def gap(theta):
    """J* - J(p) = beta KL(p || p*), exactly."""
    return BETA * kl(torch.log_softmax(theta, 0), LOG_STAR)


# KL placements around a clipped surrogate ------------------------------------

def clipped(ratio, adv, eps):
    """Negative PPO surrogate with a detached advantage, and where its gradient is cut."""
    adv = adv.detach()
    loss = -torch.minimum(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv)
    cut = ((adv > 0) & (ratio > 1 + eps)) | ((adv < 0) & (ratio < 1 - eps))
    return loss, cut


def placement_loss(name, theta, actions, weights, log_mu, adv, eps):
    """Weighted per-sample loss. KL coefficients use the current policy."""
    lp = torch.log_softmax(theta, 0)[actions]
    ratio = (lp - log_mu[actions]).exp()
    l = lp - LOG_Q[actions]
    k2, k3 = l ** 2 / 2, (-l).exp() - 1 + l
    if name == 'k1_reward':
        loss, cut = clipped(ratio, adv - BETA * l, eps)
    elif name == 'k2_loss':
        loss, cut = clipped(ratio, adv, eps)
        loss = loss + BETA * ratio.detach() * k2
    elif name == 'ratio_k3_loss':
        loss, cut = clipped(ratio, adv, eps)
        loss = loss + BETA * ratio * k3
    elif name == 'k3_split':
        loss, cut = clipped(ratio, adv - BETA * k3, eps)
        loss = loss + BETA * ratio.detach() * k3
    return (weights * loss).sum(), float((weights * cut).sum())


def grad_of(name, theta, actions, weights, log_mu, adv, eps):
    theta = theta.detach().requires_grad_()
    loss, cut = placement_loss(name, theta, actions, weights, log_mu, adv, eps)
    return torch.autograd.grad(loss, theta)[0], cut


def identity_check(seed=1, batch=256, drift=0.5):
    """Gradients of the four placements on one batch, after the policy has moved
    away from the rollout policy, with and without clipping."""
    g = torch.Generator().manual_seed(seed)
    log_mu = torch.log_softmax(LOG_Q + 0.5 * R / BETA, 0)    # partway to p*
    theta = log_mu + drift * torch.randn(K, generator=g)     # a few updates later
    actions = torch.multinomial(log_mu.exp(), batch, replacement=True, generator=g)
    adv = R[actions] - R[actions].mean()
    weights = torch.full((batch,), 1 / batch)
    out = {}
    for eps, tag in [(float('inf'), 'unclipped'), (EPS, 'clipped')]:
        grads = {n: grad_of(n, theta, actions, weights, log_mu, adv, eps) for n in PLACEMENTS}
        ref = grads['k2_loss'][0]
        out[tag] = {n: dict(relative_difference=float((gr - ref).norm() / ref.norm()), clipped_fraction=cut)
                    for n, (gr, cut) in grads.items()}
    u, c = out['unclipped'], out['clipped']
    assert all(v['relative_difference'] < 1e-12 for v in u.values())
    assert c['ratio_k3_loss']['relative_difference'] < 1e-12
    return out


# Training ---------------------------------------------------------------------

def train(name, eps, steps, inner, lr, batch=None, seed=0, theta0=None):
    """batch=None takes exact expectations under the rollout policy mu; otherwise
    each outer step samples `batch` actions from mu. mu is frozen for `inner`
    updates. Baseline: E_mu[r] (exact) or the leave-one-out batch mean (sampled),
    the same for every placement. No advantage normalization, plain SGD."""
    g = torch.Generator().manual_seed(seed)
    theta = (LOG_Q.clone() if theta0 is None else theta0.clone())
    rec = dict(gap=[gap(theta)], distortion=[], clipped=[])
    for _ in range(steps):
        log_mu = torch.log_softmax(theta, 0).detach()
        if batch is None:
            actions, weights = torch.arange(K), log_mu.exp()
            adv = R - (weights * R).sum()
        else:
            actions = torch.multinomial(log_mu.exp(), batch, replacement=True, generator=g)
            weights = torch.full((batch,), 1 / batch)
            rs = R[actions]
            adv = rs - (rs.sum() - rs) / (batch - 1)
        dist, cuts = [], []
        for _ in range(inner):
            gr, cut = grad_of(name, theta, actions, weights, log_mu, adv, eps)
            free, _ = grad_of(name, theta, actions, weights, log_mu, adv, float('inf'))
            dist.append(float((gr - free).norm() / free.norm()))
            cuts.append(cut)
            theta = theta - lr * gr
        rec['gap'].append(gap(theta))
        rec['distortion'].append(float(np.mean(dist)))
        rec['clipped'].append(float(np.mean(cuts)))
    return rec


def band(runs, key):
    """Median and interquartile range across seeds."""
    a = np.array([r[key] for r in runs])
    return dict(median=np.median(a, 0).tolist(), q25=np.quantile(a, .25, 0).tolist(), q75=np.quantile(a, .75, 0).tolist())


def main(steps=300, inner=4, lr=5.0, batch=128, seeds=16):
    global R, LOG_Q, LOG_STAR
    torch.manual_seed(0)
    identity = identity_check()
    for tag, rows in identity.items():
        print(tag, {n: (f'{v["relative_difference"]:.2e}', round(v['clipped_fraction'], 3)) for n, v in rows.items()})

    arms = ['unclipped', *PLACEMENTS]
    run = lambda arm, **kw: train('k1_reward' if arm == 'unclipped' else arm,
                                  float('inf') if arm == 'unclipped' else EPS, steps, inner, **kw)
    exact = {arm: run(arm, lr=lr) for arm in arms}
    sampled = {arm: [run(arm, lr=lr, batch=batch, seed=s) for s in range(seeds)] for arm in arms}
    for arm in arms:
        e, s = exact[arm], sampled[arm]
        print(f'{arm:14s} exact gap {e["gap"][-1]:.4f} (step 50: {e["gap"][50]:.3f}) clipped {np.mean(e["clipped"]):.3f} '
              f'distortion {np.median(e["distortion"]):.3f} | sampled gap {np.median([x["gap"][-1] for x in s]):.4f} '
              f'clipped {np.mean([x["clipped"] for x in s]):.3f}')

    # Robustness, printed only: two other bandits (exact), and a smaller step (sampled).
    for seed in [1, 2]:
        R, LOG_Q, LOG_STAR = bandit(seed)
        print(f'bandit {seed}, exact:', {arm: round(run(arm, lr=lr)['gap'][-1], 4) for arm in arms})
    R, LOG_Q, LOG_STAR = bandit()
    small = {arm: [run(arm, lr=lr / 5, batch=batch, seed=s) for s in range(seeds)] for arm in arms}
    print(f'lr {lr / 5}, sampled median gap:', {arm: round(float(np.median([x['gap'][-1] for x in v])), 4) for arm, v in small.items()},
          'clipped:', {arm: round(float(np.mean([x['clipped'] for x in v])), 3) for arm, v in small.items()})

    result = dict(
        setup=dict(actions=K, beta=BETA, eps=EPS, steps=steps, inner=inner, lr=lr, batch=batch, seeds=seeds, kl_star=kl(LOG_STAR, LOG_Q),
                   placements=PLACEMENTS),
        identity=identity,
        training=dict(exact=exact, sampled={arm: {k: band(runs, k) for k in ['gap', 'distortion', 'clipped']}
                                            for arm, runs in sampled.items()}))
    path = Path(__file__).resolve().parents[1] / 'static/klviz/clip.json'
    path.write_text(json.dumps(result, separators=(',', ':')))
    print('wrote', path)


if __name__ == '__main__':
    main()
