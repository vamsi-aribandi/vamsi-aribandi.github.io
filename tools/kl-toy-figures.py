#!/usr/bin/env python3
"""Data for the toy figures in the KL estimation post.

For random categorical pairs, compares how well k3 estimates KL with how well
its gradients estimate the gradient of the reverse KL, both relative to k1.
For a Gaussian and a two-action policy, computes (1) the variance of the KL
estimates k1 and k3, (2) the per-sample variance of three gradient estimates,
and (3) true KL during minibatch gradient descent with each gradient estimate.
For a one-Gaussian policy fit to a two-mode reference, records both KL
directions during minibatch gradient descent.

  python tools/kl-toy-figures.py   # writes static/klviz/toys.json
"""
import json
from pathlib import Path
import numpy as np

METHODS = ['k1_reward', 'k3_reward', 'k3_loss']


def per_sample(log_ratio, score):
    """Per-sample KL estimates and gradient estimates, given log p/q and score."""
    k1 = log_ratio
    k3 = log_ratio + np.expm1(-log_ratio)
    gradients = {
        'k1_reward': k1 * score,                 # k1 as a detached reward
        'k3_reward': k3 * score,                 # k3 as a detached reward
        'k3_loss': -np.expm1(-log_ratio) * score,  # d k3 / d theta, sample fixed
    }
    # Both halves of the k3 gradient together equal k1 in the reward, sample by sample.
    # (The tolerance allows for rounding where the two halves are huge and cancel.)
    with np.errstate(invalid='ignore', over='ignore'):
        halves = np.abs(gradients['k3_reward']) + np.abs(gradients['k3_loss'])
        error = np.abs(gradients['k3_reward'] + gradients['k3_loss'] - gradients['k1_reward'])
        finite = np.isfinite(halves)
        assert np.all(error[finite] <= 1e-9 * halves[finite] + 1e-12)
    return k1, k3, gradients


# Gaussian: p = N(mu, 1), q = N(0, 1), learn mu.
def gaussian_samples(mu, eps):
    x = mu + eps
    return mu * x - mu**2 / 2, x - mu   # log p/q, d log p / d mu


def gaussian_kl(mu):
    return mu**2 / 2


# Two actions: p = (a, 1 - a), a = sigmoid(theta), q = (0.5, 0.5), learn theta.
def categorical_outcomes(a):
    p = np.array([a, 1 - a])
    return p, np.log(p / .5), np.array([1 - a, -a])   # probabilities, log p/q, score


def categorical_kl(theta):
    # Stable in the logit: log a = -log(1 + e^-theta), log(1 - a) = -log(1 + e^theta).
    log_a, log_b = -np.logaddexp(0, -theta), -np.logaddexp(0, theta)
    return np.exp(log_a) * (log_a - np.log(.5)) + np.exp(log_b) * (log_b - np.log(.5))


def variances(family, value):
    """Exact variances under p (Gauss-Hermite quadrature or a sum over actions)."""
    if family == 'gaussian':
        eps, w = np.polynomial.hermite_e.hermegauss(160)
        w = w / w.sum()
        k1, k3, g = per_sample(*gaussian_samples(value, eps))
    else:
        w, log_ratio, score = categorical_outcomes(value)
        k1, k3, g = per_sample(log_ratio, score)
    var = lambda v: w @ (v - w @ v)**2
    return dict(k1=var(k1), k3=var(k3), **{m: var(g[m]) for m in METHODS})


def descend(family, start, rate, steps, batch=16, seeds=2000, seed=0):
    """Minibatch gradient descent from mu (Gaussian) or a (two actions).

    Returns, for each method, the true KL and the policy (mu or a) of every
    run at every step.
    """
    rng = np.random.default_rng(seed)
    if family == 'categorical':
        start = np.log(start / (1 - start))   # update the logit theta
    kls, policies = {}, {}
    for method in METHODS:
        theta = np.full(seeds, start, dtype=float)
        history, policy = [], []
        for step in range(steps + 1):
            if family == 'gaussian':
                history.append(gaussian_kl(theta))
                policy.append(theta)
                log_ratio, score = gaussian_samples(theta[:, None], rng.normal(size=(seeds, batch)))
            else:
                history.append(categorical_kl(theta))
                log_a, log_b = -np.logaddexp(0, -theta), -np.logaddexp(0, theta)
                policy.append(np.exp(log_a))
                a = np.exp(log_a)[:, None]
                first = rng.random((seeds, batch)) < a   # sampled action 1?
                log_ratio = np.where(first, log_a[:, None], log_b[:, None]) - np.log(.5)
                score = np.where(first, 1 - a, -a)
            gradient = per_sample(log_ratio, score)[2][method].mean(axis=1)
            theta = theta - rate * gradient
        kls[method], policies[method] = np.array(history), np.array(policy)
    return kls, policies


# Two modes: q = mixture of N(-2, 0.5^2) and N(2, 0.5^2); p = N(mu, sigma^2).
# Learn mu and log sigma. p has one mode, so it cannot match q.
def log_normal(x, mean, sd):
    return -0.5 * ((x - mean) / sd)**2 - np.log(sd) - 0.5 * np.log(2 * np.pi)


def log_q(x):
    return np.logaddexp(log_normal(x, -2, .5), log_normal(x, 2, .5)) + np.log(.5)


def bimodal_kls(mu, sigma):
    """Both KL directions by numerical integration on a grid."""
    x = np.linspace(-12, 12, 4001)
    dx, lq = x[1] - x[0], log_q(x)
    lp = log_normal(x, mu[:, None], sigma[:, None])
    reverse = (np.exp(lp) * (lp - lq)).sum(axis=1) * dx   # KL[p || q]
    forward = (np.exp(lq) * (lq - lp)).sum(axis=1) * dx   # KL[q || p]
    return reverse, forward


def bimodal_descend(rate=.05, steps=300, batch=64, seeds=1000, every=2, seed=0):
    rng = np.random.default_rng(seed)
    results = {}
    for method in METHODS:
        mu, log_sigma = np.full(seeds, 1.0), np.zeros(seeds)
        diverged = np.zeros(seeds, dtype=bool)
        history = []
        for step in range(steps + 1):
            if step % every == 0:
                reverse, forward = bimodal_kls(mu, np.exp(log_sigma))
                history.append((mu.copy(), np.exp(log_sigma), np.where(diverged, np.inf, reverse),
                                np.where(diverged, np.inf, forward), diverged.mean()))
            if step == steps:
                break
            sigma = np.exp(log_sigma)[:, None]
            eps = rng.normal(size=(seeds, batch))
            x = mu[:, None] + sigma * eps
            log_ratio = log_normal(x, mu[:, None], sigma) - log_q(x)
            score_mu, score_log_sigma = eps / sigma, eps**2 - 1
            with np.errstate(over='ignore', invalid='ignore'):
                g = per_sample(log_ratio, score_mu)[2][method].mean(axis=1)
                h = per_sample(log_ratio, score_log_sigma)[2][method].mean(axis=1)
                mu, log_sigma = mu - rate * g, log_sigma - rate * h
            # Freeze runs that leave a sane range; they count as infinite KL.
            diverged |= ~np.isfinite(mu) | ~np.isfinite(log_sigma) | (np.abs(mu) > 100) | (np.abs(log_sigma) > 5)
            mu, log_sigma = np.where(diverged, 1.0, mu), np.where(diverged, 0.0, log_sigma)
        results[method] = history
    return results


def random_pairs(count=40000, batch=16, bins=np.logspace(-4, .25, 18), seed=0):
    """Random categorical pairs: mean squared error of a batch-average KL
    estimate, and of a batch-average gradient with respect to the logits of p,
    each relative to k1. k1 is unbiased for both, so its error is variance / batch.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(count):
        size = rng.integers(2, 11)
        q = rng.dirichlet(np.ones(size))
        logits = np.log(q) + np.exp(rng.uniform(np.log(.005), np.log(2))) * rng.normal(size=size)
        p = np.exp(logits - logits.max()); p /= p.sum()
        log_ratio, score = np.log(p / q), np.eye(size) - p   # row x: d log p(x) / d logits
        k1, k3, g = per_sample(log_ratio[:, None], score)   # one row per outcome x
        k1, k3 = k1.ravel(), k3.ravel()
        kl = p @ k1
        true = p @ (k1[:, None] * score)
        def mse(v, target):
            mean = p @ v
            return ((mean - target)**2).sum() + (p @ ((v - mean)**2).reshape(size, -1)).sum() / batch
        base = mse(g['k1_reward'], true)
        rows.append([kl, mse(k3, kl) / mse(k1, kl), mse(g['k3_loss'], true) / base])
    rows = np.array(rows)
    out = dict(count=count, batch=batch, bins=[])
    for lo, hi in zip(bins[:-1], bins[1:]):
        r = rows[(rows[:, 0] >= lo) & (rows[:, 0] < hi)]
        quart = lambda c: [float(v) for v in np.quantile(r[:, c], [.25, .5, .75])]
        out['bins'].append(dict(x=float(np.sqrt(lo * hi)), lower=float(lo), upper=float(hi), count=len(r),
                                estimate=quart(1), k3_loss=quart(2)))
    return out


def figure(family, grid, start, rate, steps):
    kl = gaussian_kl if family == 'gaussian' else lambda a: categorical_kl(np.log(a / (1 - a)))
    rows = [dict(x=float(v), kl=float(kl(v)), **{k: float(x) for k, x in variances(family, v).items()}) for v in grid]
    kls, policies = descend(family, start, rate, steps)
    training = {}
    for m, t in kls.items():
        # The median run at each step, so its policy can be drawn in hover cards.
        middle = np.argsort(t, axis=1)[:, t.shape[1] // 2]
        steps_ = np.arange(len(t))
        training[m] = dict(median=t[steps_, middle].tolist(), policy=policies[m][steps_, middle].tolist(),
                           q25=np.quantile(t, .25, axis=1).tolist(), q75=np.quantile(t, .75, axis=1).tolist())
    return dict(start=start, rate=rate, steps=steps, batch=16, seeds=2000, variances=rows, training=training)


def bimodal_figure(rate=.05, steps=300, batch=64, seeds=1000, every=2):
    finite = lambda v: [None if not np.isfinite(x) else float(x) for x in v]
    training = {}
    for method, history in bimodal_descend(rate, steps, batch, seeds, every).items():
        out = dict(reverse=[], forward=[], rq25=[], rq75=[], fq25=[], fq75=[], mu=[], sigma=[], diverged=[])
        for mu, sigma, reverse, forward, diverged in history:
            middle = np.argsort(reverse)[len(reverse) // 2]   # median run by KL[p || q]
            out['reverse'].append(reverse[middle]); out['forward'].append(forward[middle])
            out['mu'].append(float(mu[middle])); out['sigma'].append(float(sigma[middle]))
            for key, v in [('rq', reverse), ('fq', forward)]:
                out[key + '25'].append(np.quantile(v, .25, method='nearest'))
                out[key + '75'].append(np.quantile(v, .75, method='nearest'))
            out['diverged'].append(float(diverged))
        training[method] = {k: finite(v) if k not in ('mu', 'sigma', 'diverged') else v for k, v in out.items()}
    return dict(start=dict(mu=1.0, sigma=1.0), rate=rate, steps=steps, every=every, batch=batch, seeds=seeds, training=training)


if __name__ == '__main__':
    data = dict(
        pairs=random_pairs(),
        bimodal=bimodal_figure(),
        gaussian=figure('gaussian', np.linspace(.01, 2, 200), start=1.5, rate=.1, steps=60),
        categorical=figure('categorical', np.linspace(.02, .98, 193), start=.05, rate=1., steps=60),
    )
    out = Path(__file__).resolve().parent.parent / 'static/klviz/toys.json'
    out.write_text(json.dumps(data, separators=(',', ':')))
    for b in data['pairs']['bins']:
        print('pairs KL %.4g n=%d  median ratio: estimate %.3g  k3 loss %.3g' % (
            b['x'], b['count'], b['estimate'][1], b['k3_loss'][1]))
    for name in ['gaussian', 'categorical']:
        print(name, 'final median KL', {m: round(data[name]['training'][m]['median'][-1], 4) for m in METHODS})
    for m, t in data['bimodal']['training'].items():
        print('bimodal', m, 'median run: mu', round(t['mu'][-1], 3), 'sigma', round(t['sigma'][-1], 3),
              'KL[p||q]', t['reverse'][-1], 'KL[q||p]', t['forward'][-1], 'diverged', t['diverged'][-1])
