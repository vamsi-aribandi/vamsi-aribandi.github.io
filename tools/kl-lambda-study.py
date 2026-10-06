#!/usr/bin/env python3
"""Minimizing KL[p || q] by stochastic gradient descent, with the k_lambda gradient.

A 1,000-action softmax policy p = softmax(theta) starts far from a fixed reference
q and descends KL[p || q] with plain SGD on the logits. Each step samples a batch
x_1..x_B ~ p and uses the full (reward + loss) gradient of
    k_lambda = log(p/q) + lambda (q/p - 1),
which is (log(p/q) + 1 - lambda) s per sample, with s the score. Every lambda gives
an unbiased gradient; lambda changes only its variance. Four choices of lambda:
    0         k1 differentiated exactly
    1         k3, which is also k1 in the reward and k2 as a loss
    value     the lambda that minimizes the variance of the KL estimate
    gradient  the lambda that minimizes the variance of the gradient
Both optimal lambdas are computed exactly from the current p (oracle), and also
estimated from the batch itself (plug-in, printed only). KL and gradient variance
are exact. Runs at several learning rates, with the rollouts' seeds shared.

  pip install numpy
  python tools/kl-lambda-study.py   # writes static/klviz/lambda.json
"""
import json
from pathlib import Path
import numpy as np

K = 1000


def problem(seed=0, spread=1.5, start=2.5):
    g = np.random.default_rng(seed)
    log_q = g.normal(0, spread, K)
    log_q -= np.logaddexp.reduce(log_q)
    theta0 = log_q + g.normal(0, start, K)
    return log_q, theta0


def log_softmax(theta):
    return theta - np.logaddexp.reduce(theta)


def moments(log_p, log_q):
    """KL, the two optimal lambdas, and the exact gradient variance as a function
    of lambda, for one sample. Gradient variance is the trace of the covariance."""
    p = np.exp(log_p)
    l = log_p - log_q
    u = np.exp(np.minimum(log_q - log_p, 700)) - 1
    kl = p @ l
    s2 = 1 - 2 * p + p @ p                      # |e_x - p|^2
    mean_grad = p * (l - kl)
    lam_value = -(p @ (l * u)) / (p @ (u * u))
    lam_grad = 1 + (p @ (l * s2)) / (p @ s2)
    grad_var = lambda lam: p @ ((l + 1 - lam) ** 2 * s2) - mean_grad @ mean_grad
    return kl, lam_value, lam_grad, grad_var


def plug_in(l, u, s2):
    """Both optimal lambdas estimated from one batch."""
    lam_value = -np.cov(l, u)[0, 1] / max(np.var(u, ddof=1), 1e-300)
    lam_grad = 1 + (l @ s2) / s2.sum()
    return lam_value, lam_grad


def run(rule, log_q, theta0, steps, batch, lr, seed):
    g = np.random.default_rng(seed)
    theta = theta0.copy()
    rec = dict(kl=[], grad_var=[], lam=[])
    for _ in range(steps + 1):
        log_p = log_softmax(theta)
        kl, lam_value, lam_grad, grad_var = moments(log_p, log_q)
        p = np.exp(log_p)
        x = g.choice(K, size=batch, p=p)
        l = log_p[x] - log_q[x]
        u = np.exp(np.minimum(-l, 700)) - 1
        s2 = 1 - 2 * p[x] + p @ p
        lam = {'zero': 0.0, 'one': 1.0, 'value': lam_value, 'gradient': lam_grad}.get(rule)
        if lam is None:
            v, gr = plug_in(l, u, s2)
            lam = v if rule == 'value_plugin' else gr
        rec['kl'].append(float(kl))
        rec['grad_var'].append(float(grad_var(lam) / batch))
        rec['lam'].append(float(lam))
        # Batch mean of (l + 1 - lambda)(e_x - p).
        coef = (l + 1 - lam) / batch
        grad = -coef.sum() * p
        np.add.at(grad, x, coef)
        theta = theta - lr * grad
    return rec


def band(runs, key, every):
    """Median and interquartile range across seeds, every `every` steps."""
    a = np.array([r[key] for r in runs])[:, ::every]
    r = lambda v: [float(f'{x:.4g}') for x in v]
    return dict(median=r(np.median(a, 0)), q25=r(np.quantile(a, .25, 0)), q75=r(np.quantile(a, .75, 0)))


def main(steps=400, batch=16, seeds=32, rates=(1, 5, 10, 15), every=2):
    log_q, theta0 = problem()
    kl0, lv, lg, _ = moments(log_softmax(theta0), log_q)
    print(f'start: KL {kl0:.3f}, lambda value {lv:.3f}, lambda gradient {lg:.3f}')
    shown = ['zero', 'one', 'value', 'gradient']
    rules = shown + ['value_plugin', 'gradient_plugin']      # plug-in rules are printed only
    curves = {}
    for rate in rates:
        runs = {r: [run(r, log_q, theta0, steps, batch, rate, s) for s in range(seeds)] for r in rules}
        print(f'lr {rate}: median KL at step 100/{steps}',
              {r: [round(float(np.median([x['kl'][t] for x in rs])), 3) for t in (100, steps)] for r, rs in runs.items()})
        curves[str(rate)] = {r: {k: band(runs[r], k, every) for k in ['kl', 'grad_var', 'lam']} for r in shown}
    result = dict(setup=dict(actions=K, steps=steps, batch=batch, seeds=seeds, kl0=float(kl0), rates=list(rates), every=every),
                  curves=curves)
    path = Path(__file__).resolve().parents[1] / 'static/klviz/lambda.json'
    path.write_text(json.dumps(result, separators=(',', ':')))
    print('wrote', path)


if __name__ == '__main__':
    main()
