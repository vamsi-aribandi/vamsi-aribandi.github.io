#!/usr/bin/env python3
"""Data for the two optimization toy figures in the KL estimation post.

For a Gaussian and a two-action policy, computes (1) the variance of the KL
estimates k1 and k3, (2) the per-sample variance of three gradient estimates,
and (3) true KL during minibatch gradient descent with each gradient estimate.

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
    """Minibatch gradient descent from mu (Gaussian) or a (two actions)."""
    rng = np.random.default_rng(seed)
    if family == 'categorical':
        start = np.log(start / (1 - start))   # update the logit theta
    kls = {}
    for method in METHODS:
        theta = np.full(seeds, start, dtype=float)
        history = []
        for step in range(steps + 1):
            if family == 'gaussian':
                history.append(gaussian_kl(theta))
                log_ratio, score = gaussian_samples(theta[:, None], rng.normal(size=(seeds, batch)))
            else:
                history.append(categorical_kl(theta))
                log_a, log_b = -np.logaddexp(0, -theta), -np.logaddexp(0, theta)
                a = np.exp(log_a)[:, None]
                first = rng.random((seeds, batch)) < a   # sampled action 1?
                log_ratio = np.where(first, log_a[:, None], log_b[:, None]) - np.log(.5)
                score = np.where(first, 1 - a, -a)
            gradient = per_sample(log_ratio, score)[2][method].mean(axis=1)
            theta = theta - rate * gradient
        kls[method] = np.array(history)
    return kls


def figure(family, grid, start, rate, steps):
    kl = gaussian_kl if family == 'gaussian' else lambda a: categorical_kl(np.log(a / (1 - a)))
    rows = [dict(x=float(v), kl=float(kl(v)), **{k: float(x) for k, x in variances(family, v).items()}) for v in grid]
    trajectories = descend(family, start, rate, steps)
    training = {m: dict(median=np.median(t, axis=1).tolist(), q25=np.quantile(t, .25, axis=1).tolist(),
                        q75=np.quantile(t, .75, axis=1).tolist()) for m, t in trajectories.items()}
    return dict(start=start, rate=rate, steps=steps, batch=16, seeds=2000, variances=rows, training=training)


if __name__ == '__main__':
    data = dict(
        gaussian=figure('gaussian', np.linspace(.01, 2, 200), start=1.5, rate=.1, steps=60),
        categorical=figure('categorical', np.linspace(.02, .98, 193), start=.05, rate=1., steps=60),
    )
    out = Path(__file__).resolve().parent.parent / 'static/klviz/toys.json'
    out.write_text(json.dumps(data, separators=(',', ':')))
    for name, d in data.items():
        print(name, 'final median KL', {m: round(d['training'][m]['median'][-1], 4) for m in METHODS})
