#!/usr/bin/env python3
"""Floating-point gradient of three exact KL-gradient implementations.

k1 in the reward, k2 as a loss and DeepSeek-V3.2's ratio-weighted k3 all have
the per-token gradient k1 * score in exact arithmetic. This differentiates each
one with autograd in float32, bfloat16 and float16, at one token with
log p - log q = x, and records the coefficient of the score that comes back.

  pip install torch numpy
  python tools/kl-precision.py   # writes static/klviz/precision.json
"""
import json
from pathlib import Path
import numpy as np
import torch

DTYPES = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}


def surrogates(logp, ref_logp):
    """Per-token losses whose gradient with respect to log p should be k1."""
    old_logp = logp.detach()   # on-policy: pi_old = pi_theta at this step
    log_ratio = ref_logp - logp   # log q/p
    k3 = torch.exp(log_ratio) - log_ratio - 1
    return {
        'k1_reward': (logp - ref_logp).detach() * logp,   # REINFORCE with -k1 as the reward
        'k2_loss': 0.5 * (logp - ref_logp)**2,
        'ratio_k3': torch.exp(logp - old_logp) * k3,      # DeepSeek-V3.2
    }


def coefficients(x, dtype):
    """Gradient with respect to log p, for log p/q = x. The likelier side has
    log-probability -0.5, so every input is a valid log-probability."""
    x = np.asarray(x, dtype=np.float64)
    ref = np.where(x < 0, -0.5, -0.5 - x)
    logp = torch.tensor(ref + x, dtype=dtype, requires_grad=True)
    ref_logp = torch.tensor(ref, dtype=dtype)
    # The exact answer for the inputs as rounded to this dtype.
    exact = (logp.detach().double() - ref_logp.double()).numpy()
    out = {}
    for name, loss in surrogates(logp, ref_logp).items():
        (g,) = torch.autograd.grad(loss.sum(), logp)
        out[name] = g.double().numpy()
    return exact, out


if __name__ == '__main__':
    x = np.round(np.arange(-30, 10.001, .1), 1)
    data = dict(x=x.tolist(), dtypes={})
    for name, dtype in DTYPES.items():
        exact, out = coefficients(x, dtype)
        finite = lambda v: [float(a) if np.isfinite(a) else None for a in v]
        data['dtypes'][name] = dict(exact=exact.tolist(), **{k: finite(v) for k, v in out.items()})
        # Where does the ratio-weighted k3 first go wrong by more than 10%?
        bad = ~np.isfinite(out['ratio_k3']) | (np.abs(out['ratio_k3'] - exact) > .1 * np.maximum(np.abs(exact), 1))
        first = x[bad & (x < 0)].max() if (bad & (x < 0)).any() else None
        for k in ['k1_reward', 'k2_loss']:
            assert np.allclose(out[k], exact, rtol=1e-2, atol=1e-2), (name, k)
        print(f'{name}: ratio-weighted k3 off by >10% from log p/q = {first}; '
              f'at -8: {out["ratio_k3"][x == -8][0]:.4g}, at -12: {out["ratio_k3"][x == -12][0]:.4g}, '
              f'at -20: {out["ratio_k3"][x == -20][0]:.4g}')
    path = Path(__file__).resolve().parents[1] / 'static/klviz/precision.json'
    path.write_text(json.dumps(data, separators=(',', ':')))
