---
title: "The Structure Behind KL Estimation for Reinforcement Learning"
date: 2026-09-30
draft: true
toc: false
klviz: true
summary: "The best KL estimator depends on whether you measure KL or train on it. Why on-policy distillation puts k1 in the reward."
---

KL divergence {{< klmath inline=true >}}\mathrm{KL}[p\,\|\,q] = \mathbb{E}_p[\log(p/q)]{{< /klmath >}} is an important quantity in AI. In reinforcement learning for language models, {{< klmath inline=true >}}\mathrm{KL}[\pi_\theta\,\|\,\pi_\mathrm{ref}]{{< /klmath >}} is used as a regularizer to keep a learned policy {{< klmath inline=true >}}\pi_\theta{{< /klmath >}} near a reference policy {{< klmath inline=true >}}\pi_\mathrm{ref}{{< /klmath >}}, and on-policy distillation (OPD) lets a student policy {{< klmath inline=true >}}\pi_student{{< /klmath >}} learn from a teacher policy {{< klmath inline=true >}}\pi_teacher{{< /klmath >}} by minimizing {{< klmath inline=true >}}\mathrm{KL}[\pi_\mathrm{student}\,\|\,\pi_\mathrm{teacher}]{{< /klmath >}}.

Computing KL between language models is generally intractable, because the expectation {{< klmath inline=true >}}\mathbb{E}_p{{< /klmath >}} is a sum over every possible sequence of tokens, which is intractable. Instead we *estimate* it from samples {{< klmath inline=true >}}x \sim p{{< /klmath >}}. Three estimators of KL are common:

{{< klmath >}}
\begin{aligned}
k_1(x) &= \log\frac{p(x)}{q(x)}\\
k_2(x) &= \frac12\left(\log\frac{p(x)}{q(x)}\right)^2\\
k_3(x) &= \log\frac{p(x)}{q(x)} + \frac{q(x)}{p(x)} - 1
\end{aligned}
{{< /klmath >}}

You might recall these estimators from a note by John Schulman[^schulman]. It shows that {{< klmath inline=true >}}k_3{{< /klmath >}} in particular is unbiased like {{< klmath inline=true >}}k_1{{< /klmath >}} and never negative like {{< klmath inline=true >}}k_2{{< /klmath >}}, with lower variance.

However, which estimator to use is far from converged. DeepSeek[^deepseek] uses {{< klmath inline=true >}}k_3{{< /klmath >}} as a surrogate loss for RLVR. Thinking Machines uses {{< klmath inline=true >}}-k_1{{< /klmath >}} as a reward for on-policy distillation[^opd], and Cursor's Composer 2 uses {{< klmath inline=true >}}k_1{{< /klmath >}} as a regularizer.

Which estimator is optimal for learning reverse KL {{< klmath inline=true >}}\min_p\mathrm{KL}[p\,\|\,q]{{< /klmath >}}?

This is best answered by answering two questions: which estimator gives a better *estimate* of KL, and which gives a better *gradient* when we train on it?

TL;DR:
- {{< klmath inline=true >}}k_1{{< /klmath >}} **in the reward** gives exactly the gradient of the reverse KL, {{< klmath inline=true >}}\mathrm{KL}[p\,\|\,q]{{< /klmath >}}. This is what OPD does.
- {{< klmath inline=true >}}k_3{{< /klmath >}} **in the reward** gives the gradient of neither KL. It can vanish, or point the wrong way.
- {{< klmath inline=true >}}k_3{{< /klmath >}} **as a loss** gives the gradient of the forward KL, {{< klmath inline=true >}}\mathrm{KL}[q\,\|\,p]{{< /klmath >}}. It converges, to something else.

## Estimating the value of KL

It's easiest to compare {{< klmath inline=true >}}k_1{{< /klmath >}} and {{< klmath inline=true >}}k_3{{< /klmath >}} by observing that {{< klmath inline=true >}}k_3{{< /klmath >}} adds one term to {{< klmath inline=true >}}k_1{{< /klmath >}}:

{{< klmath >}}
k_3 = k_1 + \left(\frac{q}{p} - 1\right).
{{< /klmath >}}

Notice that {{< klmath inline=true >}}\mathbb{E}_p[(q/p)-1] = \sum_x p(x)\,\frac{q(x)}{p(x)} - 1 = \sum_x q(x) - 1 = 0{{< /klmath >}}. So {{< klmath inline=true >}}k_3{{< /klmath >}} is {{< klmath inline=true >}}k_1{{< /klmath >}} plus a *control variate* -- it makes an estimator better when it cancels noise, and worse when it adds noise.

For {{< klmath inline=true >}}p = \mathcal{N}(\mu, 1){{< /klmath >}} and {{< klmath inline=true >}}q = \mathcal{N}(0, 1){{< /klmath >}}, as we vary {{< klmath inline=true >}}\mu{{< /klmath >}},
{{< klmath inline=true >}}k_3{{< /klmath >}} has lower variance at small KL and much higher variance at large KL:

{{< klfigure type="gaussian" subtitle=`$p = \mathcal{N}(\mu, 1)$ · $q = \mathcal{N}(0, 1)$` >}}{{< /klfigure >}}

{{< klcode title="Gaussian example" >}}
import numpy as np

rng = np.random.default_rng(0)

# p = N(mu, 1), q = N(0, 1)
# KL[p || q] = mu^2 / 2
mus = np.linspace(0.01, 2.5, 60)
n = 500_000
eps = rng.normal(size=n)

kls = []
k1_vars = []
k3_vars = []

for mu in mus:
    x = mu + eps  # Sample from p = N(mu, 1).
    # log p(x) - log q(x)
    log_ratio = mu * x - 0.5 * mu**2
    k1 = log_ratio
    k3 = log_ratio + np.expm1(-log_ratio)

    kls.append(0.5 * mu**2)
    k1_vars.append(np.var(k1))
    k3_vars.append(np.var(k3))

# The JavaScript curves use the exact variances of these same estimators.
exact_k1_vars = mus**2
exact_k3_vars = np.expm1(mus**2) - mus**2
print(np.column_stack([kls, k1_vars, k3_vars]))
{{< /klcode >}}

The size of the KL is not the whole story. Shape matters too. The worst case for {{< klmath inline=true >}}k_3{{< /klmath >}} an outcome that is likely under {{< klmath inline=true >}}q{{< /klmath >}} likes but not under {{< klmath inline=true >}}p{{< /klmath >}}. For those samples, {{< klmath inline=true >}}q/p{{< /klmath >}} explodes.

To see this, fix {{< klmath inline=true >}}q = (0.2, 0.4, 0.4){{< /klmath >}} and compare two ways to move away from it with exactly the same KL:

{{< klmath >}}
\begin{aligned}
p_1(h) &= \left(a,\ \tfrac{1-a}{2},\ \tfrac{1-a}{2}\right), \quad a = 0.2\times 10^{-h} &&\text{(a hole)}\\
p_2(h) &= \left(0.2,\ 0.4+\delta,\ 0.4-\delta\right) &&\text{(a smooth shift)}
\end{aligned}
{{< /klmath >}}

Each step in {{< klmath inline=true >}}h{{< /klmath >}} makes the first outcome ten times rarer under {{< klmath inline=true >}}p_1{{< /klmath >}}, and {{< klmath inline=true >}}\delta(h){{< /klmath >}} is chosen so that the two KLs are equal at every {{< klmath inline=true >}}h{{< /klmath >}}:

{{< klmath >}}
\mathrm{KL}[p_1(h)\,\|\,q] = \mathrm{KL}[p_2(h)\,\|\,q] = a\log\frac{a}{0.2} + (1-a)\log\frac{1-a}{0.8}.
{{< /klmath >}}

{{< klfigure type="holeness" >}}
Exact variances, logarithmic vertical axis. Solid lines use the hole {{< klmath inline=true >}}p_1{{< /klmath >}}; dashed lines use the smooth shift {{< klmath inline=true >}}p_2{{< /klmath >}}. Hover to see the distributions.
{{< /klfigure >}}

At the same KL, the hole drives the variance of {{< klmath inline=true >}}k_3{{< /klmath >}} up without bound, while the smooth shift keeps it small. Which estimator is better depends on how far {{< klmath inline=true >}}p{{< /klmath >}} is from {{< klmath inline=true >}}q{{< /klmath >}}, and on the shape of the difference.

{{< klcode title="Vary the hole depth and match the KLs" >}}
import numpy as np

q = np.array([0.2, 0.4, 0.4])


def exact_kl(p, q):
    return np.sum(p * np.log(p / q))


def matched_distributions(h):
    a = 0.2 * 10.0**(-h)
    p1 = np.array([a, (1 - a) / 2, (1 - a) / 2])
    target = exact_kl(p1, q)

    # KL([0.2, 0.4 + delta, 0.4 - delta] || q)
    # increases monotonically with delta >= 0.
    lo, hi = 0.0, 0.4 - 1e-12
    for _ in range(70):
        delta = (lo + hi) / 2
        candidate = np.array([0.2, 0.4 + delta, 0.4 - delta])
        if exact_kl(candidate, q) < target:
            lo = delta
        else:
            hi = delta
    delta = 0.0 if h == 0 else (lo + hi) / 2
    p2 = np.array([0.2, 0.4 + delta, 0.4 - delta])
    return p1, p2


def exact_variances(p, q):
    z = np.log(p / q)
    k1 = z
    k3 = z + np.expm1(-z)
    mean = exact_kl(p, q)
    return np.array([
        np.sum(p * (k1 - mean)**2),
        np.sum(p * (k3 - mean)**2),
    ])


# Inspect one setting of the hole depth.
h = 2.0
p1, p2 = matched_distributions(h)
print("p1:", p1)
print("p2:", p2)
print("KL[p1 || q]:", exact_kl(p1, q))
print("KL[p2 || q]:", exact_kl(p2, q))
print("Var_p1(k1), Var_p1(k3):", exact_variances(p1, q))
print("Var_p2(k1), Var_p2(k3):", exact_variances(p2, q))

# Columns match the four JavaScript curves.
# h=0 has zero variance and is omitted from the logarithmic plot.
hs = np.linspace(0.01, 3.5, 350)
variances = []
for h in hs:
    p1, p2 = matched_distributions(h)
    assert np.isclose(exact_kl(p1, q), exact_kl(p2, q), atol=1e-12)
    variances.append(np.concatenate([
        exact_variances(p1, q), exact_variances(p2, q)
    ]))
variances = np.array(variances)
{{< /klcode >}}

## Which estimator gives the best gradient?

In RL, the KL estimate is not the end product. We differentiate it. There are two common ways, and they keep different halves of the same derivative:

{{< klmath >}}
\nabla_\theta\,\mathbb{E}_{p_\theta}[k(x)]
=
\underbrace{\mathbb{E}_{p_\theta}\!\left[k(x)\,\nabla_\theta\log p_\theta(x)\right]}_{\text{in the reward}}
+
\underbrace{\mathbb{E}_{p_\theta}\!\left[\nabla_\theta k(x)\right]}_{\text{as a loss}}.
{{< /klmath >}}

{{< klderiv title="Derive the split" >}}
Write the expectation as a sum and differentiate both factors:

{{< klmath >}}
\nabla_\theta \sum_x p_\theta(x)\,k(x) = \sum_x \nabla_\theta p_\theta(x)\,k(x) + \sum_x p_\theta(x)\,\nabla_\theta k(x).
{{< /klmath >}}

Then use {{< klmath inline=true >}}\nabla_\theta p_\theta = p_\theta\,\nabla_\theta \log p_\theta{{< /klmath >}} in the first sum. Both sums become expectations under {{< klmath inline=true >}}p_\theta{{< /klmath >}}. For continuous {{< klmath inline=true >}}x{{< /klmath >}}, replace the sums with integrals.
{{< /klderiv >}}

- **In the reward:** treat {{< klmath inline=true >}}-k(x){{< /klmath >}} as a detached reward and take the policy gradient. Only the first term survives.
- **As a loss:** backpropagate through {{< klmath inline=true >}}k(x){{< /klmath >}} with the sample {{< klmath inline=true >}}x{{< /klmath >}} held fixed. Only the second term survives.

The KL we estimate, and want to minimize, is the **reverse KL**, {{< klmath inline=true >}}\mathrm{KL}[p_\theta\,\|\,q]{{< /klmath >}}: it averages over the policy's own samples. The **forward KL**, {{< klmath inline=true >}}\mathrm{KL}[q\,\|\,p_\theta]{{< /klmath >}}, averages over {{< klmath inline=true >}}q{{< /klmath >}} instead. Three facts finish the calculation:

{{< klmath >}}
\begin{aligned}
&\mathbb{E}_p[\nabla\log p] = 0,\\
&\mathbb{E}_p\!\left[\tfrac{q}{p}\,\nabla\log p\right] = -\nabla\,\mathrm{KL}[q\,\|\,p],\\
&\nabla k_1 = \nabla\log p, \qquad \nabla k_3 = \left(1-\tfrac{q}{p}\right)\nabla\log p.
\end{aligned}
{{< /klmath >}}

{{< klderiv title="Derive the three facts" >}}
**The score has mean zero.** {{< klmath inline=true >}}\mathbb{E}_p[\nabla\log p] = \sum_x p\,\frac{\nabla p}{p} = \nabla \sum_x p = \nabla 1 = 0.{{< /klmath >}}

**The ratio turns {{< klmath inline=true >}}p{{< /klmath >}} into {{< klmath inline=true >}}q{{< /klmath >}}.** {{< klmath inline=true >}}\mathbb{E}_p[\tfrac{q}{p}\nabla\log p] = \sum_x q\,\nabla\log p{{< /klmath >}}. The forward KL is {{< klmath inline=true >}}\mathrm{KL}[q\,\|\,p] = \sum_x q\log q - \sum_x q\log p{{< /klmath >}}, and only the second sum depends on {{< klmath inline=true >}}\theta{{< /klmath >}}, so {{< klmath inline=true >}}\nabla\,\mathrm{KL}[q\,\|\,p] = -\sum_x q\,\nabla\log p{{< /klmath >}}.

**Gradients at a fixed sample.** With {{< klmath inline=true >}}x{{< /klmath >}} fixed, {{< klmath inline=true >}}\log q(x){{< /klmath >}} does not depend on {{< klmath inline=true >}}\theta{{< /klmath >}}, so {{< klmath inline=true >}}\nabla k_1 = \nabla\log p{{< /klmath >}}. For the ratio, {{< klmath inline=true >}}\nabla(q/p) = -(q/p)\,\nabla\log p{{< /klmath >}}, so {{< klmath inline=true >}}\nabla k_3 = \nabla\log p - (q/p)\,\nabla\log p{{< /klmath >}}.
{{< /klderiv >}}

Plug them in:

| | In the reward | As a loss |
| --- | --- | --- |
| {{< klmath inline=true >}}k_1{{< /klmath >}} | {{< klmath inline=true >}}\nabla\,\mathrm{KL}[p\,\Vert\,q]{{< /klmath >}} (reverse) | {{< klmath inline=true >}}0{{< /klmath >}} |
| {{< klmath inline=true >}}k_3{{< /klmath >}} | {{< klmath inline=true >}}\nabla\,\mathrm{KL}[p\,\Vert\,q] - \nabla\,\mathrm{KL}[q\,\Vert\,p]{{< /klmath >}} | {{< klmath inline=true >}}\nabla\,\mathrm{KL}[q\,\Vert\,p]{{< /klmath >}} (forward) |

{{< klderiv title="Derive the table" >}}
**{{< klmath inline=true >}}k_1{{< /klmath >}} as a loss:** {{< klmath inline=true >}}\mathbb{E}[\nabla k_1] = \mathbb{E}[\nabla\log p] = 0{{< /klmath >}}.

**{{< klmath inline=true >}}k_1{{< /klmath >}} in the reward:** {{< klmath inline=true >}}\mathbb{E}_p[k_1] = \mathrm{KL}[p\,\|\,q]{{< /klmath >}}, so the split gives {{< klmath inline=true >}}\nabla\,\mathrm{KL}[p\,\|\,q] = \mathbb{E}[k_1\nabla\log p] + \mathbb{E}[\nabla k_1] = \mathbb{E}[k_1\nabla\log p] + 0{{< /klmath >}}.

**{{< klmath inline=true >}}k_3{{< /klmath >}} as a loss:** {{< klmath inline=true >}}\mathbb{E}[\nabla k_3] = \mathbb{E}[\nabla\log p] - \mathbb{E}[\tfrac{q}{p}\nabla\log p] = 0 + \nabla\,\mathrm{KL}[q\,\|\,p]{{< /klmath >}}.

**{{< klmath inline=true >}}k_3{{< /klmath >}} in the reward:** {{< klmath inline=true >}}\mathbb{E}[k_3\nabla\log p] = \mathbb{E}[k_1\nabla\log p] + \mathbb{E}[\tfrac{q}{p}\nabla\log p] - \mathbb{E}[\nabla\log p] = \nabla\,\mathrm{KL}[p\,\|\,q] - \nabla\,\mathrm{KL}[q\,\|\,p]{{< /klmath >}}.

As a check, the two {{< klmath inline=true >}}k_3{{< /klmath >}} cells add up to {{< klmath inline=true >}}\nabla\,\mathrm{KL}[p\,\|\,q]{{< /klmath >}}, as the split requires.
{{< /klderiv >}}

Only one cell is the gradient we asked for.

The {{< klmath inline=true >}}k_3{{< /klmath >}}-in-reward cell subtracts two gradients. Near {{< klmath inline=true >}}q{{< /klmath >}}, the reverse and forward KL agree to second order, so the difference nearly cancels: the penalty has almost no restoring force. Far from {{< klmath inline=true >}}q{{< /klmath >}}, what is left can point anywhere.

{{< klderiv title="Why the two KLs agree near q" >}}
Write {{< klmath inline=true >}}p = q\,(1+\varepsilon){{< /klmath >}}, with {{< klmath inline=true >}}\sum_x q\,\varepsilon = 0{{< /klmath >}} so that {{< klmath inline=true >}}p{{< /klmath >}} sums to 1. To second order in {{< klmath inline=true >}}\varepsilon{{< /klmath >}}:

{{< klmath >}}
\begin{aligned}
\mathrm{KL}[p\,\|\,q] &= \textstyle\sum_x q\,(1+\varepsilon)\log(1+\varepsilon) \approx \sum_x q\left(\varepsilon + \tfrac{\varepsilon^2}{2}\right) = \tfrac12\sum_x q\,\varepsilon^2,\\
\mathrm{KL}[q\,\|\,p] &= \textstyle -\sum_x q\log(1+\varepsilon) \approx \sum_x q\left(-\varepsilon + \tfrac{\varepsilon^2}{2}\right) = \tfrac12\sum_x q\,\varepsilon^2.
\end{aligned}
{{< /klmath >}}

They first differ at third order in {{< klmath inline=true >}}\varepsilon{{< /klmath >}}.
{{< /klderiv >}}

The {{< klmath inline=true >}}k_3{{< /klmath >}}-as-loss cell is a real gradient, just of the forward KL. Near {{< klmath inline=true >}}q{{< /klmath >}}, that barely matters. Far from {{< klmath inline=true >}}q{{< /klmath >}}, the two directions can prefer very different policies.

## Three small worlds

Each toy compares the three methods: {{< klmath inline=true >}}k_1{{< /klmath >}} in the reward, {{< klmath inline=true >}}k_3{{< /klmath >}} in the reward, and {{< klmath inline=true >}}k_3{{< /klmath >}} as a loss. The first two figures have three plots each:

1. the variance of the KL estimate, along a family of policies;
2. the variance of one sample's gradient, which is what a minibatch averages;
3. the true reverse KL during minibatch gradient descent, as the median of 2,000 runs, with the middle half shaded.

Hover over any plot to see the distributions.

### Gaussian: {{< klmath inline=true >}}k_3{{< /klmath >}} in the reward has no signal

Let {{< klmath inline=true >}}p_\mu = \mathcal{N}(\mu, 1){{< /klmath >}} and {{< klmath inline=true >}}q = \mathcal{N}(0, 1){{< /klmath >}}. Learn {{< klmath inline=true >}}\mu{{< /klmath >}} from {{< klmath inline=true >}}\mu_0 = 1.5{{< /klmath >}}.

{{< klfigure type="gaussianToy" data="klviz/toys.json" subtitle=`$p_\mu = \mathcal{N}(\mu, 1)$ · $q = \mathcal{N}(0, 1)$ · start $\mu_0 = 1.5$ · batch 16 · learning rate 0.1` >}}
First: variance of {{< klmath inline=true >}}k_1{{< /klmath >}} and {{< klmath inline=true >}}k_3{{< /klmath >}} as KL estimates. Second: per-sample variance of the gradient with respect to {{< klmath inline=true >}}\mu{{< /klmath >}}. Third: true KL during minibatch gradient descent.
{{< /klfigure >}}

Here the two directions have the same gradient: {{< klmath inline=true >}}\nabla\,\mathrm{KL}[p\,\|\,q] = \nabla\,\mathrm{KL}[q\,\|\,p] = \mu{{< /klmath >}}. So {{< klmath inline=true >}}k_3{{< /klmath >}} in the reward gets exactly zero, at every {{< klmath inline=true >}}\mu{{< /klmath >}}. The policy receives only noise and wanders.

{{< klderiv title="Why both gradients equal μ" >}}
For Gaussians with equal variance, both directions give the same KL:

{{< klmath >}}
\mathrm{KL}[\mathcal{N}(\mu,1)\,\|\,\mathcal{N}(0,1)] = \mathrm{KL}[\mathcal{N}(0,1)\,\|\,\mathcal{N}(\mu,1)] = \frac{\mu^2}{2}.
{{< /klmath >}}

Both have derivative {{< klmath inline=true >}}\mu{{< /klmath >}}, so their difference, the {{< klmath inline=true >}}k_3{{< /klmath >}}-in-reward gradient, is zero.
{{< /klderiv >}}

{{< klmath inline=true >}}k_3{{< /klmath >}} as a loss is correct on average. But its gradient starts out 15 times noisier than {{< klmath inline=true >}}k_1{{< /klmath >}}'s (87 versus 5.8), even though {{< klmath inline=true >}}k_3{{< /klmath >}} is only about 3 times noisier as a KL estimate. The {{< klmath inline=true >}}q/p{{< /klmath >}} in {{< klmath inline=true >}}\nabla k_3{{< /klmath >}} brings the tail back. With batches of 16, the median run keeps pace; the noise shows up as a wider spread.

### Two actions: {{< klmath inline=true >}}k_3{{< /klmath >}} in the reward pushes the wrong way

Let {{< klmath inline=true >}}p_\theta = (a, 1-a){{< /klmath >}} with {{< klmath inline=true >}}a = \sigma(\theta){{< /klmath >}}, and {{< klmath inline=true >}}q = (0.5, 0.5){{< /klmath >}}. Learn the logit {{< klmath inline=true >}}\theta{{< /klmath >}} from {{< klmath inline=true >}}a_0 = 0.05{{< /klmath >}}, so the first action is a hole.

{{< klfigure type="categoricalToy" data="klviz/toys.json" subtitle=`$p_\theta = (a, 1-a)$ · $a = \sigma(\theta)$ · $q = (0.5, 0.5)$ · start $a_0 = 0.05$ · batch 16 · learning rate 1` >}}
First: variance of {{< klmath inline=true >}}k_1{{< /klmath >}} and {{< klmath inline=true >}}k_3{{< /klmath >}} as KL estimates. Second: per-sample variance of the gradient with respect to the logit {{< klmath inline=true >}}\theta{{< /klmath >}}; both are zero at {{< klmath inline=true >}}a = 0.5{{< /klmath >}}, where {{< klmath inline=true >}}p = q{{< /klmath >}}. Third: true KL during minibatch gradient descent.
{{< /klfigure >}}

Now the forward KL has gradient {{< klmath inline=true >}}a - \tfrac12{{< /klmath >}}, which differs from the reverse KL's. The {{< klmath inline=true >}}k_3{{< /klmath >}}-in-reward gradient, their difference, has the wrong sign for every {{< klmath inline=true >}}a \neq \tfrac12{{< /klmath >}}. The KL climbs toward its maximum, {{< klmath inline=true >}}\log 2{{< /klmath >}}, as the rare action disappears.

{{< klderiv title="Derive the two gradients" >}}
With {{< klmath inline=true >}}\mathrm{d}a/\mathrm{d}\theta = a(1-a){{< /klmath >}}:

{{< klmath >}}
\begin{aligned}
\mathrm{KL}[p\,\|\,q] &= a\log 2a + (1-a)\log 2(1-a),\\
\tfrac{\mathrm{d}}{\mathrm{d}\theta}\mathrm{KL}[p\,\|\,q] &= a(1-a)\log\tfrac{a}{1-a},\\[4pt]
\mathrm{KL}[q\,\|\,p] &= -\log 2 - \tfrac12\log a - \tfrac12\log(1-a),\\
\tfrac{\mathrm{d}}{\mathrm{d}\theta}\mathrm{KL}[q\,\|\,p] &= a - \tfrac12.
\end{aligned}
{{< /klmath >}}

At {{< klmath inline=true >}}a = 0.05{{< /klmath >}} the {{< klmath inline=true >}}k_3{{< /klmath >}}-in-reward gradient is {{< klmath inline=true >}}-0.14 - (-0.45) = +0.31{{< /klmath >}}. Gradient descent then lowers {{< klmath inline=true >}}\theta{{< /klmath >}}, and with it {{< klmath inline=true >}}a{{< /klmath >}}: away from {{< klmath inline=true >}}q{{< /klmath >}}.
{{< /klderiv >}}

The intuition is simple. At the start, {{< klmath inline=true >}}k_3{{< /klmath >}} charges the rare action a penalty of 6.7 and the common one 0.17. As a reward, that discourages the rare action even more. {{< klmath inline=true >}}k_1{{< /klmath >}} charges the rare action {{< klmath inline=true >}}\log(0.05/0.5) < 0{{< /klmath >}}: a bonus, so it recovers.

{{< klmath inline=true >}}k_3{{< /klmath >}} as a loss converges, and faster than {{< klmath inline=true >}}k_1{{< /klmath >}}. It follows the forward KL, which is large when {{< klmath inline=true >}}p{{< /klmath >}} has a hole, so it pushes harder. Both KLs are minimized at {{< klmath inline=true >}}p = q{{< /klmath >}}, so here only the path differs.

### Two modes: {{< klmath inline=true >}}k_3{{< /klmath >}} as a loss fits the forward KL

The first two worlds hide the real cost of {{< klmath inline=true >}}k_3{{< /klmath >}} as a loss. In both, {{< klmath inline=true >}}p{{< /klmath >}} can match {{< klmath inline=true >}}q{{< /klmath >}} exactly, and both KLs are zero there. Take that option away.

Let {{< klmath inline=true >}}q{{< /klmath >}} have two modes, {{< klmath inline=true >}}q = \tfrac12\mathcal{N}(-2, 0.5^2) + \tfrac12\mathcal{N}(2, 0.5^2){{< /klmath >}}, and let the policy be a single Gaussian, {{< klmath inline=true >}}p = \mathcal{N}(\mu, \sigma^2){{< /klmath >}}. Learn {{< klmath inline=true >}}\mu{{< /klmath >}} and {{< klmath inline=true >}}\log\sigma{{< /klmath >}} from {{< klmath inline=true >}}\mu_0 = 1{{< /klmath >}}, {{< klmath inline=true >}}\sigma_0 = 1{{< /klmath >}}.

The two KLs now want different things:

- The reverse KL, {{< klmath inline=true >}}\mathrm{KL}[p\,\|\,q]{{< /klmath >}}, is **mode-seeking**. It punishes {{< klmath inline=true >}}p{{< /klmath >}} for putting mass where {{< klmath inline=true >}}q{{< /klmath >}} has none. Its best fit sits on one mode, at {{< klmath inline=true >}}\mathrm{KL}[p\,\|\,q] \approx \log 2 \approx 0.69{{< /klmath >}}.
- The forward KL, {{< klmath inline=true >}}\mathrm{KL}[q\,\|\,p]{{< /klmath >}}, is **mass-covering**. It punishes {{< klmath inline=true >}}p{{< /klmath >}} for missing anything {{< klmath inline=true >}}q{{< /klmath >}} does. Its best fit matches {{< klmath inline=true >}}q{{< /klmath >}}'s mean and variance, {{< klmath inline=true >}}\mathcal{N}(0, 4.25){{< /klmath >}}, and is centered on the gap between the modes.

{{< klderiv title="Derive the two best fits" >}}
**Forward KL.** {{< klmath inline=true >}}\mathrm{KL}[q\,\|\,p] = \text{const} - \mathbb{E}_q[\log p]{{< /klmath >}}, and for {{< klmath inline=true >}}p = \mathcal{N}(\mu, \sigma^2){{< /klmath >}},

{{< klmath >}}
-\mathbb{E}_q[\log p] = \tfrac12\log(2\pi\sigma^2) + \frac{\mathbb{E}_q[(x-\mu)^2]}{2\sigma^2}.
{{< /klmath >}}

This is minimized at {{< klmath inline=true >}}\mu = \mathbb{E}_q[x] = 0{{< /klmath >}} and {{< klmath inline=true >}}\sigma^2 = \operatorname{Var}_q(x) = 0.5^2 + 2^2 = 4.25{{< /klmath >}}.

**Reverse KL.** Take {{< klmath inline=true >}}p = \mathcal{N}(2, 0.5^2){{< /klmath >}}, one of the modes. Where {{< klmath inline=true >}}p{{< /klmath >}} has mass, the other mode is negligible, so {{< klmath inline=true >}}q \approx p/2{{< /klmath >}}, {{< klmath inline=true >}}\log(p/q) \approx \log 2{{< /klmath >}}, and {{< klmath inline=true >}}\mathrm{KL}[p\,\|\,q] \approx \log 2{{< /klmath >}}.
{{< /klderiv >}}

{{< klfigure type="bimodalToy" data="klviz/toys.json" subtitle=`$q = \tfrac12\mathcal{N}(-2, 0.5^2) + \tfrac12\mathcal{N}(2, 0.5^2)$ · $p = \mathcal{N}(\mu, \sigma^2)$ · start $\mu_0 = 1, \sigma_0 = 1$ · batch 64 · learning rate 0.05` >}}
First: {{< klmath inline=true >}}\mathrm{KL}[p\,\|\,q]{{< /klmath >}}, the objective we want to minimize. Second: {{< klmath inline=true >}}\mathrm{KL}[q\,\|\,p]{{< /klmath >}}. Both show the median of 1,000 runs, with the middle half shaded; runs that diverge count as infinite KL. Third: the median run's policy after 300 steps, against {{< klmath inline=true >}}q{{< /klmath >}} (gray, dashed).
{{< /klfigure >}}

{{< klmath inline=true >}}k_1{{< /klmath >}} in the reward finds a mode ({{< klmath inline=true >}}\mu \approx 2{{< /klmath >}}, {{< klmath inline=true >}}\sigma \approx 0.5{{< /klmath >}}) and reaches {{< klmath inline=true >}}\mathrm{KL}[p\,\|\,q] \approx 0.69{{< /klmath >}}.

{{< klmath inline=true >}}k_3{{< /klmath >}} as a loss lands on {{< klmath inline=true >}}\mu \approx 0{{< /klmath >}}, {{< klmath inline=true >}}\sigma \approx 2.06{{< /klmath >}}: the best fit of the forward KL. Its reverse KL is 2.10, three times worse. It puts 37% of its samples in {{< klmath inline=true >}}|x| < 1{{< /klmath >}}, where {{< klmath inline=true >}}q{{< /klmath >}} puts 2.3%.

{{< klmath inline=true >}}k_3{{< /klmath >}} in the reward also finds a mode in the median run, but 26% of its runs diverge within 300 steps. That is the {{< klmath inline=true >}}q/p{{< /klmath >}} tail again.

{{< klcode title="Reproduce the three toys" >}}
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


# Gaussian and two actions: variances at the start, and median KL after 20 steps.
print(variances('gaussian', 1.5))
print(variances('categorical', 0.05))
for family, start, rate in [('gaussian', 1.5, 0.1), ('categorical', 0.05, 1.0)]:
    kls, policies = descend(family, start, rate, steps=20)
    print(family, {m: np.median(k[-1]) for m, k in kls.items()})

# Two modes: the median run of each method after 300 steps.
for method, history in bimodal_descend().items():
    mu, sigma, reverse, forward, diverged = history[-1]
    middle = np.argsort(reverse)[len(reverse) // 2]
    print(method, 'mu', mu[middle], 'sigma', sigma[middle],
          'KL[p||q]', reverse[middle], 'KL[q||p]', forward[middle], 'diverged', diverged)
{{< /klcode >}}

## What this means for OPD

In on-policy distillation, the student samples a response and the teacher scores every token. The per-token reward is

{{< klmath >}}
-k_1 = \log \pi_\mathrm{teacher}(x_t \mid x_{<t}) - \log \pi_\mathrm{student}(x_t \mid x_{<t}).
{{< /klmath >}}

By the table, this gives exactly the gradient of the reverse KL, {{< klmath inline=true >}}\mathrm{KL}[\pi_\mathrm{student}\,\|\,\pi_\mathrm{teacher}]{{< /klmath >}}. Reverse KL is mode-seeking, which is what we want from a smaller student that cannot do everything the teacher does: commit to what it can do well.

Swapping {{< klmath inline=true >}}k_3{{< /klmath >}} into that reward does not make sense. Even where {{< klmath inline=true >}}k_3{{< /klmath >}} estimates the KL better, its gradient is the difference of the reverse and forward KL gradients.

Using {{< klmath inline=true >}}k_3{{< /klmath >}} as a loss can work. It just minimizes the forward KL, {{< klmath inline=true >}}\mathrm{KL}[\pi_\mathrm{teacher}\,\|\,\pi_\mathrm{student}]{{< /klmath >}}. The student then spreads over teacher behaviors it cannot represent, including the gaps between them.

What about GRPO, which puts {{< klmath inline=true >}}k_3{{< /klmath >}} in its loss to keep the policy near a reference?[^grpo] That is a different regime. As a leash, the KL is meant to stay small, and near the reference the two directions agree to second order. Distillation is the opposite: the student starts far from the teacher and may never reach it. That is where the direction matters. The difference between estimating KL and differentiating it is analyzed in more depth in *Rethinking KL Regularization in RLHF*.[^rethinking]

## Epilogue

A KL estimator has two jobs in RL.

**Measuring.** Neither estimator always wins. {{< klmath inline=true >}}k_3{{< /klmath >}} has lower variance near {{< klmath inline=true >}}q{{< /klmath >}}; {{< klmath inline=true >}}k_1{{< /klmath >}} is safer far from {{< klmath inline=true >}}q{{< /klmath >}} and around holes.

**Training.** Pick the estimator whose gradient is the one you want. For the reverse KL, that is {{< klmath inline=true >}}k_1{{< /klmath >}} in the reward.

{{< klmath inline=true >}}k_3{{< /klmath >}} adds a zero-mean term, {{< klmath inline=true >}}q/p - 1{{< /klmath >}}. Zero mean keeps the estimate unbiased. It does not keep the gradient unchanged: multiplied by the score, the term picks up the forward KL's gradient, and differentiated, it becomes that gradient.

**Choose a KL estimator for the gradient it produces, not the number it reports.**

[^schulman]: John Schulman, ["Approximating KL Divergence"](https://joschu.net/blog/kl-approx.html), 2020.
[^deepseek]: DeepSeek-AI, ["DeepSeek-V3.2"](https://arxiv.org/html/2512.02556v1#S3.SS1), 2025.
[^opd]: Thinking Machines Lab, ["On-Policy Distillation"](https://thinkingmachines.ai/blog/on-policy-distillation/), 2025.
[^grpo]: Zhihong Shao et al., ["DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"](https://arxiv.org/abs/2402.03300), 2024. GRPO adds {{< klmath inline=true >}}k_3{{< /klmath >}} with the reference policy directly to its loss.
[^rethinking]: ["Rethinking KL Regularization in RLHF"](https://arxiv.org/abs/2510.01555), 2025.
