---
title: "The Structure Behind KL Estimation for Reinforcement Learning"
date: 2026-09-30
draft: true
toc: false
klviz: true
summary: "Why do RL codebases and papers use different estimators, and what makes one better than another?"
---

KL divergence {{< klmath inline=true >}}\mathrm{KL}[p\,\|\,q] = \mathbb{E}_p[log(p/q)]{{< /klmath >}} is an important quantity in AI. For frontier reinforcement learning, we keep the learned policy {{< klmath inline=true >}}\pi_\theta{{< /klmath >}} close to a reference policy {{< klmath inline=true >}}\pi_\mathrm{ref}{{< /klmath >}} by adding {{< klmath inline=true >}}\mathrm{KL}[\pi_\theta\,\|\,\pi_\mathrm{ref}]{{< /klmath >}} to the loss, and we use it for on policy distillation to distill {{< klmath inline=true >}}\pi_\mathrm{teacher}{{< /klmath >}} into {{< klmath inline=true >}}\pi_\mathrm{student}{{< /klmath >}} by minimizing {{< klmath inline=true >}}\mathrm{KL}[\pi_\mathrm{student}\,\|\,\pi_\mathrm{teacher}]{{< /klmath >}}.

Since exactly computing KL is often intractable due to the expectation, you may have seen any of the following Monte-Carlo estimators used in papers / code to estimate {{< klmath inline=true >}}\mathrm{KL}[p\,\|\,q]{{< /klmath >}}, where {{< klmath inline=true >}}x \tilde p{{< /klmath >}}:

{{< klmath >}}
\begin{aligned}
k_1(x) &= \log\frac{p(x)}{q(x)}\\
k_2(x) &= \frac12\left(\log\frac{p(x)}{q(x)}\right)^2\\
k_3(x) &= \log\frac{p(x)}{q(x)} + \frac{q(x)}{p(x)} - 1
\end{aligned}
{{< /klmath >}}

The {{< klmath inline=true >}}k_3{{< /klmath >}} estimator in particular was introduced by [John Schulman](https://joschu.net/blog/kl-approx.html) to be unbiased like {{< klmath inline=true >}}k_1{{< /klmath >}} but have lower variance and stay positive like {{< klmath inline=true >}}k_2{{< /klmath >}}.

However, despite the clean story behind {{< klmath inline=true >}}k_3{{< /klmath >}} in the Schulman blog and its adoption in work as recent as [Deepseek 3.2](https://arxiv.org/html/2512.02556v1#S3.SS1), [other work](https://thinkingmachines.ai/blog/on-policy-distillation/) seems to have adopted {{< klmath inline=true >}}k_1{{< /klmath >}} to estimate KL. Why is that?

## The trivial explanation: {{< klmath inline=true >}}k_3{{< /klmath >}} is better at lower KL, but explodes at higher KL

Below, if {{< klmath inline=true >}}p = \mathcal{N}(\mu, 1){{< /klmath >}}; {{< klmath inline=true >}}q = \mathcal{N}(0, 1){{< /klmath >}}, we can see that {{< klmath inline=true >}}k_1{{< /klmath >}} remains stable when estimating higher values of {{< klmath inline=true >}}\mathrm{KL}[p\,\|\,q]{{< /klmath >}}, while {{< klmath inline=true >}}k_3{{< /klmath >}} diverges.

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

This explains why {{< klmath inline=true >}}k_1{{< /klmath >}} can be a better estimator at higher KL. However, it does not reveal the full story.

## Diving deeper: same underlying KL can yield wildly different estimates

Let’s consider three categorical distributions over three variables:

{{< klmath >}}
\begin{aligned}
q   &= (0.1,\;0.45,\;0.45),\\
p_1 &= (0.001,\;0.4995,\;0.4995),\\
p_2 &= (0.1,\;0.6577534291783038,\;0.2422465708216962).
\end{aligned}
{{< /klmath >}}

{{< klmath inline=true >}}p_1{{< /klmath >}} and {{< klmath inline=true >}}p_2{{< /klmath >}} were chosen so that {{< klmath inline=true >}}\mathrm{KL}[p_1\,\|\,q] = \mathrm{KL}[p_2\,\|\,q] = 0.0996504851{{< /klmath >}}, but have different distribution shapes. {{< klmath inline=true >}}p_1{{< /klmath >}} differs from q in that one category's probability mass is reduced to almost 0, whereas {{< klmath inline=true >}}p_2{{< /klmath >}} has a smoother shift in probability mass across the other two categories.

With {{< klmath inline=true >}}q{{< /klmath >}} fixed, the exact estimator variances are (in nats²):

{{< klmath >}}
\begin{aligned}
\operatorname{Var}_{p_1}(k_1)&\approx 0.0222, &\operatorname{Var}_{p_1}(k_3)&\approx 8.9005,\\
\operatorname{Var}_{p_2}(k_1)&\approx 0.1777, &\operatorname{Var}_{p_2}(k_3)&\approx 0.00650.
\end{aligned}
{{< /klmath >}}

As we can see, {{< klmath inline=true >}}k_1{{< /klmath >}} is better for {{< klmath inline=true >}}p_1{{< /klmath >}}, and {{< klmath inline=true >}}k_3{{< /klmath >}} is better for {{< klmath inline=true >}}p_2{{< /klmath >}}.

Instead of looking at just one example, we can gradually make the hole deeper. Keep {{< klmath inline=true >}}q=(0.2,0.4,0.4){{< /klmath >}} fixed and define:

{{< klmath >}}
\begin{aligned}
a(h)&=0.2\times 10^{-h},\\
p_1(h)&=\left(a(h),\frac{1-a(h)}{2},\frac{1-a(h)}{2}\right),\\
p_2(h)&=\left(0.2,0.4+\delta(h),0.4-\delta(h)\right).
\end{aligned}
{{< /klmath >}}

Here {{< klmath inline=true >}}h{{< /klmath >}} controls the depth of the hole: each increase of one makes the first outcome ten times less likely under {{< klmath inline=true >}}p_1{{< /klmath >}}. We choose {{< klmath inline=true >}}\delta(h)\geq 0{{< /klmath >}} so that {{< klmath inline=true >}}\mathrm{KL}[p_1(h)\,\|\,q]=\mathrm{KL}[p_2(h)\,\|\,q]{{< /klmath >}}. The two KLs match at every setting; their shared value changes with {{< klmath inline=true >}}h{{< /klmath >}}.

Hover over the curves to compare the distributions and their estimator variances.

{{< klfigure type="holeness" >}}
Exact variances, with a logarithmic vertical axis. Solid lines use {{< klmath inline=true >}}p_1{{< /klmath >}}; dashed lines use {{< klmath inline=true >}}p_2{{< /klmath >}}. At {{< klmath inline=true >}}h=0{{< /klmath >}}, both distributions equal {{< klmath inline=true >}}q{{< /klmath >}} and all variances are zero, so that endpoint is omitted from the log plot.
{{< /klfigure >}}

The shared KL approaches {{< klmath inline=true >}}\log(1/0.8)\approx 0.22314{{< /klmath >}} as the hole gets deeper. Meanwhile, the variance of {{< klmath inline=true >}}k_3{{< /klmath >}} under {{< klmath inline=true >}}p_1{{< /klmath >}} keeps growing: the first outcome becomes rarer, but its {{< klmath inline=true >}}q/p_1{{< /klmath >}} contribution becomes larger. The smooth distribution has the same KL without that increasingly rare, large contribution.

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

## Estimating KL is not the same as optimizing KL

So far, we have treated KL as a number to measure. But in reinforcement learning, we often use the estimate to update the policy. An unbiased estimate of the value need not give an unbiased policy-gradient coefficient.

Let {{< klmath inline=true >}}p_\theta{{< /klmath >}} be the policy we are learning, and keep {{< klmath inline=true >}}q{{< /klmath >}} fixed. We will sample on-policy, with no task reward, and try to minimize {{< klmath inline=true >}}D(\theta)=\mathrm{KL}[p_\theta\,\|\,q]{{< /klmath >}}. There are two ways to use a sample's KL estimate:

- **In reward:** use {{< klmath inline=true >}}-k_i(x){{< /klmath >}} as a detached reward. The corresponding penalty gradient is {{< klmath inline=true >}}g_i=\mathbb{E}_{p_\theta}[k_i(x)\nabla_\theta\log p_\theta(x)]{{< /klmath >}}.
- **As a direct loss:** backpropagate through {{< klmath inline=true >}}k_i(x){{< /klmath >}}, holding the sampled action fixed. The expected gradient is {{< klmath inline=true >}}\mathbb{E}_{p_\theta}[\nabla_\theta k_i(x)]{{< /klmath >}}.

In either case, we subtract the penalty gradient when updating the policy. We compare three combinations: {{< klmath inline=true >}}k_1{{< /klmath >}} in reward, {{< klmath inline=true >}}k_3{{< /klmath >}} in reward, and {{< klmath inline=true >}}k_3{{< /klmath >}} as a direct loss. ({{< klmath inline=true >}}k_1{{< /klmath >}} as a direct loss has zero expected gradient, so we leave it out.)

We use two toy policies, one for each distribution shape from earlier: a Gaussian and a categorical distribution with a rare action. Each gets one figure with three plots:

1. **Variance of the KL estimate**, as the policy moves away from {{< klmath inline=true >}}q{{< /klmath >}}. This is what the earlier sections measured.
2. **Variance of the gradient**: how noisy a single sample's contribution to the update is. This is what a minibatch averages.
3. **Gradient descent**: the true {{< klmath inline=true >}}\mathrm{KL}[p_\theta\,\|\,q]{{< /klmath >}} after each step, using minibatches of 16 on-policy samples. Lines show the median of 2,000 runs, and shading covers the middle half.

Hover over any plot to see the distributions and read off values.

### Gaussian

Let {{< klmath inline=true >}}p_\mu=\mathcal{N}(\mu,1){{< /klmath >}} and {{< klmath inline=true >}}q=\mathcal{N}(0,1){{< /klmath >}}, and learn {{< klmath inline=true >}}\mu{{< /klmath >}}, starting from {{< klmath inline=true >}}\mu_0=1.5{{< /klmath >}} with learning rate 0.1.

{{< klfigure type="gaussianToy" data="klviz/toys.json" subtitle=`$p_\mu = \mathcal{N}(\mu, 1)$ · $q = \mathcal{N}(0, 1)$ · start $\mu_0 = 1.5$ · batch 16 · learning rate 0.1` >}}
First: variance of {{< klmath inline=true >}}k_1{{< /klmath >}} and {{< klmath inline=true >}}k_3{{< /klmath >}} as KL estimates. Second: per-sample variance of the gradient with respect to {{< klmath inline=true >}}\mu{{< /klmath >}}. Third: true KL during minibatch gradient descent; median of 2,000 runs, with the middle 50% shaded. All variances are exact.
{{< /klfigure >}}

- **{{< klmath inline=true >}}k_1{{< /klmath >}} in reward** gives an unbiased estimate of the KL gradient, {{< klmath inline=true >}}\mu{{< /klmath >}}, and converges.
- **{{< klmath inline=true >}}k_3{{< /klmath >}} in reward** has an expected gradient of exactly zero, for every {{< klmath inline=true >}}\mu{{< /klmath >}}:

  {{< klmath >}}
  \mathbb{E}_{p_\mu}[k_3(x)(x-\mu)]=\underbrace{\mathbb{E}_{p_\mu}[k_1(x)(x-\mu)]}_{=\,\mu}+\underbrace{\mathbb{E}_{p_\mu}\!\left[\left(\tfrac{q(x)}{p_\mu(x)}-1\right)(x-\mu)\right]}_{=\,-\mu}=0.
  {{< /klmath >}}

  The policy receives only noise and wanders: some runs drift closer to {{< klmath inline=true >}}q{{< /klmath >}} by chance, but the typical run never gets there.
- **{{< klmath inline=true >}}k_3{{< /klmath >}} as a loss** also has expected gradient {{< klmath inline=true >}}\mu{{< /klmath >}} here, so on average it moves exactly like {{< klmath inline=true >}}k_1{{< /klmath >}} in reward. But at the start its gradient is 15 times noisier (87 versus 5.8), even though {{< klmath inline=true >}}k_3{{< /klmath >}} is only about 3 times noisier as a KL estimate. With batches of 16, the median run still keeps pace; the extra noise shows up as a wider spread between runs.

### Two actions with a rare action

Let {{< klmath inline=true >}}p_\theta=(a,1-a){{< /klmath >}} with {{< klmath inline=true >}}a=\sigma(\theta){{< /klmath >}}, and {{< klmath inline=true >}}q=(0.5,0.5){{< /klmath >}}. We learn the logit {{< klmath inline=true >}}\theta{{< /klmath >}}, starting from {{< klmath inline=true >}}a_0=0.05{{< /klmath >}}, so the first action is rare under the policy, like the probability hole earlier. The learning rate is 1.

{{< klfigure type="categoricalToy" data="klviz/toys.json" subtitle=`$p_\theta = (a, 1-a)$ · $a = \sigma(\theta)$ · $q = (0.5, 0.5)$ · start $a_0 = 0.05$ · batch 16 · learning rate 1` >}}
First: variance of {{< klmath inline=true >}}k_1{{< /klmath >}} and {{< klmath inline=true >}}k_3{{< /klmath >}} as KL estimates. Second: per-sample variance of the gradient with respect to the logit {{< klmath inline=true >}}\theta{{< /klmath >}}. Both are zero at {{< klmath inline=true >}}a=0.5{{< /klmath >}}, where {{< klmath inline=true >}}p=q{{< /klmath >}}. Third: true KL during minibatch gradient descent; median of 2,000 runs, with the middle 50% shaded.
{{< /klfigure >}}

- **{{< klmath inline=true >}}k_1{{< /klmath >}} in reward** again follows the KL gradient and converges.
- **{{< klmath inline=true >}}k_3{{< /klmath >}} in reward** goes the **wrong way**: the KL rises toward its maximum, {{< klmath inline=true >}}\log 2{{< /klmath >}}, as the rare action disappears. In expectation, its gradient is {{< klmath inline=true >}}g_1+\tfrac12-a{{< /klmath >}}, which points the wrong way for every {{< klmath inline=true >}}a\neq\tfrac12{{< /klmath >}}. The intuition: at the start, {{< klmath inline=true >}}k_3{{< /klmath >}} gives the rare action a penalty of 6.7 and the common action only 0.17, so as a detached reward it discourages the rare action even more. {{< klmath inline=true >}}k_1{{< /klmath >}} gives the rare action a negative penalty, {{< klmath inline=true >}}\log(0.05/0.5)<0{{< /klmath >}}, encouraging it to recover.
- **{{< klmath inline=true >}}k_3{{< /klmath >}} as a loss** converges, and faster than {{< klmath inline=true >}}k_1{{< /klmath >}}, even though its gradient is the noisiest of the three. Its expected gradient is {{< klmath inline=true >}}a-\tfrac12{{< /klmath >}}, which is not the gradient of {{< klmath inline=true >}}\mathrm{KL}[p_\theta\,\|\,q]{{< /klmath >}}. As the next section shows, it is the gradient of the opposite direction, {{< klmath inline=true >}}\mathrm{KL}[q\,\|\,p_\theta]{{< /klmath >}}. That KL is large when {{< klmath inline=true >}}p{{< /klmath >}} has a hole, so it pushes harder here. Both KLs are minimized at {{< klmath inline=true >}}p=q{{< /klmath >}}, so with no task reward this only changes the path. With a task reward, the two directions would settle at different policies.

{{< klcode title="Reproduce both figures" >}}
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


# Values at the starting points, and median KL after 20 steps.
print(variances('gaussian', 1.5))
print(variances('categorical', 0.05))
for family, start, rate in [('gaussian', 1.5, 0.1), ('categorical', 0.05, 1.0)]:
    kls, policies = descend(family, start, rate, steps=20)
    print(family, {m: np.median(k[-1]) for m, k in kls.items()})
{{< /klcode >}}

### Why the gradients differ

Why can two unbiased estimates of KL lead to different updates? Differentiating an expectation has two terms:

{{< klmath >}}
\nabla_\theta\mathbb{E}_{p_\theta}[k_i]
=
\underbrace{\mathbb{E}_{p_\theta}[k_i\nabla_\theta\log p_\theta]}_{\text{in reward}}
+
\underbrace{\mathbb{E}_{p_\theta}[\nabla_\theta k_i]}_{\text{as a direct loss}}.
{{< /klmath >}}

The first term accounts for the changing sampling distribution; the second differentiates the estimator at a fixed sample. Their **sum** is the same KL gradient for both estimators, but each implementation keeps only one term. Write {{< klmath inline=true >}}D_R=\mathrm{KL}[p_\theta\,\|\,q]{{< /klmath >}} and {{< klmath inline=true >}}D_F=\mathrm{KL}[q\,\|\,p_\theta]{{< /klmath >}}. With on-policy samples and a fixed {{< klmath inline=true >}}q{{< /klmath >}}, the expected gradients are:

| Estimator | In reward | As a direct loss |
| --- | --- | --- |
| {{< klmath inline=true >}}k_1{{< /klmath >}} | {{< klmath inline=true >}}\nabla D_R{{< /klmath >}} | {{< klmath inline=true >}}0{{< /klmath >}} |
| {{< klmath inline=true >}}k_3{{< /klmath >}} | {{< klmath inline=true >}}\nabla D_R-\nabla D_F{{< /klmath >}} | {{< klmath inline=true >}}\nabla D_F{{< /klmath >}} |

For the equal-variance Gaussians, {{< klmath inline=true >}}\nabla D_R=\nabla D_F=\mu{{< /klmath >}}, which is why {{< klmath inline=true >}}k_3{{< /klmath >}} in reward gets exactly zero and {{< klmath inline=true >}}k_3{{< /klmath >}} as a loss gets the right answer. For the two-action policy, {{< klmath inline=true >}}\nabla D_F=a-\tfrac12{{< /klmath >}} differs from {{< klmath inline=true >}}\nabla D_R{{< /klmath >}}. This distinction between value estimation and gradient estimation is also analyzed in [Rethinking KL Regularization in RLHF](https://arxiv.org/abs/2510.01555).

## Epilogue

The shape of the distributions matters when choosing an estimator to **measure** KL. When using that estimate to **optimize** a policy, two more questions matter: which gradient does the implementation actually produce, and how noisy is it?

With on-policy samples, {{< klmath inline=true >}}k_1{{< /klmath >}} as a detached reward gives an unbiased gradient of {{< klmath inline=true >}}\mathrm{KL}[p_\theta\,\|\,q]{{< /klmath >}}. Putting {{< klmath inline=true >}}k_3{{< /klmath >}} in the reward instead keeps the expected KL value but changes the expected update: it can cancel the learning signal or reverse it. Using {{< klmath inline=true >}}k_3{{< /klmath >}} as a direct loss optimizes the opposite direction, {{< klmath inline=true >}}\mathrm{KL}[q\,\|\,p_\theta]{{< /klmath >}}, and its gradient can be much noisier than {{< klmath inline=true >}}k_1{{< /klmath >}}'s, even where {{< klmath inline=true >}}k_3{{< /klmath >}} is the better estimate of the KL value.

A good estimate of KL is not necessarily a good estimate of its gradient. The objective, the gradient path, and the gradient's variance determine which estimator is appropriate.
