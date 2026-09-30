---
title: "The Structure Behind KL Estimation for Reinforcement Learning"
date: 2026-09-30
draft: false
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

This could be a reason that {{< klmath inline=true >}}k_1{{< /klmath >}} is better than {{< klmath inline=true >}}k_3{{< /klmath >}} for LLMs. However, it does not reveal the full story.

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


## How do LLMs compare to classical RL?

Moving beyond the toy distributions above, let’s try using both estimators for LLMs, and for classical RL.

For LLMs, we’ll run on policy distillation for mathematical reasoning. Specifically, we’ll use Qwen3-32B as the teacher and Qwen3-4B as the student. We’ll use Hendrycks math as the train set and AIME 2024 as the test set. Here, the KL estimate is directly used as advantage.

{{< klmath >}}
A_t = -\widehat{\mathrm{KL}}_t[\pi_\mathrm{student}\,\|\,\pi_\mathrm{teacher}].
{{< /klmath >}}

For classical RL, we’ll PPO on 3 MuJoCo environments: Hopper, Half Cheetah and Walker2D, and learn a policy using PPO. Here, KL is used to determine when to stop training on a collected batch of data, i.e. we constrain {{< klmath inline=true >}}\mathrm{KL}[\pi_\mathrm{old}\,\|\,\pi_\theta]{{< /klmath >}} to be below 0.015, and use an estimator to measure it.

{{< klfigure type="empirical">}}
Shading is the 25th–75th percentile range. Bins with fewer than 10 rows are omitted. LLM: {{< klmath inline=true >}}\mathrm{KL}[\pi_\mathrm{student}\,\|\,\pi_\mathrm{teacher}]{{< /klmath >}} at fixed prefixes; PPO: {{< klmath inline=true >}}\mathrm{KL}[\pi_\mathrm{old}\,\|\,\pi_\theta]{{< /klmath >}}, averaged over states.
{{< /klfigure >}}

As we can see, {{< klmath inline=true >}}k_3{{< /klmath >}} seems to be a better estimator for the classical RL experiments, but {{< klmath inline=true >}}k_1{{< /klmath >}} seems to be better for the LLM experiments. This is likely due to the existence of probability "holes" similar to the toy experiment we showed in the previous section.

## Epilogue
Ultimately, does this mean you should blindly use {{< klmath inline=true >}}k_1{{< /klmath >}} for LLMs and {{< klmath inline=true >}}k_3{{< /klmath >}} for classical RL? Maybe - That's not a bad rule. However, a more comprehensive conclusion is that the shape of distribution matters when choosing an estimator for KL. For RL, this means your policy's action space and where KL is used in your algorithm will ultimately decide what estimator is best for you.
