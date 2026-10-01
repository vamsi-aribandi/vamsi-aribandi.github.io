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

In either case, we subtract the penalty gradient when updating the policy. These two implementations can behave very differently.

### The same Gaussians, a different question

Return to {{< klmath inline=true >}}p_\mu=\mathcal{N}(\mu,1){{< /klmath >}} and {{< klmath inline=true >}}q=\mathcal{N}(0,1){{< /klmath >}}. Both estimators have exactly the same expectation:

{{< klmath >}}
\mathbb{E}_{p_\mu}[k_1]=\mathbb{E}_{p_\mu}[k_3]=\frac{\mu^2}{2},
\qquad \frac{\partial D}{\partial\mu}=\mu.
{{< /klmath >}}

The score is {{< klmath inline=true >}}\partial_\mu\log p_\mu(x)=x-\mu{{< /klmath >}}. Multiplying by that score changes the comparison:

{{< klmath >}}
\begin{aligned}
g_1&=\mathbb{E}_{p_\mu}[k_1(x)(x-\mu)]=\mu,\\
g_3&=\mathbb{E}_{p_\mu}[k_3(x)(x-\mu)]=0.
\end{aligned}
{{< /klmath >}}

As a reward coefficient, {{< klmath inline=true >}}k_1{{< /klmath >}} gives the exact KL gradient. {{< klmath inline=true >}}k_3{{< /klmath >}} gives no expected update at all, even with infinitely many samples. Its control-variate term cancels the learning signal:

{{< klmath >}}
\mathbb{E}_{p_\mu}\!\left[\left(\frac{q(x)}{p_\mu(x)}-1\right)(x-\mu)\right]=-\mu.
{{< /klmath >}}

Directly differentiating {{< klmath inline=true >}}k_3{{< /klmath >}}, on the other hand, gives an expected gradient of {{< klmath inline=true >}}\mu{{< /klmath >}} in this example. The plots below compare the gradients and the resulting optimization. Hover to inspect the distributions, KL values, and updates.

{{< klfigure type="gaussianGradients" subtitle=`$p_\mu = \mathcal{N}(\mu, 1)$ · $q = \mathcal{N}(0, 1)$` >}}
Left: expected penalty gradients with respect to {{< klmath inline=true >}}\mu{{< /klmath >}}; the blue curve is the exact gradient of {{< klmath inline=true >}}\mathrm{KL}[p_\mu\,\|\,q]{{< /klmath >}}. Right: exact gradient updates from {{< klmath inline=true >}}\mu_0=1{{< /klmath >}}, with learning rate 0.15. The dashed green curve overlaps the blue curve. All curves use exact expectations, without sampling noise.
{{< /klfigure >}}

### Two actions: the wrong direction

A two-action policy makes the failure more striking. Let

{{< klmath >}}
p_\theta=(a,1-a),\qquad a=\sigma(\theta),\qquad q=(0.5,0.5).
{{< /klmath >}}

Both estimators still have the same expectation, {{< klmath inline=true >}}D=a\log(2a)+(1-a)\log(2(1-a)){{< /klmath >}}. Their reward-coefficient gradients, with respect to the logit {{< klmath inline=true >}}\theta{{< /klmath >}}, are

{{< klmath >}}
\begin{aligned}
g_1&=a(1-a)\log\frac{a}{1-a}=\frac{\partial D}{\partial\theta},\\
g_3&=g_1+\frac12-a.
\end{aligned}
{{< /klmath >}}

At {{< klmath inline=true >}}a=0.8{{< /klmath >}}, both estimate a KL of 0.1927 nats, but {{< klmath inline=true >}}g_1\approx 0.2218{{< /klmath >}} and {{< klmath inline=true >}}g_3\approx -0.0782{{< /klmath >}}. Subtracting {{< klmath inline=true >}}g_1{{< /klmath >}} moves the policy toward the reference. Subtracting {{< klmath inline=true >}}g_3{{< /klmath >}} moves it **away**.

There is a simple intuition here. At this policy, {{< klmath inline=true >}}k_3{{< /klmath >}} assigns a penalty of about 0.095 to the common action and 0.584 to the rare one. As a detached negative reward, it discourages the already underrepresented action even more. {{< klmath inline=true >}}k_1{{< /klmath >}} assigns the rare action a negative penalty, encouraging its probability to recover.

{{< klfigure type="categoricalGradients" subtitle=`$p_\theta = (a, 1-a)$ · $a = \sigma(\theta)$ · $q = (0.5, 0.5)$` >}}
Left: expected penalty gradients with respect to the logit {{< klmath inline=true >}}\theta{{< /klmath >}}. Right: exact gradient updates from {{< klmath inline=true >}}a_0=0.8{{< /klmath >}}, with learning rate 0.5. Using {{< klmath inline=true >}}k_3{{< /klmath >}} in reward increases the true KL. Directly differentiating {{< klmath inline=true >}}k_3{{< /klmath >}} pulls toward the reference, but its gradient differs from the blue curve.
{{< /klfigure >}}

{{< klcode title="Reproduce the gradients and optimization" >}}
import numpy as np


def gaussian(mu):
    # KL, k1 in reward, k3 in reward, k3 as direct loss.
    return np.array([mu**2 / 2, mu, 0.0, mu])


def categorical(theta):
    a = 1 / (1 + np.exp(-theta))
    p = np.array([a, 1 - a])
    q = np.array([0.5, 0.5])
    score = np.array([1 - a, -a])  # d log p / d theta
    k1 = np.log(p / q)
    k3 = k1 + q / p - 1
    assert np.isclose(p @ k1, p @ k3)
    return np.array([
        p @ k1,
        p @ (k1 * score),
        p @ (k3 * score),
        p @ ((1 - q / p) * score),
    ])


def optimize(metrics, initial, learning_rate, steps):
    # Each column follows its own policy, updated with exact expectations.
    parameters = np.full(3, initial, dtype=float)
    kls = []
    for step in range(steps + 1):
        values = [metrics(t) for t in parameters]
        kls.append([v[0] for v in values])
        if step < steps:
            parameters -= learning_rate * np.array([
                values[i][i + 1] for i in range(3)
            ])
    return np.array(kls)


gaussian_kls = optimize(gaussian, 1.0, 0.15, 30)
categorical_kls = optimize(categorical, np.log(4.0), 0.5, 80)
print("KL and gradients at a=0.8:", categorical(np.log(4.0)))
# [0.19274476, 0.22180710, -0.07819290, 0.30000000]
{{< /klcode >}}

### What changes when we differentiate the loss?

Why can two unbiased estimates lead to different updates? Differentiating an expectation has two terms:

{{< klmath >}}
\nabla_\theta\mathbb{E}_{p_\theta}[k_i]
=
\underbrace{\mathbb{E}_{p_\theta}[k_i\nabla_\theta\log p_\theta]}_{\text{detached reward coefficient}}
+
\underbrace{\mathbb{E}_{p_\theta}[\nabla_\theta k_i]}_{\text{direct loss gradient}}.
{{< /klmath >}}

The first accounts for the changing sampling distribution; the second differentiates the estimator at a fixed sample. The **sum** is the same KL gradient for both estimators. The implementations above each keep only one term.

Write {{< klmath inline=true >}}D_R=\mathrm{KL}[p_\theta\,\|\,q]{{< /klmath >}} and {{< klmath inline=true >}}D_F=\mathrm{KL}[q\,\|\,p_\theta]{{< /klmath >}}. With on-policy samples, fixed {{< klmath inline=true >}}q{{< /klmath >}}, and common support, the expected gradients are:

| Estimator | In reward | As a direct loss |
| --- | --- | --- |
| {{< klmath inline=true >}}k_1{{< /klmath >}} | {{< klmath inline=true >}}\nabla D_R{{< /klmath >}} | {{< klmath inline=true >}}0{{< /klmath >}} |
| {{< klmath inline=true >}}k_3{{< /klmath >}} | {{< klmath inline=true >}}\nabla D_R-\nabla D_F{{< /klmath >}} | {{< klmath inline=true >}}\nabla D_F{{< /klmath >}} |

So {{< klmath inline=true >}}k_3{{< /klmath >}} can provide a restoring gradient as a direct loss, but it generally gives the gradient of the **opposite KL direction**. The two directions happen to coincide for our equal-variance Gaussians. For the two-action policy, its direct-loss gradient is {{< klmath inline=true >}}a-0.5{{< /klmath >}}, which differs from {{< klmath inline=true >}}\partial_\theta D_R{{< /klmath >}}. This distinction between value estimation and gradient estimation is also analyzed in [Rethinking KL Regularization in RLHF](https://arxiv.org/abs/2510.01555).

## A better KL estimate is not a better gradient

{{< klmath inline=true >}}k_3{{< /klmath >}} is usually chosen because it estimates the KL value with lower variance. In training, though, what we average over a minibatch is the per-sample **gradient**, not the KL estimate. Does the variance advantage carry over?

We compare two common choices: {{< klmath inline=true >}}k_1{{< /klmath >}} in reward, which is unbiased for {{< klmath inline=true >}}\nabla D_R{{< /klmath >}}, and {{< klmath inline=true >}}k_3{{< /klmath >}} as a direct loss, which is how {{< klmath inline=true >}}k_3{{< /klmath >}} is typically used as a KL penalty.

### The Gaussian case: unbiased, but noisier

For {{< klmath inline=true >}}p_\mu=\mathcal{N}(\mu,1){{< /klmath >}} and {{< klmath inline=true >}}q=\mathcal{N}(0,1){{< /klmath >}}, both choices have expected gradient {{< klmath inline=true >}}\mu{{< /klmath >}}, so only their noise differs. With {{< klmath inline=true >}}t=\mu^2{{< /klmath >}}, the exact per-sample variances are:

{{< klmath >}}
\begin{aligned}
\operatorname{Var}(k_1)&=t, &\operatorname{Var}(k_3)&=e^t-1-t,\\
\operatorname{Var}(k_1\,\partial_\mu\log p_\mu)&=2t+\tfrac14 t^2, &\operatorname{Var}(\partial_\mu k_3)&=e^t(1+4t)-1-3t.
\end{aligned}
{{< /klmath >}}

At {{< klmath inline=true >}}\mu=0.5{{< /klmath >}}, {{< klmath inline=true >}}k_3{{< /klmath >}} estimates the KL with 7.3× lower variance (0.034 versus 0.25), yet its gradient has 1.6× **higher** variance (0.82 versus 0.52). Near the reference, both per-sample gradients reduce to {{< klmath inline=true >}}\mu(x-\mu)^2{{< /klmath >}} to first order, so their variances agree at about {{< klmath inline=true >}}2\mu^2{{< /klmath >}}, even though the variance of {{< klmath inline=true >}}k_3{{< /klmath >}} as a value estimate is much smaller, about {{< klmath inline=true >}}\mu^4/2{{< /klmath >}}. The control variate that makes {{< klmath inline=true >}}k_3{{< /klmath >}} a good value estimate does not carry over to the gradient, and farther from the reference the {{< klmath inline=true >}}q/p{{< /klmath >}} tail makes the gradient noisier. This ordering also holds when each method uses its own variance-minimizing scalar baseline (0.50 versus 0.76 at {{< klmath inline=true >}}\mu=0.5{{< /klmath >}}).

{{< klfigure type="gradientNoise" subtitle=`$p_\mu = \mathcal{N}(\mu, 1)$ · $q = \mathcal{N}(0, 1)$ · only $\mu$ is learned` >}}
Left: exact variance of the KL estimates. Right: exact per-sample variance of the gradient with respect to {{< klmath inline=true >}}\mu{{< /klmath >}}, for {{< klmath inline=true >}}k_1{{< /klmath >}} in reward (blue) and {{< klmath inline=true >}}k_3{{< /klmath >}} as a direct loss (dashed green). Both gradients are unbiased in this example. Logarithmic vertical axes.
{{< /klfigure >}}

{{< klcode title="Check the Gaussian gradient variances" >}}
import numpy as np

rng = np.random.default_rng(0)

# p = N(mu, 1), q = N(0, 1); only mu is learned.
mu = 0.5
eps = rng.normal(size=4_000_000)
x = mu + eps
log_ratio = mu * x - 0.5 * mu**2   # log p(x) - log q(x)
score = x - mu                     # d log p(x) / d mu

k1 = log_ratio
k3 = log_ratio + np.expm1(-log_ratio)
k1_reward = k1 * score                       # detached reward coefficient
k3_loss = -np.expm1(-log_ratio) * score      # d k3 / d mu at fixed x

t = mu**2
print("KL estimate variance:", k1.var(), k3.var())   # t, e^t - 1 - t
print("gradient mean:", k1_reward.mean(), k3_loss.mean())  # both mu
print("gradient variance:", k1_reward.var(), k3_loss.var())
print("exact:", 2 * t + t**2 / 4, np.exp(t) * (1 + 4 * t) - 1 - 3 * t)
{{< /klcode >}}

### Same KL, three different answers

Return to {{< klmath inline=true >}}q=(0.1,0.45,0.45){{< /klmath >}} and the two distributions from earlier, and add a third, {{< klmath inline=true >}}p_3\approx(0.257,0.372,0.372){{< /klmath >}}, which moves mass **toward** the rare outcome. All three have exactly the same {{< klmath inline=true >}}\mathrm{KL}[p\,\|\,q]\approx 0.0997{{< /klmath >}}. Gradients are taken with respect to three softmax logits, each method uses its own variance-minimizing scalar baseline, and the {{< klmath inline=true >}}k_3{{< /klmath >}} column also lists the squared bias of its expected gradient relative to {{< klmath inline=true >}}\nabla D_R{{< /klmath >}}:

| | {{< klmath inline=true >}}\operatorname{Var}(k_1){{< /klmath >}} | {{< klmath inline=true >}}\operatorname{Var}(k_3){{< /klmath >}} | {{< klmath inline=true >}}k_1{{< /klmath >}} in reward: gradient variance | {{< klmath inline=true >}}k_3{{< /klmath >}} as loss: gradient variance | {{< klmath inline=true >}}k_3{{< /klmath >}} as loss: squared bias |
| --- | --- | --- | --- | --- | --- |
| {{< klmath inline=true >}}p_1{{< /klmath >}}, hole | 0.022 | 8.90 | 0.033 | 14.6 | 0.013 |
| {{< klmath inline=true >}}p_2{{< /klmath >}}, smooth shift | 0.178 | 0.0065 | 0.025 | 0.047 | 0.0018 |
| {{< klmath inline=true >}}p_3{{< /klmath >}}, more mass on the rare outcome | 0.245 | 0.019 | 0.115 | 0.060 | 0.0053 |

For a minibatch of {{< klmath inline=true >}}N{{< /klmath >}} samples, the gradient's mean squared error is the squared bias plus the variance divided by {{< klmath inline=true >}}N{{< /klmath >}}. For the hole, {{< klmath inline=true >}}k_3{{< /klmath >}} is worse on every count. For the smooth shift, {{< klmath inline=true >}}k_3{{< /klmath >}} is 27× better as a value estimate, but its gradient is both noisier and biased, so it is worse at every batch size. Only for {{< klmath inline=true >}}p_3{{< /klmath >}} does the lower value variance carry over to the gradient, and even there {{< klmath inline=true >}}k_3{{< /klmath >}} has lower gradient error only for batches smaller than about 10. Beyond that, the bias floor dominates.

These examples are not unusual. On a grid of 2,964 three-outcome pairs {{< klmath inline=true >}}(p,q){{< /klmath >}}, {{< klmath inline=true >}}k_3{{< /klmath >}} had lower value variance in 2,274. In 1,757 of those, its direct-loss gradient had **higher** variance than {{< klmath inline=true >}}k_1{{< /klmath >}} in reward, with baselines for both. Counting bias, {{< klmath inline=true >}}k_3{{< /klmath >}} had lower gradient error in only 236 of them at batch size 4, and 2 at batch size 256.

### What happens during optimization

Finally, we ran stochastic gradient descent on {{< klmath inline=true >}}\mathbb{E}_{p_\theta}[-R]+\mathrm{KL}[p_\theta\,\|\,q]{{< /klmath >}} with minibatches of 32 on-policy samples and 512 samples in total. Each method used its own optimal scalar baseline and its own learning rate, tuned on 1,024 seeds; the table reports the mean excess objective above the optimum on 4,096 separate seeds (lower is better). Only the policy mean is learned.

| Policy and reward | {{< klmath inline=true >}}k_1{{< /klmath >}} in reward | {{< klmath inline=true >}}k_3{{< /klmath >}} as loss |
| --- | --- | --- |
| {{< klmath inline=true >}}\mathcal{N}(\mu,1){{< /klmath >}}, {{< klmath inline=true >}}\mu_0=2{{< /klmath >}}, no reward | {{< klmath inline=true >}}<10^{-20}{{< /klmath >}} | 0.14 |
| {{< klmath inline=true >}}\mathcal{N}(\mu,0.6){{< /klmath >}}, {{< klmath inline=true >}}\mu_0=0.5{{< /klmath >}}, no reward | 0.0011 | 0.012 |
| {{< klmath inline=true >}}\mathcal{N}(\mu,2){{< /klmath >}}, {{< klmath inline=true >}}\mu_0=0.5{{< /klmath >}}, no reward | 0.0021 | **0.0010** |
| {{< klmath inline=true >}}\mathcal{N}(\mu,2){{< /klmath >}}, {{< klmath inline=true >}}\mu_0=0.5{{< /klmath >}}, {{< klmath inline=true >}}R(x)=0.1x{{< /klmath >}} | **0.0021** | 0.0066 |

{{< klmath inline=true >}}k_3{{< /klmath >}} as a loss won once: for the wide policy without a task reward. There, its gradient is far less noisy (0.09 versus 1.25 per sample at the start), and its target, {{< klmath inline=true >}}D_F{{< /klmath >}}, has the same minimizer as {{< klmath inline=true >}}D_R{{< /klmath >}}, so its bias costs nothing. Adding a small task reward separates the two objectives: {{< klmath inline=true >}}D_R{{< /klmath >}} puts the optimum at {{< klmath inline=true >}}\mu=0.1{{< /klmath >}}, but {{< klmath inline=true >}}D_F{{< /klmath >}} pulls the policy to {{< klmath inline=true >}}\mu=0.2{{< /klmath >}}, leaving an excess of 0.005 that no amount of data removes. In the narrow and far cases, the {{< klmath inline=true >}}q/p{{< /klmath >}} tail dominates and {{< klmath inline=true >}}k_3{{< /klmath >}} is an order of magnitude worse or more.

The categorical hole is a reminder that a toy can still surprise: with {{< klmath inline=true >}}p(x_1)=0.001{{< /klmath >}}, the rare outcome is seldom sampled, and the median run of every method barely moved in 512 samples. At batch size 32, the runs that did escape made {{< klmath inline=true >}}k_3{{< /klmath >}} as a loss better on average (0.061 versus 0.095 excess). These are small, fixed-budget experiments with oracle baselines and plain SGD, without clipping, Adam, or off-policy samples. They are not LLM results, but they show that value variance alone does not predict which estimator trains better.

## Epilogue

The shape of the distributions matters when choosing an estimator to **measure** KL. When using that estimate to **optimize** a policy, we also need to ask where gradients flow, and how noisy those gradients are.

For the on-policy setting above, {{< klmath inline=true >}}k_1{{< /klmath >}} as a detached reward coefficient gives the gradient of {{< klmath inline=true >}}\mathrm{KL}[p_\theta\,\|\,q]{{< /klmath >}}. Substituting {{< klmath inline=true >}}k_3{{< /klmath >}} preserves the expected KL value but changes the expected update: it can cancel the learning signal or even reverse its direction. Directly differentiating {{< klmath inline=true >}}k_3{{< /klmath >}} gives yet another update, corresponding to {{< klmath inline=true >}}\mathrm{KL}[q\,\|\,p_\theta]{{< /klmath >}}.

Even when {{< klmath inline=true >}}k_3{{< /klmath >}} is the better estimate of the KL value, its gradient is often the noisier one, and its bias toward the opposite KL direction sets a floor that larger batches cannot remove. An unbiased, low-variance estimate of KL is not necessarily a good estimate of its gradient. The objective, the gradient path, and the gradient's own variance determine which estimator is appropriate.
