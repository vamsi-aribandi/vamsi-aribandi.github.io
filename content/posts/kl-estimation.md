---
title: "The Structure Behind KL Estimation for Reinforcement Learning"
date: 2026-09-30
draft: true
toc: false
klviz: true
summary: "k1 in the reward, k2 as a loss and k3 in both give the same gradient of KL. The real choice is a baseline, and the λ that gives the best KL estimate is not the one that gives the best gradient."
---

KL divergence {{< klmath inline=true >}}\mathrm{KL}[p\,\|\,q] = \mathbb{E}_p[\log(p/q)]{{< /klmath >}} is an important quantity in AI. In reinforcement learning for language models, {{< klmath inline=true >}}\mathrm{KL}[\pi_\theta\,\|\,\pi_\mathrm{ref}]{{< /klmath >}} is used as a regularizer to keep a learned policy {{< klmath inline=true >}}\pi_\theta{{< /klmath >}} near a reference policy {{< klmath inline=true >}}\pi_\mathrm{ref}{{< /klmath >}}, and on-policy distillation (OPD) lets a student policy {{< klmath inline=true >}}\pi_\mathrm{student}{{< /klmath >}} learn from a teacher policy {{< klmath inline=true >}}\pi_\mathrm{teacher}{{< /klmath >}} by minimizing {{< klmath inline=true >}}\mathrm{KL}[\pi_\mathrm{student}\,\|\,\pi_\mathrm{teacher}]{{< /klmath >}}.

Computing KL between language models is generally intractable, because the expectation {{< klmath inline=true >}}\mathbb{E}_p{{< /klmath >}} is a sum over every possible sequence of tokens. Instead we *estimate* it from samples {{< klmath inline=true >}}x \sim p{{< /klmath >}}. Three estimators of KL are common:

{{< klmath >}}
\begin{aligned}
k_1(x) &= \log\frac{p(x)}{q(x)}\\
k_2(x) &= \frac12\left(\log\frac{p(x)}{q(x)}\right)^2\\
k_3(x) &= \log\frac{p(x)}{q(x)} + \frac{q(x)}{p(x)} - 1
\end{aligned}
{{< /klmath >}}

You might recall these estimators from a note by John Schulman[^schulman]. It shows that {{< klmath inline=true >}}k_3{{< /klmath >}} in particular is unbiased like {{< klmath inline=true >}}k_1{{< /klmath >}} and never negative like {{< klmath inline=true >}}k_2{{< /klmath >}}, with lower variance.

How should reverse KL be optimized ({{< klmath inline=true >}}\min_p\mathrm{KL}[p\,\|\,q]{{< /klmath >}}), and does the best estimate of KL lead to the best learning curve?

## Estimating the value of KL

Let's compare the estimators on two Gaussians, {{< klmath inline=true >}}p = \mathcal{N}(\mu, 1){{< /klmath >}} and {{< klmath inline=true >}}q = \mathcal{N}(0, 1){{< /klmath >}}. As we vary {{< klmath inline=true >}}\mu{{< /klmath >}}, {{< klmath inline=true >}}\mathrm{KL}[p\,\|\,q]{{< /klmath >}} is {{< klmath inline=true >}}\mu^2/2{{< /klmath >}}.

First, bias. {{< klmath inline=true >}}k_1{{< /klmath >}} and {{< klmath inline=true >}}k_3{{< /klmath >}} are unbiased, so their expected value is the KL itself. {{< klmath inline=true >}}k_2{{< /klmath >}} overestimates, and the gap grows with the KL:

{{< klfigure type="gaussianBias" subtitle=`$p = \mathcal{N}(\mu, 1)$ · $q = \mathcal{N}(0, 1)$` >}}
Expected value of each estimator. {{< klmath inline=true >}}k_1{{< /klmath >}} and {{< klmath inline=true >}}k_3{{< /klmath >}} lie on the true KL; {{< klmath inline=true >}}k_2{{< /klmath >}} has expectation {{< klmath inline=true >}}\mathrm{KL} + \mathrm{KL}^2/2{{< /klmath >}}. Hover to see {{< klmath inline=true >}}p{{< /klmath >}} and {{< klmath inline=true >}}q{{< /klmath >}}.
{{< /klfigure >}}

Second, variance. {{< klmath inline=true >}}k_3{{< /klmath >}} adds one term to {{< klmath inline=true >}}k_1{{< /klmath >}}:

{{< klmath >}}
k_3 = k_1 + \left(\frac{q}{p} - 1\right).
{{< /klmath >}}

Notice that {{< klmath inline=true >}}\mathbb{E}_p[(q/p)-1] = \sum_x p(x)\,\frac{q(x)}{p(x)} - 1 = \sum_x q(x) - 1 = 0{{< /klmath >}}. So {{< klmath inline=true >}}k_3{{< /klmath >}} is {{< klmath inline=true >}}k_1{{< /klmath >}} plus a *control variate* -- it makes an estimator better when it cancels noise, and worse when it adds noise. {{< klmath inline=true >}}k_3{{< /klmath >}} has lower variance at small KL and much higher variance at large KL:

{{< klfigure type="gaussian" subtitle=`$p = \mathcal{N}(\mu, 1)$ · $q = \mathcal{N}(0, 1)$` >}}
Variance of each estimator, logarithmic scale. Hover to see {{< klmath inline=true >}}p{{< /klmath >}} and {{< klmath inline=true >}}q{{< /klmath >}}.
{{< /klfigure >}}

{{< klcode title="Gaussian example" >}}
import numpy as np

rng = np.random.default_rng(0)

# p = N(mu, 1), q = N(0, 1), so KL[p || q] = mu^2 / 2.
eps = rng.normal(size=1_000_000)
for mu in [0.5, 1.0, 1.5, 2.0]:
    x = mu + eps                      # samples from p
    log_ratio = mu * x - mu**2 / 2    # log p(x) - log q(x)
    k1 = log_ratio
    k2 = log_ratio**2 / 2
    k3 = log_ratio + np.expm1(-log_ratio)
    kl = mu**2 / 2
    # Exact: E[k2] = kl + kl^2 / 2; Var(k1) = 2 kl, Var(k2) = 2 kl^2 + 2 kl^3,
    # Var(k3) = exp(2 kl) - 1 - 2 kl.
    print(f'KL {kl:.3f}  means {k1.mean():.3f} {k2.mean():.3f} {k3.mean():.3f}  '
          f'variances {k1.var():.3f} {k2.var():.3f} {k3.var():.3f}')
{{< /klcode >}}

So as an estimate, the answer depends on how far {{< klmath inline=true >}}p{{< /klmath >}} is from {{< klmath inline=true >}}q{{< /klmath >}}. If all you want is to log the KL, {{< klmath inline=true >}}k_3{{< /klmath >}} is the best choice near {{< klmath inline=true >}}q{{< /klmath >}} and {{< klmath inline=true >}}k_1{{< /klmath >}} far from it.

## The gradient of KL

In RL, the KL estimate is not the end product. We differentiate it. The gradient of KL is as follows:

{{< klmath >}}
\begin{aligned}
\nabla\,\mathrm{KL}[p\,\|\,q] &= \nabla \sum_x p \log\frac{p}{q} = \sum_x \nabla p\,\log\frac{p}{q} + \sum_x p\,\nabla \log p\\
&= \mathbb{E}_p\!\left[\log\frac{p}{q}\,\nabla\log p\right] + \mathbb{E}_p\!\left[\nabla\log p\right]\\
&= \mathbb{E}_p\!\left[\Big(\log\frac{p}{q} + 1\Big)\nabla\log p\right].
\end{aligned}
{{< /klmath >}}

The first sum becomes an expectation through {{< klmath inline=true >}}\nabla p = p\,\nabla\log p{{< /klmath >}}.

We can drop the second term, {{< klmath inline=true >}}\mathbb{E}_p[\nabla\log p]{{< /klmath >}}, as its expectation is zero{{< klhint >}}Probabilities always sum to one, so their changes sum to zero: {{< klmath inline=true >}}\displaystyle\mathbb{E}_p[\nabla\log p] = \sum_x p\,\frac{\nabla p}{p} = \sum_x \nabla p = \nabla \sum_x p = \nabla 1 = 0.{{< /klmath >}}{{< /klhint >}}:

{{< klmath >}}
\nabla\,\mathrm{KL}[p\,\|\,q] = \mathbb{E}_p\!\left[\log\frac{p}{q}\,\nabla\log p\right].
{{< /klmath >}}

This tidies up the math, but there is more going on. {{< klmath inline=true >}}\mathbb{E}_p[\nabla\log p]{{< /klmath >}} is zero, i.e. *in expectation*. It is *not* true that {{< klmath inline=true >}}\nabla\log p{{< /klmath >}} for any single sample. Dropping {{< klmath inline=true >}}\mathbb{E}_p[\nabla\log p]{{< /klmath >}} leaves the expectation of the gradient unchanged, but changes its variance. Keep an eye on this term: every gradient in this section differs only in how much of it we include.

### {{< klmath inline=true >}}\nabla\,\mathrm{KL} \equiv k_1{{< /klmath >}} in the reward {{< klmath inline=true >}}\equiv k_2{{< /klmath >}} as a loss

{{< klmath inline=true >}}k_1{{< /klmath >}} in the reward yields the correct gradient. In policy gradient, a sample {{< klmath inline=true >}}x{{< /klmath >}} with a fixed reward {{< klmath inline=true >}}R(x){{< /klmath >}} contributes {{< klmath inline=true >}}R(x)\,\nabla\log p(x){{< /klmath >}}. Notice that directly setting {{< klmath inline=true >}}R(x){{< /klmath >}} as {{< klmath inline=true >}}k_1{{< /klmath >}} yields the correct gradient:

{{< klmath >}}
\begin{aligned}
&k_1\,\nabla\log p = \log\frac{p}{q}\,\nabla\log p\\
\therefore\ &\mathbb{E}_p[k_1\,\nabla\log p] = \nabla\,\mathrm{KL}[p\,\|\,q].
\end{aligned}
{{< /klmath >}}

Similarly, using {{< klmath inline=true >}}k_2{{< /klmath >}} directly as a loss function also yields the correct gradient:

{{< klmath >}}
\begin{aligned}
&\nabla k_2 = \nabla\,\frac12\left(\log\frac{p}{q}\right)^2 = \log\frac{p}{q}\,\nabla\!\left(\log p - \log q\right) = \log\frac{p}{q}\,\nabla\log p\\
\therefore\ &\mathbb{E}_p[\nabla k_2] = \nabla\,\mathrm{KL}[p\,\|\,q].
\end{aligned}
{{< /klmath >}}

Note that this is *not* the gradient of its expectation {{< klmath inline=true >}}\nabla\mathbb{E}_p[k_2]{{< /klmath >}}.

### The gradients of {{< klmath inline=true >}}k_3{{< /klmath >}}, and {{< klmath inline=true >}}k_\lambda{{< /klmath >}} control variates

Let's introduce a new estimator parameterized by {{< klmath inline=true >}}\lambda{{< /klmath >}}, denoted as {{< klmath inline=true >}}k_\lambda{{< /klmath >}}.
Since {{< klmath inline=true >}}\mathbb{E}_p[\frac{q}{p} - 1] = \sum_x q - 1 = 0{{< /klmath >}}, any multiple of {{< klmath inline=true >}}\frac{q}{p} - 1{{< /klmath >}} can be added to {{< klmath inline=true >}}k_1{{< /klmath >}} without changing its expectation. We will use this fact to define:

{{< klmath >}}
k_\lambda = \log\frac{p}{q} + \lambda\left(\frac{q}{p} - 1\right).
{{< /klmath >}}

Notice that {{< klmath inline=true >}}k_3 = k_{\lambda=1}{{< /klmath >}} and {{< klmath inline=true >}}k_1 = k_{\lambda=0}{{< /klmath >}}. In other words, they can be viewed as instantiations of {{< klmath inline=true >}}k_\lambda{{< /klmath >}}, but with different magnitudes of the control variate added, which is controlled by {{< klmath inline=true >}}\lambda{{< /klmath >}}.

Let's compute the gradient of the expectation of {{< klmath inline=true >}}k_\lambda{{< /klmath >}}:

{{< klmath >}}
\begin{aligned}
\nabla\mathbb{E}_p[k_\lambda] &= \nabla \sum_x p\,k_\lambda = \sum_x \nabla p\,k_\lambda + \sum_x p\,\nabla k_\lambda\\
&= \mathbb{E}_p\!\left[k_\lambda\,\nabla\log p + \nabla k_\lambda\right].
\end{aligned}
{{< /klmath >}}

Since {{< klmath inline=true >}}\nabla \log\frac{p}{q} = \nabla\log p{{< /klmath >}} and {{< klmath inline=true >}}\nabla\frac{q}{p} = -\frac{q}{p}\,\nabla\log p{{< /klmath >}}{{< klhint label="how?" >}}{{< klmath inline=true >}}\displaystyle\nabla \frac{q}{p} = q\,\nabla\frac{1}{p} = -\frac{q}{p^2}\,\nabla p = -\frac{q}{p}\,\frac{\nabla p}{p} = -\frac{q}{p}\,\nabla\log p.{{< /klmath >}}{{< /klhint >}}, every term is a multiple of {{< klmath inline=true >}}\nabla\log p{{< /klmath >}}, and the {{< klmath inline=true >}}\frac{q}{p}{{< /klmath >}} terms cancel:

{{< klmath >}}
\begin{aligned}
k_\lambda\,\nabla\log p + \nabla k_\lambda &= \Big(\log\frac{p}{q} + \cancel{\lambda\,\frac{q}{p}} - \lambda\Big)\nabla\log p + \Big(1 - \cancel{\lambda\,\frac{q}{p}}\Big)\nabla\log p\\
&= \log\frac{p}{q}\,\nabla\log p + (1 - \lambda)\,\nabla\log p.
\end{aligned}
{{< /klmath >}}

Notice that the second term {{< klmath inline=true >}}\mathbb{E}_p[ (1 - \lambda)\,\nabla\log p] = 0{{< /klmath >}}{{< klhint >}}Probabilities always sum to one, so their changes sum to zero: {{< klmath inline=true >}}\displaystyle\mathbb{E}_p[\nabla\log p] = \sum_x p\,\frac{\nabla p}{p} = \sum_x \nabla p = \nabla \sum_x p = \nabla 1 = 0.{{< /klmath >}}{{< /klhint >}}. Therefore:

{{< klmath >}}
\begin{aligned}
\nabla\mathbb{E}_p[k_\lambda] &= \mathbb{E}_p[\log\frac{p}{q}\,\nabla\log p + (1 - \lambda)\,\nabla\log p]\\
&= \mathbb{E}_p[\log\frac{p}{q}\,\nabla\log p]\\
&= \nabla\,\mathrm{KL}[p\,\|\,q]\\
\end{aligned}
{{< /klmath >}}

In canceling this second term, notice that it is a control variate. However, the control variate in {{< klmath inline=true >}}k_\lambda{{< /klmath >}} is {{< klmath inline=true >}}\lambda\left(\frac{q}{p} - 1\right){{< /klmath >}}, and in the per-sample gradient of {{< klmath inline=true >}}\mathbb{E}_p[k_\lambda]{{< /klmath >}} it is {{< klmath inline=true >}}(1 - \lambda)\,\nabla\log p{{< /klmath >}}. This suggests that optimally estimating the value of KL may not correspond to optimally estimating its gradient.

## Choosing {{< klmath inline=true >}}\lambda{{< /klmath >}}

Schulman's note arrives at {{< klmath inline=true >}}k_3{{< /klmath >}} through this same family. It observes that the variance-minimizing {{< klmath inline=true >}}\lambda{{< /klmath >}} "depends on p and q and is hard to calculate analytically", and takes {{< klmath inline=true >}}\lambda = 1{{< /klmath >}} instead, because then the estimate can never be negative[^schulman].

What if we did want to choose {{< klmath inline=true >}}\lambda{{< /klmath >}}? As an estimate of the value, {{< klmath inline=true >}}k_\lambda{{< /klmath >}}'s variance is a quadratic in {{< klmath inline=true >}}\lambda{{< /klmath >}}, smallest at

{{< klmath >}}
\lambda_\text{value} = -\frac{\operatorname{Cov}_p\!\left(\log\frac{p}{q},\ \frac{q}{p}\right)}{\operatorname{Var}_p\!\left(\frac{q}{p}\right)}.
{{< /klmath >}}

But in training we never use the value. We use the gradient {{< klmath inline=true >}}\big(\log\frac{p}{q} + 1 - \lambda\big)\nabla\log p{{< /klmath >}}, and there {{< klmath inline=true >}}\lambda{{< /klmath >}} sets the size of a different control variate {{< klmath inline=true >}}(1 - \lambda)\,\nabla\log p{{< /klmath >}}.
In policy-gradient language, it shifts the reward {{< klmath inline=true >}}\log\frac{p}{q}{{< /klmath >}} by a *baseline*, the standard tool for reducing a policy gradient's variance. The variance of this gradient is smallest at

{{< klmath >}}
\lambda_\text{gradient} = 1 + \frac{\mathbb{E}_p\!\left[\log\frac{p}{q}\,\|\nabla\log p\|^2\right]}{\mathbb{E}_p\!\left[\|\nabla\log p\|^2\right]},
{{< /klmath >}}

which is close to {{< klmath inline=true >}}1 + \mathrm{KL}{{< /klmath >}} when {{< klmath inline=true >}}\|\nabla\log p\|^2{{< /klmath >}} varies little.

For the Gaussians, {{< klmath inline=true >}}p = \mathcal{N}(\mu, 1){{< /klmath >}} and {{< klmath inline=true >}}q = \mathcal{N}(0, 1){{< /klmath >}} with the gradient taken with respect to {{< klmath inline=true >}}\mu{{< /klmath >}}, so that {{< klmath inline=true >}}\nabla\log p = x - \mu{{< /klmath >}}, both have closed forms. Write {{< klmath inline=true >}}g_\lambda = \big(\log\frac{p}{q} + 1 - \lambda\big)\nabla\log p{{< /klmath >}} for the gradient. Then

{{< klmath >}}
\begin{aligned}
\operatorname{Var}(k_\lambda) &= \mu^2\,(1 - 2\lambda) + \lambda^2\left(e^{\mu^2} - 1\right),\\
\operatorname{Var}(g_\lambda) &= \left(\frac{\mu^2}{2} + 1 - \lambda\right)^2 + 2\mu^2,
\end{aligned}
{{< /klmath >}}

so they are smallest at

{{< klmath >}}
\lambda_\text{value} = \frac{\mu^2}{e^{\mu^2} - 1}
\qquad\text{and}\qquad
\lambda_\text{gradient} = 1 + \frac{\mu^2}{2}.
{{< /klmath >}}

Since {{< klmath inline=true >}}\mathrm{KL} = \mu^2/2{{< /klmath >}} here, {{< klmath inline=true >}}\lambda_\text{gradient}{{< /klmath >}} is exactly {{< klmath inline=true >}}1 + \mathrm{KL}{{< /klmath >}}.

{{< klderiv title="Derive the two variances" >}}
Write {{< klmath inline=true >}}x = \mu + z{{< /klmath >}} with {{< klmath inline=true >}}z \sim \mathcal{N}(0, 1){{< /klmath >}}. Then {{< klmath inline=true >}}\log\frac{p}{q} = \frac{\mu^2}{2} + \mu z{{< /klmath >}}, {{< klmath inline=true >}}\frac{q}{p} = e^{-\mu z - \mu^2/2}{{< /klmath >}}, and {{< klmath inline=true >}}\nabla_\mu \log p = x - \mu = z{{< /klmath >}}.

**The estimate.** {{< klmath inline=true >}}\operatorname{Var}(\log\frac{p}{q}) = \mu^2{{< /klmath >}}. From {{< klmath inline=true >}}\mathbb{E}[e^{-2\mu z}] = e^{2\mu^2}{{< /klmath >}}, {{< klmath inline=true >}}\operatorname{Var}(\frac{q}{p}) = e^{\mu^2} - 1{{< /klmath >}}. From {{< klmath inline=true >}}\mathbb{E}[z\,e^{-\mu z}] = -\mu\,e^{\mu^2/2}{{< /klmath >}}, {{< klmath inline=true >}}\operatorname{Cov}(\log\frac{p}{q}, \frac{q}{p}) = -\mu^2{{< /klmath >}}. So {{< klmath inline=true >}}\operatorname{Var}(k_\lambda) = \mu^2 - 2\lambda\mu^2 + \lambda^2(e^{\mu^2} - 1){{< /klmath >}}, a quadratic in {{< klmath inline=true >}}\lambda{{< /klmath >}} with its minimum at {{< klmath inline=true >}}\mu^2 / (e^{\mu^2} - 1){{< /klmath >}}.

**The gradient.** Let {{< klmath inline=true >}}a = \frac{\mu^2}{2} + 1 - \lambda{{< /klmath >}}, so the gradient is {{< klmath inline=true >}}(a + \mu z)\,z{{< /klmath >}}. Its mean is {{< klmath inline=true >}}\mu{{< /klmath >}}, the gradient of {{< klmath inline=true >}}\mu^2/2{{< /klmath >}}. Since {{< klmath inline=true >}}\mathbb{E}[z^2] = 1{{< /klmath >}}, {{< klmath inline=true >}}\mathbb{E}[z^3] = 0{{< /klmath >}} and {{< klmath inline=true >}}\mathbb{E}[z^4] = 3{{< /klmath >}}, its second moment is {{< klmath inline=true >}}a^2 + 3\mu^2{{< /klmath >}}, and its variance is {{< klmath inline=true >}}a^2 + 2\mu^2{{< /klmath >}}, smallest at {{< klmath inline=true >}}a = 0{{< /klmath >}}.
{{< /klderiv >}}

The two optima move in opposite directions. As {{< klmath inline=true >}}p{{< /klmath >}} moves away from {{< klmath inline=true >}}q{{< /klmath >}}, the best estimate slides from {{< klmath inline=true >}}k_3{{< /klmath >}} toward {{< klmath inline=true >}}k_1{{< /klmath >}}, while the best gradient wants a larger and larger baseline. Move the slider to see it.

{{< klfigure type="gaussianLambda" subtitle=`$p = \mathcal{N}(\mu, 1)$ · $q = \mathcal{N}(0, 1)$ · $k_\lambda = k_1 + \lambda\,(q/p - 1)$` >}}
Diamonds mark each minimum. Near {{< klmath inline=true >}}q{{< /klmath >}}, both minima sit at {{< klmath inline=true >}}\lambda = 1{{< /klmath >}}, which is {{< klmath inline=true >}}k_3{{< /klmath >}}. Far from {{< klmath inline=true >}}q{{< /klmath >}}, they are on opposite sides of it.
{{< /klfigure >}}

### A toy: descending KL with each {{< klmath inline=true >}}\lambda{{< /klmath >}}

Does the difference matter in training? Take a softmax policy over 1,000 actions and a fixed reference {{< klmath inline=true >}}q{{< /klmath >}}, start the policy far from {{< klmath inline=true >}}q{{< /klmath >}} ({{< klmath inline=true >}}\mathrm{KL} = 2.43{{< /klmath >}}), and minimize {{< klmath inline=true >}}\mathrm{KL}[p\,\|\,q]{{< /klmath >}} alone by plain SGD on the logits. Each step samples 16 actions and averages {{< klmath inline=true >}}\big(\log\frac{p}{q} + 1 - \lambda\big)\nabla\log p{{< /klmath >}}, with {{< klmath inline=true >}}\lambda{{< /klmath >}} set four ways: {{< klmath inline=true >}}\lambda = 0{{< /klmath >}} ({{< klmath inline=true >}}k_1{{< /klmath >}} differentiated exactly); {{< klmath inline=true >}}\lambda = 1{{< /klmath >}} ({{< klmath inline=true >}}k_3{{< /klmath >}}); {{< klmath inline=true >}}\lambda_\text{value}{{< /klmath >}}; and {{< klmath inline=true >}}\lambda_\text{gradient}{{< /klmath >}}, the last two computed exactly from the current policy. Every choice gives an unbiased gradient, so any difference comes from variance.

{{< klfigure type="lambdaDescent" data="klviz/lambda.json" subtitle=`1,000 actions · 16 samples per step · 32 seeds · median and interquartile range` >}}
KL and gradient variance are exact at each step. A policy that collapses onto a single action stops moving, because {{< klmath inline=true >}}\nabla\log p{{< /klmath >}} is then zero for the only action it samples.
{{< /klfigure >}}

## Epilogue

A KL estimator has two jobs in RL.

**Measuring.** Neither estimator always wins. {{< klmath inline=true >}}k_3{{< /klmath >}} has lower variance near {{< klmath inline=true >}}q{{< /klmath >}}; {{< klmath inline=true >}}k_1{{< /klmath >}} is safer far from {{< klmath inline=true >}}q{{< /klmath >}}; {{< klmath inline=true >}}k_2{{< /klmath >}} is biased.

**Training.** The estimator does not matter. {{< klmath inline=true >}}k_1{{< /klmath >}} in the reward, {{< klmath inline=true >}}k_2{{< /klmath >}} as a loss, and {{< klmath inline=true >}}k_3{{< /klmath >}} in both give the same gradient for every sample. What matters is keeping the right terms, and, if you tune anything, tuning the baseline for the gradient rather than the control variate for the estimate.

**Choose a KL estimator for the number it reports, and a baseline for the gradient it produces.**

[^schulman]: John Schulman, ["Approximating KL Divergence"](https://joschu.net/blog/kl-approx.html), 2020.
[^deepseek]: DeepSeek-AI, ["DeepSeek-V3.2"](https://arxiv.org/html/2512.02556v1#S3.SS1), 2025.
[^opd]: Thinking Machines Lab, ["On-Policy Distillation"](https://thinkingmachines.ai/blog/on-policy-distillation/), 2025.
[^cursor]: Cursor Research, ["Composer 2 Technical Report"](https://arxiv.org/abs/2603.24477), 2026, Section 4.1.
[^tang]: Yunhao Tang and Rémi Munos, ["On a few pitfalls in KL divergence gradient estimation for RL"](https://arxiv.org/abs/2506.09477), 2025.
[^rethinking]: Kezhao Liu et al., ["Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization"](https://arxiv.org/abs/2510.01555), 2025.
[^wang]: Xihuai Wang, ["Choosing KL Estimators in RL: From Value Unbiasedness to Gradient Correctness"](https://xihuai18.github.io/reinforcement-learning/2025/12/01/kl-estimators-en.html), 2025.
