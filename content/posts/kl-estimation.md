---
title: "The best estimate of KL ≢ the best estimate its gradient"
date: 2026-10-02
draft: false
toc: false
klviz: true
summary: "Even after finding an estimate of KL that produces the correct gradient, reducing the variance of the estimator doesn't necessarily translate to reducing the variance of the corresponding gradient."
---

KL divergence {{< klmath inline=true >}}\mathrm{KL}[p\,\|\,q] = \mathbb{E}_p[\log(p/q)]{{< /klmath >}} is an important quantity in AI.

In reinforcement learning for language models, {{< klmath inline=true >}}\mathrm{KL}[\pi_\theta\,\|\,\pi_\mathrm{ref}]{{< /klmath >}} is used as a regularizer to keep a learned policy {{< klmath inline=true >}}\pi_\theta{{< /klmath >}} near a reference policy {{< klmath inline=true >}}\pi_\mathrm{ref}{{< /klmath >}}, and on-policy distillation (OPD) lets a student policy {{< klmath inline=true >}}\pi_\mathrm{student}{{< /klmath >}} learn from a teacher policy {{< klmath inline=true >}}\pi_\mathrm{teacher}{{< /klmath >}} by minimizing {{< klmath inline=true >}}\mathrm{KL}[\pi_\mathrm{student}\,\|\,\pi_\mathrm{teacher}]{{< /klmath >}}.

Exactly computing KL between two language models is intractable, because the expectation {{< klmath inline=true >}}\mathbb{E}_p{{< /klmath >}} is a sum over every possible sequence of tokens. Instead we *estimate* it from samples {{< klmath inline=true >}}x \sim p{{< /klmath >}}. 

Three estimators of KL are common:

{{< klmath >}}
\begin{aligned}
k_1(x) &= \log\frac{p(x)}{q(x)}\\
k_2(x) &= \frac12\left(\log\frac{p(x)}{q(x)}\right)^2\\
k_3(x) &= \log\frac{p(x)}{q(x)} + \frac{q(x)}{p(x)} - 1
\end{aligned}
{{< /klmath >}}

You might recall these estimators from an excellent post by John Schulman[^schulman]. It shows that {{< klmath inline=true >}}k_3{{< /klmath >}} in particular is unbiased like {{< klmath inline=true >}}k_1{{< /klmath >}} and never negative like {{< klmath inline=true >}}k_2{{< /klmath >}}, with lower variance.

We can use these functions to estimate KL, but how should we use them to *minimize* it? Let's explore how to minimize reverse KL ({{< klmath inline=true >}}\min_p\mathrm{KL}[p\,\|\,q]{{< /klmath >}}), since it is relevant for reinforcement learning.

## Estimating the value of KL

Before trying to minimize KL, let's compare the estimators on two Gaussians, {{< klmath inline=true >}}p = \mathcal{N}(\mu, 1){{< /klmath >}} and {{< klmath inline=true >}}q = \mathcal{N}(0, 1){{< /klmath >}}. Here, {{< klmath inline=true >}}\mathrm{KL}[p\,\|\,q]{{< /klmath >}} is exactly {{< klmath inline=true >}}\mu^2/2{{< /klmath >}}{{< klhint >}}The two Gaussian densities have the same normalizing constant, so {{< klmath inline=true >}}\displaystyle\log\frac{p(x)}{q(x)} = -\frac{(x-\mu)^2}{2} + \frac{x^2}{2} = \mu x - \frac{\mu^2}{2}{{< /klmath >}}. Taking the expectation under {{< klmath inline=true >}}p{{< /klmath >}}, where {{< klmath inline=true >}}\mathbb{E}_p[x] = \mu{{< /klmath >}}, gives {{< klmath inline=true >}}\displaystyle\mathrm{KL}[p\,\|\,q] = \mu\mathbb{E}_p[x] - \frac{\mu^2}{2} = \frac{\mu^2}{2}{{< /klmath >}}.{{< /klhint >}}.

Let's first examine bias. An estimator is unbiased when its expectation equals the true KL.

{{< klmath inline=true >}}k_1{{< /klmath >}} is unbiased by the definition of KL:

{{< klmath >}}
\mathbb{E}_p[k_1] = \mathbb{E}_p\!\left[\log\frac{p}{q}\right] = \mathrm{KL}[p\,\|\,q].
{{< /klmath >}}

{{< klmath inline=true >}}k_2{{< /klmath >}} is biased. For these Gaussians,

{{< klmath >}}
\mathbb{E}_p[k_2] = \frac12\mathbb{E}_p\!\left[\left(\mu x - \frac{\mu^2}{2}\right)^2\right] = \frac{\mu^2}{2} + \frac{\mu^4}{8} = \mathrm{KL}[p\,\|\,q] + \frac{\mu^4}{8}.
{{< /klmath >}}

{{< klmath inline=true >}}k_3{{< /klmath >}} adds a zero-mean term to {{< klmath inline=true >}}k_1{{< /klmath >}}, so it stays unbiased:

{{< klmath >}}
\begin{aligned}
&\mathbb{E}_p\!\left[\frac{q}{p} - 1\right] = \int p(x)\frac{q(x)}{p(x)}\,dx - 1 = 0,\\
\Longrightarrow\quad &\mathbb{E}_p[k_3] = \mathbb{E}_p[k_1] + \underbrace{\mathbb{E}_p\!\left[\frac{q}{p} - 1\right]}_{0} = \mathrm{KL}[p\,\|\,q].
\end{aligned}
{{< /klmath >}}

{{< klfigure type="gaussianBias" subtitle=`$p = \mathcal{N}(\mu, 1)$ · $q = \mathcal{N}(0, 1)$` >}}
Expected value of each estimator. Hover to see {{< klmath inline=true >}}p{{< /klmath >}} and {{< klmath inline=true >}}q{{< /klmath >}}.
{{< /klfigure >}}

As the graph shows, {{< klmath inline=true >}}k_1{{< /klmath >}} and {{< klmath inline=true >}}k_3{{< /klmath >}} are unbiased, while {{< klmath inline=true >}}k_2{{< /klmath >}} overestimates KL. The key to {{< klmath inline=true >}}k_3{{< /klmath >}} staying unbiased is that its added term, {{< klmath inline=true >}}q/p - 1{{< /klmath >}}, is zero *in expectation*. It can be nonzero for an individual sample, but it does not change the estimator's mean.

This zero-mean term is a *control variate*: we add it to an estimator to adjust its variance without changing its expectation. It can cancel noise and decrease variance, or add noise and increase variance. For these Gaussians, {{< klmath inline=true >}}k_3{{< /klmath >}} has lower variance near {{< klmath inline=true >}}q{{< /klmath >}} and higher variance farther away:

{{< klfigure type="gaussian" subtitle=`$p = \mathcal{N}(\mu, 1)$ · $q = \mathcal{N}(0, 1)$` >}}
Variance of each estimator, logarithmic scale. Hover to see {{< klmath inline=true >}}p{{< /klmath >}} and {{< klmath inline=true >}}q{{< /klmath >}}.
{{< /klfigure >}}

{{< klderiv title="How the control variate changes variance" >}}
Write {{< klmath inline=true >}}c = q/p - 1{{< /klmath >}}, so {{< klmath inline=true >}}k_3 = k_1 + c{{< /klmath >}}. The variance of their sum separates into the original noise, the added noise, and their covariance:

{{< klmath >}}
\operatorname{Var}_p(k_3) = \operatorname{Var}_p(k_1) + \operatorname{Var}_p(c) + 2\operatorname{Cov}_p(k_1,c).
{{< /klmath >}}

For the Gaussians, write a sample as {{< klmath inline=true >}}x = \mu + z{{< /klmath >}}, where {{< klmath inline=true >}}z \sim \mathcal{N}(0,1){{< /klmath >}}. Substituting into the log ratio derived above gives

{{< klmath >}}
\begin{aligned}
k_1 &= \mu x - \frac{\mu^2}{2} = \frac{\mu^2}{2} + \mu z,\\
c &= e^{-\mu^2/2 - \mu z} - 1.
\end{aligned}
{{< /klmath >}}

Since {{< klmath inline=true >}}\mathbb{E}[z] = 0{{< /klmath >}} and {{< klmath inline=true >}}\operatorname{Var}(z) = 1{{< /klmath >}}, {{< klmath inline=true >}}\operatorname{Var}_p(k_1) = \mu^2{{< /klmath >}}. The Gaussian identity {{< klmath inline=true >}}\mathbb{E}[e^{tz}] = e^{t^2/2}{{< /klmath >}} gives the added term's variance:

{{< klmath >}}
\operatorname{Var}_p(c) = \mathbb{E}_p[c^2] = e^{-\mu^2}\mathbb{E}[e^{-2\mu z}] - 1 = e^{\mu^2} - 1.
{{< /klmath >}}

Its covariance with {{< klmath inline=true >}}k_1{{< /klmath >}} is negative. Using {{< klmath inline=true >}}\mathbb{E}[z e^{tz}] = t e^{t^2/2}{{< /klmath >}},

{{< klmath >}}
\operatorname{Cov}_p(k_1,c) = \mathbb{E}[\mu z\,c] = \mu e^{-\mu^2/2}\mathbb{E}[z e^{-\mu z}] = -\mu^2.
{{< /klmath >}}

Putting these together,

{{< klmath >}}
\operatorname{Var}_p(k_3) = \underbrace{\mu^2}_{\text{original noise}} + \underbrace{(e^{\mu^2} - 1)}_{\text{added noise}} \underbrace{-\,2\mu^2}_{\text{cancellation}} = e^{\mu^2} - 1 - \mu^2.
{{< /klmath >}}

For completeness, {{< klmath inline=true >}}k_2 = \frac12(\mu^2/2 + \mu z)^2{{< /klmath >}}. Using {{< klmath inline=true >}}\mathbb{E}[z^3] = 0{{< /klmath >}} and {{< klmath inline=true >}}\mathbb{E}[z^4] = 3{{< /klmath >}} gives

{{< klmath >}}
\operatorname{Var}_p(k_2) = \frac{\mu^4}{2} + \frac{\mu^6}{4}.
{{< /klmath >}}

If we weight the control variate by {{< klmath inline=true >}}\lambda{{< /klmath >}}, the cancellation scales linearly while the added variance scales quadratically:

{{< klmath >}}
\operatorname{Var}_p(k_1 + \lambda c) = \mu^2 - 2\lambda\mu^2 + \lambda^2(e^{\mu^2} - 1).
{{< /klmath >}}
{{< /klderiv >}}

Because the control variate is zero-mean, any multiple of it can be added without bising the estimator, yielding the family of estimators:

{{< klmath >}}
k_\lambda = \log\frac{p(x)}{q(x)} + \lambda\left(\frac{q}{p} - 1\right).
{{< /klmath >}}

Here, {{< klmath inline=true >}}\lambda = 0{{< /klmath >}} gives {{< klmath inline=true >}}k_1{{< /klmath >}}, and {{< klmath inline=true >}}\lambda = 1{{< /klmath >}} gives {{< klmath inline=true >}}k_3{{< /klmath >}}. We'll revisit how to choose {{< klmath inline=true >}}\lambda{{< /klmath >}}.

So choosing the best estimate depends on how far {{< klmath inline=true >}}p{{< /klmath >}} is from {{< klmath inline=true >}}q{{< /klmath >}}. If all you want is to estimate KL, {{< klmath inline=true >}}k_3{{< /klmath >}} is the best choice near {{< klmath inline=true >}}q{{< /klmath >}} and {{< klmath inline=true >}}k_1{{< /klmath >}} far from it.

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

This tidies up the math, but there is more going on. {{< klmath inline=true >}}\mathbb{E}_p[\nabla\log p]{{< /klmath >}} is zero, i.e. *in expectation* it won't change the gradient. In other words, {{< klmath inline=true >}}\nabla\log p{{< /klmath >}} plays the same role for the gradient that {{< klmath inline=true >}}\frac{q}{p} - 1{{< /klmath >}} plays for the estimate: a control variate with mean zero, which can be kept in any amount without changing the expectation.

### {{< klmath inline=true >}}\lambda=1{{< /klmath >}}: {{< klmath inline=true >}}\nabla\,\mathrm{KL} \equiv k_1{{< /klmath >}} in the reward {{< klmath inline=true >}}\equiv k_2{{< /klmath >}} as a loss

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

Both {{< klmath inline=true >}}k_1{{< /klmath >}} in the reward and {{< klmath inline=true >}}k_2{{< /klmath >}} as a loss keep none of the {{< klmath inline=true >}}\nabla\log p{{< /klmath >}} term.

### From control variate to baseline

Now return to the family {{< klmath inline=true >}}k_\lambda = k_1 + \lambda\left(\frac{q}{p} - 1\right){{< /klmath >}} from the estimation section, where {{< klmath inline=true >}}\lambda{{< /klmath >}} sets the magnitude of the estimate's control variate. Since {{< klmath inline=true >}}k_\lambda{{< /klmath >}} is unbiased, we can differentiate its expectation the same way we differentiated KL:

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

The second term can be canceled, as {{< klmath inline=true >}}\mathbb{E}_p[ (1 - \lambda)\,\nabla\log p] = 0{{< /klmath >}}{{< klhint >}}Probabilities always sum to one, so their changes sum to zero: {{< klmath inline=true >}}\displaystyle\mathbb{E}_p[\nabla\log p] = \sum_x p\,\frac{\nabla p}{p} = \sum_x \nabla p = \nabla \sum_x p = \nabla 1 = 0.{{< /klmath >}}{{< /klhint >}}. Therefore, it is a control variate:

{{< klmath >}}
\begin{aligned}
\nabla\mathbb{E}_p[k_\lambda] &= \mathbb{E}_p[\log\frac{p}{q}\,\nabla\log p + (1 - \lambda)\,\nabla\log p]\\
&= \mathbb{E}_p[\log\frac{p}{q}\,\nabla\log p]\\
&= \nabla\,\mathrm{KL}[p\,\|\,q]\\
\end{aligned}
{{< /klmath >}}

Notice that the control variate in {{< klmath inline=true >}}k_\lambda{{< /klmath >}} is {{< klmath inline=true >}}\lambda\left(\frac{q}{p} - 1\right){{< /klmath >}}, but in the {{< klmath inline=true >}}\nabla\mathbb{E}_p[k_\lambda]{{< /klmath >}} it is {{< klmath inline=true >}}(1 - \lambda)\,\nabla\log p{{< /klmath >}}. This suggests that choosing {{< klmath inline=true >}}\lambda{{< /klmath >}} to minimize the variance of KL might not minimize the variance of its gradient.
{{< klmath inline=true >}}k_1{{< /klmath >}} ({{< klmath inline=true >}}\lambda = 0{{< /klmath >}}) keeps all of the {{< klmath inline=true >}}\nabla\log p{{< /klmath >}} term, the same gradient as differentiating KL directly, and {{< klmath inline=true >}}k_3{{< /klmath >}} ({{< klmath inline=true >}}\lambda = 1{{< /klmath >}}) keeps none of it, the same gradient as {{< klmath inline=true >}}k_1{{< /klmath >}} in the reward and {{< klmath inline=true >}}k_2{{< /klmath >}} as a loss.

## The best estimate is not the best gradient

Schulman's note arrives at {{< klmath inline=true >}}k_3{{< /klmath >}} through this same family. It observes that the variance-minimizing {{< klmath inline=true >}}\lambda{{< /klmath >}} "depends on p and q and is hard to calculate analytically", and takes {{< klmath inline=true >}}\lambda = 1{{< /klmath >}} instead, because then the estimate can never be negative[^schulman].

What if we did want to choose {{< klmath inline=true >}}\lambda{{< /klmath >}}? As an estimate of the value, {{< klmath inline=true >}}k_\lambda{{< /klmath >}}'s variance is a quadratic in {{< klmath inline=true >}}\lambda{{< /klmath >}}, smallest at

{{< klmath >}}
\lambda_\text{value} = -\frac{\operatorname{Cov}_p\!\left(\log\frac{p}{q},\ \frac{q}{p}\right)}{\operatorname{Var}_p\!\left(\frac{q}{p}\right)}.
{{< /klmath >}}

But in training we usually don't use the value of KL. We use its gradient {{< klmath inline=true >}}\big(\log\frac{p}{q} + 1 - \lambda\big)\nabla\log p{{< /klmath >}}, and there {{< klmath inline=true >}}\lambda{{< /klmath >}} sets the size of a different control variate {{< klmath inline=true >}}(1 - \lambda)\,\nabla\log p{{< /klmath >}}.
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

### A simple bandit experiment: optimizing KL with each {{< klmath inline=true >}}\lambda{{< /klmath >}}

Does the difference matter in training? Let's take a softmax policy over 1,000 actions (i.e. a bandit) and a fixed reference {{< klmath inline=true >}}q{{< /klmath >}}, start the policy far from {{< klmath inline=true >}}q{{< /klmath >}} ({{< klmath inline=true >}}\mathrm{KL} = 2.43{{< /klmath >}}), and minimize {{< klmath inline=true >}}\mathrm{KL}[p\,\|\,q]{{< /klmath >}} alone by plain SGD on the logits. Each step samples 16 actions and averages {{< klmath inline=true >}}\big(\log\frac{p}{q} + 1 - \lambda\big)\nabla\log p{{< /klmath >}}, with {{< klmath inline=true >}}\lambda{{< /klmath >}} set four ways: {{< klmath inline=true >}}\lambda = 0{{< /klmath >}} ({{< klmath inline=true >}}k_1{{< /klmath >}} differentiated exactly); {{< klmath inline=true >}}\lambda = 1{{< /klmath >}} ({{< klmath inline=true >}}k_3{{< /klmath >}}); {{< klmath inline=true >}}\lambda_\text{value}{{< /klmath >}}; and {{< klmath inline=true >}}\lambda_\text{gradient}{{< /klmath >}}, the last two computed exactly from the current policy. Every choice gives an unbiased gradient, so any difference comes from variance.

{{< klfigure type="lambdaDescent" data="klviz/lambda.json" subtitle=`1,000 actions · 16 samples per step · 32 seeds · median and interquartile range` >}}
KL and gradient variance are exact at each step. A policy that collapses onto a single action stops moving, because {{< klmath inline=true >}}\nabla\log p{{< /klmath >}} is then zero for the only action it samples.
{{< /klfigure >}}

## Epilogue

A KL estimator has two jobs in RL: reporting a number, and producing a gradient.

**Measuring.** Neither estimator always wins. {{< klmath inline=true >}}k_3{{< /klmath >}} has lower variance near {{< klmath inline=true >}}q{{< /klmath >}}; {{< klmath inline=true >}}k_1{{< /klmath >}} is safer far from {{< klmath inline=true >}}q{{< /klmath >}}; {{< klmath inline=true >}}k_2{{< /klmath >}} is biased. Within the family {{< klmath inline=true >}}k_\lambda{{< /klmath >}}, the variance-minimizing {{< klmath inline=true >}}\lambda{{< /klmath >}} slides from {{< klmath inline=true >}}k_3{{< /klmath >}} toward {{< klmath inline=true >}}k_1{{< /klmath >}} as {{< klmath inline=true >}}p{{< /klmath >}} moves away from {{< klmath inline=true >}}q{{< /klmath >}}.

**Training.** {{< klmath inline=true >}}k_1{{< /klmath >}} in the reward, {{< klmath inline=true >}}k_2{{< /klmath >}} as a loss, and {{< klmath inline=true >}}k_3{{< /klmath >}} differentiated through its expectation give the same gradient for every sample. Across {{< klmath inline=true >}}k_\lambda{{< /klmath >}}, the gradient changes only by the zero-mean term {{< klmath inline=true >}}(1 - \lambda)\,\nabla\log p{{< /klmath >}}, a baseline, and the {{< klmath inline=true >}}\lambda{{< /klmath >}} that minimizes its variance sits near {{< klmath inline=true >}}1 + \mathrm{KL}{{< /klmath >}}, on the other side of {{< klmath inline=true >}}k_3{{< /klmath >}}. In the bandit experiment, that {{< klmath inline=true >}}\lambda{{< /klmath >}} kept training at step sizes where the others collapsed.

**The best estimate of KL is not the best gradient of KL. Choose an estimator for the number it reports, and a baseline for the gradient it produces.**

[^schulman]: John Schulman, ["Approximating KL Divergence"](https://joschu.net/blog/kl-approx.html), 2020.
[^deepseek]: DeepSeek-AI, ["DeepSeek-V3.2"](https://arxiv.org/html/2512.02556v1#S3.SS1), 2025.
[^opd]: Thinking Machines Lab, ["On-Policy Distillation"](https://thinkingmachines.ai/blog/on-policy-distillation/), 2025.
[^cursor]: Cursor Research, ["Composer 2 Technical Report"](https://arxiv.org/abs/2603.24477), 2026, Section 4.1.
[^tang]: Yunhao Tang and Rémi Munos, ["On a few pitfalls in KL divergence gradient estimation for RL"](https://arxiv.org/abs/2506.09477), 2025.
[^rethinking]: Kezhao Liu et al., ["Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization"](https://arxiv.org/abs/2510.01555), 2025.
[^wang]: Xihuai Wang, ["Choosing KL Estimators in RL: From Value Unbiasedness to Gradient Correctness"](https://xihuai18.github.io/reinforcement-learning/2025/12/01/kl-estimators-en.html), 2025.
