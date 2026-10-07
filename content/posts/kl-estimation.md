---
title: "The best estimate of KL ≢ the best estimate of its gradient"
date: 2026-10-02
draft: false
toc: false
klviz: true
summary: "Even after finding an estimate of KL that produces the correct gradient, reducing the variance of the estimator doesn't necessarily translate to reducing the variance of the corresponding gradient."
---

KL divergence {{< klmath inline=true >}}\mathrm{KL}[p\,\|\,q] = \mathbb{E}_{x\sim p}\!\left[\log\frac{p(x)}{q(x)}\right]{{< /klmath >}} is an important quantity in AI.

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

You might recall these estimators from an excellent [post by John Schulman](https://joschu.net/blog/kl-approx.html). It shows that {{< klmath inline=true >}}k_3{{< /klmath >}} in particular is unbiased like {{< klmath inline=true >}}k_1{{< /klmath >}} and never negative like {{< klmath inline=true >}}k_2{{< /klmath >}}, with lower variance.

We can use these functions to estimate KL, but how should we use them to *minimize* it? Let's explore how to minimize reverse KL ({{< klmath inline=true >}}\min_p\mathrm{KL}[p\,\|\,q]{{< /klmath >}}), since it is relevant for reinforcement learning.

## Estimating the value of KL

Before trying to minimize KL, let's compare the estimators on two Gaussians, {{< klmath inline=true >}}p = \mathcal{N}(\mu, 1){{< /klmath >}} and {{< klmath inline=true >}}q = \mathcal{N}(0, 1){{< /klmath >}}. Here, {{< klmath inline=true >}}\mathrm{KL}[p\,\|\,q]{{< /klmath >}} is exactly {{< klmath inline=true >}}\mu^2/2{{< /klmath >}}{{< klhint >}}{{< klmath inline=true >}}\displaystyle\log\frac{p}{q} = -\frac{(x-\mu)^2}{2} + \frac{x^2}{2} = \mu x - \frac{\mu^2}{2}{{< /klmath >}}. Taking the expectation under {{< klmath inline=true >}}p{{< /klmath >}}, where {{< klmath inline=true >}}\mathbb{E}_p[x] = \mu{{< /klmath >}}, gives {{< klmath inline=true >}}\displaystyle\mathrm{KL}[p\,\|\,q] = \mu\mathbb{E}_p[x] - \frac{\mu^2}{2} = \frac{\mu^2}{2}{{< /klmath >}}.{{< /klhint >}}.

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
&\mathbb{E}_p\!\left[\frac{q}{p} - 1\right] = \int p\frac{q}{p}\,dx - 1 = 0,\\
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

Because the control variate is zero-mean, any multiple of it can be added without biasing the estimator, yielding the family of estimators:

{{< klmath >}}
k_\lambda = \log\frac{p}{q} + \lambda\left(\frac{q}{p} - 1\right).
{{< /klmath >}}

Here, {{< klmath inline=true >}}\lambda = 0{{< /klmath >}} gives {{< klmath inline=true >}}k_1{{< /klmath >}}, and {{< klmath inline=true >}}\lambda = 1{{< /klmath >}} gives {{< klmath inline=true >}}k_3{{< /klmath >}}. We'll revisit how to choose {{< klmath inline=true >}}\lambda{{< /klmath >}}.

So choosing the best estimate depends on how far {{< klmath inline=true >}}p{{< /klmath >}} is from {{< klmath inline=true >}}q{{< /klmath >}}. If all you want is to estimate KL, {{< klmath inline=true >}}k_3{{< /klmath >}} is the best choice near {{< klmath inline=true >}}q{{< /klmath >}} and {{< klmath inline=true >}}k_1{{< /klmath >}} far from it.

## The gradient of KL

In RL, the KL estimate is not the end product. We differentiate it. Using the log derivative trick {{< klmath inline=true >}}\nabla p = p\,\nabla\log p{{< /klmath >}}{{< klhint >}}By the chain rule, {{< klmath inline=true >}}\displaystyle\nabla\log p = \frac{\nabla p}{p}{{< /klmath >}}. Multiplying by {{< klmath inline=true >}}p{{< /klmath >}} gives {{< klmath inline=true >}}\nabla p = p\,\nabla\log p{{< /klmath >}}. This lets us rewrite {{< klmath inline=true >}}\sum_x \nabla p\,f(x){{< /klmath >}} as {{< klmath inline=true >}}\mathbb{E}_p[f(x)\,\nabla\log p]{{< /klmath >}}.{{< /klhint >}}, the gradient of KL is as follows:

{{< klmath >}}
\begin{aligned}
\nabla\,\mathrm{KL}[p\,\|\,q] &= \nabla \sum_x p \log\frac{p}{q} = \sum_x \nabla p\,\log\frac{p}{q} + \sum_x p\,\nabla \log p\\
&= \mathbb{E}_p\!\left[\log\frac{p}{q}\,\nabla\log p\right] + \mathbb{E}_p\!\left[\nabla\log p\right]\\
&= \mathbb{E}_p\!\left[\Big(\log\frac{p}{q} + 1\Big)\nabla\log p\right].
\end{aligned}
{{< /klmath >}}

We can drop the second term, {{< klmath inline=true >}}\mathbb{E}_p[\nabla\log p]{{< /klmath >}}, as its expectation is zero{{< klhint >}}Probabilities always sum to one, so their changes sum to zero: {{< klmath inline=true >}}\displaystyle\mathbb{E}_p[\nabla\log p] = \sum_x p\,\frac{\nabla p}{p} = \sum_x \nabla p = \nabla \sum_x p = \nabla 1 = 0.{{< /klmath >}}{{< /klhint >}}:

{{< klmath >}}
\nabla\,\mathrm{KL}[p\,\|\,q] = \mathbb{E}_p\!\left[\log\frac{p}{q}\,\nabla\log p\right].
{{< /klmath >}}

This tidies up the math. However, notice that {{< klmath inline=true >}}\nabla\log p{{< /klmath >}} plays the same role for the gradient that {{< klmath inline=true >}}\frac{q}{p} - 1{{< /klmath >}} plays for the estimate: a control variate with mean zero.

### {{< klmath inline=true >}}\nabla\,\mathrm{KL} \equiv k_1{{< /klmath >}} in the reward {{< klmath inline=true >}}\equiv k_2{{< /klmath >}} as a loss

The most common implementations of minimizing KL use one of two mechanisms: adding {{< klmath inline=true >}}k_1{{< /klmath >}} to the reward, or {{< klmath inline=true >}}k_2{{< /klmath >}} as a surrogate loss.

{{< klmath inline=true >}}k_1{{< /klmath >}} in the reward yields the correct gradient. In policy gradient, a sample {{< klmath inline=true >}}x{{< /klmath >}} with a fixed reward {{< klmath inline=true >}}R(x){{< /klmath >}} contributes {{< klmath inline=true >}}R(x)\,\nabla\log p{{< /klmath >}}. Notice that directly setting {{< klmath inline=true >}}R(x){{< /klmath >}} as {{< klmath inline=true >}}k_1{{< /klmath >}} yields the correct gradient:

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

### {{< klmath inline=true >}}\nabla\,\mathrm{KL} \equiv \nabla\mathbb{E}_p[k_\lambda] \equiv{{< /klmath >}} scaling the control variate

Now return to the family {{< klmath inline=true >}}k_\lambda = \log\frac{p}{q} + \lambda\left(\frac{q}{p} - 1\right){{< /klmath >}} from the estimation section, where {{< klmath inline=true >}}\lambda{{< /klmath >}} sets the magnitude of the estimate's control variate. Since {{< klmath inline=true >}}k_\lambda{{< /klmath >}} is unbiased, we should be able to show that the gradient of its expectation is the same as the gradient of KL:

{{< klmath >}}
\begin{aligned}
\nabla\mathbb{E}_p[k_\lambda] &= \nabla \sum_x p\,k_\lambda = \sum_x \nabla p\,k_\lambda + \sum_x p\,\nabla k_\lambda\\
&= \mathbb{E}_p\!\left[k_\lambda\,\nabla\log p + \nabla k_\lambda\right]\\
&= \mathbb{E}_p\Big[\Big(\log\frac{p}{q} + \cancel{\lambda\,\frac{q}{p}} - \lambda\Big)\nabla\log p + \Big(1 - \cancel{\lambda\,\frac{q}{p}}\Big)\nabla\log p\Big]\\
&= \mathbb{E}_p\Big[\log\frac{p}{q}\,\nabla\log p + (1 - \lambda)\,\nabla\log p\Big].
\end{aligned}
{{< /klmath >}}

The second term can be canceled, as {{< klmath inline=true >}}\mathbb{E}_p[ (1 - \lambda)\,\nabla\log p] = 0{{< /klmath >}}{{< klhint >}}Probabilities always sum to one, so their changes sum to zero: {{< klmath inline=true >}}\displaystyle\mathbb{E}_p[\nabla\log p] = \sum_x p\,\frac{\nabla p}{p} = \sum_x \nabla p = \nabla \sum_x p = \nabla 1 = 0.{{< /klmath >}}{{< /klhint >}}:

{{< klmath >}}
\begin{aligned}
\nabla\mathbb{E}_p[k_\lambda] &= \mathbb{E}_p[\log\frac{p}{q}\,\nabla\log p + (1 - \lambda)\,\nabla\log p]\\
&= \mathbb{E}_p[\log\frac{p}{q}\,\nabla\log p]\\
&= \nabla\,\mathrm{KL}[p\,\|\,q]\\
\end{aligned}
{{< /klmath >}}

Notice that again, we've found a zero-mean term {{< klmath inline=true >}}(1 - \lambda)\,\nabla\log p{{< /klmath >}} that does not bias the expectation when added, i.e. a control variate. In {{< klmath inline=true >}}k_\lambda{{< /klmath >}} it is {{< klmath inline=true >}}\lambda\left(\frac{q}{p} - 1\right){{< /klmath >}}, but in the {{< klmath inline=true >}}\nabla\mathbb{E}_p[k_\lambda]{{< /klmath >}} it is {{< klmath inline=true >}}(1 - \lambda)\,\nabla\log p{{< /klmath >}}. This suggests that choosing {{< klmath inline=true >}}\lambda{{< /klmath >}} to minimize the variance of KL might not minimize the variance of its gradient.

{{< klmath inline=true >}}\nabla\mathbb{E}_p[k_1]{{< /klmath >}} ({{< klmath inline=true >}}\lambda=0{{< /klmath >}}) keeps all of the {{< klmath inline=true >}}\nabla\log p{{< /klmath >}} term in the gradient, and is the exact same sample gradient as differentiating KL directly. Similarly, {{< klmath inline=true >}}\nabla\mathbb{E}_p[k_3]{{< /klmath >}} ({{< klmath inline=true >}}\lambda=1{{< /klmath >}}) keeps none of it, and is the exact same sample gradient as setting {{< klmath inline=true >}}k_1{{< /klmath >}} as the reward or {{< klmath inline=true >}}k_2{{< /klmath >}} as a loss. All of these are correct and unbiased gradients of KL, they just have different variances as a result of adding different magnitudes of the control variate.

In policy-gradient language, we've shifted the reward ({{< klmath inline=true >}}\log\frac{p}{q}{{< /klmath >}}) by a *baseline* ({{< klmath inline=true >}}1-\lambda{{< /klmath >}}), the standard tool for reducing a policy gradient's variance.

## The best estimate does not yield the best gradient

[Schulman's note](https://joschu.net/blog/kl-approx.html) says that choosing {{< klmath inline=true >}}\lambda{{< /klmath >}} to minimize the variance of {{< klmath inline=true >}}k_\lambda{{< /klmath >}} depends on {{< klmath inline=true >}}p{{< /klmath >}} and {{< klmath inline=true >}}q{{< /klmath >}} and is hard to calculate analytically. Instead, it uses {{< klmath inline=true >}}\lambda = 1{{< /klmath >}}, because then the estimate can never be negative.

What if we wanted to choose {{< klmath inline=true >}}\lambda{{< /klmath >}} optimally? The variance of {{< klmath inline=true >}}k_\lambda{{< /klmath >}} is quadratic in {{< klmath inline=true >}}\lambda{{< /klmath >}}, smallest at

{{< klmath >}}
\lambda_\text{value} = -\frac{\operatorname{Cov}_p\!\left(\log\frac{p}{q},\ \frac{q}{p}\right)}{\operatorname{Var}_p\!\left(\frac{q}{p}\right)}.
{{< /klmath >}}

However, in training we usually don't use the value of KL. We use its gradient. The corresponding per-sample gradient is {{< klmath inline=true >}}\big(\log\frac{p}{q} + 1 - \lambda\big)\nabla\log p{{< /klmath >}}, where {{< klmath inline=true >}}\lambda{{< /klmath >}} sets the size of a control variate {{< klmath inline=true >}}(1 - \lambda)\,\nabla\log p{{< /klmath >}}. The variance of this gradient is smallest at

{{< klmath >}}
\lambda_\text{gradient} = 1 + \frac{\mathbb{E}_p\!\left[\log\frac{p}{q}\,\|\nabla\log p\|^2\right]}{\mathbb{E}_p\!\left[\|\nabla\log p\|^2\right]},
{{< /klmath >}}

Let's consider the Gaussians again, {{< klmath inline=true >}}p = \mathcal{N}(\mu, 1){{< /klmath >}} and {{< klmath inline=true >}}q = \mathcal{N}(0, 1){{< /klmath >}}.

Taking the gradient taken with respect to {{< klmath inline=true >}}\mu{{< /klmath >}}, {{< klmath inline=true >}}\nabla\log p = x - \mu{{< /klmath >}}. Let {{< klmath inline=true >}}g_\lambda = \big(\log\frac{p}{q} + 1 - \lambda\big)\nabla\log p{{< /klmath >}} for the gradient. Then

{{< klmath >}}
\begin{aligned}
\operatorname{Var}(k_\lambda) &= \mu^2\,(1 - 2\lambda) + \lambda^2\left(e^{\mu^2} - 1\right)\\
\operatorname{Var}(g_\lambda) &= \left(\frac{\mu^2}{2} + 1 - \lambda\right)^2 + 2\mu^2
\end{aligned}
{{< /klmath >}}

These variances are smallest at

{{< klmath >}}
\lambda_\text{value} = \frac{\mu^2}{e^{\mu^2} - 1}
\qquad\text{and}\qquad
\lambda_\text{gradient} = 1 + \frac{\mu^2}{2}.
{{< /klmath >}}

{{< klderiv title="Derive the two variances" >}}
Write {{< klmath inline=true >}}x = \mu + z{{< /klmath >}} with {{< klmath inline=true >}}z \sim \mathcal{N}(0, 1){{< /klmath >}}. Then {{< klmath inline=true >}}\log\frac{p}{q} = \frac{\mu^2}{2} + \mu z{{< /klmath >}}, {{< klmath inline=true >}}\frac{q}{p} = e^{-\mu z - \mu^2/2}{{< /klmath >}}, and {{< klmath inline=true >}}\nabla_\mu \log p = x - \mu = z{{< /klmath >}}.

**The estimate.** {{< klmath inline=true >}}\operatorname{Var}(\log\frac{p}{q}) = \mu^2{{< /klmath >}}. From {{< klmath inline=true >}}\mathbb{E}[e^{-2\mu z}] = e^{2\mu^2}{{< /klmath >}}, {{< klmath inline=true >}}\operatorname{Var}(\frac{q}{p}) = e^{\mu^2} - 1{{< /klmath >}}. From {{< klmath inline=true >}}\mathbb{E}[z\,e^{-\mu z}] = -\mu\,e^{\mu^2/2}{{< /klmath >}}, {{< klmath inline=true >}}\operatorname{Cov}(\log\frac{p}{q}, \frac{q}{p}) = -\mu^2{{< /klmath >}}. So {{< klmath inline=true >}}\operatorname{Var}(k_\lambda) = \mu^2 - 2\lambda\mu^2 + \lambda^2(e^{\mu^2} - 1){{< /klmath >}}, a quadratic in {{< klmath inline=true >}}\lambda{{< /klmath >}} with its minimum at {{< klmath inline=true >}}\mu^2 / (e^{\mu^2} - 1){{< /klmath >}}.

**The gradient.** Let {{< klmath inline=true >}}a = \frac{\mu^2}{2} + 1 - \lambda{{< /klmath >}}, so the gradient is {{< klmath inline=true >}}(a + \mu z)\,z{{< /klmath >}}. Its mean is {{< klmath inline=true >}}\mu{{< /klmath >}}, the gradient of {{< klmath inline=true >}}\mu^2/2{{< /klmath >}}. Since {{< klmath inline=true >}}\mathbb{E}[z^2] = 1{{< /klmath >}}, {{< klmath inline=true >}}\mathbb{E}[z^3] = 0{{< /klmath >}} and {{< klmath inline=true >}}\mathbb{E}[z^4] = 3{{< /klmath >}}, its second moment is {{< klmath inline=true >}}a^2 + 3\mu^2{{< /klmath >}}, and its variance is {{< klmath inline=true >}}a^2 + 2\mu^2{{< /klmath >}}, smallest at {{< klmath inline=true >}}a = 0{{< /klmath >}}.
{{< /klderiv >}}

The two optima move in opposite directions. As {{< klmath inline=true >}}p{{< /klmath >}} moves away from {{< klmath inline=true >}}q{{< /klmath >}}, the best estimate slides from {{< klmath inline=true >}}k_3{{< /klmath >}} toward {{< klmath inline=true >}}k_1{{< /klmath >}}, while the best gradient wants a larger and larger baseline. Move the slider to see it.

{{< klfigure type="gaussianLambda">}}
Diamonds mark each minimum. Near {{< klmath inline=true >}}q{{< /klmath >}}, both minima sit at {{< klmath inline=true >}}\lambda = 1{{< /klmath >}}, which is {{< klmath inline=true >}}k_3{{< /klmath >}}. Far from {{< klmath inline=true >}}q{{< /klmath >}}, they are on opposite sides of it.
{{< /klfigure >}}

### A simple bandit experiment: optimizing KL with each {{< klmath inline=true >}}\lambda{{< /klmath >}}

Does the difference matter in training?

Let's take a softmax policy over 1,000 actions (i.e. a bandit) and a fixed reference {{< klmath inline=true >}}q{{< /klmath >}}, and initializing the policy {{< klmath inline=true >}}q{{< /klmath >}} far from {{< klmath inline=true >}}q{{< /klmath >}} ({{< klmath inline=true >}}\mathrm{KL} = 2.43{{< /klmath >}}). We'll minimize {{< klmath inline=true >}}\mathrm{KL}[p\,\|\,q]{{< /klmath >}} alone by plain SGD on the logits using policy gradient. Each step samples 16 actions and averages {{< klmath inline=true >}}\big(\log\frac{p}{q} + 1 - \lambda\big)\nabla\log p{{< /klmath >}}, with {{< klmath inline=true >}}\lambda{{< /klmath >}} set four ways: {{< klmath inline=true >}}\lambda = 0{{< /klmath >}} ({{< klmath inline=true >}}k_1{{< /klmath >}} differentiated exactly); {{< klmath inline=true >}}\lambda = 1{{< /klmath >}} ({{< klmath inline=true >}}k_3{{< /klmath >}}); {{< klmath inline=true >}}\lambda_\text{value}{{< /klmath >}}; and {{< klmath inline=true >}}\lambda_\text{gradient}{{< /klmath >}}, the last two computed exactly from the current policy. Every choice gives an unbiased gradient, so any difference comes from variance.

{{< klfigure type="lambdaDescent" data="klviz/lambda.json" >}}
KL and gradient variance are exact at each step. A policy that collapses onto a single action stops moving, because {{< klmath inline=true >}}\nabla\log p{{< /klmath >}} is then zero for the only action it samples.
{{< /klfigure >}}

## Epilogue

A KL estimator can be used to either estimate KL, or estimate the gradient of KL. Usually, we want the latter, which can be unstable if we mistakenly optimize for lower KL estimate variance instead of lower KL gradient variance.

Practically, this doesn't change much for the LLM status-quo of using {{< klmath inline=true >}}k_1{{< /klmath >}} in the reward or {{< klmath inline=true >}}k_2{{< /klmath >}} as a loss, but it is useful to understand the distinction.

## Appendix

Mixing up "the estimator" with "how the estimator enters the gradient" is prevalent despite others pointing out these pitfalls before ([Tang and Munos](https://arxiv.org/abs/2506.09477); [Liu et al.](https://arxiv.org/abs/2510.01555)). This shows up in popular code and reports: open-source RL frameworks, the Composer 2 report, and DeepSeek's GRPO papers.

{{< klderiv title="Libraries that were wrong: TRL, NeMo-RL, OpenInstruct" tag="Note" >}}
Popular RL libraries computed {{< klmath inline=true >}}k_3{{< /klmath >}} for each sampled token and added it to the loss: Hugging Face [TRL](https://github.com/huggingface/trl/pull/6503), NVIDIA [NeMo-RL](https://github.com/NVIDIA-NeMo/RL/pull/2506), and AI2 [open-instruct](https://github.com/allenai/open-instruct/blob/11826255077617a46919ce75cadf1f3d53f30dac/open_instruct/grpo_utils.py). This means we're using {{< klmath inline=true >}}\nabla k_3{{< /klmath >}} instead of the expectation {{< klmath inline=true >}}\nabla\mathbb{E}_p[k_3]{{< /klmath >}}, which is wrong.

TRL and NeMo-RL have since fixed their defaults. At the time of writing, open-instruct's GRPO still uses it by default.

Let's see what their implementation was actually optimizing. The gradient of the expectation has two terms, and they kept only one:

{{< klmath >}}
\nabla\mathbb{E}_p[k_3] = \mathbb{E}_p\big[\underbrace{k_3\,\nabla\log p}_{\text{dropped}} + \underbrace{\nabla k_3}_{\text{kept}}\big].
{{< /klmath >}}

The kept term is the gradient of *forward* KL instead of *reverse* KL:

{{< klmath >}}
\mathbb{E}_p[\nabla k_3] = \mathbb{E}_p\!\left[\Big(1 - \frac{q}{p}\Big)\nabla\log p\right] = \underbrace{\mathbb{E}_p[\nabla\log p]}_{0} - \sum_x q\,\nabla\log p = \nabla\,\mathrm{KL}[q\,\|\,p].
{{< /klmath >}}

The mistake can be quiet because forward KL is also minimized at {{< klmath inline=true >}}p = q{{< /klmath >}}, so the penalty still pulls toward the reference. But each sample is weighted by {{< klmath inline=true >}}1 - q/p{{< /klmath >}}, which is huge for a token the policy rarely picks but the reference likes.
{{< /klderiv >}}

{{< klderiv title="DeepSeek's original mistake corrected in V3.2" tag="Note" >}}
[DeepSeekMath](https://arxiv.org/abs/2402.03300), which introduced GRPO, makes the same mistake as the libraries: it adds a per-token {{< klmath inline=true >}}k_3{{< /klmath >}} to the loss. Its appendix even writes out the resulting gradient coefficient for each token,

{{< klmath >}}
\hat{A}_t + \beta\left(\frac{q}{p} - 1\right),
{{< /klmath >}}

which is the advantage minus {{< klmath inline=true >}}\beta(1 - q/p){{< /klmath >}}, the forward-KL gradient from the previous note. [DeepSeek-V3](https://arxiv.org/abs/2412.19437) and [DeepSeek-R1](https://arxiv.org/abs/2501.12948) reuse the same term.

[DeepSeek-V3.2](https://arxiv.org/html/2512.02556v1#S3.SS1) puts {{< klmath inline=true >}}k_3{{< /klmath >}} *inside* the expectation the objective takes over sampled responses, which is correct.

{{< /klderiv >}}

{{< klderiv title="Composer 2's wrong reason for the correct implementation" tag="Note" >}}
The [Composer 2 report](https://arxiv.org/abs/2603.24477) (Section 4.1) notes that {{< klmath inline=true >}}k_3{{< /klmath >}} is unbiased with low variance when the two policies are close, but its variance grows quickly as they drift apart, so they "use the standard estimator {{< klmath inline=true >}}k_1{{< /klmath >}} instead."

It's unclear how they use either estimator, but the report seems to imply they are comparing {{< klmath inline=true >}}k_1{{< /klmath >}} in reward with {{< klmath inline=true >}}k_3{{< /klmath >}} in reward. {{< klmath inline=true >}}k_3{{< /klmath >}} in reward is not a valid choice. With {{< klmath inline=true >}}R = k_3{{< /klmath >}}, the expected policy gradient is

{{< klmath >}}
\begin{aligned}
\mathbb{E}_p[k_3\,\nabla\log p] &= \mathbb{E}_p\!\left[\log\frac{p}{q}\,\nabla\log p\right] + \mathbb{E}_p\!\left[\Big(\frac{q}{p} - 1\Big)\nabla\log p\right]\\
&= \nabla\,\mathrm{KL}[p\,\|\,q] + \sum_x q\,\nabla\log p\\
&= \nabla\,\mathrm{KL}[p\,\|\,q] - \nabla\,\mathrm{KL}[q\,\|\,p],
\end{aligned}
{{< /klmath >}}

which is not a divergence and is unbounded below: the policy can lower it forever by moving mass away from actions the reference likes. {{< klmath inline=true >}}k_1{{< /klmath >}} in the reward is the right choice, but because its gradient is correct, not because its estimate has lower variance.
{{< /klderiv >}}
