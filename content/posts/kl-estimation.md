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

However, which estimator to use is far from converged. DeepSeek[^deepseek] uses {{< klmath inline=true >}}k_3{{< /klmath >}} as a surrogate loss for RLVR. Thinking Machines uses {{< klmath inline=true >}}-k_1{{< /klmath >}} as a reward for on-policy distillation[^opd], and Cursor's Composer 2 uses {{< klmath inline=true >}}k_1{{< /klmath >}} as a regularizer[^cursor].

Which estimator is optimal for learning reverse KL {{< klmath inline=true >}}\min_p\mathrm{KL}[p\,\|\,q]{{< /klmath >}}?

This is best answered by answering two questions: which estimator gives a better *estimate* of KL, and which gives a better *gradient* when we train on it?

TL;DR:
- As **estimates**, they differ. {{< klmath inline=true >}}k_1{{< /klmath >}} and {{< klmath inline=true >}}k_3{{< /klmath >}} are unbiased and {{< klmath inline=true >}}k_2{{< /klmath >}} is not; {{< klmath inline=true >}}k_3{{< /klmath >}} has the lowest variance near {{< klmath inline=true >}}q{{< /klmath >}} and the highest far from it.
- As **gradients**, they do not. {{< klmath inline=true >}}k_1{{< /klmath >}} in the reward, {{< klmath inline=true >}}k_2{{< /klmath >}} as a loss, and {{< klmath inline=true >}}k_3{{< /klmath >}} in both the reward and the loss (as DeepSeek-V3.2 does) give the same gradient for every sample: exactly the gradient of the reverse KL.
- The family {{< klmath inline=true >}}k_\lambda = k_1 + \lambda(\frac{q}{p} - 1){{< /klmath >}}, which contains {{< klmath inline=true >}}k_3{{< /klmath >}}, gives the right gradient for every {{< klmath inline=true >}}\lambda{{< /klmath >}}. In the gradient, {{< klmath inline=true >}}\lambda{{< /klmath >}} is just a baseline, and the {{< klmath inline=true >}}\lambda{{< /klmath >}} that minimizes the gradient's variance is not the one that minimizes the estimate's. Several published reasons for choosing an estimator are about the estimate, not the gradient.

## Estimating the value of KL

It's easiest to compare the estimators on two Gaussians, {{< klmath inline=true >}}p = \mathcal{N}(\mu, 1){{< /klmath >}} and {{< klmath inline=true >}}q = \mathcal{N}(0, 1){{< /klmath >}}, where everything has a closed form. As we vary {{< klmath inline=true >}}\mu{{< /klmath >}}, the KL is {{< klmath inline=true >}}\mu^2/2{{< /klmath >}}.

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

In RL, the KL estimate is not the end product. We differentiate it. Write {{< klmath inline=true >}}s(x) = \nabla_\theta \log p_\theta(x){{< /klmath >}} for the score. Two facts do all the work. The score has mean zero:

{{< klmath >}}
\mathbb{E}_p[s] = \sum_x p\,\frac{\nabla p}{p} = \nabla \sum_x p = \nabla 1 = 0,
{{< /klmath >}}

and the gradient of the reverse KL is {{< klmath inline=true >}}\log\frac{p}{q}{{< /klmath >}} times the score, on average:

{{< klmath >}}
\begin{aligned}
\nabla\,\mathrm{KL}[p\,\|\,q] &= \nabla \sum_x p \log\frac{p}{q} = \sum_x \nabla p\,\log\frac{p}{q} + \sum_x p\,\nabla \log p\\
&= \mathbb{E}_p\!\left[\log\frac{p}{q}\,s\right] + \mathbb{E}_p[s] = \mathbb{E}_p\!\left[\log\frac{p}{q}\,s\right].
\end{aligned}
{{< /klmath >}}

Here {{< klmath inline=true >}}q{{< /klmath >}} does not depend on {{< klmath inline=true >}}\theta{{< /klmath >}}, so {{< klmath inline=true >}}\nabla \log q = 0{{< /klmath >}}. Any per-sample gradient whose mean is {{< klmath inline=true >}}\mathbb{E}_p[\log\frac{p}{q}\,s]{{< /klmath >}} follows the reverse KL. The rest of this section finds that gradient in each estimator.

An estimate {{< klmath inline=true >}}k(x){{< /klmath >}} can enter training in two ways. The gradient of its expectation has one term for each:

{{< klmath >}}
\nabla_\theta\,\mathbb{E}_{p_\theta}[k(x)]
=
\underbrace{\mathbb{E}_{p_\theta}\!\left[k(x)\,s(x)\right]}_{\text{in the reward}}
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

- **In the reward:** treat {{< klmath inline=true >}}-k(x){{< /klmath >}} as a detached reward, or advantage, and take the policy gradient. Only the first term survives.
- **As a loss:** backpropagate through {{< klmath inline=true >}}k(x){{< /klmath >}} with the sample {{< klmath inline=true >}}x{{< /klmath >}} held fixed. Only the second term survives.

### {{< klmath inline=true >}}k_1{{< /klmath >}} in the reward is the gradient

The policy gradient with {{< klmath inline=true >}}k_1{{< /klmath >}} as the reward is

{{< klmath >}}
k_1\,s = \log\frac{p}{q}\,s,
{{< /klmath >}}

which is the gradient of KL sample by sample. The loss term it drops is {{< klmath inline=true >}}\nabla k_1 = \nabla\log p = s{{< /klmath >}}, which has mean zero.

### {{< klmath inline=true >}}k_2{{< /klmath >}} as a loss is the same gradient

Differentiate the square with the chain rule:

{{< klmath >}}
\nabla k_2 = \nabla\,\frac12\left(\log\frac{p}{q}\right)^2 = \log\frac{p}{q}\,\nabla\!\left(\log p - \log q\right) = \log\frac{p}{q}\,s.
{{< /klmath >}}

This is the same number as {{< klmath inline=true >}}k_1\,s{{< /klmath >}} for every sample. Note what it is not: the gradient of {{< klmath inline=true >}}\mathbb{E}_p[k_2]{{< /klmath >}}. That expectation is {{< klmath inline=true >}}\mathrm{KL} + \mathrm{KL}^2/2{{< /klmath >}} for the Gaussians above, a different function, and by the split its gradient also has the reward term {{< klmath inline=true >}}\mathbb{E}_p[k_2\,s]{{< /klmath >}}, which is not zero. Using {{< klmath inline=true >}}k_2{{< /klmath >}} as a loss drops that term, and dropping it is what makes the gradient right. Putting {{< klmath inline=true >}}k_2{{< /klmath >}} in the reward as well would follow the wrong objective.

### {{< klmath inline=true >}}k_\lambda{{< /klmath >}} in both is the gradient, for every {{< klmath inline=true >}}\lambda{{< /klmath >}}

Since {{< klmath inline=true >}}\mathbb{E}_p[\frac{q}{p} - 1] = \sum_x q - 1 = 0{{< /klmath >}}, any multiple of {{< klmath inline=true >}}\frac{q}{p} - 1{{< /klmath >}} can be added to {{< klmath inline=true >}}k_1{{< /klmath >}} without changing its mean:

{{< klmath >}}
k_\lambda = \log\frac{p}{q} + \lambda\left(\frac{q}{p} - 1\right).
{{< /klmath >}}

This is a control variate: a zero-mean term that can cancel part of {{< klmath inline=true >}}k_1{{< /klmath >}}'s noise. {{< klmath inline=true >}}\lambda = 0{{< /klmath >}} is {{< klmath inline=true >}}k_1{{< /klmath >}} and {{< klmath inline=true >}}\lambda = 1{{< /klmath >}} is {{< klmath inline=true >}}k_3{{< /klmath >}}. Since {{< klmath inline=true >}}k_\lambda{{< /klmath >}} is unbiased, using it in both the reward and the loss gives the gradient of its expectation, KL. Here is each step. The reward term is

{{< klmath >}}
k_\lambda\,s = \log\frac{p}{q}\,s + \lambda\,\frac{q}{p}\,s - \lambda\,s.
{{< /klmath >}}

For the loss term, {{< klmath inline=true >}}\nabla \log\frac{p}{q} = s{{< /klmath >}}, {{< klmath inline=true >}}\nabla 1 = 0{{< /klmath >}}, and

{{< klmath >}}
\nabla \frac{q}{p} = q\,\nabla\frac{1}{p} = -\frac{q}{p^2}\,\nabla p = -\frac{q}{p}\,\frac{\nabla p}{p} = -\frac{q}{p}\,s,
{{< /klmath >}}

so

{{< klmath >}}
\nabla k_\lambda = s - \lambda\,\frac{q}{p}\,s.
{{< /klmath >}}

Add the two. The {{< klmath inline=true >}}\frac{q}{p}\,s{{< /klmath >}} terms cancel:

{{< klmath >}}
k_\lambda\,s + \nabla k_\lambda = \log\frac{p}{q}\,s + \cancel{\lambda\,\frac{q}{p}\,s} - \lambda\,s + s - \cancel{\lambda\,\frac{q}{p}\,s} = \Big(\log\frac{p}{q} + 1 - \lambda\Big)\,s.
{{< /klmath >}}

Since {{< klmath inline=true >}}\mathbb{E}_p[s] = 0{{< /klmath >}}, the mean is {{< klmath inline=true >}}\nabla\,\mathrm{KL}{{< /klmath >}} for every {{< klmath inline=true >}}\lambda{{< /klmath >}}. At {{< klmath inline=true >}}\lambda = 1{{< /klmath >}}, which is {{< klmath inline=true >}}k_3{{< /klmath >}}, the constant vanishes, and {{< klmath inline=true >}}k_3{{< /klmath >}} in both gives {{< klmath inline=true >}}\log\frac{p}{q}\,s{{< /klmath >}}: once again, the same number as {{< klmath inline=true >}}k_1{{< /klmath >}} in the reward and {{< klmath inline=true >}}k_2{{< /klmath >}} as a loss, for every sample.

Using only one of {{< klmath inline=true >}}k_3{{< /klmath >}}'s two terms is a different matter. Since {{< klmath inline=true >}}\mathbb{E}_p[\frac{q}{p}\,s] = \sum_x q\,\nabla\log p = -\nabla\,\mathrm{KL}[q\,\|\,p]{{< /klmath >}}, each term alone brings in the *forward* KL:

| | Expected gradient |
| --- | --- |
| {{< klmath inline=true >}}k_1{{< /klmath >}} in the reward, {{< klmath inline=true >}}k_2{{< /klmath >}} as a loss, {{< klmath inline=true >}}k_\lambda{{< /klmath >}} in both | {{< klmath inline=true >}}\nabla\,\mathrm{KL}[p\,\Vert\,q]{{< /klmath >}}, the reverse KL |
| {{< klmath inline=true >}}k_1{{< /klmath >}} as a loss | {{< klmath inline=true >}}0{{< /klmath >}} |
| {{< klmath inline=true >}}k_3{{< /klmath >}} as a loss | {{< klmath inline=true >}}\nabla\,\mathrm{KL}[q\,\Vert\,p]{{< /klmath >}}, the forward KL |
| {{< klmath inline=true >}}k_3{{< /klmath >}} in the reward | {{< klmath inline=true >}}\nabla\,\mathrm{KL}[p\,\Vert\,q] - \nabla\,\mathrm{KL}[q\,\Vert\,p]{{< /klmath >}} |

{{< klderiv title="Derive the last three rows" >}}
**{{< klmath inline=true >}}k_1{{< /klmath >}} as a loss:** {{< klmath inline=true >}}\mathbb{E}[\nabla k_1] = \mathbb{E}[s] = 0{{< /klmath >}}.

**{{< klmath inline=true >}}k_3{{< /klmath >}} as a loss:** {{< klmath inline=true >}}\mathbb{E}[\nabla k_3] = \mathbb{E}[s] - \mathbb{E}[\tfrac{q}{p}\,s] = 0 + \nabla\,\mathrm{KL}[q\,\|\,p]{{< /klmath >}}, using {{< klmath inline=true >}}\mathrm{KL}[q\,\|\,p] = \sum_x q\log q - \sum_x q\log p{{< /klmath >}}, so {{< klmath inline=true >}}\nabla\,\mathrm{KL}[q\,\|\,p] = -\sum_x q\,\nabla\log p{{< /klmath >}}.

**{{< klmath inline=true >}}k_3{{< /klmath >}} in the reward:** the two terms add up to {{< klmath inline=true >}}\nabla\,\mathrm{KL}[p\,\|\,q]{{< /klmath >}}, so the reward term is that minus the loss term.
{{< /klderiv >}}

These results are known: Tang and Munos show that {{< klmath inline=true >}}k_3{{< /klmath >}} as a loss gives the forward KL's gradient[^tang], Liu et al. that {{< klmath inline=true >}}k_2{{< /klmath >}} as a loss matches {{< klmath inline=true >}}k_1{{< /klmath >}} in the reward in expectation[^rethinking], and Wang that the three correct forms agree sample by sample[^wang].

## Choosing {{< klmath inline=true >}}\lambda{{< /klmath >}}

Schulman's note arrives at {{< klmath inline=true >}}k_3{{< /klmath >}} through this same family. It observes that the variance-minimizing {{< klmath inline=true >}}\lambda{{< /klmath >}} "depends on p and q and is hard to calculate analytically", and takes {{< klmath inline=true >}}\lambda = 1{{< /klmath >}} instead, because then the estimate can never be negative[^schulman].

What if we did want to choose {{< klmath inline=true >}}\lambda{{< /klmath >}}? As an estimate of the value, {{< klmath inline=true >}}k_\lambda{{< /klmath >}}'s variance is a quadratic in {{< klmath inline=true >}}\lambda{{< /klmath >}}, smallest at

{{< klmath >}}
\lambda_\text{value} = -\frac{\operatorname{Cov}_p\!\left(\log\frac{p}{q},\ \frac{q}{p}\right)}{\operatorname{Var}_p\!\left(\frac{q}{p}\right)}.
{{< /klmath >}}

But in training we never use the value. We use the gradient {{< klmath inline=true >}}\big(\log\frac{p}{q} + 1 - \lambda\big)s{{< /klmath >}}, and there {{< klmath inline=true >}}\lambda{{< /klmath >}} plays a different role: it only shifts the reward {{< klmath inline=true >}}\log\frac{p}{q}{{< /klmath >}} by a constant. It is a *baseline*, the standard tool for reducing a policy gradient's variance, and has nothing to do with the control variate it was in the estimate. The gradient's variance, the trace of its covariance, is smallest at

{{< klmath >}}
\lambda_\text{gradient} = 1 + \frac{\mathbb{E}_p\!\left[\log\frac{p}{q}\,\|s\|^2\right]}{\mathbb{E}_p\!\left[\|s\|^2\right]},
{{< /klmath >}}

which is close to {{< klmath inline=true >}}1 + \mathrm{KL}{{< /klmath >}} when {{< klmath inline=true >}}\|s\|^2{{< /klmath >}} varies little: subtract the mean reward, as usual.

For the Gaussians, {{< klmath inline=true >}}p = \mathcal{N}(\mu, 1){{< /klmath >}} and {{< klmath inline=true >}}q = \mathcal{N}(0, 1){{< /klmath >}} with the gradient taken with respect to {{< klmath inline=true >}}\mu{{< /klmath >}}, both have closed forms. Write {{< klmath inline=true >}}g_\lambda = \big(\log\frac{p}{q} + 1 - \lambda\big)s{{< /klmath >}} for the gradient. In terms of {{< klmath inline=true >}}\mathrm{KL} = \mu^2/2{{< /klmath >}},

{{< klmath >}}
\begin{aligned}
\operatorname{Var}(k_\lambda) &= 2\,\mathrm{KL}\,(1 - 2\lambda) + \lambda^2\left(e^{2\,\mathrm{KL}} - 1\right),\\
\operatorname{Var}(g_\lambda) &= \left(\mathrm{KL} + 1 - \lambda\right)^2 + 4\,\mathrm{KL},
\end{aligned}
{{< /klmath >}}

so they are smallest at

{{< klmath >}}
\lambda_\text{value} = \frac{2\,\mathrm{KL}}{e^{2\,\mathrm{KL}} - 1}
\qquad\text{and}\qquad
\lambda_\text{gradient} = 1 + \mathrm{KL}.
{{< /klmath >}}

{{< klderiv title="Derive the two variances" >}}
Write {{< klmath inline=true >}}x = \mu + z{{< /klmath >}} with {{< klmath inline=true >}}z \sim \mathcal{N}(0, 1){{< /klmath >}}. Then {{< klmath inline=true >}}\log\frac{p}{q} = \frac{\mu^2}{2} + \mu z{{< /klmath >}}, {{< klmath inline=true >}}\frac{q}{p} = e^{-\mu z - \mu^2/2}{{< /klmath >}}, and {{< klmath inline=true >}}s = z{{< /klmath >}}.

**The estimate.** {{< klmath inline=true >}}\operatorname{Var}(\log\frac{p}{q}) = \mu^2{{< /klmath >}}. From {{< klmath inline=true >}}\mathbb{E}[e^{-2\mu z}] = e^{2\mu^2}{{< /klmath >}}, {{< klmath inline=true >}}\operatorname{Var}(\frac{q}{p}) = e^{\mu^2} - 1{{< /klmath >}}. From {{< klmath inline=true >}}\mathbb{E}[z\,e^{-\mu z}] = -\mu\,e^{\mu^2/2}{{< /klmath >}}, {{< klmath inline=true >}}\operatorname{Cov}(\log\frac{p}{q}, \frac{q}{p}) = -\mu^2{{< /klmath >}}. So {{< klmath inline=true >}}\operatorname{Var}(k_\lambda) = \mu^2 - 2\lambda\mu^2 + \lambda^2(e^{\mu^2} - 1){{< /klmath >}}, a quadratic in {{< klmath inline=true >}}\lambda{{< /klmath >}} with its minimum at {{< klmath inline=true >}}\mu^2 / (e^{\mu^2} - 1){{< /klmath >}}.

**The gradient.** Let {{< klmath inline=true >}}a = \frac{\mu^2}{2} + 1 - \lambda{{< /klmath >}}, so the gradient is {{< klmath inline=true >}}(a + \mu z)\,z{{< /klmath >}}. Its mean is {{< klmath inline=true >}}\mu{{< /klmath >}}, the gradient of {{< klmath inline=true >}}\mu^2/2{{< /klmath >}}. Since {{< klmath inline=true >}}\mathbb{E}[z^2] = 1{{< /klmath >}}, {{< klmath inline=true >}}\mathbb{E}[z^3] = 0{{< /klmath >}} and {{< klmath inline=true >}}\mathbb{E}[z^4] = 3{{< /klmath >}}, its second moment is {{< klmath inline=true >}}a^2 + 3\mu^2{{< /klmath >}}, and its variance is {{< klmath inline=true >}}a^2 + 2\mu^2{{< /klmath >}}, smallest at {{< klmath inline=true >}}a = 0{{< /klmath >}}.
{{< /klderiv >}}

The two optima move in opposite directions. As {{< klmath inline=true >}}p{{< /klmath >}} moves away from {{< klmath inline=true >}}q{{< /klmath >}}, the best estimate slides from {{< klmath inline=true >}}k_3{{< /klmath >}} toward {{< klmath inline=true >}}k_1{{< /klmath >}}, while the best gradient wants a larger and larger baseline. Move the slider to see it.

{{< klfigure type="gaussianLambda" subtitle=`$p = \mathcal{N}(\mu, 1)$ · $q = \mathcal{N}(0, 1)$ · $k_\lambda = k_1 + \lambda\,(q/p - 1)$` >}}
Diamonds mark each minimum. Near {{< klmath inline=true >}}q{{< /klmath >}}, both minima sit at {{< klmath inline=true >}}\lambda = 1{{< /klmath >}}, which is {{< klmath inline=true >}}k_3{{< /klmath >}}. Far from {{< klmath inline=true >}}q{{< /klmath >}}, they are on opposite sides of it.
{{< /klfigure >}}

### A toy: descending KL with each {{< klmath inline=true >}}\lambda{{< /klmath >}}

Does the difference matter in training? Take a softmax policy over 1,000 actions and a fixed reference {{< klmath inline=true >}}q{{< /klmath >}}, start the policy far from {{< klmath inline=true >}}q{{< /klmath >}} ({{< klmath inline=true >}}\mathrm{KL} = 2.43{{< /klmath >}}), and minimize {{< klmath inline=true >}}\mathrm{KL}[p\,\|\,q]{{< /klmath >}} alone by plain SGD on the logits. Each step samples 16 actions and averages {{< klmath inline=true >}}\big(\log\frac{p}{q} + 1 - \lambda\big)s{{< /klmath >}}, with {{< klmath inline=true >}}\lambda{{< /klmath >}} set three ways: {{< klmath inline=true >}}\lambda = 1{{< /klmath >}}; {{< klmath inline=true >}}\lambda_\text{value}{{< /klmath >}}; and {{< klmath inline=true >}}\lambda_\text{gradient}{{< /klmath >}}, both computed exactly from the current policy. Every choice gives an unbiased gradient, so any difference comes from variance.

{{< klfigure type="lambdaDescent" data="klviz/lambda.json" subtitle=`1,000 actions · 16 samples per step · 32 seeds · median and interquartile range` >}}
Training curves use learning rate 5. KL and gradient variance are exact at each step. A policy that collapses onto a single action stops moving, because its score is zero.
{{< /klfigure >}}

At learning rate 5, the estimate's best {{< klmath inline=true >}}\lambda{{< /klmath >}} trains the slowest: median KL 0.47 after 400 steps, against 0.32 for the other two, with five to ten times their gradient variance. Here {{< klmath inline=true >}}\lambda_\text{value}{{< /klmath >}} stays near 0. A few actions where {{< klmath inline=true >}}p{{< /klmath >}} is tiny but {{< klmath inline=true >}}q{{< /klmath >}} is not make {{< klmath inline=true >}}\frac{q}{p}{{< /klmath >}} heavy-tailed, so the estimate barely uses its control variate. Across learning rates, the three agree when steps are small. As steps grow, {{< klmath inline=true >}}\lambda_\text{value}{{< /klmath >}} fails first, collapsing onto a single action at learning rate 10. {{< klmath inline=true >}}\lambda = 1{{< /klmath >}} collapses at 20, and {{< klmath inline=true >}}\lambda_\text{gradient}{{< /klmath >}} still converges there. Estimating {{< klmath inline=true >}}\lambda_\text{gradient}{{< /klmath >}} from each batch instead of exactly, as a practical method would, gives nearly the same results (`tools/kl-lambda-study.py` runs both). This is one toy problem, but the direction is what the formulas predict: a good estimate and a good gradient ask for different {{< klmath inline=true >}}\lambda{{< /klmath >}}, and training needs the gradient.

## Examples

**DeepSeek-V3.2.** The original GRPO adds {{< klmath inline=true >}}k_3{{< /klmath >}} to its loss for every token, with {{< klmath inline=true >}}p = \pi_\theta{{< /klmath >}} and {{< klmath inline=true >}}q = \pi_\mathrm{ref}{{< /klmath >}}[^grpo]. Autodiff keeps only the loss term, so by the table it follows the forward KL. DeepSeek-V3.2 multiplies the term by the importance ratio {{< klmath inline=true >}}r = \pi_\theta/\pi_\mathrm{old}{{< /klmath >}} to make the gradient unbiased, citing the "unbounded" weights the original gives tokens with {{< klmath inline=true >}}\pi_\theta \ll \pi_\mathrm{ref}{{< /klmath >}}[^deepseek]. The fix works, and the product rule shows why. Since {{< klmath inline=true >}}\nabla r = r\,s{{< /klmath >}},

{{< klmath >}}
\nabla_\theta\!\left(r\,k_3\right) = r\Big(\underbrace{k_3\,s}_{\text{reward term}} + \underbrace{\nabla k_3}_{\text{loss term}}\Big) = r\,\log\frac{p}{q}\,s.
{{< /klmath >}}

The ratio supplies the missing reward term, and the unbounded weight is the {{< klmath inline=true >}}\frac{q}{p}\,s{{< /klmath >}} that it cancels. On the first optimizer step, where {{< klmath inline=true >}}r = 1{{< /klmath >}}, this is exactly {{< klmath inline=true >}}k_2{{< /klmath >}} as a loss. On later steps over the same batch, it is {{< klmath inline=true >}}k_2{{< /klmath >}} as a loss weighted by the detached ratio, which keeps it unbiased off-policy. Either way, the extra machinery recovers a gradient that {{< klmath inline=true >}}k_2{{< /klmath >}} gives directly.

{{< klcode title="Check the DeepSeek-V3.2 gradient numerically" >}}
import numpy as np

rng = np.random.default_rng(1)
size = 5
softmax = lambda z: np.exp(z) / np.exp(z).sum()
old = rng.normal(size=size)                  # logits of pi_old
theta = old + 0.3 * rng.normal(size=size)    # pi_theta, a few steps later
ref = rng.dirichlet(np.ones(size))           # pi_ref


def kl_term(logits, x):
    """DeepSeek-V3.2's per-token KL term: (pi_theta / pi_old) * k3."""
    p, p_old = softmax(logits), softmax(old)
    ratio = ref[x] / p[x]
    return p[x] / p_old[x] * (ratio - np.log(ratio) - 1)


for x in range(size):
    h = 1e-6
    numeric = np.array([(kl_term(theta + h * e, x) - kl_term(theta - h * e, x)) / (2 * h)
                        for e in np.eye(size)])
    p = softmax(theta)
    score = np.eye(size)[x] - p
    k1 = np.log(p[x] / ref[x])
    predicted = p[x] / softmax(old)[x] * k1 * score   # r * k1 * score
    assert np.allclose(numeric, predicted, atol=1e-7)
print('gradient of r * k3 = r * k1 * score at every token')
{{< /klcode >}}

**Composer 2.** Cursor chooses {{< klmath inline=true >}}k_1{{< /klmath >}} over {{< klmath inline=true >}}k_3{{< /klmath >}} because the variance of the {{< klmath inline=true >}}k_3{{< /klmath >}} estimate "increases drastically" as the distributions diverge, and rules out {{< klmath inline=true >}}k_2{{< /klmath >}} because it is biased[^cursor]. Both reasons are about the estimate. Neither reaches the gradient: {{< klmath inline=true >}}k_3{{< /klmath >}} in both terms has exactly {{< klmath inline=true >}}k_1{{< /klmath >}}'s gradient, and {{< klmath inline=true >}}k_2{{< /klmath >}} as a loss has exactly the right gradient despite its bias. The report does not say how {{< klmath inline=true >}}k_1{{< /klmath >}} enters training. In the reward, it gives exactly the right gradient. As a loss alone, its gradient is zero on average.

**GRPO and verl.** GRPO's original {{< klmath inline=true >}}k_3{{< /klmath >}} loss, which is also verl's default KL loss[^verl], cites Schulman's note, whose case for {{< klmath inline=true >}}k_3{{< /klmath >}} is its quality as an estimate. As a loss alone, it follows the forward KL. Near the reference the two directions nearly agree, so as a light leash this is mostly harmless. It is not what the estimator's reputation promises.

**On-policy distillation.** Thinking Machines puts {{< klmath inline=true >}}-k_1{{< /klmath >}} in the reward[^opd]. The loss term it drops has mean zero, so this is the reverse KL's gradient, the right direction for a smaller student: mode-seeking, committing to what it can do well.

## Epilogue

A KL estimator has two jobs in RL.

**Measuring.** Neither estimator always wins. {{< klmath inline=true >}}k_3{{< /klmath >}} has lower variance near {{< klmath inline=true >}}q{{< /klmath >}}; {{< klmath inline=true >}}k_1{{< /klmath >}} is safer far from {{< klmath inline=true >}}q{{< /klmath >}}; {{< klmath inline=true >}}k_2{{< /klmath >}} is biased.

**Training.** The estimator does not matter. {{< klmath inline=true >}}k_1{{< /klmath >}} in the reward, {{< klmath inline=true >}}k_2{{< /klmath >}} as a loss, and {{< klmath inline=true >}}k_3{{< /klmath >}} in both give the same gradient for every sample. What matters is keeping the right terms, and, if you tune anything, tuning the baseline for the gradient rather than the control variate for the estimate.

**Choose a KL estimator for the number it reports, and a baseline for the gradient it produces.**

[^schulman]: John Schulman, ["Approximating KL Divergence"](https://joschu.net/blog/kl-approx.html), 2020.
[^deepseek]: DeepSeek-AI, ["DeepSeek-V3.2"](https://arxiv.org/html/2512.02556v1#S3.SS1), 2025.
[^opd]: Thinking Machines Lab, ["On-Policy Distillation"](https://thinkingmachines.ai/blog/on-policy-distillation/), 2025.
[^cursor]: Cursor Research, ["Composer 2 Technical Report"](https://arxiv.org/abs/2603.24477), 2026, Section 4.1.
[^grpo]: Zhihong Shao et al., ["DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"](https://arxiv.org/abs/2402.03300), 2024. GRPO adds {{< klmath inline=true >}}k_3{{< /klmath >}} with the reference policy directly to its loss.
[^verl]: verl, [`kl_penalty_forward` in `core_algos.py`](https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py); `kl_loss_type` defaults to `low_var_kl`, which is {{< klmath inline=true >}}k_3{{< /klmath >}}.
[^tang]: Yunhao Tang and Rémi Munos, ["On a few pitfalls in KL divergence gradient estimation for RL"](https://arxiv.org/abs/2506.09477), 2025.
[^rethinking]: Kezhao Liu et al., ["Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization"](https://arxiv.org/abs/2510.01555), 2025.
[^wang]: Xihuai Wang, ["Choosing KL Estimators in RL: From Value Unbiasedness to Gradient Correctness"](https://xihuai18.github.io/reinforcement-learning/2025/12/01/kl-estimators-en.html), 2025.
