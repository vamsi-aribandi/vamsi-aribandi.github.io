# Editing “The Structure Behind KL Estimation for Reinforcement Learning”

The article lives in **`content/posts/kl-estimation.md`**. Edit its ordinary Markdown paragraphs, headings, links, and captions. The article currently has `draft: true` and is excluded from production builds. No JavaScript knowledge is needed to edit the writing.

## Preview

The current preview is running at:

http://localhost:6778/posts/kl-estimation/

The preview includes drafts and future-dated posts and binds to `127.0.0.1` in the current workspace. Forward port 6778 from the workspace if accessing it from another machine. The previous preview on port 6779 has been stopped.

The server polls for changes every second because this repository is on shared NFS storage, where edits from another node may not generate local filesystem notifications. Save `content/posts/kl-estimation.md` and the preview will rebuild automatically; the browser normally reloads itself.

To start a new preview (Hugo extended 0.145.0, matching the site's CI):

```sh
HUGO_RESOURCEDIR=/tmp/kl-blog-resources hugo server \
  --buildDrafts --buildFuture --bind 127.0.0.1 --port 6778 \
  --baseURL http://localhost:6778/ --disableFastRender --poll 1s \
  --destination /tmp/kl-blog-preview
```

This session's temporary Hugo executable is `/tmp/kl-blog-tools/hugo`. Its server log and PID are in `/tmp/kl-blog-tools/preview-server.log` and `preview-server.pid`.

## Article components

Display equations use editable LaTeX:

```text
{{< klmath >}}
D_{\mathrm{KL}}[p\,\|\,q] = \mathbb{E}_p[\log(p/q)].
{{< /klmath >}}
```

Collapsible code uses Python directly, without a surrounding Markdown code fence:

```text
{{< klcode title="Reproduce the example" >}}
import numpy as np
# Edit the Python here.
{{< /klcode >}}
```

For figures, keep the `type` and edit the subtitle and Markdown caption; plots have no titles:

```text
{{< klfigure type="categoricalGradients"
    subtitle=`$p_\theta = (a, 1-a)$ · $a = \sigma(\theta)$ · $q = (0.5, 0.5)$` >}}
Your caption, including **Markdown emphasis** and [links](https://example.com).
{{< /klfigure >}}
```

Figure subtitles also accept inline LaTeX between `$…$`; use Hugo backtick-quoted parameters when the formula contains backslashes (see the Gaussian figure in the Markdown).

The article uses `gaussian`, `holeness`, `gaussianGradients`, `gaussianToy`, `categoricalToy`, and `bimodalToy`. The variance figures remain unchanged. The LLM-versus-classical-RL section has been replaced with two analytical policy-gradient examples and a conclusion distinguishing KL values from gradient coefficients. The optional `empirical` and `rewards` components and their frozen data remain available but are not inserted into this draft.

The gradient figures each have two plots: expected penalty gradient and true KL during optimization. A single three-entry legend identifies k1 in reward, k3 in reward, and k3 as direct loss. They have no plot titles, cards, or controls. Hovering the first plot shows the distributions, common expected estimator value, and gradients; hovering the second shows the evolving policy distributions and each policy's KL. The compact hover cards can extend outside the figure. Placement scores all four corners around the cursor by overlap with both plots, stays within the viewport, and uses a small preference for the current corner to avoid jitter. The cards retain the distribution inset and numerical values while omitting repeated explanations. They dismiss on scroll or blur. Arrow keys also inspect values, and Escape dismisses the tooltip. The plots stack on narrow screens.

All gradient curves use exact on-policy expectations with a fixed reference. Gaussian updates start at mu=1, with learning rate 0.15 and 30 steps. Two-action updates start at a=0.8, with learning rate 0.5 and 80 steps; updates are in logit space. The collapsible NumPy block reproduces the three trajectories. Direct k3 gives the forward-KL gradient, which coincides with the reverse-KL gradient for the equal-variance Gaussians but differs for the categorical example. Keep this distinction in the prose and captions.

Structure and message: the article argues that k3 is a better estimate of KL but not a better gradient. Sections: intro and thesis; “k3 is k1 plus a zero-mean term” (Gaussian variance and hole figures); “The estimate becomes a gradient” (one identity, three facts, the 2×2 table); “Three small worlds” (Gaussian, two actions, two modes); “What this means for OPD”; “The mental model”. Citations are Markdown footnotes at the end of the file. The prose style is deliberately short and declarative.

`bimodalToy` fits one Gaussian p = N(mu, sigma^2) to q = ½N(−2, 0.5²) + ½N(2, 0.5²), learning mu and log sigma from (1, 1), batch 64, learning rate 0.05, 300 steps, 1,000 runs. Its plots are KL[p ∥ q] and KL[q ∥ p] over steps (median run by KL[p ∥ q], middle 50% shaded, diverged runs counted as infinite) and the median runs' final densities. Hover cards show q and each method's median-run Gaussian.

The optimization part of the article also uses two three-panel figures, `gaussianToy` (p = N(mu, 1), q = N(0, 1), learning mu from 1.5, learning rate 0.1) and `categoricalToy` (p = (a, 1 - a), q = (0.5, 0.5), learning the logit from a = 0.05, learning rate 1). Each shows the exact variance of the k1/k3 KL estimates, the exact per-sample variance of three gradients (k1 in reward, k3 in reward, k3 as direct loss), and true KL during minibatch gradient descent (batch 16, median of 2,000 runs, middle 50% shaded). Their data are in `static/klviz/toys.json`, generated by `python tools/kl-toy-figures.py`; the article's collapsible code block is that script's core. Pass the data file with `data="klviz/toys.json"` on the shortcode. Hovering the variance plots shows the policy against q; hovering the training plot shows each method's median run (its policy and KL), whose parameters are stored in `toys.json`. In every plot, a marker on each curve appears only while hovering and follows the pointer; there are no fixed markers. To change a start point, learning rate, or batch size, edit the script, rerun it, and update the numbers quoted in the prose. `gaussianGradients` and `categoricalGradients` are no longer inserted but remain available.

`tools/kl-gradient-study.py` is the broader exploratory study (archived in the kl-estimation-experiments repository under `results/kl-gradient-study-20261001/`); the article no longer quotes it.

Use `inline=true` on `klmath` for an equation within a paragraph. Empty figure captions are omitted.

## Figures, themes, and data

- `assets/klviz/klviz.js`: SVG rendering, controls, tooltips, keyboard inspection, and the analytical toy examples. No plotting library or runtime build step.
- `assets/klviz/klviz.css`: scoped figure/code styling. Gruvbox accents match the parallel-transformers post. It follows the site's OS-driven light/dark theme and also supports an explicit `data-theme` attribute.
- `static/klviz/data.json`: frozen numerical plot data and source SHA256 hashes. These archived experimental data are not used by the current article; the new gradient figures are evaluated analytically in JavaScript.
- `static/klviz/vendor/katex/`: KaTeX 0.16.22, bundled locally with its MIT license and fonts. No external CDN calls.
- `layouts/shortcodes/kl*.html`: Markdown-facing components.
- `layouts/partials/head/head-end.html`: loads content-fingerprinted assets only on pages with `klviz: true`, so browsers do not reuse an older math renderer after an edit.

The Gaussian example moves p = N(mu, 1) against fixed q = N(0, 1), sampling from p to estimate KL[p || q]. The plot uses exact variances; the collapsible NumPy example also estimates them by sampling. Categorical values are exact three-outcome sums. The Gaussian figure has a fixed logarithmic variance axis and a pointer-following inset showing the two densities at the hovered KL[p || q]. It has no slider, scale selector, or text beneath it. The fixed categorical example retains its distributions in the prose; its figure and separate code blocks have been removed. The varying-hole figure retains its self-contained NumPy code. Figures fit the article column and use the page background, with no outer card, border, or shadow. Plot titles are omitted; subtitles retain model and distribution details. The posts index uses each post’s front-matter `summary` as its subtext, with Markdown formatting. Edit `summary` to change that text.

To regenerate the experimental data from the completed studies:

```sh
python tools/export-kl-data.py --experiments ../kl-estimation-experiments
```

This exporter reads existing results only. It checks PPO inputs against the frozen audit hashes and never starts training. Keep the experiment repository's frozen results unchanged.

## Publication and archived experiments

The article remains a draft. Pushes to `main` run the site's Hugo build and GitHub Pages deployment workflow, which excludes drafts. Preview it with `--buildDrafts`; leave generated output in `/tmp/kl-blog-preview` rather than the tracked `public/` directory. Article-to-article footer navigation is disabled in `layouts/partials/single/page_nav.html`.

The frozen experimental exporter and data are retained for historical use. They do not supply the new gradient plots. The old variance data describe the completed four-update, three-seed LLM study; the separate 50-update reward study has one paired seed. If experimental figures are reintroduced, audit the results and update seed counts, checkpoints, captions, and surrounding prose together. Do not modify the experiment repository's frozen results.
