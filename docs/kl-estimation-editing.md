# Editing “The Structure Behind KL Estimation for Reinforcement Learning”

The article lives in **`content/posts/kl-estimation.md`**. Edit its ordinary Markdown paragraphs, headings, links, and captions. `draft: false` includes it in production builds. No JavaScript knowledge is needed to edit the writing.

## Preview

The current preview is running at:

http://localhost:6779/posts/kl-estimation/

It includes drafts and future-dated posts (the KL article is dated September 30, 2026), and reloads when the Markdown, styles, or scripts change. The preview runs on compute node `denvrbm-2002` (`10.223.239.102`). From your laptop, use a separate terminal for the tunnel; leave the existing Codex SSH session untouched:

```sh
ssh -N -o ExitOnForwardFailure=yes -L 6779:10.223.239.102:6779 npi
```

Keep that terminal open while viewing the preview. The server polls for changes every second because this repository is on shared NFS storage, where edits from another node may not generate local filesystem notifications. Save `content/posts/kl-estimation.md` and the preview will rebuild automatically; the browser normally reloads itself.

To start a new preview (Hugo extended 0.145.0, matching the site's CI):

```sh
HUGO_RESOURCEDIR=/tmp/kl-blog-resources hugo server \
  --buildDrafts --buildFuture --bind 0.0.0.0 --port 6779 \
  --baseURL http://localhost:6779/ --disableFastRender --poll 1s
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
{{< klfigure type="empirical"
    subtitle="LLM RL: Qwen3 32B to 4B on-policy distillation"
    subtitle2="Classical RL: PPO on 3 environments" >}}
Your caption, including **Markdown emphasis** and [links](https://example.com).
{{< /klfigure >}}
```

Figure subtitles also accept inline LaTeX between `$…$`; use Hugo backtick-quoted parameters when the formula contains backslashes (see the Gaussian figure in the Markdown).

The article uses `gaussian`, `holeness`, and `empirical`. The new “Turning up the probability hole” section is standalone Markdown: `holeness` varies h from 0 to 3.5, matches the two KLs by bisection, with q=(0.2, 0.4, 0.4). Its NumPy block is self-contained. The figure has no slider or probability panels below it; hovering displays the matched distributions p1, p2 and q and estimator variances. The optional `rewards` component is retained in JavaScript but is not inserted into this draft. The prose follows the original supplied draft; the additional explanation and reward section from the first implementation have been removed. Use `inline=true` on `klmath` for an equation within a paragraph. Empty figure captions are omitted.

## Figures, themes, and data

- `assets/klviz/klviz.js`: SVG rendering, controls, tooltips, keyboard inspection, and the two analytical toy examples. No plotting library or runtime build step.
- `assets/klviz/klviz.css`: scoped figure/code styling. Gruvbox accents match the parallel-transformers post. It follows the site's OS-driven light/dark theme and also supports an explicit `data-theme` attribute.
- `static/klviz/data.json`: frozen numerical plot data and source SHA256 hashes. The experimental charts are not fitted curves or simulated data.
- `static/klviz/vendor/katex/`: KaTeX 0.16.22, bundled locally with its MIT license and fonts. No external CDN calls.
- `layouts/shortcodes/kl*.html`: Markdown-facing components.
- `layouts/partials/head/head-end.html`: loads content-fingerprinted assets only on pages with `klviz: true`, so browsers do not reuse an older math renderer after an edit.

The Gaussian example moves p = N(mu, 1) against fixed q = N(0, 1), sampling from p to estimate KL[p || q]. The plot uses exact variances; the collapsible NumPy example also estimates them by sampling. Categorical values are exact three-outcome sums. The Gaussian figure has a fixed logarithmic variance axis and a pointer-following inset showing the two densities at the hovered KL[p || q]. It has no slider, scale selector, or text beneath it. The fixed categorical example retains its distributions in the prose; its figure and separate code blocks have been removed. The varying-hole figure retains its self-contained NumPy code. The empirical chart shows the full observed range using the existing decade-bin medians and within-bin interquartile bands for all four raw lines (LLM k1/k3 and PPO k1/k3). Bins with fewer than ten rows remain gaps. No shaded band is a confidence interval. Figures fit the article column and use the page background, with no outer card, border, or shadow. Plot titles are omitted; subtitles retain model and distribution details. The posts index preserves and renders inline math in each opening-paragraph excerpt.

To regenerate the experimental data from the completed studies:

```sh
python tools/export-kl-data.py --experiments ../kl-estimation-experiments
```

This exporter reads existing results only. It checks PPO inputs against the frozen audit hashes and never starts training. Keep the experiment repository's frozen results unchanged.

## When the 50-update study finishes

The empirical variance figure still uses the completed **four-update, three-seed** LLM study. No reward curve is currently embedded. The ongoing 50-update pair is not silently substituted. Its native reward report will be under `../kl-estimation-experiments/results/reward-50-20260928/`.

Once it is complete and audited, update the exporter and the reward figure's seed count, evaluation checkpoints, captions, and surrounding prose together. The 50-update study has **one paired seed**; its curve must not inherit the older three-seed band. The existing variance data still describe the initial/two/four-update fixed prefix bank, even after a longer reward curve is added. Add reward findings to the article only when requested; preserve the supplied wording in the meantime.

The article is enabled for publication. Pushes to `main` run the site’s Hugo build and GitHub Pages deployment workflow. Article-to-article footer navigation is disabled in `layouts/partials/single/page_nav.html`.
