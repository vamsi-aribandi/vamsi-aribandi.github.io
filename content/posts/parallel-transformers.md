---
title: "A Visual Guide to Parallel Transformers"
date: 2026-09-28
toc: false
summary: "Interactive figures for how a transformer's forward and backward passes are sharded across devices — each axis of parallelism on its own, then the combinations frontier models actually train with, in the JAX scaling book's notation."
---

<link rel="stylesheet" href="/tpviz/tpviz.css">
<script src="/tpviz/data.js" defer></script>
<script src="/tpviz/tpviz.js" defer></script>

How a transformer's forward and backward passes are sharded across devices: each axis of parallelism on its own, then the combinations frontier models train with, in the notation of the <a href="https://jax-ml.github.io/scaling-book/">JAX scaling book</a>.

## How to read the notation

Everything here uses the scaling book's <a href="https://jax-ml.github.io/scaling-book/sharding/">sharding notation</a>. Devices form a <em>mesh</em>: <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 4})</span> is four devices along one axis named <span class="meq">X</span>. A subscript names the mesh axis a dimension is split over: <span class="meq"><span class="mu">In</span>[B<sub>X</sub>, T, D]</span> means the batch is sharded four ways over <span class="meq">X</span>, so each device holds a <span class="meq">[B/4, T, D]</span> slice. A dimension without a subscript is not split, and a tensor with no subscripts is replicated. A trailing <span class="meq">{U<sub>X</sub>}</span> marks an <strong>unreduced partial sum</strong>: each device holds a full-shaped tensor that is only its contribution, and the contributions still have to be summed over <span class="meq">X</span>. That sum is what AllReduce and ReduceScatter do.

Sizes assumed by the tooltips: <span class="meq">B=8</span> sequences of <span class="meq">T=128</span> tokens, <span class="meq">D=1024</span> model width, <span class="meq">F=4096</span> MLP width, <span class="meq">H=1024</span> the attention heads concatenated, <span class="meq">E=4</span> experts with <span class="meq">S=256</span> routed tokens each, all in bf16.

## How to read the figures

Model depth runs left to right: the input enters at <span class="meq"><span class="mu">In</span></span>, passes each layer's attention and MLP stations, and leaves at <span class="meq"><span class="mu">Out</span></span>. Devices are horizontal lanes; when they communicate, tensors fly vertically between lanes. Weights are <span class="sw wt"></span> blue, activations <span class="sw act"></span> amber, keys and values <span class="sw kv"></span> teal, gradients <span class="sw grad"></span> rose. A dashed outline is the full logical tensor; the solid patch inside it is the slice this device holds, drawn at its offset. Hover a tensor for its shape, sharding and memory.

Each segment of the step bar is one operation. Click to jump to the state after it, or press play. While the current operation is a collective, the panel under the canvas shows how it runs on the wire, hop by hop. The program beside the canvas lists every operation in book notation and follows along. Figures open forward-only; <em>+ Backward</em> adds the rest of the training step, where saved activations park under the weights that need them and every backward operation cites the forward operation it differentiates.

## The axes of transformer sharding

Each way of splitting a transformer is a choice of dimension to shard: the batch, the sequence, the weights' features, the experts, or the layers. Here each runs alone on a four-device mesh, so its communication is visible in isolation.

### Data parallelism

Replicate the weights, shard the batch: <span class="meq"><span class="mu">In</span>[B<sub>X</sub>, T, D]</span>. The forward pass communicates nothing. The backward does: each device computes weight gradients from its quarter of the batch, so every <span class="meq"><span class="mu">d</span>W[D, F]{U<sub>X</sub>}</span> is a partial sum, and each one is AllReduced over <span class="meq">X</span> before the optimizer step.

<figure class="tpv-outer">
  <tpviz-figure strategy="dp"></tpviz-figure>
  <figcaption>
    Data parallelism over <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 4})</span>: a silent forward pass, one AllReduce per weight gradient.
  </figcaption>
</figure>

### ZeRO-1: shard the optimizer, keep the weights

Data parallelism also replicates the optimizer state, which for Adam is two more model-sized tensors. <strong>ZeRO-1</strong> keeps the weights replicated and shards the optimizer state over <span class="meq">X</span>. Only the gradient step changes. Each <span class="meq"><span class="mu">d</span>W[D, F]{U<sub>X</sub>}</span> is <strong>ReduceScattered</strong> rather than AllReduced, so a device holds <span class="meq"><span class="mu">d</span>W[D<sub>X</sub>, F]</span>; it updates its slice <span class="meq">W[D<sub>X</sub>, F]</span>; and an <strong>AllGather</strong> of the updated slices makes the weights whole again. An AllReduce is a ReduceScatter followed by an AllGather, so ZeRO-1 moves the same bytes as data parallelism, with the optimizer step in between. This is the data axis DeepSeek-V3 and Kimi K2 train with.

<figure class="tpv-outer">
  <tpviz-figure strategy="zero1"></tpviz-figure>
  <figcaption>
    ZeRO-1 over <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 4})</span>: ReduceScatter, a sharded optimizer step, then an AllGather of the updated weights.
  </figcaption>
</figure>

### Fully-sharded data parallelism (ZeRO-3)

ZeRO-1 still replicates every weight. FSDP shards them too, along the same axis: <span class="meq">W<sub><span class="mu">in</span></sub>[D<sub>X</sub>, F]</span>, <span class="meq">W<sub><span class="mu">out</span></sub>[F, D<sub>X</sub>]</span>. Each weight is <strong>AllGathered just before its matmul</strong> and discarded right after; the station header shows the gathered shape and back. The backward gathers each weight again, and each gradient <strong>ReduceScatters</strong> back to the layout the weights live in. Memory of a shard, communication of a gather, in both passes.

<figure class="tpv-outer">
  <tpviz-figure strategy="fsdp"></tpviz-figure>
  <figcaption>
    FSDP over <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 4})</span>: just-in-time weight gathers forward, re-gathers and gradient ReduceScatters backward.
  </figcaption>
</figure>

### Tensor parallelism

Tensor parallelism shards the feature dimensions: the heads in attention (<span class="meq">W<sub><span class="mu">qkv</span></sub>[D, H<sub>Y</sub>]</span>, <span class="meq">W<sub><span class="mu">o</span></sub>[H<sub>Y</sub>, D]</span>) and the hidden width in the MLP (<span class="meq">W<sub><span class="mu">in</span></sub>[D, F<sub>Y</sub>]</span>, <span class="meq">W<sub><span class="mu">out</span></sub>[F<sub>Y</sub>, D]</span>). Activations travel sharded on <span class="meq">D</span>. Each block <strong>AllGathers on the way in and ReduceScatters on the way out</strong>: the first matmul needs all of <span class="meq">D</span>, so the activations are gathered; the second contracts a sharded dimension, so each device is left with a partial sum <span class="meq">{U<sub>Y</sub>}</span>, which the ReduceScatter resolves while re-sharding for the next block.

<figure class="tpv-outer">
  <tpviz-figure strategy="tp"></tpviz-figure>
  <figcaption>
    Tensor parallelism over <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;Y&#x27;</span>: 4})</span>: AllGather into every block, ReduceScatter out of it, mirrored in the backward.
  </figcaption>
</figure>

### Context parallelism

Context parallelism shards the sequence: <span class="meq"><span class="mu">In</span>[B, T<sub>X</sub>, D]</span>. The MLP acts per token and never communicates. Attention needs every key and value for every query, so <span class="meq">K</span> and <span class="meq">V</span> are <strong>AllGathered</strong> over the sequence shards while <span class="meq">Q</span> stays local; the gathered K and V are used and dropped, and the backward gathers them again. In the backward every device contributes to every token's <span class="meq"><span class="mu">d</span>K</span> and <span class="meq"><span class="mu">d</span>V</span>, so those are partial sums that <strong>ReduceScatter</strong> back over the sequence, and weight gradients AllReduce over <span class="meq">X</span> as in data parallelism.

<figure class="tpv-outer">
  <tpviz-figure strategy="cp"></tpviz-figure>
  <figcaption>
    Context parallelism over <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 4})</span>: K/V gathers in attention, a silent MLP, sequence-scattered dK/dV in the backward.
  </figcaption>
</figure>

### Expert parallelism

In a mixture-of-experts layer the MLP becomes <span class="meq">E</span> experts, one per device, <span class="meq">W<sub><span class="mu">in</span></sub>[E<sub>Z</sub>, D, F]</span>, and a router sends each token to one of them. Tokens are drawn as squares colored by their expert. The <strong>AllToAll dispatch</strong> sends every device the tokens for its expert; the experts run an ordinary MLP; a second <strong>AllToAll</strong> sends every token home. The backward mirrors this: token gradients AllToAll out and back, expert weight gradients stay local because each expert owns its weights, and attention weight gradients AllReduce over the batch axis as in data parallelism.

<figure class="tpv-outer">
  <tpviz-figure strategy="ep"></tpviz-figure>
  <figcaption>
    Expert parallelism over <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;Z&#x27;</span>: 4})</span>: tokens sorted by expert in the AllToAll, expert MLPs, mirrored gradient AllToAlls.
  </figcaption>
</figure>

### Pipeline parallelism

Pipeline parallelism shards the layers: stage 0 owns layer 1, stage 1 owns layer 2, and activations cross the boundary with one point-to-point send, the cheapest communication here. The cost is idleness. With one batch each stage would wait on the other, so the batch is split into four microbatches that follow each other through the pipe. The schedule under the lanes fills as you step: the staircase is the pipe filling, the empty cells are the bubble. In the backward each cell is twice as wide, because backward is about twice the compute, so the bubble grows during training.

<figure class="tpv-outer">
  <tpviz-figure strategy="pp"></tpviz-figure>
  <figcaption>
    Pipeline parallelism, 2 stages × 4 microbatches: activations hop forward, gradients hop back, and the schedule shows who idles.
  </figcaption>
</figure>

## Examples from frontier models

Training runs combine these axes. From recent reports (<a href="https://arxiv.org/abs/2412.19437">DeepSeek-V3</a>, <a href="https://arxiv.org/abs/2507.20534">Kimi K2</a>, <a href="https://arxiv.org/abs/2507.01006">GLM-4.5V</a>, <a href="https://arxiv.org/abs/2406.11704">Nemotron-4</a>, <a href="https://arxiv.org/abs/2504.03624">Nemotron-H</a>, <a href="https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf">Nemotron 3</a>, <a href="https://arxiv.org/abs/2511.21631">Qwen3-VL</a>, <a href="https://arxiv.org/abs/2407.21783">Llama 3</a>) the recipes fall into three families. The figures use two devices per axis so the grids stay readable; the reported degrees are in the text.

The canvas changes here. Each pipeline stage is a grid of devices, arranged so that every mesh axis is a fixed direction: an FSDP partner is two rows away, a tensor-parallel partner is the next column, and so on. Every collective flies along exactly one axis, wrapped in that axis's color, with a badge under the grid and a tag in the program naming the parallelism responsible. The tally under the step bar counts collectives per axis.

### Dense models: Llama 3

<strong>FSDP × TP × CP × PP.</strong> Llama 3 405B trained with TP 8, PP 16 and FSDP 64 at 8K tokens, then TP 8, <strong>CP 16</strong>, PP 16 and FSDP 8 at 128K: context parallelism is turned on only when sequences get long. Axes are ordered by bandwidth, TP within a server and data parallelism across the cluster. Nemotron-4 340B (TP 8 × PP 12 × DP) and Nemotron-H (TP 8 × DP 768, no PP) are the same family without the context axis. Here <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 2, <span class="mu">&#x27;Y&#x27;</span>: 2, <span class="mu">&#x27;C&#x27;</span>: 2, <span class="mu">&#x27;stage&#x27;</span>: 2})</span> and <span class="meq"><span class="mu">In</span>[B<sub>X</sub>, T<sub>C</sub>, D<sub>Y</sub>]</span>. Every matmul opens with FSDP's weight gather and TP's activation gather; CP speaks only in attention; the pipeline is one send per direction.

<figure class="tpv-outer">
  <tpviz-figure strategy="dense4d"></tpviz-figure>
  <figcaption>
    Dense 4D over <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 2, <span class="mu">&#x27;Y&#x27;</span>: 2, <span class="mu">&#x27;C&#x27;</span>: 2, <span class="mu">&#x27;stage&#x27;</span>: 2})</span>, Llama 3 style: FSDP × TP × CP × PP on 16 devices.
  </figcaption>
</figure>

### Open MoE models: DeepSeek and Kimi

<strong>EP × CP × PP × ZeRO-1, and no TP.</strong> DeepSeek-V3 trained with PP 16 (DualPipe), EP 64 across 8 nodes and ZeRO-1 data parallelism, "without costly tensor parallelism"; Kimi K2 with PP 16, EP 16 and ZeRO-1. Neither reports context parallelism (V3 reached 128K with YaRN), but the long-context work that followed does: GLM-4.5V adds CP 4 for its long-context stage, and <a href="https://arxiv.org/abs/2606.19348">DeepSeek-V4</a> and <a href="https://arxiv.org/abs/2607.24653">Kimi K3</a> each describe context-parallel attention for million-token training. Here <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 2, <span class="mu">&#x27;C&#x27;</span>: 2, <span class="mu">&#x27;Z&#x27;</span>: 2, <span class="mu">&#x27;stage&#x27;</span>: 2})</span> and <span class="meq"><span class="mu">In</span>[B<sub>XZ</sub>, T<sub>C</sub>, D]</span>: the batch is split over the data axis and the expert axis, since outside the MoE block the expert axis is more data parallelism, and the sequence over <span class="meq">C</span>. The forward has no weight gathers at all: K and V gather over <span class="meq">C</span>, the <span class="meq"><span class="mu">AllToAll</span><sub>Z</sub></span> pair runs per MoE layer, and the activations hop between stages. In the backward an attention weight gradient scatters over <span class="meq">X</span> (ZeRO-1) and sums over <span class="meq">Z</span> and <span class="meq">C</span>; an expert weight gradient scatters over <span class="meq">X</span> and sums over <span class="meq">C</span> only, because each expert owns its weights. Then each device steps on its shard and the updated weights gather back.

<figure class="tpv-outer">
  <tpviz-figure strategy="moe4d"></tpviz-figure>
  <figcaption>
    MoE 4D over <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 2, <span class="mu">&#x27;C&#x27;</span>: 2, <span class="mu">&#x27;Z&#x27;</span>: 2, <span class="mu">&#x27;stage&#x27;</span>: 2})</span>, DeepSeek / Kimi style: EP × CP × PP × ZeRO-1, no tensor parallelism, 16 devices.
  </figcaption>
</figure>

### Everything at once: Nemotron 3 and Qwen3-VL

<strong>FSDP × TP × CP × EP × PP.</strong> Nemotron 3 Nano's long-context phase ran CP 8 × TP 8 × EP 8 × PP 4 (Nemotron 3 Ultra pushes EP to 128), and Qwen3-VL lists TP, PP, CP, EP and ZeRO-1 DP on up to 10,000 GPUs. Here, two of everything: 32 devices, <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 2, <span class="mu">&#x27;Y&#x27;</span>: 2, <span class="mu">&#x27;C&#x27;</span>: 2, <span class="mu">&#x27;Z&#x27;</span>: 2, <span class="mu">&#x27;stage&#x27;</span>: 2})</span>, <span class="meq"><span class="mu">In</span>[B<sub>XZ</sub>, T<sub>C</sub>, D<sub>Y</sub>]</span>. Each stage is a 4 × 4 grid, rows <span class="meq">X</span> then <span class="meq">C</span>, columns <span class="meq">Z</span> then <span class="meq">Y</span>. The tally is the summary: FSDP and TP do most of the talking, CP speaks only in attention, EP is two AllToAlls per MoE layer, PP one send per direction. In the backward a weight gradient is summed over every axis that split the tokens, <span class="meq">X</span>, <span class="meq">Z</span> and <span class="meq">C</span>, so one gradient takes three collectives to settle.

<figure class="tpv-outer">
  <tpviz-figure strategy="5d"></tpviz-figure>
  <figcaption>
    Everything at once over <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 2, <span class="mu">&#x27;Y&#x27;</span>: 2, <span class="mu">&#x27;C&#x27;</span>: 2, <span class="mu">&#x27;Z&#x27;</span>: 2, <span class="mu">&#x27;stage&#x27;</span>: 2})</span>, Nemotron 3 / Qwen3-VL style: 32 devices, every collective colored by the axis that causes it.
  </figcaption>
</figure>
