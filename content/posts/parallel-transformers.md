---
title: "A Visual Guide to Parallel Transformers"
date: 2026-09-28
toc: false
summary: "Interactive figures for how a transformer's forward and backward passes are sharded across devices — each axis of parallelism on its own, then the combinations frontier models actually train with, in the JAX scaling book's notation."
---

<link rel="stylesheet" href="/tpviz/tpviz.css">
<script src="/tpviz/data.js" defer></script>
<script src="/tpviz/tpviz.js" defer></script>

Illustrating how a transformer's forward and backward passes are sharded across devices —
data, tensor, context, pipeline, and expert parallelism, in the
notation of the [JAX scaling book](https://jax-ml.github.io/scaling-book/).

## How to read the notation

Everything here is written in the scaling book's [sharding notation](https://jax-ml.github.io/scaling-book/sharding/#partitioning-notation-and-collective-operations). The devices form a *mesh* — <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 4})</span> is four devices along
one axis named <span class="meq">X</span> — and <span class="meq"><span class="mu">In</span>[B<sub>X</sub>,  T,  D]</span> means the batch dimension <span class="meq">B</span> is
sharded four ways over <span class="meq">X</span>, so each device holds a <span class="meq">[B/4,  T,  D]</span> slice. A
tensor with no subscripts anywhere is fully replicated. A trailing <span class="meq">{U<sub>X</sub>}</span> marks an unreduced partial
sum — each device holds a full-shaped tensor that is only its *contribution*
to the true result, and the shards must still be summed over <span class="meq">X</span>.

## How to read the figures

Each figure is a small machine you can drive. Model depth runs left to
right: the input enters at <span class="meq"><span class="mu">In</span></span>, passes through each layer's attention and MLP
stations, and leaves at <span class="meq"><span class="mu">Out</span></span>. Devices are horizontal lanes, stacked
vertically — when devices communicate, tensors fly *vertically* between lanes.
Weights are <span class="sw wt"></span> blue, activations
<span class="sw act"></span> amber, keys and values
<span class="sw kv"></span> teal, and gradients
<span class="sw grad"></span> rose. Hover any tensor for its shape, sharding,
and memory at our nominal sizes (<span class="meq">B=8, T=128, D=1024, F=4096</span>, bf16).
Each figure opens in forward-only mode; switch to *+ Backward* for the full training
step, where every backward operation cites the forward operation it differentiates.

## The axes of transformer sharding

Every way of splitting a transformer across devices is a choice of which dimension to shard: the batch, the sequence, the weights' features, the experts, or the layers. This section takes them one at a time, each on its own four-device mesh, so the communication each one costs is visible in isolation. The next section combines them the way real training runs do.

### Data parallelism

The simplest strategy: replicate the weights everywhere and shard the *batch* —
each device gets <span class="meq"><span class="mu">In</span>[B<sub>X</sub>,  T,  D]</span>, a quarter of the sequences, and runs the whole
model on them. The forward pass needs **zero communication**; step through it
and watch nothing ever cross between lanes. The price appears in the backward
pass: every device computes a weight gradient from only its shard of the batch,
so the gradients are *unreduced partial sums* — <span class="meq"><span class="mu">d</span>W[D,  F]{U<sub>X</sub>}</span> — and each one
must be **AllReduced** across the data axis before the optimizer can step. That
per-weight AllReduce is the classic data-parallel gradient sync.

<figure class="tpv-outer">
  <tpviz-figure strategy="dp"></tpviz-figure>
  <figcaption>
    Data parallelism over <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 4})</span>: a silent forward pass, and
    an AllReduce per weight gradient in the backward.
  </figcaption>
</figure>

### ZeRO-1: shard the optimizer, keep the weights

Plain data parallelism replicates more than the weights: every device also keeps a full copy of the optimizer state, which for Adam is two more tensors the size of the model. <strong>ZeRO-1</strong> keeps the weights replicated but shards the optimizer state over the data axis, and that changes only the gradient step. Instead of AllReducing each <span class="meq"><span class="mu">d</span>W[D, F]{U<sub>X</sub>}</span>, it <strong>ReduceScatters</strong> it — <span class="meq"><span class="mu">ReduceScatter</span><sub>X,D</sub></span> — so each device holds one slice, <span class="meq"><span class="mu">d</span>W[D<sub>X</sub>, F]</span>. Each device then updates just its slice of the weight, <span class="meq">W[D<sub>X</sub>, F]</span>, and an <strong>AllGather</strong> of the updated slices makes the weights whole again, <span class="meq">W[D, F]</span>. Watch the wire panel: an AllReduce <em>is</em> a ReduceScatter followed by an AllGather, so ZeRO-1 moves the same bytes as data parallelism — it has only pulled the optimizer step in between the two halves. The forward pass is untouched. This is the data axis DeepSeek-V3 and Kimi K2 train with.

<figure class="tpv-outer">
  <tpviz-figure strategy="zero1"></tpviz-figure>
  <figcaption>
    ZeRO-1 over <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 4})</span>: a silent forward pass, then ReduceScatter, a sharded optimizer step, and an AllGather of the updated weights.
  </figcaption>
</figure>

### Fully-sharded data parallelism (ZeRO-3)

ZeRO-1 sharded the optimizer state but still replicated every weight four times —
FSDP (ZeRO-3) refuses to pay that memory either. Weights are sharded along the same data axis (<span class="meq">W<sub><span class="mu">in</span></sub>[D<sub>X</sub>,  F]</span>,
<span class="meq">W<sub><span class="mu">out</span></sub>[F,  D<sub>X</sub>]</span>), and each one is **AllGathered just in time**, used for its
matmul, and immediately discarded — watch the station header switch to the
gathered shape and back. The backward pass pays the gather *again* (the weight
is long gone), and the gradient flows the other way: each <span class="meq"><span class="mu">d</span>W</span> is a partial sum
that **ReduceScatters** back to exactly the shard layout the weights live in —
memory of a shard, communication of a gather, in both passes.

<figure class="tpv-outer">
  <tpviz-figure strategy="fsdp"></tpviz-figure>
  <figcaption>
    FSDP over <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 4})</span>: just-in-time weight gathers forward,
    re-gathers plus gradient ReduceScatters backward.
  </figcaption>
</figure>

### Tensor parallelism

Tensor parallelism shards the *feature* dimensions:
attention heads across devices (<span class="meq">W<sub><span class="mu">qkv</span></sub>[D,  H<sub>Y</sub>]</span>, <span class="meq">W<sub><span class="mu">o</span></sub>[H<sub>Y</sub>,  D]</span>) and the MLP's
hidden width (<span class="meq">W<sub><span class="mu">in</span></sub>[D,  F<sub>Y</sub>]</span>, <span class="meq">W<sub><span class="mu">out</span></sub>[F<sub>Y</sub>,  D]</span>). Activations travel sharded on
<span class="meq">D</span>. Every block then **AllGathers on the way in,
ReduceScatters on the way out.** The first matmul needs the full <span class="meq">D</span>, so the
sharded activations are gathered; the second matmul contracts a sharded
dimension, so each device is left holding an *unreduced partial sum* — the
dashed <span class="meq">{U<sub>Y</sub>}</span> tensor — which the ReduceScatter resolves while re-sharding for
the next block.

<figure class="tpv-outer">
  <tpviz-figure strategy="tp"></tpviz-figure>
  <figcaption>
    Tensor parallelism over <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;Y&#x27;</span>: 4})</span>: AllGather into every block,
    ReduceScatter out of it, and the exact mirror image in the backward.
  </figcaption>
</figure>

### Context parallelism

Long sequences don't fit on one device, so shard the *sequence*:
<span class="meq"><span class="mu">In</span>[B,  T<sub>X</sub>,  D]</span>, each device holding a quarter of the tokens. The MLP never
notices — tokens are independent there. Attention is where every query must see every key and value,
so <span class="meq">K</span> and <span class="meq">V</span> are **AllGathered** across the sequence shards while <span class="meq">Q</span> stays
local. The gathered K/V are used and dropped and the backward
pass re-gathers them — the same trade FSDP makes with weights. During the backward pass every device produces gradient contributions
for *all* tokens' keys and values, so <span class="meq"><span class="mu">d</span>K</span> and <span class="meq"><span class="mu">d</span>V</span> are partial sums that
**ReduceScatter** back over the sequence — and the weight gradients AllReduce
over the context axis, exactly like data parallelism sums over the batch.

<figure class="tpv-outer">
  <tpviz-figure strategy="cp"></tpviz-figure>
  <figcaption>
    Context parallelism over <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 4})</span>: K/V gathers in
    attention, a silent MLP, and sequence-scattered dK/dV in the backward.
  </figcaption>
</figure>

### Expert parallelism

In a mixture-of-experts layer the MLP becomes four experts, one per device
(<span class="meq">W<sub><span class="mu">in</span></sub>[E<sub>Z</sub>,  D,  F]</span>), and each token is routed to one of them. The figure's
tokens are small squares colored by their assigned expert: the **AllToAll
dispatch** is the moment every device sends every other device the tokens that
belong to it — watch the colors sort themselves into lanes. The experts run an
ordinary MLP on their guests, and a second **AllToAll** sends every token home.
The backward pass mirrors it: token *gradients* AllToAll out to the experts,
expert weight gradients stay local (each expert owns its weights outright), and
the attention weight gradients AllReduce over the batch axis as in data
parallelism.

<figure class="tpv-outer">
  <tpviz-figure strategy="ep"></tpviz-figure>
  <figcaption>
    Expert parallelism over <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;Z&#x27;</span>: 4})</span>: router-colored tokens
    crossing in the AllToAll, expert MLPs, and mirrored gradient AllToAlls.
  </figcaption>
</figure>

### Pipeline parallelism

Pipeline parallelism shards the *layers*: stage 0 owns the first layer, stage 1 owns
the second, and activations hop across the boundary with a single point-to-point
send — the cheapest communication in this whole article. The catch is idleness:
with one batch, each stage would wait on the other, so the batch splits into
four microbatches that chase each other through the pipe. The staircase is the pipeline filling, the empty
corners are **the bubble**. Switch to *+ Backward* and the gradients flow
right-to-left through the same pipe — each backward cell is drawn twice as wide
because backward costs roughly twice the compute, which is why the bubble grows
during training.

<figure class="tpv-outer">
  <tpviz-figure strategy="pp"></tpviz-figure>
  <figcaption>
    Pipeline parallelism, 2 stages × 4 microbatches: activations hop forward,
    gradients hop back, and the Gantt chart shows who idles.
  </figcaption>
</figure>

## Examples from frontier models

Training reports rarely use one of these alone. Reading the recent ones — DeepSeek-V3 and V4, Kimi K2, GLM-4.5/5, Nemotron-4, -H and 3, Qwen3-VL, and Llama 3 — the recipes fall into three families. The figures below show each one with two devices per axis so the grids stay readable; the real degrees are in the text. The canvas changes shape here: each pipeline stage is a grid of devices, arranged so that every mesh axis is a fixed direction — an FSDP partner is always two rows away, a tensor-parallel partner the next column, and so on — and every collective flies along exactly one of them, wrapped in that axis's color, with a badge and a program tag naming the parallelism responsible. The tally under the step bar counts collectives per axis.

### Dense models: Llama 3

<strong>Dense models use 4D: FSDP × TP × CP × PP.</strong> Llama 3 405B trained with TP 8, PP 16 and FSDP 64 at 8K context, then TP 8, <strong>CP 16</strong>, PP 16 and FSDP 8 for the 128K stage — context parallelism is switched on only when the sequences get long. The dimensions are ordered by bandwidth: TP stays inside a server, DP spans the cluster. Nemotron-4 340B (TP 8 × PP 12 × DP) and Nemotron-H (TP 8 × DP 768, no PP) are the same family without the context axis. Here: <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 2, <span class="mu">&#x27;Y&#x27;</span>: 2, <span class="mu">&#x27;C&#x27;</span>: 2, <span class="mu">&#x27;stage&#x27;</span>: 2})</span>, activations <span class="meq"><span class="mu">In</span>[B<sub>X</sub>, T<sub>C</sub>, D<sub>Y</sub>]</span>. Every matmul opens with FSDP's weight gather and TP's activation gather; CP only speaks in attention; the pipeline is one send per direction.

<figure class="tpv-outer">
  <tpviz-figure strategy="dense4d"></tpviz-figure>
  <figcaption>
    Dense 4D over <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 2, <span class="mu">&#x27;Y&#x27;</span>: 2, <span class="mu">&#x27;C&#x27;</span>: 2, <span class="mu">&#x27;stage&#x27;</span>: 2})</span> (Llama 3 style): FSDP × TP × CP × PP, 16 devices.
  </figcaption>
</figure>

### Open MoE models: DeepSeek-V3 and Kimi K2

<strong>Open MoE models use 3D: EP × PP × ZeRO-1, and skip TP.</strong> DeepSeek-V3 trained with PP 16 (DualPipe), EP 64 across 8 nodes and ZeRO-1 data parallelism, explicitly "without costly tensor parallelism"; Kimi K2 with PP 16, EP 16 and ZeRO-1; GLM-4.5V with EP 8 × PP 4, adding CP 4 only for its long-context stage. Here: <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 2, <span class="mu">&#x27;Z&#x27;</span>: 2, <span class="mu">&#x27;stage&#x27;</span>: 2})</span>, activations <span class="meq"><span class="mu">In</span>[B<sub>XZ</sub>, T, D]</span> — the batch is split over both the data axis and the expert axis, because outside the MoE block the expert axis is just more data parallelism. The forward pass has <em>no gathers at all</em>: only the <span class="meq"><span class="mu">AllToAll</span><sub>Z</sub></span> pair per MoE layer and the stage hop. The backward is where the two data-like axes show their difference: an attention weight gradient scatters over <span class="meq">X</span> (ZeRO-1) and sums over <span class="meq">Z</span> (the expert axis replicates attention), while an expert weight gradient only scatters over <span class="meq">X</span> — each expert owns its weights. The optimizer then steps on each shard and the updated weights gather back.

<figure class="tpv-outer">
  <tpviz-figure strategy="moe3d"></tpviz-figure>
  <figcaption>
    MoE 3D over <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 2, <span class="mu">&#x27;Z&#x27;</span>: 2, <span class="mu">&#x27;stage&#x27;</span>: 2})</span> (DeepSeek-V3 / Kimi K2 style): EP × PP × ZeRO-1, no tensor parallelism, 8 devices.
  </figcaption>
</figure>

### Everything at once: Nemotron 3 and Qwen3-VL

<strong>Megatron-based labs use everything at once.</strong> Nemotron 3 Nano's long-context phase ran CP 8 × TP 8 × EP 8 × PP 4 (Nemotron 3 Ultra pushes EP to 128), and Qwen3-VL lists TP, PP, CP, EP and ZeRO-1 DP on up to 10,000 GPUs. Here it is with two of everything: 32 devices, <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 2, <span class="mu">&#x27;Y&#x27;</span>: 2, <span class="mu">&#x27;C&#x27;</span>: 2, <span class="mu">&#x27;Z&#x27;</span>: 2, <span class="mu">&#x27;stage&#x27;</span>: 2})</span>, activations <span class="meq"><span class="mu">In</span>[B<sub>XZ</sub>, T<sub>C</sub>, D<sub>Y</sub>]</span> — sharded four ways at once. Each stage is a 4×4 grid; rows are <span class="meq">X</span> then <span class="meq">C</span>, columns <span class="meq">Z</span> then <span class="meq">Y</span>. The tally row is the takeaway: FSDP and TP do most of the talking, CP speaks only in attention, EP is two AllToAlls per MoE layer, and PP is one send per direction. In the backward, a weight gradient has to be summed over every axis that split the tokens — <span class="meq">X</span>, <span class="meq">Z</span> and <span class="meq">C</span> — which is why one gradient takes three collectives to settle.

<figure class="tpv-outer">
  <tpviz-figure strategy="5d"></tpviz-figure>
  <figcaption>
    Everything at once over <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 2, <span class="mu">&#x27;Y&#x27;</span>: 2, <span class="mu">&#x27;C&#x27;</span>: 2, <span class="mu">&#x27;Z&#x27;</span>: 2, <span class="mu">&#x27;stage&#x27;</span>: 2})</span> (Nemotron 3 / Qwen3-VL style): 32 devices, every collective colored by the axis that causes it.
  </figcaption>
</figure>
