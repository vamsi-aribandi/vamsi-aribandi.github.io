---
title: "A Visual Guide to Parallel Transformers"
date: 2026-09-28
toc: false
summary: "Interactive figures for how a transformer's forward and backward passes are sharded across devices — data, tensor, context, pipeline, and expert parallelism, one at a time and then all five at once on a 32-device mesh, in the JAX scaling book's notation."
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
The final figure combines all five strategies on one 32-device mesh; there, every
collective is also tagged with the mesh axis — and so the parallelism — it belongs to.

## Data parallelism

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

## Fully-sharded data parallelism (ZeRO-3)

Data parallelism replicates every weight four times — FSDP refuses to pay that
memory. Weights are sharded along the same data axis (<span class="meq">W<sub><span class="mu">in</span></sub>[D<sub>X</sub>,  F]</span>,
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

## Tensor parallelism

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

## Context parallelism

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

## Expert parallelism

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

## Pipeline parallelism

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

## Putting it all together: 5D parallelism

Real training runs use all five at once, and the point of the singles was to make this figure readable. Here is the same two-layer MoE transformer on <strong>32 devices</strong>, two of everything: <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 2, <span class="mu">&#x27;Y&#x27;</span>: 2, <span class="mu">&#x27;C&#x27;</span>: 2, <span class="mu">&#x27;Z&#x27;</span>: 2, <span class="mu">&#x27;stage&#x27;</span>: 2})</span>. FSDP shards weights over <span class="meq">X</span>, tensor parallelism shards features over <span class="meq">Y</span>, context parallelism shards the sequence over <span class="meq">C</span>, the two experts live on <span class="meq">Z</span>, and the two layers sit on two pipeline stages. The activations are sharded four ways <em>at once</em> — <span class="meq"><span class="mu">In</span>[B<sub>XZ</sub>, T<sub>C</sub>, D<sub>Y</sub>]</span> — the batch over both <span class="meq">X</span> and <span class="meq">Z</span> (outside the MoE block the expert axis is just more data parallelism), the sequence over <span class="meq">C</span>, the model width over <span class="meq">Y</span>. Weights are <span class="meq">W<sub><span class="mu">qkv</span></sub>[D<sub>X</sub>, H<sub>Y</sub>]</span>: FSDP-sharded on their input dimension, tensor-sharded on their output dimension; expert weights add the expert axis, <span class="meq">W<sub><span class="mu">in</span></sub>[E<sub>Z</sub>, D<sub>X</sub>, F<sub>Y</sub>]</span>.

<strong>Reading the grid.</strong> Each stage is a 4×4 grid of devices, laid out so that every mesh axis is a fixed geometric relation: rows are <span class="meq">X</span> outside and <span class="meq">C</span> inside, columns are <span class="meq">Z</span> outside and <span class="meq">Y</span> inside (the margin labels give every device's coordinates). So an FSDP partner is two rows away, a context partner is the adjacent row, an expert partner is two columns away, a tensor partner is the adjacent column, and the pipeline partner is the same cell in the other grid. Every collective flies along exactly one of those relations, wrapped in a halo of that axis's color, with a badge under the grid naming the axis and the parallelism it implements; the same tag appears on every line of the program, the step bar colors each collective by axis, and the tally under it counts them (hover a chip to isolate that axis's steps, click to jump to its next collective). Inside each box the weights of the active station sit on top and the device's activation shard below; watch the four distinct weight-shard quadrants across an <span class="meq">X</span>×<span class="meq">Y</span> block, and the one owned slab, half-band, half-width of every activation deck.

<strong>What to notice.</strong> The program now interleaves the singles' signatures in a fixed order, and the engine derives all of it from the shardings above. Every matmul opens with FSDP's just-in-time <span class="meq"><span class="mu">AllGather</span><sub>X</sub></span> of the weight, then tensor parallelism's <span class="meq"><span class="mu">AllGather</span><sub>Y</sub></span> of the activations, and closes with <span class="meq"><span class="mu">ReduceScatter</span><sub>Y,D</sub></span>; those two axes account for most of the communication in both passes. Context parallelism only speaks in attention — <span class="meq"><span class="mu">AllGather</span><sub>C</sub></span> of <span class="meq">K</span> and <span class="meq">V</span> forward, the mirrored ReduceScatters of their gradients backward. The expert axis has two faces: inside the MoE block it is the <span class="meq"><span class="mu">AllToAll</span><sub>Z</sub></span> pair, dispatching tokens into <span class="meq">X[E<sub>Z</sub>, S<sub>XC</sub>, D<sub>Y</sub>]</span> — note that <span class="meq">S</span> stays sharded over <span class="meq">X</span> and <span class="meq">C</span>, because the AllToAll re-sorts tokens along <span class="meq">Z</span> <em>only</em> — and everywhere else it behaves like plain data parallelism. The pipeline contributes exactly one point-to-point send per direction, paid for in bubbles rather than bytes.

The backward pass is where the axes visibly stack. An attention weight gradient comes out of its matmul as <span class="meq"><span class="mu">d</span>W<sub><span class="mu">qkv</span></sub>[D, H<sub>Y</sub>]{U<sub>XZC</sub>}</span>: an unreduced partial sum over <em>every axis that sharded tokens</em>. It resolves in three attributed steps — <span class="meq"><span class="mu">ReduceScatter</span><sub>X,D</sub></span> lands it on the FSDP shards, then <span class="meq"><span class="mu">AllReduce</span><sub>Z</sub></span> and <span class="meq"><span class="mu">AllReduce</span><sub>C</sub></span> sum the replicas the expert and context axes hold — until it is <span class="meq"><span class="mu">d</span>W<sub><span class="mu">qkv</span></sub>[D<sub>X</sub>, H<sub>Y</sub>]</span>, exactly the layout the weight lives in. Expert weight gradients come out as <span class="meq"><span class="mu">d</span>W<sub><span class="mu">in</span></sub>[E<sub>Z</sub>, D, F<sub>Y</sub>]{U<sub>XC</sub>}</span> and skip the <span class="meq">Z</span> step: each expert owns its weights outright, so there is nothing to sum over the expert axis. In practice those three collectives fuse into a single ReduceScatter over the whole <span class="meq">X</span>×<span class="meq">Z</span>×<span class="meq">C</span> group (and FSDP would shard over that whole group too); keeping them apart here is what lets you see which parallelism is paying for each one. Activation saving is not drawn in this figure — the singles cover the memory story — and the pipeline runs one microbatch so the stages take turns; the bubble math is in the pipeline section above.

<figure class="tpv-outer">
  <tpviz-figure strategy="5d"></tpviz-figure>
  <figcaption>
    5D parallelism over <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 2, <span class="mu">&#x27;Y&#x27;</span>: 2, <span class="mu">&#x27;C&#x27;</span>: 2, <span class="mu">&#x27;Z&#x27;</span>: 2, <span class="mu">&#x27;stage&#x27;</span>: 2})</span>: 32 devices, every collective wrapped in the color of the mesh axis — and so the parallelism — that causes it.
  </figcaption>
</figure>
