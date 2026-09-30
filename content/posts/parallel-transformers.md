---
title: "A Visual Guide to Parallel Transformers"
date: 2026-09-28
toc: false
summary: "Illustrating a transformer's forward and backward passes across different sharding strategies; and their combinations used to train frontier LLMs"
---

<link rel="stylesheet" href="/tpviz/tpviz.css">
<script src="/tpviz/data.js" defer></script>
<script src="/tpviz/tpviz.js" defer></script>

How a transformer's forward and backward passes are sharded across devices, first one strategy at a time and then in the combinations used to train recent models. The notation follows the <a href="https://jax-ml.github.io/scaling-book/">JAX scaling book</a>.

## Notation

We use the <a href="https://jax-ml.github.io/scaling-book/sharding/">sharding notation</a> from the scaling book. A mesh such as <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 4})</span> is a set of four devices along one axis named <span class="meq">X</span>. A subscript on a dimension gives the mesh axis that dimension is sharded over. For example, <span class="meq"><span class="mu">In</span>[B<sub>X</sub>, T, D]</span> is sharded along the batch over <span class="meq">X</span>, so each device holds an array of shape <span class="meq">[B/4, T, D]</span>. A dimension with no subscript is replicated. A trailing <span class="meq">{U<sub>X</sub>}</span> marks a partial sum: each device holds an array of the full shape, and the true value is the sum of these arrays over <span class="meq">X</span>. AllReduce and ReduceScatter compute this sum.

The tooltips assume <span class="meq">B=8</span>, <span class="meq">T=128</span>, <span class="meq">D=1024</span>, <span class="meq">F=4096</span>, <span class="meq">H=1024</span> (all attention heads concatenated), <span class="meq">E=4</span> experts with <span class="meq">S=256</span> tokens routed to each, and bf16.

## Reading the figures

The model runs from left to right: <span class="meq"><span class="mu">In</span></span>, then the attention and MLP blocks of each layer, then <span class="meq"><span class="mu">Out</span></span>. Each device is a horizontal lane, so tensors sent between devices move vertically. Weights are <span class="sw wt"></span> blue, activations <span class="sw act"></span> amber, keys and values <span class="sw kv"></span> teal, and gradients <span class="sw grad"></span> rose. A dashed outline is the full array and the solid part is the shard held by that device. Hovering over a tensor shows its shape, sharding and memory.

Each segment of the step bar is one operation. Click a segment to see the state after that operation, or press play. For a collective, the panel below the canvas shows how it runs on a ring of devices. The program on the right lists every operation. The figures start with the forward pass; <em>+ Backward</em> adds the backward pass, including the activations saved for it, and links each backward operation to the forward operation it comes from.

## The axes of transformer sharding

Each strategy below shards one thing: the batch, the optimizer state, the weights, the feature dimensions, the sequence, the experts, or the layers. Each runs on four devices.

### Data parallelism

The batch is sharded, <span class="meq"><span class="mu">In</span>[B<sub>X</sub>, T, D]</span>, and the weights are replicated. The forward pass needs no communication. In the backward pass, each device computes the weight gradients from its part of the batch, so each gradient <span class="meq"><span class="mu">d</span>W[D, F]{U<sub>X</sub>}</span> is a partial sum. It is AllReduced over <span class="meq">X</span> before the optimizer step.

<figure class="tpv-outer">
  <tpviz-figure strategy="dp"></tpviz-figure>
  <figcaption>
    Data parallelism over <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 4})</span>. The forward pass has no communication; the backward pass has one AllReduce per weight gradient.
  </figcaption>
</figure>

### ZeRO-1

Data parallelism also replicates the optimizer state, which for Adam is two more arrays the size of the weights. ZeRO-1 shards the optimizer state over <span class="meq">X</span> and keeps the weights replicated. The forward pass is unchanged. In the backward pass, each gradient <span class="meq"><span class="mu">d</span>W[D, F]{U<sub>X</sub>}</span> is ReduceScattered instead of AllReduced, giving <span class="meq"><span class="mu">d</span>W[D<sub>X</sub>, F]</span>. Each device updates its shard of the weights, <span class="meq">W[D<sub>X</sub>, F]</span>, and an AllGather gives every device the full updated weights. An AllReduce is a ReduceScatter followed by an AllGather, so ZeRO-1 has the same communication cost as data parallelism.

<figure class="tpv-outer">
  <tpviz-figure strategy="zero1"></tpviz-figure>
  <figcaption>
    ZeRO-1 over <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 4})</span>. Each weight gradient is ReduceScattered, each device updates its shard, and the updated shards are AllGathered.
  </figcaption>
</figure>

### Fully-sharded data parallelism (ZeRO-3)

FSDP also shards the weights over <span class="meq">X</span>: <span class="meq">W<sub><span class="mu">in</span></sub>[D<sub>X</sub>, F]</span> and <span class="meq">W<sub><span class="mu">out</span></sub>[F, D<sub>X</sub>]</span>. Each weight is AllGathered just before its matmul and freed just after. The backward pass AllGathers each weight again, and each weight gradient is ReduceScattered so that it is sharded like its weight.

<figure class="tpv-outer">
  <tpviz-figure strategy="fsdp"></tpviz-figure>
  <figcaption>
    FSDP over <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 4})</span>. Weights are AllGathered in both passes, and weight gradients are ReduceScattered in the backward pass.
  </figcaption>
</figure>

### Tensor parallelism

Tensor parallelism shards the attention heads, <span class="meq">W<sub><span class="mu">qkv</span></sub>[D, H<sub>Y</sub>]</span> and <span class="meq">W<sub><span class="mu">o</span></sub>[H<sub>Y</sub>, D]</span>, and the MLP hidden dimension, <span class="meq">W<sub><span class="mu">in</span></sub>[D, F<sub>Y</sub>]</span> and <span class="meq">W<sub><span class="mu">out</span></sub>[F<sub>Y</sub>, D]</span>. Activations are sharded along <span class="meq">D</span>. Each block starts with an AllGather and ends with a ReduceScatter. The first matmul contracts over <span class="meq">D</span>, so the activations are AllGathered first. The second matmul contracts over the sharded <span class="meq">H</span> or <span class="meq">F</span>, so each device holds a partial sum <span class="meq">{U<sub>Y</sub>}</span>. The ReduceScatter sums it and shards the result along <span class="meq">D</span> for the next block. The backward pass does the same in reverse.

<figure class="tpv-outer">
  <tpviz-figure strategy="tp"></tpviz-figure>
  <figcaption>
    Tensor parallelism over <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;Y&#x27;</span>: 4})</span>. Each block has one AllGather and one ReduceScatter in each pass.
  </figcaption>
</figure>

### Context parallelism

Context parallelism shards the sequence: <span class="meq"><span class="mu">In</span>[B, T<sub>X</sub>, D]</span>. The MLP acts on each token separately and needs no communication. In attention, each query needs every key and value, so <span class="meq">K</span> and <span class="meq">V</span> are AllGathered over <span class="meq">X</span> while <span class="meq">Q</span> stays sharded. The backward pass AllGathers <span class="meq">K</span> and <span class="meq">V</span> again. Each device computes gradients for all keys and values, so <span class="meq"><span class="mu">d</span>K</span> and <span class="meq"><span class="mu">d</span>V</span> are partial sums and are ReduceScattered back over <span class="meq">T</span>. The weight gradients are AllReduced over <span class="meq">X</span>, as in data parallelism.

<figure class="tpv-outer">
  <tpviz-figure strategy="cp"></tpviz-figure>
  <figcaption>
    Context parallelism over <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 4})</span>. K and V are AllGathered in attention; dK and dV are ReduceScattered in the backward pass.
  </figcaption>
</figure>

### Expert parallelism

In a mixture-of-experts layer, the MLP is replaced by <span class="meq">E</span> experts, one per device, <span class="meq">W<sub><span class="mu">in</span></sub>[E<sub>Z</sub>, D, F]</span>, and a router assigns each token to one expert. Tokens are drawn as squares colored by their expert. An AllToAll sends each token to the device that holds its expert, each expert runs its MLP, and a second AllToAll sends the tokens back. In the backward pass, the token gradients go through the same two AllToAlls. The expert weight gradients need no communication, since each device holds its own expert. The attention weight gradients are AllReduced over <span class="meq">Z</span>, as in data parallelism.

<figure class="tpv-outer">
  <tpviz-figure strategy="ep"></tpviz-figure>
  <figcaption>
    Expert parallelism over <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;Z&#x27;</span>: 4})</span>. Each MoE layer has two AllToAlls in each pass.
  </figcaption>
</figure>

### Pipeline parallelism

Pipeline parallelism shards the layers: stage 0 holds layer 1 and stage 1 holds layer 2. The activations are sent from one stage to the next with a point-to-point send. With a single batch, one stage is always idle, so the batch is split into four microbatches that go through the stages in turn. The grid below the lanes shows which microbatch each stage runs at each time step. The empty cells are idle time, called the bubble. The backward pass takes about twice the compute of the forward pass, so its cells are twice as wide.

<figure class="tpv-outer">
  <tpviz-figure strategy="pp"></tpviz-figure>
  <figcaption>
    Pipeline parallelism with 2 stages and 4 microbatches.
  </figcaption>
</figure>

## Examples from frontier models

Large training runs combine several of these strategies. The technical reports for <a href="https://arxiv.org/abs/2412.19437">DeepSeek-V3</a>, <a href="https://arxiv.org/abs/2507.20534">Kimi K2</a>, <a href="https://arxiv.org/abs/2507.01006">GLM-4.5V</a>, <a href="https://arxiv.org/abs/2406.11704">Nemotron-4</a>, <a href="https://arxiv.org/abs/2504.03624">Nemotron-H</a>, <a href="https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf">Nemotron 3</a>, <a href="https://arxiv.org/abs/2511.21631">Qwen3-VL</a> and <a href="https://arxiv.org/abs/2407.21783">Llama 3</a> use three combinations. The figures use two devices per axis; the sizes used in the reports are given in the text.

In these figures, each pipeline stage is a grid of devices. The row and column labels give each device's coordinate along each mesh axis. Each collective runs over one axis and is drawn in that axis's color, and the label under the grid names the axis and the strategy. The row under the step bar counts the collectives on each axis.

### Dense models: Llama 3

FSDP × TP × CP × PP. Llama 3 405B used TP 8, PP 16 and FSDP 64 for 8K-token sequences, and TP 8, CP 16, PP 16 and FSDP 8 for 128K-token sequences. TP is placed within a server, where bandwidth is highest, and data parallelism across servers. Nemotron-4 340B used TP 8, PP 12 and data parallelism; Nemotron-H used TP 8 and 768-way data parallelism. The figure uses <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 2, <span class="mu">&#x27;Y&#x27;</span>: 2, <span class="mu">&#x27;C&#x27;</span>: 2, <span class="mu">&#x27;stage&#x27;</span>: 2})</span> with <span class="meq"><span class="mu">In</span>[B<sub>X</sub>, T<sub>C</sub>, D<sub>Y</sub>]</span>. Each matmul has an FSDP AllGather of the weight and a TP AllGather of the activations. CP adds the K and V AllGathers in attention, and PP adds one send in each direction.

<figure class="tpv-outer">
  <tpviz-figure strategy="dense4d"></tpviz-figure>
  <figcaption>
    FSDP × TP × CP × PP over <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 2, <span class="mu">&#x27;Y&#x27;</span>: 2, <span class="mu">&#x27;C&#x27;</span>: 2, <span class="mu">&#x27;stage&#x27;</span>: 2})</span>, 16 devices.
  </figcaption>
</figure>

### Open MoE models: DeepSeek and Kimi

EP × CP × PP × ZeRO-1, without TP. DeepSeek-V3 used PP 16, EP 64 across 8 nodes and ZeRO-1, with no tensor parallelism. Kimi K2 used PP 16, EP 16 and ZeRO-1. Neither used context parallelism; DeepSeek-V3 extends its context to 128K with YaRN. Their successors add it: <a href="https://arxiv.org/abs/2606.19348">DeepSeek-V4</a> and <a href="https://arxiv.org/abs/2607.24653">Kimi K3</a> both use context parallelism for long-context training, and GLM-4.5V adds CP 4 for its long-context stage. The figure uses <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 2, <span class="mu">&#x27;C&#x27;</span>: 2, <span class="mu">&#x27;Z&#x27;</span>: 2, <span class="mu">&#x27;stage&#x27;</span>: 2})</span> with <span class="meq"><span class="mu">In</span>[B<sub>XZ</sub>, T<sub>C</sub>, D]</span>. The batch is sharded over both <span class="meq">X</span> and <span class="meq">Z</span>, since outside the MoE layers the expert axis is another data axis. The forward pass has no weight AllGathers. It has the K and V AllGathers over <span class="meq">C</span>, two AllToAlls over <span class="meq">Z</span> per MoE layer, and the send between stages. In the backward pass, each attention weight gradient is ReduceScattered over <span class="meq">X</span> and AllReduced over <span class="meq">Z</span> and <span class="meq">C</span>. Each expert weight gradient is ReduceScattered over <span class="meq">X</span> and AllReduced over <span class="meq">C</span> only, since each expert is held at a single <span class="meq">Z</span> coordinate.

<figure class="tpv-outer">
  <tpviz-figure strategy="moe4d"></tpviz-figure>
  <figcaption>
    EP × CP × PP × ZeRO-1 over <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 2, <span class="mu">&#x27;C&#x27;</span>: 2, <span class="mu">&#x27;Z&#x27;</span>: 2, <span class="mu">&#x27;stage&#x27;</span>: 2})</span>, 16 devices.
  </figcaption>
</figure>

### All five: Nemotron 3 and Qwen3-VL

FSDP × TP × CP × EP × PP. The long-context stage of Nemotron 3 Nano used CP 8, TP 8, EP 8 and PP 4, and Nemotron 3 Ultra used EP 128. Qwen3-VL uses TP, PP, CP, EP and ZeRO-1 on up to 10,000 GPUs. The figure uses 32 devices, <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 2, <span class="mu">&#x27;Y&#x27;</span>: 2, <span class="mu">&#x27;C&#x27;</span>: 2, <span class="mu">&#x27;Z&#x27;</span>: 2, <span class="mu">&#x27;stage&#x27;</span>: 2})</span>, with <span class="meq"><span class="mu">In</span>[B<sub>XZ</sub>, T<sub>C</sub>, D<sub>Y</sub>]</span>. Each stage is a 4 × 4 grid with rows <span class="meq">X</span> and <span class="meq">C</span> and columns <span class="meq">Z</span> and <span class="meq">Y</span>. Most collectives are the FSDP and TP AllGathers and ReduceScatters. CP communicates only in attention, EP adds two AllToAlls per MoE layer, and PP adds one send in each direction. In the backward pass, each attention weight gradient is summed over <span class="meq">X</span>, <span class="meq">Z</span> and <span class="meq">C</span>, the three axes that shard the tokens, which takes three collectives.

<figure class="tpv-outer">
  <tpviz-figure strategy="5d"></tpviz-figure>
  <figcaption>
    FSDP × TP × CP × EP × PP over <span class="meq"><span class="mu">Mesh</span>({<span class="mu">&#x27;X&#x27;</span>: 2, <span class="mu">&#x27;Y&#x27;</span>: 2, <span class="mu">&#x27;C&#x27;</span>: 2, <span class="mu">&#x27;Z&#x27;</span>: 2, <span class="mu">&#x27;stage&#x27;</span>: 2})</span>, 32 devices.
  </figcaption>
</figure>
