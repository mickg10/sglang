# GLM‑5.3‑Flash NVFP4 on 4×/8× RTX PRO 6000 Blackwell (SM120)

## Deployment recommendation, competing hypotheses, and falsifiable validation plan

**Status:** RFC / decision report  
**Prepared for:** `mickg10`  
**Date:** 2026-08-27 (America/New_York)  
**Author role:** BigOracle systems analysis  
**Operational verdict:** **APPROVE a four-GPU bring-up; HOLD production until the gates in this report pass.**

---

## 0. Executive decision

### Recommended first deployment

Run the LibertAI experts-only NVFP4 checkpoint as a **single four-GPU TP4 / EP1 SGLang instance**, confined to one coherent four-GPU PCIe/NUMA island:

- Model: `LibertAIDAI/GLM-5.3-Flash-NVFP4`
- Pinned model revision: `aa28e1f54130286c95fee10d0705c74ce8743734` or a later revision that is re-audited
- Tensor parallelism: `TP=4`
- Expert parallelism: `EP=1`
- Prefill context parallelism: off
- Decode context parallelism: off
- DSA prefill/decode: TileLang
- KV cache: BF16
- Routed-expert NVFP4 MoE: FlashInfer CUTLASS
- Shared-expert fusion: off
- DeepGEMM: fully disabled
- Speculative decoding: off for first boot
- HiCache: off for first boot
- Text-only mode for the first correctness pass
- SM120 TileLang shared-memory schedule patched or otherwise proven to stay below the device limit

### Recommended eight-GPU production shape

On a normal eight-GPU PCIe server—especially a dual-socket 4+4 topology—the expected production winner is:

> **Two independent TP4 / EP1 replicas, one per four-GPU island, with sticky prefix affinity.**

Use one TP8 process only if measurements show that the actual topology and workload justify the extra collective participants. Weight capacity does not require TP8.

### Verdict on the proposed command

The proposed command is **not an RTX-specific NVFP4 recipe**. It is essentially the SGLang **GB300 low-latency official-FP8 recipe**, reduced to TP4, with prefill context parallelism and HiCache added:

```bash
sglang serve \
  --model-path zai-org/GLM-5.3-Flash \
  --tp-size 4 \
  --attn-cp-size 4 \
  --enable-prefill-cp \
  --cp-strategy interleave \
  --dsa-prefill-backend trtllm \
  --dsa-decode-backend trtllm \
  --kv-cache-dtype fp8_e4m3 \
  --moe-runner-backend deep_gemm \
  --ep-size 4 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --speculative-adaptive \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --host 0.0.0.0 \
  --port 30000 \
  --enable-hierarchical-cache \
  --hicache-size 32
```

Only the following parts are approved unchanged for the intended four-GPU RTX deployment:

- `--tp-size 4`
- `--reasoning-parser glm45`
- `--tool-call-parser glm47`
- host and port

Everything else is either for a different checkpoint, validated on a different Blackwell architecture, or an optimization that must be isolated behind an A/B gate.

---

## 1. Evidence vocabulary

This report deliberately separates evidence from inference.

- **OBSERVED** — directly present in source, checkpoint metadata, code, or a first-hand field report.
- **DERIVED** — arithmetic or a direct logical consequence of observed facts.
- **INFERRED** — a systems prediction supported by evidence but not yet measured on this exact machine.
- **HYPOTHESIS** — a plausible optimization claim that requires an experiment.
- **UNKNOWN** — missing evidence that materially affects the decision.

A reviewer can reject an inference without disputing the observations beneath it.

---

## 2. Exact scope and source identity

### Hardware scope

- NVIDIA RTX PRO 6000 Blackwell
- Compute capability SM120
- 96 GiB-class VRAM per card
- Four or eight GPUs
- PCIe-only GPU interconnect; no NVLink
- Exact CPU, root-complex, switch, ACS, and NUMA topology still unknown

### Model scope

Primary production candidate:

- [`LibertAIDAI/GLM-5.3-Flash-NVFP4`](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)
- Current report pin: [`aa28e1f54130286c95fee10d0705c74ce8743734`](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4/commit/aa28e1f54130286c95fee10d0705c74ce8743734)
- Important config repair: [`cf5434c00bf69bd0e6b58420c9636999472a2291`](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4/commit/cf5434c00bf69bd0e6b58420c9636999472a2291)

Reference/canary checkpoint:

- [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash)
- This is the official FP8-oriented serving payload used by the SGLang datacenter recipes.
- It is **not** interchangeable with the LibertAI experts-only NVFP4 checkpoint.

### Runtime scope

- SGLang GLM-5.3 support PR: [`sgl-project/sglang#36507`](https://github.com/sgl-project/sglang/pull/36507)
- State at report time: open
- Head at report time: [`c4d5d45e506dcd978a65661a503eda1a272c39a4`](https://github.com/sgl-project/sglang/commit/c4d5d45e506dcd978a65661a503eda1a272c39a4)
- Dedicated mutable image tag used in field reports: `lmsysorg/sglang:glm-5.3-flash`
- **UNKNOWN:** the image digest and source SHA on the target host

Supporting field knowledge:

- [`local-inference-lab/rtx6kpro`](https://github.com/local-inference-lab/rtx6kpro)
- Source snapshot consulted: `7f15d620217af442235aca44b52f77ee287246e0`

Emerging SM120 sparse-MLA implementation consulted:

- [`flashinfer-ai/flashinfer`](https://github.com/flashinfer-ai/flashinfer)
- Source snapshot consulted: `286eee4e2999a825716eab68e597cb1ee0881e1b`

---

## 3. Architecture facts that determine the serving design

### 3.1 Model structure

**OBSERVED** from the current checkpoint configuration and model card:

- 320B total parameters
- 18B active parameters per token
- 45 language layers
- 34 KDA linear-attention layers
- 11 DeepSeek-style sparse-attention/MLA layers
- Hidden width 4096
- 288 routed experts
- 8 routed experts selected per token
- 1 shared expert
- First 3 MLP layers dense; the remaining 42 are sparse MoE layers
- Sparse index `topk=2048`
- `kv_lora_rank=512`
- `qk_nope_head_dim=256`
- `qk_rope_head_dim=0`
- NoPE MLA
- `hc_mult=4` manifold-constrained Hyper-Connections
- One native NEXTN/MTP draft layer
- 24-layer vision tower
- Nominal maximum context: 1,048,576 tokens

### 3.2 Quantization partition

**OBSERVED:**

The LibertAI checkpoint quantizes essentially only the routed-expert FFN matrices:

- Routed experts: weight-only NVFP4, group size 16
- Attention, including KDA and DSA: BF16
- Sparse indexer: BF16
- Shared expert: BF16
- Routers: BF16/FP32 as configured
- Dense MLP layers: BF16
- MTP layer: BF16
- mHC tensors: BF16
- Vision tower: BF16
- Embeddings, `lm_head`, and norms: BF16

Checkpoint footprint:

- Routed experts: about 175 GB
- Everything else: about 19 GB
- Total: about 181 GiB

### 3.3 Immediate consequences

**DERIVED:**

1. **Four GPUs are sufficient for weights.**  
   A perfectly even TP4 split is roughly `181 / 4 = 45.25 GiB` per card before replication and runtime overhead.

2. **TP8 is not required for model residency.**  
   The case for TP8 must be performance or cache capacity, not basic fit.

3. **Shared-expert fusion is unsafe for this checkpoint unless the runtime explicitly supports mixed BF16/NVFP4 fusion.**  
   The current checkpoint publisher requires it disabled.

4. **DeepGEMM is not the NVFP4 expert backend for this deployment.**  
   The checkpoint’s verified SGLang path uses `flashinfer_cutlass`.

5. **The scheduler has two different memory constraints.**
   - DSA layers have context-growing cache.
   - KDA layers have per-request recurrent state.
   - Concurrency can exhaust KDA state slots before ordinary KV storage is full.

6. **NoPE matters.**  
   Kernels written around a 512+64 or other RoPE-bearing MLA layout cannot be assumed to support a zero-width RoPE tail.

---

## 4. Memory model

### 4.1 Weight residency on TP4

Nominal even split:

```text
181 GiB / 4 = 45.25 GiB per GPU
```

A deliberately pessimistic upper-bound thought experiment puts all 19 GB of BF16 non-expert material on every rank while sharding the 175 GB expert payload:

```text
175 / 4 + 19 = 62.75 GB per GPU
```

The real number depends on which BF16 modules are sharded or replicated, allocator layouts, scale metadata, transformed expert layouts, graphs, and workspaces. Even the pessimistic estimate leaves roughly 30 GiB on a 96 GiB card.

**INFERRED:** TP4 has sufficient memory margin for practical DSA cache, KDA states, graphs, and workspaces. The startup allocation report remains authoritative.

### 4.2 DSA latent-KV lower bound

There are 11 context-growing DSA layers and a 512-element BF16 latent per token:

```text
11 layers × 512 values × 2 bytes = 11,264 bytes/token/GPU
```

Lower-bound latent storage:

| Context | Bare latent-KV lower bound |
|---:|---:|
| 65,536 | 0.69 GiB |
| 131,072 | 1.38 GiB |
| 262,144 | 2.75 GiB |
| 524,288 | 5.50 GiB |
| 1,048,576 | 11.00 GiB |

This excludes the sparse index/key pool, page tables, metadata, fragmentation, graph workspaces, multimodal state, and any duplicated draft cache.

### 4.3 KDA recurrent-state estimate

The checkpoint publisher measured 31 KDA state slots consuming roughly 2.19 GB per rank on TP2. That is about 70.6 MB per slot per rank at TP2.

**DERIVED, assuming the head-sharded component scales inversely with TP:**

```text
TP4 ≈ 35 MB per active request/state slot per GPU
```

This is an estimate, not a sizing contract. Speculative decoding can require draft/verification state and intermediate copies. Record the server’s actual hybrid-state allocation.

### 4.4 Why HiCache is not the first move

TP4 NVFP4 leaves substantial local VRAM. Host-backed hierarchical caching adds:

- Host-RAM reservation
- GPU↔host transfers on the same PCIe fabric used by collectives
- Eviction and consistency behavior
- A larger hybrid KDA/DSA/MTP state surface
- Dependence on very recent HiCache+MTP fixes in the GLM-5.3 branch

**Decision:** leave HiCache off until a measured workload proves that local VRAM is insufficient or host-resident prefix reuse produces a net latency/throughput win.

---

## 5. What the proposed command actually changes

### 5.1 It selects the wrong checkpoint for the original objective

```bash
--model-path zai-org/GLM-5.3-Flash
```

This selects the official checkpoint, not the LibertAI experts-only NVFP4 checkpoint.

**Verdict:** **HOLD** for the NVFP4 production objective.  
**Use:** valuable as a reference/canary because a four-card RTX field report exists.

### 5.2 TP4 is correct

```bash
--tp-size 4
```

**Verdict:** **APPROVE.**

TP4 is enough for NVFP4 weight residency and keeps collectives inside a four-GPU group.

### 5.3 The added CP flags are prefill CP, not DCP

```bash
--attn-cp-size 4
--enable-prefill-cp
--cp-strategy interleave
```

**OBSERVED in current SGLang code:**

- This is prefill context parallelism.
- In GLM-5.3 KDA paths, prefill CP can all-gather hidden states before projection and reduce-scatter outputs.
- It does not substitute for SGLang decode context parallelism.
- DCP is a separate `--dcp-size N` facility with separate communication and correctness contracts.

**Verdict:** **HOLD for baseline; APPROVE as a long-prefill A/B experiment.**

Strongest case for CP4:

- Long prompts can distribute sequence work across the four cards.
- Interleaving can balance causal sequence segments.
- Compute-heavy prefill may amortize communication.

Counterargument:

- It adds collectives through a hybrid model with 34 KDA layers and 11 DSA layers.
- PCIe latency and bandwidth are much weaker than NVLink/NVSwitch.
- It does nothing by itself to accelerate ordinary decode.
- It can regress short and medium prompts where communication dominates.

### 5.4 TRTLLM DSA plus FP8 KV is a datacenter recipe, not a proven SM120 recipe

```bash
--dsa-prefill-backend trtllm
--dsa-decode-backend trtllm
--kv-cache-dtype fp8_e4m3
```

**OBSERVED:**

- The official SGLang cookbook validates this pair on GB300-class systems.
- A first-hand 4× RTX PRO 6000 report using the dedicated GLM-5.3 image found automatic TRTLLM selection failed with an unsupported-architecture error.
- That report reached correct serving with TileLang DSA and BF16 KV.
- Current FlashInfer main contains emerging SM120 sparse-MLA code.
- The current SM120 d_qk=512 decode instantiation set does not list `topk=2048`, while GLM-5.3 uses `index_topk=2048`.
- FlashInfer has had recent open dispatch issues where unsupported top-k shapes fell through to a prefill path.

**Verdict:** **HOLD for baseline. Experimental only after exact-build feature proof and numerical validation.**

This is not a claim that TRTLLM+FP8 KV can never work on SM120. It is a claim that the specific command is ahead of the currently proven end-to-end stack.

### 5.5 DeepGEMM is a hard mismatch on stock SM120

```bash
--moe-runner-backend deep_gemm
```

**OBSERVED:**

The upstream DeepGEMM README lists SM90 and SM100 as its supported NVIDIA architectures. The RTX PRO 6000 is SM120. The four-card GLM-5.3 field report explicitly disabled DeepGEMM and required the mHC pre-normalization fallback.

**Verdict:** **REJECT on SM120.**

For the LibertAI NVFP4 checkpoint:

```bash
--moe-runner-backend flashinfer_cutlass
```

For the official FP8 canary, start with a portable SM120-capable FP8 MoE backend—such as Triton—or leave auto-selection enabled only after confirming that DeepGEMM is disabled and logging the resolved backend.

### 5.6 EP4 is legal but unproven on PCIe

```bash
--ep-size 4
```

EP4 does not multiply the GPU count; it uses the same four ranks. It changes the routed-expert execution from TP-style participation to expert sharding plus token dispatch/combine.

**Strongest case for EP4:**

- Experts dominate the model.
- Expert sharding can reduce local expert storage and GEMM participation.
- On NVSwitch systems, dispatch/combine can be overlapped with expert compute.

**Counterargument on RTX PCIe:**

- Every sparse MoE layer introduces token dispatch/combine communication.
- The model has 42 sparse MoE layers.
- A low-concurrency request has few tokens, making all-to-all startup latency hard to amortize.
- Only eight experts are active per token, creating possible rank imbalance.
- RTX6kPro community measurements generally found EP slower or equivalent on PCIe unless substantial custom work was applied.

**Verdict:** **HOLD for baseline; benchmark as a separate EP1↔EP4 arm.**

### 5.7 EAGLE 5/1/6 adaptive is valid, but not a first-boot setting

```bash
--speculative-algorithm EAGLE
--speculative-num-steps 5
--speculative-eagle-topk 1
--speculative-num-draft-tokens 6
--speculative-adaptive
```

Current SGLang treats native NEXTN through the EAGLE speculative framework. The settings are valid for the official datacenter low-latency recipe.

**OBSERVED on four RTX PRO 6000 cards:** a field report served the official FP8 model with a smaller 3-step / 4-draft / top-k-1 profile and observed about 2.9 accepted tokens per four drafted.

**Verdict:** **HOLD for first boot.**

Promotion order:

1. No speculation
2. EAGLE 3/4/1
3. EAGLE adaptive 5/6/1
4. Keep the winner per workload, not globally

### 5.8 Parser flags are correct

```bash
--reasoning-parser glm45
--tool-call-parser glm47
```

**Verdict:** **APPROVE.**

The checkpoint publisher specifically warns that the older `glm` tool parser can fail silently by consuming output without producing a tool call.

### 5.9 HiCache is a workload-specific experiment

```bash
--enable-hierarchical-cache
--hicache-size 32
```

**Verdict:** **HOLD for baseline.**

Enable only after:

- local-cache behavior is correct;
- MTP behavior is correct;
- host memory and transfer bandwidth are measured;
- the target workload has enough large-prefix reuse or context spill to justify it.

---

## 6. The core disagreement: GB300 recipe versus RTX recipe

### Position A: “The official GB300 recipe should be copied because both are Blackwell”

Best argument:

- Both GPUs expose Blackwell tensor cores.
- The SGLang recipe is tuned for GLM-5.3 itself.
- FP8 KV, TRTLLM DSA, DeepGEMM, EP4, and adaptive MTP are individually plausible optimizations.
- Starting from the vendor recipe avoids leaving performance on the table.

Why this is not sufficient:

- SM100/GB300 and SM120/RTX are materially different kernel targets.
- DeepGEMM upstream does not list SM120 support.
- The first-hand RTX report found TRTLLM unsupported in the tested image.
- RTX cards communicate over PCIe, not NVLink/NVSwitch.
- TileLang’s default shared-memory schedule also exceeded the RTX per-block limit.
- The official command targets the official FP8 checkpoint, not mixed BF16/NVFP4 experts-only weights.

**Current ruling:** the GB300 recipe is a source of hypotheses, not a baseline.

### Position B: “Start from the known-good SM120 path even if it is slower”

Best argument:

- It isolates checkpoint loading, NoPE DSA, KDA states, mHC, MTP, and mixed-precision MoE.
- It matches the only published end-to-end GLM-5.3 RTX field report.
- It gives a trustworthy reference before adding communication-heavy features.

Cost:

- BF16 KV uses more memory.
- TileLang may be slower than a mature TRTLLM path.
- EP1 may leave potential expert parallel speedups unused.
- Initial no-MTP service sacrifices latency.

**Current ruling:** this is the correct bring-up strategy because each later optimization can be tested against a healthy reference.

---

## 7. Source-level/runtime hazards

### 7.1 Stale checkpoint config causes the old NVFP4 load failure

An earlier version of the LibertAI config named unfused BF16 attention modules in the ModelOpt ignore list. SGLang constructs fused runtime modules, so it treated BF16 attention as packed FP4 and allocated the wrong shape.

The checkpoint was repaired in commit:

- `cf5434c00bf69bd0e6b58420c9636999472a2291`

Required fused names now include:

```text
*.self_attn.qkv_proj
*.self_attn.fused_qkvbfg_a_proj
*.self_attn.fused_fg_b_proj
*.self_attn.qkv_conv1d
*.self_attn.fused_qkv_a_proj_with_mqa
*.self_attn.q_conv1d
*.self_attn.k_conv1d
*.self_attn.v_conv1d
*.self_attn.o_norm
visual.*
*.visual.*
```

**Gate:** reject any local snapshot whose `config.json` lacks these entries.

Do not begin by patching the generic SGLang wildcard matcher. Refresh and pin the checkpoint first.

### 7.2 DeepGEMM mHC pre-normalization fallback

The RTX field report found that disabling DeepGEMM alone still left an mHC pre-normalization path trying to call a DeepGEMM symbol.

Set:

```bash
SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
```

Also disable DeepGEMM and its JIT path explicitly in the chosen source snapshot.

### 7.3 TileLang shared-memory schedule on SM120

The four-card RTX report found:

- Default two-stage schedule requested about 151,552 bytes
- The card permitted about 99 KiB per block
- Setting `num_stages=1` reduced the requirement to roughly 86 KiB and worked end-to-end

Conceptual patch:

```python
props = torch.cuda.get_device_properties(q.device)

if getattr(props, "shared_memory_per_block_optin", 1 << 20) < 120 * 1024:
    kernel = kernel_factory(
        num_heads,
        d_v,
        tail_dim,
        topk,
        sm_scale=sm_scale,
        num_stages=1,
    )
else:
    kernel = kernel_factory(
        num_heads,
        d_v,
        tail_dim,
        topk,
        sm_scale=sm_scale,
    )
```

Do not blindly copy the GB10/SM121 tile. The checkpoint publisher reports a different GB10 retune involving `block_I=32`, `num_stages=1`, and 128 threads. The RTX field report specifically found that its attempted `block_I=32` variant failed. Select by measured device properties and validate numerics.

### 7.4 Moving upstream branch

At report time, SGLang PR #36507 is open and has changed rapidly. Recent commits have addressed:

- HiCache + MTP pool assembly
- Shared-expert fusion gating
- Hybrid linear-KV pool initialization
- DCP TileLang LSE output
- NEXTN→EAGLE alias resolution
- DFlash hidden-state capture

A floating image tag cannot be treated as a reproducible dependency.

**Gate:** record all of the following:

```text
Docker image digest
SGLang git SHA
FlashInfer version and git SHA
PyTorch version
CUDA toolkit/runtime
cuDNN version
NVIDIA driver
checkpoint revision
config.json SHA-256
GPU firmware/driver and topology output
```

---

## 8. Recommended operating profiles

## Profile R0 — NVFP4 correctness baseline

Purpose: prove loading, text generation, NoPE DSA, KDA state handling, mixed-precision expert placement, parsers, and cache integrity.

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3

export SAFETENSORS_FAST_GPU=1
export OMP_NUM_THREADS=8

# Branch/image-version dependent; verify each is recognized in logs.
export SGLANG_ENABLE_DEEP_GEMM=0
export SGLANG_DISABLE_DEEP_GEMM=1
export SGLANG_ENABLE_JIT_DEEPGEMM=0
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0

sglang serve \
  --model-path /models/GLM-5.3-Flash-NVFP4-aa28e1f \
  --tp-size 4 \
  --ep-size 1 \
  --attention-backend dsa \
  --dsa-prefill-backend tilelang \
  --dsa-decode-backend tilelang \
  --kv-cache-dtype bfloat16 \
  --moe-runner-backend flashinfer_cutlass \
  --disable-shared-experts-fusion \
  --disable-custom-all-reduce \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --mem-fraction-static 0.84 \
  --context-length 65536 \
  --max-running-requests 2 \
  --chunked-prefill-size 8192 \
  --host 0.0.0.0 \
  --port 30000 \
  --language-only
```

Notes:

- If this source snapshot lacks `--language-only`, omit only that flag.
- Do not pass `--quantization modelopt_fp4` initially; verify auto-detection.
- If graph capture obscures a failure, add `--disable-cuda-graph` temporarily.
- Do not add CP, DCP, EP4, MTP, HiCache, or custom PCIe all-reduce yet.

## Profile R1 — normal TP4 service

After R0 passes:

```text
context-length:       131072
max-running-requests: 4
mem-fraction-static:  begin at 0.84; raise only from measured headroom
CUDA graphs:          enabled
vision:               enabled after a separate multimodal gate
```

Increase one memory dimension at a time. KDA slots, not nominal KV tokens, may be the limiting resource.

## Profile R2 — low-latency MTP 3/4/1

```bash
export SGLANG_ENABLE_SPEC_V2=True
```

Add:

```bash
--speculative-algorithm EAGLE \
--speculative-num-steps 3 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4
```

Promotion gate:

- no quality/parser regression;
- no cache corruption;
- accepted draft length measured;
- target-workload p50 and p95 improve;
- no scheduler instability.

## Profile R3 — adaptive MTP 5/6/1

Replace the R2 speculative settings with:

```bash
--speculative-algorithm EAGLE \
--speculative-num-steps 5 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 6 \
--speculative-adaptive
```

Do not assume R3 beats R2. Larger verification batches can increase TP communication and graph/workspace pressure on PCIe.

## Profile R4 — long-prefill CP4 experiment

Start from R1, keep EP1, MTP off, and add:

```bash
--attn-cp-size 4 \
--enable-prefill-cp \
--cp-strategy interleave
```

Test first at 32K, 128K, and 256K unique prompts. Measure both TTFT and PCIe traffic. Reject CP4 if it improves only synthetic maximum-length prefill while materially regressing normal prompts.

## Profile R5 — HiCache experiment

Start from a fully green R1 or R2 profile and add:

```bash
--enable-hierarchical-cache \
--hicache-size 32
```

Require:

- documented host-RAM budget;
- measured L1/L2 hit rates;
- cache-correctness tests with abort/eviction;
- no MTP state mismatch;
- no PCIe contention that degrades decode.

## Profile R6 — official FP8 reference/canary

Use the official model but retain the SM120-safe attention policy:

```bash
sglang serve \
  --model-path zai-org/GLM-5.3-Flash \
  --tp-size 4 \
  --ep-size 1 \
  --attention-backend dsa \
  --dsa-prefill-backend tilelang \
  --dsa-decode-backend tilelang \
  --kv-cache-dtype bfloat16 \
  --moe-runner-backend triton \
  --disable-shared-experts-fusion \
  --disable-custom-all-reduce \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --mem-fraction-static 0.82 \
  --context-length 65536 \
  --max-running-requests 2 \
  --host 0.0.0.0 \
  --port 30010 \
  --language-only
```

This is a conservative reference profile, not a claim that Triton is the fastest FP8 MoE backend. Its purpose is to produce a healthy higher-precision comparator on the same hardware.

After it is green, reproduce the field-tested EAGLE 3/4/1 profile.

## Profile X1 — experimental TRTLLM/FP8-KV

This profile is **not authorized by default**.

Prerequisites:

1. Exact FlashInfer commit includes SM120 NoPE sparse MLA.
2. The d_qk/top-k shape used by GLM-5.3 dispatches to a dedicated decode kernel rather than a fallback.
3. The SGLang integration emits an explicit SM120 backend marker.
4. A micro-kernel test matches a gathered FP32/BF16 reference.
5. End-to-end output matches the TileLang/BF16 reference within an agreed tolerance.
6. Cache capacity and throughput improvements are measured.

Run this against the official FP8 checkpoint first. Do not combine it with EP4, CP4, HiCache, or MTP in its first experiment.

---

## 9. Eight-GPU deployment

### 9.1 Topology discovery

Before assigning ranks:

```bash
nvidia-smi -L
nvidia-smi topo -m
nvidia-smi topo -p2p r
numactl -H
lspci -tv

nvidia-smi \
  --query-gpu=index,name,pci.bus_id,memory.total,pstate,power.limit,clocks.sm,clocks.mem \
  --format=csv
```

Choose four-card groups that minimize `SYS` links.

Typical dual-socket grouping:

```text
Replica A: GPUs 0,1,2,3
Replica B: GPUs 4,5,6,7
```

Do not assume numerical adjacency; derive it from the topology matrix.

### 9.2 Two TP4 replicas

Replica A:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PORT=30000 \
./launch-glm53-nvfp4-tp4.sh
```

Replica B:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
PORT=30001 \
./launch-glm53-nvfp4-tp4.sh
```

Route with:

- request affinity;
- common-prefix affinity;
- health-aware failover;
- separate warmup and cache telemetry per replica.

### 9.3 When TP8 remains worth testing

Test TP8 only if at least one is true:

- all eight GPUs share a high-quality switch fabric;
- there is no cross-socket `SYS` path;
- the target workload is dominated by very large prefill;
- one process needs more long-context concurrency than TP4 can supply;
- measured TP8 throughput or p95 latency wins.

TP8 is a benchmark arm, not a capacity requirement.

---

## 10. Communication model

### TP4 / EP1

- All four ranks participate in tensor-parallel layers.
- Small collectives occur frequently.
- No token-routing all-to-all is introduced by expert parallelism.
- This is predictable and generally friendly to a coherent four-GPU PCIe island.

### TP4 / EP4

- Expert weights are sharded by expert ownership.
- Tokens are dispatched to expert-owning ranks and combined afterward.
- There are 42 sparse MoE layers, so dispatch/combine is frequent.
- Low-batch decode may underutilize ranks and expose all-to-all latency.

### Prefill CP4

- Sequence work is distributed during prefill.
- SGLang GLM-5.3 paths perform CP all-gather/reduce-scatter operations.
- It can help long compute-heavy prompts.
- It is not a free way to “use all four GPUs more.”

### DCP4

- Separate feature from prefill CP.
- Shards context/KV work during decode.
- Requires partial-output/LSE merge behavior.
- Current GLM-5.3 DCP validation is centered on GB300.
- PCIe SM120 requires its own correctness and performance campaign.

### PCIe one-shot all-reduce

The RTX6kPro field wiki reports meaningful TP4 decode gains for its custom one-shot all-reduce on coherent four-GPU PCIe topologies, while eight-GPU cross-socket cases can regress—especially with MTP and longer context.

Do not enable it in R0. Later:

1. Verify direct GPU P2P behavior.
2. Run the standalone crossover benchmark.
3. Compare NCCL and one-shot at decode-relevant 8 KiB–512 KiB sizes.
4. Test with and without MTP.
5. Reject on any correctness or topology-dependent instability.

---

## 11. Required preflight audit

### 11.1 Pin the checkpoint

```bash
MODEL_ID=LibertAIDAI/GLM-5.3-Flash-NVFP4
MODEL_REV=aa28e1f54130286c95fee10d0705c74ce8743734
MODEL_DIR=/models/GLM-5.3-Flash-NVFP4-aa28e1f

huggingface-cli download "$MODEL_ID" \
  --revision "$MODEL_REV" \
  --local-dir "$MODEL_DIR"
```

Audit the config:

```bash
python3 - <<'PY'
import json
from pathlib import Path

p = Path("/models/GLM-5.3-Flash-NVFP4-aa28e1f/config.json")
cfg = json.loads(p.read_text())
ignore = set(cfg["quantization_config"]["ignore"])

required = {
    "*.self_attn.qkv_proj",
    "*.self_attn.fused_qkvbfg_a_proj",
    "*.self_attn.fused_fg_b_proj",
    "*.self_attn.qkv_conv1d",
    "*.self_attn.fused_qkv_a_proj_with_mqa",
    "visual.*",
    "*.visual.*",
}
missing = sorted(required - ignore)
if missing:
    raise SystemExit(f"STALE/INVALID CONFIG; missing ignore entries: {missing}")

tc = cfg["text_config"]
assert tc["qk_rope_head_dim"] == 0
assert tc["kv_lora_rank"] == 512
assert tc["index_topk"] == 2048
assert tc["n_routed_experts"] == 288
assert tc["num_experts_per_tok"] == 8

print("config audit: OK")
PY

sha256sum "$MODEL_DIR/config.json"
```

### 11.2 Pin and inspect the image

```bash
docker pull lmsysorg/sglang:glm-5.3-flash
docker image inspect lmsysorg/sglang:glm-5.3-flash \
  --format '{{json .RepoDigests}}'
```

Inside the image, record:

```bash
python3 - <<'PY'
import torch
import sglang
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("sglang", getattr(sglang, "__version__", "unknown"))
try:
    import flashinfer
    print("flashinfer", getattr(flashinfer, "__version__", "unknown"))
except Exception as e:
    print("flashinfer import error", repr(e))
print("device", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
print("smem optin", torch.cuda.get_device_properties(0).shared_memory_per_block_optin)
PY
```

If the source tree is present:

```bash
git -C /sgl-workspace/sglang rev-parse HEAD || true
git -C /opt/sglang rev-parse HEAD || true
```

### 11.3 Quantization placement audit

Before trusting any output, enumerate runtime module methods and assert:

```text
model.layers.*.mlp.experts.*           ModelOpt NVFP4
model.layers.*.self_attn.*             unquantized/BF16
model.layers.*.mlp.shared_experts.*    unquantized/BF16
model.layers.0-2.mlp.*                 unquantized/BF16
visual.*                               unquantized/BF16
MTP/NEXTN layer                        unquantized/BF16
routers, mHC, embeddings, lm_head      native configured dtype
```

Any BF16 attention projection with packed-FP4 storage shape is a hard failure.

---

## 12. Correctness gates

No performance verdict is valid before these pass.

### Gate C0 — process health

- All four workers initialize.
- `/health` becomes ready.
- No CUDA Xid.
- No NaN/Inf.
- No unsupported-architecture fallback.
- No repeated worker restart.
- Startup log names the intended DSA and MoE backends.

### Gate C1 — deterministic text

Run greedy decoding on a fixed prompt set:

- arithmetic
- factual short answer
- code generation
- structured JSON
- long-form reasoning
- multilingual text
- stop-token behavior

Repeat each case and verify stable output under the same configuration.

### Gate C2 — parser behavior

Reasoning:

- `reasoning_content` populated correctly
- final `content` excludes reasoning trace
- no stray unmatched `</think>`

Tools:

- valid tool schema
- one tool call
- multiple tool calls
- tool result continuation
- no empty content + null tool_calls silent failure

### Gate C3 — quantization sanity

Compare NVFP4 against the official FP8 canary:

- token-level logit KL on a fixed prompt corpus, where instrumentation permits;
- greedy-token agreement rate;
- GSM8K-style arithmetic;
- coding tests;
- long-generation coherence;
- no systematic early stops or loops.

Loading successfully is not a quality verdict.

### Gate C4 — context staircase

Test:

```text
8K → 32K → 64K → 128K → 256K
```

At each length:

```text
concurrency 1 → 2 → 4
```

Then test 512K and 1M as dedicated low-concurrency profiles.

### Gate C5 — cache semantics

Exercise:

- exact repeated prefix;
- shared prefix with divergent suffixes;
- interleaved unrelated requests;
- abort during prefill;
- abort during decode;
- cache eviction under pressure;
- MTP on/off transitions;
- server restart;
- image request followed by text requests once multimodal is enabled.

### Gate C6 — multimodal

After removing `--language-only`:

- single image;
- multiple images;
- video sample;
- mixed text/image turn;
- repeated visual prefix;
- malformed and oversized media handling.

The vision tower is BF16 and should not be treated as covered by text-only success.

---

## 13. Performance experiment sequence

Change one major mechanism at a time.

### Phase P0 — establish R1 baseline

Measure:

- TTFT
- prefill tokens/s
- inter-token latency
- output tokens/s/request
- aggregate output tokens/s
- p50/p95/p99
- GPU memory
- KDA state slots
- DSA KV tokens
- PCIe RX/TX
- GPU utilization, clocks, power, temperature
- scheduler retractions

### Phase P1 — MTP

Compare:

```text
off
3 steps / 4 draft / topk 1
adaptive 5 steps / 6 draft / topk 1
```

Record:

- accepted draft tokens;
- draft time;
- target verification time;
- effective output speed;
- graph sizes;
- memory delta;
- acceptance by workload class.

### Phase P2 — prefill CP4

Compare CP off/on at:

```text
context:     8K, 32K, 128K, 256K
concurrency: 1, 4
output:      1 token and 256 tokens
```

This separates pure prefill from mixed prefill/decode.

### Phase P3 — EP4

Compare EP1 and EP4 with all other mechanisms fixed.

Required telemetry:

- per-rank expert token counts;
- dispatch/combine time;
- PCIe bytes;
- expert imbalance;
- MoE compute time;
- end-to-end TTFT/TPOT.

### Phase P4 — TRTLLM + FP8 KV

Only after the backend feature gate. Compare against TileLang/BF16 at equal:

- prompt;
- concurrency;
- memory fraction;
- graph policy;
- MTP policy.

### Phase P5 — HiCache

Use a workload with controlled prefix reuse:

```text
0%, 25%, 50%, 75%, 100% shared-prefix rate
```

Record L1/L2 hit rates, host bandwidth, eviction, TTFT, and decode interference.

### Phase P6 — collective tuning

Compare:

- NCCL baseline
- NCCL topology/channel tuning
- PCIe one-shot all-reduce
- one-shot + MTP
- one-shot + CP

Run small-message collective tests, not only large-bandwidth `all_reduce_perf`.

### Phase P7 — deployment topology

Compare:

```text
one TP8 instance
one TP4 instance
two TP4 replicas, aggregate
```

Test both low-concurrency interactive and high-concurrency throughput workloads.

---

## 14. Benchmark matrix

A bounded matrix avoids an unmanageable factorial explosion.

| Arm | Checkpoint | TP | EP | Prefill CP | DSA/KV | MTP | HiCache |
|---|---|---:|---:|---:|---|---|---|
| A0 | NVFP4 | 4 | 1 | off | TileLang/BF16 | off | off |
| A1 | NVFP4 | 4 | 1 | off | TileLang/BF16 | 3/4/1 | off |
| A2 | NVFP4 | 4 | 1 | off | TileLang/BF16 | adaptive 5/6/1 | off |
| A3 | NVFP4 | 4 | 1 | 4 | TileLang/BF16 | off | off |
| A4 | NVFP4 | 4 | 4 | off | TileLang/BF16 | off | off |
| A5 | NVFP4 | 4 | 1 | off | TileLang/BF16 | winning MTP | on |
| F0 | official FP8 | 4 | 1 | off | TileLang/BF16 | off | off |
| F1 | official FP8 | 4 | 1 | off | TileLang/BF16 | 3/4/1 | off |
| X1 | official FP8 | 4 | 1 | off | TRTLLM/FP8 | off | off |
| T8 | NVFP4 | 8 | 1 | off | TileLang/BF16 | winning MTP | off |
| 2R | NVFP4 | 2×4 | 1 | off | TileLang/BF16 | winning MTP | off |

Suggested workload cells:

| Prompt context | Output | Concurrency |
|---:|---:|---:|
| 1K | 256 | 1, 4, 16 |
| 8K | 256 | 1, 4, 16 |
| 32K | 256 | 1, 4, 8 |
| 128K | 256 | 1, 2, 4 |
| 256K | 256 | 1, 2 |
| 32K | 2K | 1, 4 |
| 128K | 2K | 1, 2 |

---

## 15. Fail-able promotion criteria

These thresholds are intentionally explicit.

### MTP adoption

Adopt only if:

- target-workload p50 output speed improves at least 5%;
- p95 does not regress more than 5%;
- no task-quality regression is detected;
- accepted draft length is stable;
- no scheduler/cache fault appears in soak.

### CP4 adoption

Adopt only for a dedicated long-prefill profile if:

- 128K+ TTFT improves at least 10%;
- 8K–32K TTFT regression is acceptable for that profile;
- decode TPOT does not materially regress;
- PCIe traffic remains below the topology’s saturation point.

### EP4 adoption

Adopt only if:

- aggregate target-workload throughput improves at least 8%;
- low-concurrency latency is not unacceptable;
- expert imbalance is bounded;
- dispatch/combine is not the dominant layer cost.

The larger threshold reflects EP’s operational and correctness complexity.

### TRTLLM/FP8 adoption

Adopt only if:

- exact SM120 NoPE/topk2048 dispatch is proven;
- numerical reference tests pass;
- no wrong-output or clean-stop echo failure appears;
- useful cache capacity increases materially;
- target performance improves at least 10%.

### HiCache adoption

Adopt only if:

- measured workload has sustained useful L2 hit rate;
- host transfers do not reduce decode throughput;
- eviction/abort/restart tests pass;
- hybrid KDA/DSA/MTP state remains coherent.

### TP8 adoption

Adopt one TP8 process over two TP4 replicas only if it wins the actual service objective:

- lower target p95 latency, or
- higher aggregate throughput, or
- required per-instance cache capacity.

---

## 16. Explicit “do not combine yet” list

Do not begin with any of these combinations:

1. LibertAI NVFP4 + `deep_gemm`
2. LibertAI NVFP4 + shared-expert fusion
3. TileLang DSA + FP8 KV
4. TRTLLM/FP8 KV without an exact SM120 NoPE/topk2048 feature proof
5. EP4 + CP4 in the same first experiment
6. MTP + HiCache in the first correctness run
7. Prefill CP as a substitute for DCP
8. TP8 across unknown cross-socket topology
9. PCIe one-shot all-reduce before P2P/crossover validation
10. Floating image tag + floating Hugging Face revision
11. An old cached `config.json` predating `cf5434c`
12. Performance claims based only on successful startup

---

## 17. Soak and failure testing

Run a sustained mixed workload while monitoring:

```bash
nvidia-smi dmon
nvidia-smi -q -d PERFORMANCE,POWER,TEMPERATURE,ECC
dmesg --follow
```

Inject:

- client disconnects;
- request cancellation;
- concurrent long and short prompts;
- malformed tool output;
- cache pressure;
- maximum-running-request pressure;
- one worker process kill in a non-production test;
- router failover between TP4 replicas.

Reject a build on:

- CUDA Xid;
- NaN/Inf;
- silent empty tool outputs;
- repeated scheduler retractions;
- unreleased KDA state slots;
- prefix corruption;
- inconsistent MTP result after retry;
- thermal/power throttling that invalidates benchmark comparisons;
- worker restart without clean recovery.

---

## 18. Questions for external reviewers

A useful technical response should address one or more of these with exact source identities or measurements.

1. Which exact SGLang image digest and FlashInfer commit first serve GLM-5.3 NoPE `topk=2048` through TRTLLM/FP8 KV on SM120?
2. Does the current SM120 sparse-MLA d_qk=512 decode path have a dedicated topk=2048 instantiation, or does it still fall through?
3. Has `--attn-cp-size 4 --enable-prefill-cp --cp-strategy interleave` been benchmarked on four PCIe RTX PRO 6000 cards for GLM-5.3 specifically?
4. In that CP4 path, what fraction of time is spent in KDA all-gather/reduce-scatter versus DSA attention?
5. Is there a first-hand EP1 versus EP4 GLM-5.3 result on PCIe-only four-card SM120?
6. Which FP8 MoE runner is currently fastest and correct for the official checkpoint on SM120 with DeepGEMM disabled?
7. Has the one-stage TileLang SM120 schedule been upstreamed, and at what SHA?
8. Is the correct RTX tile only `num_stages=1`, or are there model/batch shapes requiring a block/thread retune?
9. What is the exact per-slot KDA state allocation formula for TP4 with and without MTP?
10. Has HiCache passed abort, eviction, MTP, and multimodal tests on GLM-5.3 hybrid state?
11. Has DCP4 been validated on SM120 PCIe, not only GB300?
12. What MTP profile wins on RTX: 3/4/1, adaptive 5/6/1, or another bounded point?
13. Does two-replica prefix affinity outperform TP8 for the real request distribution?
14. Are there any quality measurements for this exact NVFP4 checkpoint beyond per-expert round-trip cosine?

---

## 19. Reviewer response template

```markdown
### Reviewer identity / environment

- Hardware:
- CPU / NUMA:
- GPU topology:
- Driver:
- CUDA:
- Docker image digest:
- SGLang SHA:
- FlashInfer SHA:
- Checkpoint and revision:
- Exact launch command:

### Verdict

APPROVE / HOLD

### Findings

1. ...
2. ...
3. ...

### Reproduction evidence

- Logs:
- Benchmark command:
- Raw results:
- Correctness comparison:
- Backend markers:

### Counterexample to this report

State the exact claim being challenged and provide the smallest reproducer or measurement that falsifies it.

### Recommended plan change

...
```

---

## 20. Final recommendation

### APPROVE

- Four-GPU bring-up
- TP4 / EP1
- LibertAI NVFP4 at a pinned repaired revision
- TileLang DSA with BF16 KV
- SM120 shared-memory-safe TileLang schedule
- FlashInfer CUTLASS NVFP4 MoE
- shared-expert fusion disabled
- DeepGEMM disabled
- parsers `glm45` and `glm47`
- no CP, DCP, MTP, HiCache, or custom all-reduce in the first run

### HOLD

- Production traffic before correctness, cache, and soak gates
- EP4
- prefill CP4 as a default
- DCP4
- HiCache
- adaptive 5/6 MTP
- TP8
- TRTLLM/FP8 KV on the currently unproven SM120 image

### REJECT

- `--moe-runner-backend deep_gemm` on stock SM120
- treating the official FP8 command as an NVFP4 recipe
- mixed-precision shared-expert fusion for this checkpoint
- floating dependencies
- interpreting “server started” as “model is correct”

### Expected production winner

> **Two independent NUMA-/switch-local TP4 / EP1 NVFP4 replicas across eight GPUs, with a separate low-latency MTP profile and a conservative non-speculative throughput/reference profile.**

This is an inference, not a measured result. The experiment sequence above is designed to either confirm it or replace it with a better configuration without losing the healthy baseline.

---

## 21. Primary sources

- [LibertAI GLM-5.3-Flash NVFP4 model card](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)
- [LibertAI current pinned revision used by this report](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4/commit/aa28e1f54130286c95fee10d0705c74ce8743734)
- [LibertAI fused-ignore config repair](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4/commit/cf5434c00bf69bd0e6b58420c9636999472a2291)
- [SGLang GLM-5.3 support PR #36507](https://github.com/sgl-project/sglang/pull/36507)
- [Four-card RTX PRO 6000 SM120 field report in PR #36507](https://github.com/sgl-project/sglang/pull/36507#issuecomment-5432203047)
- [NVFP4 load-failure diagnosis in PR #36507](https://github.com/sgl-project/sglang/pull/36507#issuecomment-5433433658)
- [Current SGLang GLM-5.3 cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/GLM/GLM-5.3-Flash.mdx)
- [DeepGEMM architecture requirements](https://github.com/deepseek-ai/DeepGEMM)
- [FlashInfer SM120 sparse MLA implementation](https://github.com/flashinfer-ai/flashinfer/blob/286eee4e2999a825716eab68e597cb1ee0881e1b/flashinfer/mla/_sparse_mla_sm120.py)
- [FlashInfer SM120 sparse-MLA dispatch issue #4541](https://github.com/flashinfer-ai/flashinfer/issues/4541)
- [RTX PRO 6000 field wiki](https://github.com/local-inference-lab/rtx6kpro)
- [RTX PRO 6000 topology notes](https://github.com/local-inference-lab/rtx6kpro/blob/master/hardware/topology.md)
- [RTX PRO 6000 PCIe one-shot all-reduce notes](https://github.com/local-inference-lab/rtx6kpro/blob/master/optimization/pcie-oneshot-allreduce.md)

---

## 22. Non-claims

This report does **not** claim:

- that the proposed optimized command can never work;
- that TP4 always beats TP8;
- that EP4 can never win on PCIe;
- that TRTLLM/FP8 KV will remain unavailable on SM120;
- that the current NVFP4 checkpoint has been quality-validated for every task;
- that the author has booted this exact stack on the target host;
- that the target host has a particular topology;
- that any throughput number from another machine transfers directly.

It claims that the recommended baseline has the strongest current evidence, that the rejected combinations contain concrete architecture/checkpoint mismatches, and that the remaining performance questions can be resolved by a bounded, fail-able experiment campaign.
