# DGX Spark Quantization — Qwen3.6-35B-A3B (Claude-4.7-Opus distill)

**Base model**: [`lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled`](https://huggingface.co/lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled)
35 B parameters / ~3 B active (top-8 of 256 experts), Qwen3.5-MoE multimodal architecture, reasoning-distilled from Claude 4.7 Opus.

**Hardware**: NVIDIA DGX Spark (single node, GB10 / SM121a, Ubuntu 24.04 aarch64, 128 GiB unified memory).

**Goal**: produce vLLM-loadable INT8/INT4 quantizations and measure the quality delta against the bf16 base on reasoning + knowledge benchmarks, so we can pick which artifact to ship.

This doc covers two quantized builds plus the bf16 baseline, all evaluated on the same battery (GSM8K + full MMLU + ARC-Challenge) on 2026-04-27.

---

## TL;DR

| | **bf16 (base)** | **FP8 W8A8 dynamic** | **AWQ-INT4 GEMM** |
|---|---|---|---|
| Bits / weight | 16 | 8 | 4 |
| Activation precision | 16 | 8 dynamic per-token | 16 |
| Calibration | n/a | none | none (data-free RTN) |
| Vision tower preserved | yes | **no** (stripped) | yes (kept fp16) |
| MTP head preserved | yes | **no** (stripped) | yes (kept fp16) |
| Disk size | ~67 GiB | ~35 GiB | **~24 GiB** |
| On-disk format | safetensors / bf16 | compressed-tensors `float-quantized` | AutoAWQ GEMM |
| Quantizer | n/a | custom in-place per-layer (script) | custom shard-streaming RTN (script) |
| Smoke validated | n/a | yes (`<think>` survives) | yes (live vLLM endpoint) |
| Eval status | **full eval done** (2026-04-27) | **full eval done** (2026-04-27) | **full eval done** (2026-04-27) |
| GSM8K (full 1,319, 5-shot CoT) | **0.9447 strict / 0.9447 flexible** | **0.9447 strict / 0.9439 flexible** | **0.9386 strict / 0.9416 flexible** |
| MMLU (full 14,042, 5-shot raw MC) | **0.8341** | **0.8332** | **0.8068** |
| ARC-Challenge (full 1,172, raw MC) | **0.5478 acc / 0.5648 acc_norm** | **0.5529 acc / 0.5708 acc_norm** | **0.5648 acc / 0.5606 acc_norm** |
| Δ MMLU vs bf16 | — | **−0.09 pp** (within 0.3 σ) | **−2.73 pp** (~9 σ) |
| Δ GSM8K strict vs bf16 | — | **0.00 pp** (identical) | **−0.61 pp** (~1 σ) |
| Δ ARC-C acc_norm vs bf16 | — | +0.60 pp (within stderr) | −0.42 pp (within stderr) |

**Headline**: FP8 is statistically indistinguishable from bf16 on this battery. AWQ-INT4 loses ~2.7 pp on MMLU but holds GSM8K within ~1 σ. ARC-Challenge is a wash for all three (deltas inside ±1.45 pp stderr).

---

## Architecture context (so the recipe choices make sense)

Qwen3.5-MoE has ~25 distinct module classes. They split cleanly into "quantize-friendly" and "leave alone":

| component | per-layer count | total | precision-sensitive? | quantize? |
|---|---|---|---|---|
| Routed experts (gate/up/down per expert) | 256 × 3 = 768 | huge — the bulk of params | no (large redundancy) | **yes** |
| Shared expert (always-active dense MLP) | 1 set of 3 | small | yes | usually no |
| Self-attention (Q/K/V/O), full-attn layers only | 4 projs × 12 layers | small-medium | yes | depends |
| Linear-attention (Gated DeltaNet, Mamba-style) | 5 projs × 28 layers | small-medium | **very** | no |
| Router gate (`mlp.gate`) | 1 per layer | tiny | extremely (one wrong nibble = bad routing) | no |
| Vision tower (27-block ViT) | self-contained | medium | yes | depends |
| MTP head (speculative decoding) | 1 layer | tiny | yes (affects accept rate) | no |
| `lm_head`, embeddings, norms | n/a | small | yes | no |

The two builds make different calls about which "yes/depends" rows to actually quantize — see the side-by-side recipe table below.

---

## Build A — FP8 W8A8 Dynamic (compressed-tensors)

Artifact: [`Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-FP8-Dynamic/`](../../artifacts/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-FP8-Dynamic/)
Per-artifact model card: [`Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-FP8-Dynamic/README.md`](../../artifacts/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-FP8-Dynamic/README.md)
Quantizer: [`recipes/fp8_dynamic.py`](./recipes/fp8_dynamic.py)
Generic scheme reference: [`docs/schemes/fp8-dynamic.md`](../../docs/schemes/fp8-dynamic.md)

### Why FP8 dynamic was the first try

- **One-shot, no calibration data.** ~30–60 min total runtime; the scheme is "compute weight scales from min/max and write them out", with activations quantized at runtime per token.
- **Native vLLM support.** compressed-tensors `float-quantized` is the canonical FP8 path; no third-party kernels.
- **Conservative quality envelope.** The Claude-4.7-Opus reasoning distill is an attention-LoRA merge over base Qwen3.6 — the fine-tune signal lives in attention projections, which FP8 handles cleanly. Expert FFNs are still base weights so quantizing them is low-risk.
- **Halves the disk footprint** (~67 → ~35 GiB) without exotic kernels.

### Implementation note (the painful bit)

The straightforward path — `llmcompressor.oneshot(scheme="FP8_DYNAMIC")` — does not fit on Spark. `llmcompressor`'s `CalibrationQwen3_5MoeSparseMoeBlock` unfuses each layer's 256 experts into individual `nn.Linear` modules so they can be targeted with `targets="Linear"`. While it does this, the parent block still holds a reference to the original fused 3-D tensors, so transient memory roughly doubles the MoE weight footprint. On Spark's 121 GiB unified-memory ceiling we hit OOM around layer 34 of 40.

Workaround: a custom in-place quantizer that

1. Loads the bf16 model once at `device_map="cpu"`.
2. Strips `visual.*` and `mtp.*` immediately (we're going text-only here).
3. Walks layers one at a time. For each layer:
   - Replaces every targeted `nn.Linear`'s `weight` with `float8_e4m3fn` in place and registers a per-output-channel `weight_scale` parameter (fp32, shape `[out, 1]`).
   - Quantizes the fused `experts.gate_up_proj` (`[E, 2I, H]`) and `experts.down_proj` (`[E, H, I]`) **per-expert per-output-channel** without unfusing — same fp8 shape as input, plus paired `*_scale` of shape `[E, O, 1]`.
4. `gc.collect()` + `empty_cache()` after every layer; aborts if RSS exceeds 105 GiB so we get a Python traceback rather than a kernel SIGKILL.
5. On disk, keeps experts **fused** in 3-D under `language_model.experts.gate_up_proj` / `down_proj` (plus `_weight_scale`) — that's the layout vLLM's `Qwen3_5MoeForConditionalGeneration` `hf_to_vllm_mapper` recognizes via its `is_fused_expert` switch.

Peak RSS during quantization: ~92 GiB anon + page cache, hit around layer 39/40.

### Settings (canonical compressed-tensors `FP8_DYNAMIC`)

| key | value |
|---|---|
| `quant_method` | `compressed-tensors` |
| `format` | `float-quantized` |
| weight `num_bits` / `type` / `strategy` | 8 / `float` / `channel` |
| weight `symmetric` / `dynamic` / `observer` | true / false / `minmax` |
| activation `num_bits` / `type` / `strategy` | 8 / `float` / `token` |
| activation `symmetric` / `dynamic` / `observer` | true / true / `null` |
| storage dtype | `torch.float8_e4m3fn` (1 byte/elem) + fp32 scales |
| MoE expert layout | **fused** 3-D on disk |
| 191-name `ignore` list | `lm_head`, all `*.mlp.gate` (router), `*.shared_expert_gate`, all `*.linear_attn.*` projections, `model.embed_tokens`, `model.norm` |

### What was stripped vs what was quantized

- **Stripped before save**: `visual.*` (vision tower), `mtp.*` (multi-token-prediction head). The text-only stack is what shipped.
- **Quantized to FP8**: every `nn.Linear` in `model.layers.*` not on the 191-name ignore list — this includes attention Q/K/V/O on full-attention layers, shared-expert projections, and routed experts (per-expert per-channel).
- **Kept at bf16**: router gates (`*.mlp.gate`), shared-expert sigmoid gates, linear-attention inner projections, `lm_head`, embeddings, final norm.

### Smoke result

Live serving works. The reasoning `<think>...</think>` blocks survive — i.e. the distill signal is intact.

```bash
vllm serve Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-FP8-Dynamic \
  --quantization compressed-tensors \
  --kv-cache-dtype fp8_e4m3 \
  --enforce-eager \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --reasoning-parser qwen3
```

> **Serving gotcha**: the FP8 build still declares `Qwen3_5MoeForConditionalGeneration` as its architecture (the same multimodal class as the bf16 source) so vLLM's startup runs the multimodal init path even though we shipped no vision tower. That path tries to load an image processor — supply `processor_config.json` (copy from the AWQ sibling or the base repo) into the FP8 directory or vLLM will fail at startup with `OSError: Can't load image processor`. The vision path is never exercised at runtime since we only send text.

### Full eval (2026-04-27, total wall-clock 1 h 59 min)

Driver: `run_eval_full.sh qwen3.6-fp8 <fp8-dir> fp8_full`.
Endpoint: vLLM `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:20260415` on `:8000`, `num_concurrent=4`, `max_length=4096`.
Result files under [`./results/fp8_full/`](./results/fp8_full/).

| benchmark | n | metric | score |
|---|---|---|---|
| GSM8K (5-shot CoT, chat-templated) | 1,319 | exact_match strict | **0.9447 ± 0.0063** |
| GSM8K (5-shot CoT, chat-templated) | 1,319 | exact_match flexible | **0.9439 ± 0.0063** |
| MMLU overall (5-shot, raw MC) | 14,042 | acc | **0.8332 ± 0.0030** |
| MMLU social sciences | — | acc | 0.9061 |
| MMLU other | — | acc | 0.8671 |
| MMLU STEM | — | acc | 0.8148 |
| MMLU humanities | — | acc | 0.7756 |
| ARC-Challenge (raw MC) | 1,172 | acc | **0.5529 ± 0.0145** |
| ARC-Challenge (raw MC) | 1,172 | acc_norm | **0.5708 ± 0.0145** |

MMLU best subtasks: `high_school_government_and_politics` 0.984, `high_school_microeconomics` 0.962, `high_school_biology` 0.961, `conceptual_physics` 0.945, `marketing` 0.944. Worst: `global_facts` 0.580, `virology` 0.584, `high_school_mathematics` 0.589, `moral_scenarios` 0.632.

**Wall-clock note**: FP8 finished in 1 h 59 min vs AWQ's 1 h 33 min on identical hardware/concurrency. GSM8K phase 47 min (vs AWQ ~30 min), MMLU 67 min (vs AWQ 50 min), ARC-C 4 min. FP8 W8A8 has higher per-step memory-bandwidth pressure than AWQ W4A16 in vLLM's kernels — quality wins, throughput loses.

---

## BF16 baseline — full eval (2026-04-27, total wall-clock 2 h 45 min)

Driver: `run_eval_full.sh qwen3.6-bf16 <bf16-snapshot-dir> bf16_full`.
Endpoint: vLLM `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:20260415` on `:8000`, `num_concurrent=4`, `max_length=4096`, `--gpu-memory-utilization=0.7`, `--kv-cache-dtype fp8_e4m3` (matched to the FP8/AWQ runs for parity).
Result files under [`./results/bf16_full/`](./results/bf16_full/).

| benchmark | n | metric | score |
|---|---|---|---|
| GSM8K (5-shot CoT, chat-templated) | 1,319 | exact_match strict | **0.9447 ± 0.0063** |
| GSM8K (5-shot CoT, chat-templated) | 1,319 | exact_match flexible | **0.9447 ± 0.0063** |
| MMLU overall (5-shot, raw MC) | 14,042 | acc | **0.8341 ± 0.0030** |
| MMLU social sciences | — | acc | 0.9074 |
| MMLU other | — | acc | 0.8642 |
| MMLU STEM | — | acc | 0.8208 |
| MMLU humanities | — | acc | 0.7751 |
| ARC-Challenge (raw MC) | 1,172 | acc | **0.5478 ± 0.0145** |
| ARC-Challenge (raw MC) | 1,172 | acc_norm | **0.5648 ± 0.0145** |

**Wall-clock**: GSM8K 66 min, MMLU 94 min, ARC-C 6 min — total 2 h 45 min. Slower than FP8 (1 h 59 min) and AWQ (1 h 33 min) on the same hardware, as expected: bf16 weights are heavier per token through the kernel and the KV cache is bandwidth-shared with the larger weight footprint.

**Memory note**: bf16 needed `--max-model-len 4096` with `--gpu-memory-utilization=0.7` to fit on Spark's 121 GiB unified memory (67 GiB resident weights + KV cache + CUDA overhead). Peak RSS during eval: ~100 GiB.

---

## Three-way head-to-head: bf16 vs FP8 vs AWQ

Same harness, same prompts, same `temperature=0`, same KV-cache dtype. Deltas are signed — positive means the quantized build *beat* bf16 on that metric (within stderr that's noise; outside stderr it's a real signal).

| benchmark | metric | **bf16** | **FP8** | Δ FP8 | **AWQ** | Δ AWQ |
|---|---|---|---|---|---|---|
| GSM8K | exact_match strict | **0.9447** | 0.9447 | **0.00 pp** | 0.9386 | −0.61 pp (≈1 σ) |
| GSM8K | exact_match flexible | **0.9447** | 0.9439 | −0.08 pp (within σ) | 0.9416 | −0.31 pp (within σ) |
| MMLU | overall acc | **0.8341** | 0.8332 | **−0.09 pp** (within σ) | 0.8068 | **−2.73 pp** (~9 σ) |
| MMLU humanities | acc | 0.7751 | 0.7756 | +0.05 pp | 0.7458 | −2.93 pp |
| MMLU social sciences | acc | 0.9074 | 0.9061 | −0.13 pp | 0.8902 | −1.72 pp |
| MMLU STEM | acc | 0.8208 | 0.8148 | −0.60 pp | 0.8059 | −1.49 pp |
| MMLU other | acc | 0.8642 | 0.8671 | +0.29 pp | 0.8175 | −4.67 pp |
| ARC-C | acc | 0.5478 | 0.5529 | +0.51 pp | 0.5648 | +1.70 pp (within σ) |
| ARC-C | acc_norm | 0.5648 | 0.5708 | +0.60 pp | 0.5606 | −0.42 pp (within σ) |

**What the deltas say**

- **FP8 ≈ bf16.** Every FP8 metric sits inside ±1 σ of the bf16 baseline. GSM8K strict-match is bit-for-bit (0.9447 / 0.9447). MMLU overall is −0.09 pp on a 0.30 pp stderr — about as quiet as a quantizer can be. The "wins" on MMLU other (+0.29 pp) and ARC-C (+0.51 / +0.60 pp) are noise around the baseline. **The dynamic FP8 recipe is effectively lossless on this battery.**
- **AWQ-INT4 takes a measurable but bounded MMLU hit.** Overall −2.73 pp at ~9 σ on the tight overall-MMLU stderr — clearly real. The "other" cluster (general knowledge, miscellany) takes the biggest single hit (−4.67 pp), consistent with INT4 routed-experts losing some of the long-tail-knowledge precision FP8 retains. STEM and social sciences hold tighter (~−1.5 pp).
- **GSM8K barely moves under either quantization.** AWQ −0.61 pp strict / −0.31 pp flexible, both within ~1 σ. Reasoning chains stay coherent at INT4 weights / fp16 activations — the chain-of-thought signal is robust.
- **ARC-Challenge is a tie three ways.** All deltas inside ±1.45 pp stderr. The raw-`acc` ranking (AWQ > FP8 > bf16) flips on `acc_norm` (FP8 ≈ bf16 > AWQ). 1,172 samples isn't enough to separate the three on this task; treat as noise.

**Surprise that isn't a surprise**: bf16 and FP8 have *identical* GSM8K strict-match numbers (0.9447). Two reads of this:

1. The FP8 round-off is small enough that for the deterministic greedy-decode path on this benchmark the two models output the same answers on the same questions. (We didn't check exact-token-match — only exact-answer-match.)
2. The KV cache is `fp8_e4m3` for both runs, which means both models share the same KV-quantization noise floor — the only thing that varies is the weight precision, and FP8 weights apparently don't perturb the answer string at GSM8K's evaluation granularity.

---

## Build B — AWQ-INT4 GEMM (data-free RTN)

Artifact: [`Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-AWQ-INT4/`](../../artifacts/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-AWQ-INT4/)
Per-artifact model card: [`Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-AWQ-INT4/README.md`](../../artifacts/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-AWQ-INT4/README.md)
Quantizer: [`recipes/awq_gemm.py`](./recipes/awq_gemm.py)
Generic scheme reference: [`docs/schemes/awq-gemm.md`](../../docs/schemes/awq-gemm.md)

### Why an AWQ-INT4 build at all

- **Smaller** (~24 GiB vs ~35 GiB FP8 vs ~67 GiB bf16).
- **Multimodal preserved.** Unlike the FP8 build, the vision tower and MTP head stay at fp16 — you can serve image inputs.
- **Layout matches an existing public build** (`QuantTrio/Qwen3.6-35B-A3B-AWQ`), so vLLM's `moe_wna16` kernel path picks it up unchanged.

### Implementation note

A shard-streaming pure-Python RTN packer that emits AutoAWQ GEMM-format weights bit-for-bit identical to QuantTrio's reference. No model load — it reads one bf16 source shard at a time, quantizes, packs, and writes. Stays well under 24 GiB host RAM the entire time.

For each in-scope `[out, in]` weight (and per-expert sub-tensor of the fused 3-D MoE tensors), per-group asymmetric quantization along `in_features` with `group_size = 128`:

```
scale  = (max(W_g) - min(W_g)) / 15        # round-trip via fp16
zp     = round(-min(W_g) / scale)          # int, clamped to [0, 15]
q      = clip(round(W / scale) + zp, 0, 15)
```

Then 8 nibbles along `out_features` are packed into one `int32` using AWQ's canonical lane permutation `[0, 4, 1, 5, 2, 6, 3, 7]`, producing the qweight / qzeros / scales tensor layout vLLM's `convert_awq_tensor` (`moe_wna16.py`) consumes.

The fused 3-D Qwen3.5-MoE expert tensors are **unfused** on the way out (the AWQ-MoE convention):

- `experts.gate_up_proj` (`[256, 2·512, 2048]`) → 256 × `gate_proj` + 256 × `up_proj` (`[512, 2048]` each).
- `experts.down_proj` (`[256, 2048, 512]`) → 256 × `down_proj` per layer.

### Settings

| key | value |
|---|---|
| `quant_method` | `awq` |
| `version` | `gemm` |
| `bits` | 4 |
| `group_size` | 128 |
| `zero_point` | true (asymmetric per-group min/max) |
| pack order | `[0, 4, 1, 5, 2, 6, 3, 7]` (AWQ canonical) |
| calibration | **none** (data-free RTN) |
| activation precision | fp16 (W4A16) |
| storage dtype | `int32` (packed nibbles) + `float16` scales |
| MoE expert layout | **unfused** → per-expert `gate_proj` / `up_proj` / `down_proj` |
| `modules_to_not_convert` | `["visual", "linear_attn", "self_attn", "shared_expert", "mlp.gate", "model.layers.0.", "mtp"]` |

### What was kept at fp16/bf16 vs quantized to INT4

- **Kept at fp16**: vision tower, all linear-attention layers, all self-attention layers, shared-expert MLP, router gates, **layer 0 entirely** (early layers are most sensitive in MoE models), MTP head, `lm_head`.
- **Quantized to INT4**: routed expert MLPs (`gate_proj`, `up_proj`, `down_proj`) for all 256 experts × ~39 routed-MoE layers. That's the bulk of the parameter count.

### Partial eval (n=100/200/30-per-subtask)

| benchmark | metric | score |
|---|---|---|
| GSM8K (5-shot CoT, chat-templated) | exact_match (strict & flexible) | **0.940 ± 0.024** (n=100) |
| MMLU overall (5-shot, raw multiple-choice) | acc | **0.815 ± 0.009** (n=1,710 = 30/subtask × 57) |
| ARC-Challenge (raw multiple-choice) | acc | 0.570 ± 0.035 (n=200) |

MMLU domain breakdown: social sciences 0.886 / humanities 0.803 / other 0.797 / stem 0.791. Best subtasks (saturated): `college_biology` 1.00, `high_school_computer_science` 1.00, `international_law` 0.97. Worst: `global_facts` 0.37, `high_school_mathematics` 0.50.

### Full eval (2026-04-27, total wall-clock 1 h 33 min)

Driver: `run_eval_full.sh qwen3.6-awq-smoke <tokenizer> awq_full`.
Endpoint: vLLM `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:20260415` on `:8000`, `num_concurrent=4`, `max_length=4096`.
Result files under [`./results/awq_full/`](./results/awq_full/).

| benchmark | n | metric | score |
|---|---|---|---|
| GSM8K (5-shot CoT, chat-templated) | 1,319 | exact_match strict | **0.9386 ± 0.0066** |
| GSM8K (5-shot CoT, chat-templated) | 1,319 | exact_match flexible | **0.9416 ± 0.0065** |
| MMLU overall (5-shot, raw MC) | 14,042 | acc | **0.8068 ± 0.0032** |
| MMLU social sciences | — | acc | 0.8902 |
| MMLU other | — | acc | 0.8175 |
| MMLU STEM | — | acc | 0.8059 |
| MMLU humanities | — | acc | 0.7458 |
| ARC-Challenge (raw MC) | 1,172 | acc | **0.5648 ± 0.0145** |
| ARC-Challenge (raw MC) | 1,172 | acc_norm | **0.5606 ± 0.0145** |

MMLU best subtasks: `high_school_government_and_politics` 0.974, `high_school_microeconomics` 0.958, `conceptual_physics` 0.945. Worst: `virology` 0.548, `global_facts` 0.550, `moral_scenarios` 0.590.

**Partial → full delta**: every metric stayed inside the partial-eval CIs. GSM8K strict 0.940 → 0.9386 (4× tighter CI), MMLU 0.815 → 0.8068 (3× tighter CI), ARC-C acc 0.570 → 0.5648. ARC-C `acc_norm` shifted from 0.520 → 0.5606 with the larger n — the partial run's `acc_norm` was the noisiest of the four numbers.

---

## Side-by-side recipe table

| dimension | FP8 W8A8 dynamic | AWQ-INT4 GEMM |
|---|---|---|
| Format on disk | compressed-tensors `float-quantized` | AutoAWQ GEMM |
| Bits / weight | 8 | 4 |
| Activation precision | 8 (dynamic per-token) | 16 |
| Symmetry | symmetric (zero point = 0) | asymmetric (zero point stored) |
| Quant axis | per-output-channel along `in_features` | per-group of 128 along `in_features`, packed 8-nibbles along `out_features` |
| Calibration data | none (one-shot) | none (data-free RTN) |
| Scale dtype | fp32 | fp16 |
| MoE expert layout on disk | **fused** 3-D | **unfused** per-expert |
| Vision tower | **stripped** | preserved at fp16 |
| MTP head | **stripped** | preserved at fp16 |
| Layer 0 quantized? | yes | **no** (kept fully fp16) |
| Router gate (`mlp.gate`) | no | no |
| Linear-attention | no | no |
| Self-attention (Q/K/V/O) | yes | no |
| Shared expert | yes | no |
| `lm_head` | no | no |
| Disk size | ~35 GiB | ~24 GiB |
| Compression vs bf16 | ~1.9× | ~2.8× |
| Runtime kernel (vLLM) | compressed-tensors FP8 | `moe_wna16` / AWQ |
| Smoke result | `<think>` block survives | live endpoint, GSM8K 0.94 |

The takeaway: FP8 plays it safe (more modules quantized but at higher precision per module) while AWQ trades aggressive bit-width on the routed experts for keeping every other component dense, including the entire vision stack and layer 0.

---

## Disk-space accounting (what we shipped vs threw away)

Currently on disk under the project root:

| item | size | status |
|---|---|---|
| `hf-cache/.../Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled` | ~67 GiB | bf16 source, kept |
| `Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-FP8-Dynamic/` | ~35 GiB | shipped artifact |
| `Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-AWQ-INT4/` | ~24 GiB | shipped artifact |
| `results/` | ~5 MB | summary JSONs + run logs for all three full evals |

Removed earlier (failed attempts, ~38 GiB total):

- `qwen36-35b-distill-autoround-awq-int4-smoke/` — abandoned AutoRound path
- `qwen36-35b-distill-autoround-awq-int4-smoke-vllm/` — abandoned AutoRound + vLLM rekey path
- `qwen36-bf16-shadow-textonly/` — old text-only symlink scratch dir

---

## The full-eval plan (what we're doing next)

We're running the same eval battery against all three artifacts so we get an apples-to-apples quality comparison.

### Battery

| benchmark | metric | n | reasoning |
|---|---|---|---|
| **GSM8K** (5-shot, chat-templated, CoT) | `exact_match` strict + flexible | full 1,319 | Math reasoning. Chat path needed because `--apply_chat_template` produces message lists that `/v1/completions` rejects; we hit `/v1/chat/completions` via lm-eval's `local-chat-completions`. |
| **MMLU** (5-shot, raw multiple-choice loglikelihood) | `acc`, per-subdomain | full ~14,042 across 57 subtasks | Knowledge breadth. Raw loglikelihood scoring against `local-completions`. |
| **ARC-Challenge** (raw multiple-choice loglikelihood) | `acc`, `acc_norm` | full 1,172 | Science-flavored MC. Raw scoring underrates chat-tuned reasoning models — treat as a floor. |

Each model uses identical `lm-evaluation-harness 0.4.11` configs, same chat template, same `max_length=4096`, same `num_concurrent=4`, same `temperature=0`/`top_p=1`. The only thing that changes between runs is the served vLLM checkpoint and the local tokenizer path.

Estimated wall-clock per artifact: ~30 min GSM8K + ~2–3 hr MMLU + ~30 min ARC-C ≈ 3–4 hr. Three artifacts → roughly a working day, sequential, dominated by MMLU.

### Sequence

We can't run them in parallel — the Spark serves one vLLM at a time.

1. **AWQ full eval** — vLLM is currently up on the AWQ artifact. Run first while the container is hot.
2. **FP8 full eval** — stop AWQ, start FP8 vLLM (compressed-tensors), re-run identical battery.
3. **BF16 baseline eval** — stop FP8, start vLLM serving the unquantized bf16 source from the HF cache, re-run identical battery. This is the reference; deltas are computed against this row.

### Output layout

```
results/
├── awq_full/
│   ├── gsm8k/qwen3.6-awq-smoke/...
│   ├── mmlu/qwen3.6-awq-smoke/...
│   ├── arc_challenge/qwen3.6-awq-smoke/...
│   └── run.log
├── fp8_full/
│   └── ... (same shape, served-name `qwen3.6-fp8`)
└── bf16_full/
    └── ... (same shape, served-name `qwen3.6-bf16`)
```

### Why this gives us what we want

- **Absolute scores** for each artifact on standard benchmarks → comparable to public model leaderboards.
- **Delta vs bf16** quantifies actual quantization damage (the missing piece in the partial-eval table above).
- **Same items, same prompts, same seed** for all three runs → the only variable is the quantization.

### Risks / things to watch

- **Run time creep.** MMLU at full size can stretch beyond the 3 hr estimate at `num_concurrent=4`. If we see < 1 req/s steady-state we'll bump concurrency.
- **BF16 memory pressure.** Spark has 128 GiB unified RAM; the bf16 model is ~67 GiB on disk and roughly the same resident. With KV cache and CUDA overhead this is tight — we'll pick `--gpu-memory-utilization` carefully and may need `--max-model-len 4096` to avoid OOM.
- **Tokenizer drift.** All three tokenizers should be identical (each artifact carries the same `tokenizer.json` from the source), but we'll point lm-eval at the local artifact dir per-run rather than a shared cache to be safe.
- **Sampling determinism.** `temperature=0` is deterministic for greedy decoding, but FP8 vs INT4 vs BF16 have different round-off behavior at the kernel level — small score differences within ±1 stderr on chunky CoT tasks are expected.

### Where the deltas land — preliminary expectations

These are working hypotheses, not predictions:

- FP8 dynamic on attention-LoRA-merged distills: < 1 pp loss vs bf16 on reasoning. Knowledge tasks (MMLU) typically tighter (< 0.5 pp).
- AWQ-INT4 on routed experts only (with attention/layer-0/vision dense): typically 1–3 pp loss on MMLU, < 1 pp on math reasoning if the chain-of-thought isn't truncated.
- Both quantizations should preserve domain ranking on MMLU (social sciences > humanities > other > stem).

If we see a much larger gap than that on any single subtask, that's a signal something specific (e.g. a low-resource expert getting clamped) needs a closer look.

---

## Decision matrix (with measured numbers)

All three artifacts are now eval-validated. The call between FP8 and AWQ-INT4:

| if you care about... | pick | why |
|---|---|---|
| Maximum quality preservation | **FP8** | −0.09 pp MMLU vs bf16, 0.00 pp GSM8K strict — effectively lossless |
| Smallest disk + multimodal | **AWQ** | ~24 GiB vs ~35 GiB FP8, vision tower preserved |
| Knowledge-heavy tasks (MMLU "other", broad recall) | **FP8** | AWQ is −4.67 pp on this cluster vs bf16; FP8 is +0.29 pp |
| Math reasoning (GSM8K-class) | **either** | both within 1 σ of bf16; pick on disk/multimodal |
| Faster decode at small batch | **AWQ** | W4A16 has lower memory-bandwidth pressure |
| Faster compute-bound throughput at large batch | **FP8** | FP8 GEMMs hit native FP8 tensor cores |
| Vision-language capability | **AWQ** | FP8 build is text-only (vision tower stripped) |

Default recommendation: ship **FP8** as the headline artifact (lossless) and keep **AWQ** as the multimodal / disk-constrained alternative. Both are functional; the choice is determined by deployment constraints, not by quality.

---

## Out of scope (for now)

- Static / calibrated FP8 (with a small calibration corpus). Open as a follow-up if dynamic FP8 underperforms on a specific subtask.
- Calibrated AWQ (real activation-aware salience pass over a calibration set). The data-free RTN AWQ already lands at GSM8K 0.94 / MMLU 0.81 on the partial subset — calibration may or may not be worth re-running for.
- HumanEval (sandboxed Docker only — we don't run model-generated Python on the host).
- Long-context probe (RULER / NIAH at ≥ 32 K).
- Multi-Spark sharded serving.
- NVFP4 / MXFP4. The relevant vLLM kernels aren't stable on SM121a yet.

---

## Roadmap checklist

- [x] FP8 W8A8 dynamic build (text-only)
- [x] AWQ-INT4 GEMM build (multimodal-preserving)
- [x] AWQ partial eval (subset)
- [x] AWQ full-set eval (GSM8K 0.939 / MMLU 0.807 / ARC-C 0.565)
- [x] Per-artifact HuggingFace model cards (this doc + per-artifact READMEs)
- [x] FP8 full-set eval (GSM8K 0.945 / MMLU 0.833 / ARC-C 0.571 acc_norm)
- [x] FP8 vs AWQ head-to-head delta table (in this doc)
- [x] BF16 baseline full-set eval (GSM8K 0.945 / MMLU 0.834 / ARC-C 0.565 acc_norm)
- [x] Update this report with the three-way delta table (bf16 vs FP8 vs AWQ)
- [ ] Pick the artifact to publish on HF (or publish both, with this report as the README on the parent collection)

---

## File index

- `recipes/fp8_dynamic.py` — FP8 quantizer (with `--selftest`)
- `recipes/awq_gemm.py` — AWQ-INT4 quantizer
- `recipes/inspect_modules.py` — module-tree diagnostic for the source model
- `../../tools/run_eval_full.sh` — full-eval driver (generic, accepts absolute output dir)
- `../../tools/serve_vllm_docker.sh` — DGX Spark vLLM container launcher (generic)
- `../../tools/run_under_memcap.sh` — generic systemd-run cgroup wrapper
- `../../artifacts/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-FP8-Dynamic/` — FP8 artifact + its README (gitignored — too big for GitHub)
- `../../artifacts/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-AWQ-INT4/` — AWQ artifact + its README (gitignored — too big for GitHub)
- `results/` — `results_*.json` summaries + `run.log` for each of awq/fp8/bf16 full evals
