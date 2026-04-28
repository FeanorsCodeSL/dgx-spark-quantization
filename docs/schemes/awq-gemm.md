# Scheme: AWQ-INT4 GEMM (data-free RTN)

Data-free round-to-nearest INT4 quantization producing an AutoAWQ
GEMM-format checkpoint. Smallest disk footprint, lets vLLM serve via the
`moe_wna16` / AWQ kernel path.

---

## When to pick this scheme

- You want **maximum compression** (~4× vs bf16, ~2× vs FP8).
- You need to **preserve a vision tower** or other components at fp16 — AWQ
  recipes typically only quantize the bulk-parameter components (routed
  expert MLPs in MoE, dense MLPs in transformer) and leave attention,
  embeddings, layer 0, and any multimodal tower at fp16.
- You don't have a calibration corpus (the data-free RTN variant) **or** you
  do (use a real AWQ calibration run for ~0.5–1 pp of recovery on knowledge
  benchmarks).
- The deployment runtime is **vLLM**, **TGI**, or any AutoAWQ-compatible
  loader.

Skip this scheme if:

- You need < 1 pp loss on knowledge-heavy benchmarks (use FP8 instead).
- The runtime can't load AutoAWQ's pack format.
- The model has unusual layout (unfused MoE experts, custom attention) that
  AutoAWQ's loader doesn't handle.

---

## What "AWQ-INT4 GEMM" means

| axis | choice |
|---|---|
| **W** weight precision | INT4, asymmetric per-group |
| **A** activation precision | fp16 (W4A16) — no activation quantization |
| Group size | 128 along `in_features` (the AWQ canonical group size) |
| Pack order | `[0, 4, 1, 5, 2, 6, 3, 7]` — AWQ's canonical 8-nibble lane permutation |
| Scale dtype | fp16 |
| Zero point | stored (asymmetric) |
| Quant axis | per-group of 128 along `in_features`, packed 8-nibbles along `out_features` |
| Calibration | optional. Data-free RTN works; activation-aware AWQ recovers ~0.5–1 pp on knowledge tasks |

Math (per group of 128 in `in_features`, asymmetric):

```
qmax  = 15
scale = (max(W_g) - min(W_g)) / qmax     # fp16
zp    = round(-min(W_g) / scale)         # int, clamped to [0, 15]
q     = clip(round(W / scale) + zp, 0, 15)
```

Then 8 nibbles along `out_features` are packed into one `int32` using AWQ's
canonical lane permutation `[0, 4, 1, 5, 2, 6, 3, 7]`, producing the
`qweight` / `qzeros` / `scales` tensor layout that vLLM's
`convert_awq_tensor` (`moe_wna16.py`) consumes.

### Tensor shapes (for a `[out, in]` linear)

```
qweight: int32  [in,             out // 8]
qzeros : int32  [in // 128,      out // 8]
scales : fp16   [in // 128,      out]
```

---

## Required `quantization_config` block

```json
{
  "quant_method": "awq",
  "bits": 4,
  "group_size": 128,
  "version": "gemm",
  "zero_point": true,
  "modules_to_not_convert": [
    "lm_head",
    "<embed/norm>",
    "<router gates>",
    "<linear_attention or recurrent inner>",
    "<self_attention if you want to keep it dense>",
    "<vision tower>",
    "<layer 0 — early layers are most sensitive>",
    "<MTP head>"
  ]
}
```

`modules_to_not_convert` is a list of substrings; AutoAWQ matches modules
whose name *contains* any entry. Be specific to avoid accidental wildcards
(e.g. `mlp.gate.weight` vs `mlp.gate_proj.weight` — use the exact suffix).

---

## What to quantize vs leave alone

Stricter than FP8 by default. The bulk of compression comes from the routed
expert MLPs — keep almost everything else dense:

| component | quantize? | reason |
|---|---|---|
| Routed expert MLPs (gate/up/down per expert) | **yes** | bulk of param count, redundant across experts |
| Dense MLPs (non-MoE models) | **yes** | bulk of params |
| Attention Q/K/V/O | **no** by default | INT4 attention is more disruptive than INT4 MLP; can cost ~1-2 pp |
| Shared expert (always-on dense MLP in MoE) | optional | smaller params, more sensitive |
| Router gate | **no** | tiny, sensitive |
| Linear-attention / Mamba / DeltaNet | **no** | recurrent state is sensitive |
| **Layer 0** | **no** | first layer is the most sensitive in MoE models |
| Embeddings, lm_head | **no** | sensitive, small |
| Vision tower | **no** (keep at fp16) | preserves multimodal capability |
| MTP / speculative decoding head | **no** | preserves accept rate |

Compare against a public reference of similar architecture before shipping —
`QuantTrio/<model>-AWQ` and `casper-hansen/<model>-AWQ` are the canonical
sources for skip-list conventions.

---

## Implementation notes

### Memory: shard streaming beats full model load

For multi-tens-of-billions models on commodity hardware, don't load the
whole bf16 model. Instead:

1. Read the source's `model.safetensors.index.json` for the weight map.
2. For each source shard, open with `safe_open`, iterate keys, decide
   quantize-or-passthrough per key, write to the output buffer.
3. Flush output shards at ~4 GB each.

Peak host RAM stays at ~one shard's worth (a few GB) regardless of model
size. No GPU needed.

### Fused MoE experts: unfuse on disk

AWQ-MoE convention is **unfused** experts on disk. If the source has fused
3-D MoE tensors:

- `experts.gate_up_proj` (`[E, 2*intermediate, hidden]`) → split into per-expert
  `experts.<i>.gate_proj` + `experts.<i>.up_proj` (each `[intermediate, hidden]`).
- `experts.down_proj` (`[E, hidden, intermediate]`) → per-expert `experts.<i>.down_proj`.

vLLM's `moe_wna16` kernel expects this unfused layout for AWQ-MoE.

### Pack-order correctness

The pack order must match the loader's `reverse_awq_pack_order` exactly. For
canonical AWQ-GEMM that's `[0, 4, 1, 5, 2, 6, 3, 7]`. Implement a
pack/unpack roundtrip test that asserts dequant error < ~0.05 max-abs on a
random tensor before trusting any output.

### Scale rounding through fp16

Compute the scale in fp32 (`(max - min) / qmax`), but **save it as fp16**.
Then for the actual quantization math, cast the saved fp16 back to fp32 — so
the saved scales exactly reproduce dequant. If you don't round through fp16
during quant math, the saved fp16 scales drift from the math used to compute
`q`, and dequant accuracy drops measurably on outlier groups.

---

## Worked example

[`runs/qwen3.6-35b-distill/recipes/awq_gemm.py`](../../runs/qwen3.6-35b-distill/recipes/awq_gemm.py)
applies this scheme (data-free RTN variant) to the same 35 B Qwen3.5-MoE
model. Matches `QuantTrio/Qwen3.6-35B-A3B-AWQ`'s layout bit-for-bit. See
[`runs/qwen3.6-35b-distill/REPORT.md`](../../runs/qwen3.6-35b-distill/REPORT.md)
for the eval — MMLU **−2.73 pp** vs bf16, GSM8K strict-match **−0.61 pp**
(within 1 σ), vision tower preserved at fp16.

A calibrated AWQ pass would likely close ~half to all of the MMLU gap; not
yet measured.

---

## References

- AutoAWQ: <https://github.com/casper-hansen/AutoAWQ>
- vLLM AWQ-MoE loader: `vllm/model_executor/layers/quantization/moe_wna16.py`
- Pack-order spec: `reverse_awq_pack_order` in the same file
- Reference checkpoints: `QuantTrio/<model>-AWQ`, `casper-hansen/<model>-AWQ`
- Original AWQ paper: <https://arxiv.org/abs/2306.00978>
