# Scheme: FP8 W8A8 Dynamic (compressed-tensors)

One-shot, calibration-free FP8 quantization producing a vLLM-loadable
`compressed-tensors` / `float-quantized` checkpoint. Effectively lossless on
most architectures we've measured.

---

## When to pick this scheme

- You want **maximum quality preservation** under a 2× compression factor
  (16 → 8 bits per weight).
- You don't have a calibration corpus and don't want to gather one.
- The deployment runtime is **vLLM** (native loader) or any other inference
  stack that consumes `compressed-tensors`.
- The target hardware has **native FP8 tensor cores** (Hopper / GB10 / SM121a /
  Ada Lovelace L4) — at small batch FP8 trades some memory-bandwidth headroom
  for tensor-core throughput at large batch.

Skip this scheme if:

- You need ≤ 5-bit compression (FP8 is only 2× — use AWQ or GPTQ for INT4).
- The runtime can't load `compressed-tensors`.
- Activation outliers in the model are extreme (rare) — switch to *static*
  FP8 with calibration.

---

## What "W8A8 dynamic" means

| axis | choice | rationale |
|---|---|---|
| **W** weight precision | `torch.float8_e4m3fn` (1 byte/elem) | E4M3 has a wider dynamic range than E5M2 and matches what vLLM's compressed-tensors loader expects |
| **A** activation precision | FP8 dynamic per-token | activations get quantized at runtime per token — no on-disk activation scales, no calibration data needed |
| Symmetry | symmetric (zero point = 0) | E4M3 is symmetric around 0 |
| Weight quant axis | per-output-channel along `in_features` | reduction axis = input features, scale shape `[out_features, 1]` fp32 |
| Calibration data | **none** | scales come from `max(|w|) / fp8_max` — purely deterministic |
| Storage scales | fp32 per-channel weights, no activation scales | matches the canonical compressed-tensors `FP8_DYNAMIC` preset |

Math (per output channel, with E4M3 max ≈ 448):

```
scale  = max(|w|) / FP8_MAX                # fp32, shape [out, 1]
w_fp8  = clamp(w / scale, -FP8_MAX, +FP8_MAX).to(float8_e4m3fn)
```

---

## Required `quantization_config` block

This is the JSON block written into the artifact's `config.json`. vLLM uses it
to dispatch the FP8 loader path:

```json
{
  "quant_method": "compressed-tensors",
  "format": "float-quantized",
  "version": "<compressed_tensors version>",
  "config_groups": {
    "group_0": {
      "targets": ["Linear"],
      "weights": {
        "num_bits": 8,
        "type": "float",
        "strategy": "channel",
        "symmetric": true,
        "dynamic": false,
        "observer": "minmax"
      },
      "input_activations": {
        "num_bits": 8,
        "type": "float",
        "strategy": "token",
        "symmetric": true,
        "dynamic": true,
        "observer": null
      },
      "output_activations": null,
      "format": null
    }
  },
  "ignore": ["lm_head", "<router gates>", "<embed>", "<norms>", ...],
  "quantization_status": "compressed",
  "kv_cache_scheme": null,
  "global_compression_ratio": null,
  "sparsity_config": {},
  "transform_config": {}
}
```

Validate with `compressed_tensors.QuantizationConfig.model_validate(...)`
before saving — vLLM rejects malformed configs at load time and the error
message is unhelpful.

---

## What to quantize vs leave alone

Generic recommendation. Override per-architecture if needed:

| component | quantize? | reason |
|---|---|---|
| Attention Q/K/V/O | **yes** | linear, well-conditioned |
| Dense MLP up/gate/down | **yes** | linear, well-conditioned |
| MoE routed experts | **yes** (per-expert per-channel) | bulk of params, redundancy across experts |
| Shared expert (always-on dense MLP in MoE) | usually yes | acts like a normal dense MLP |
| Router gate (`mlp.gate`) | **no** | tiny; one wrong nibble degrades routing |
| Linear-attention / Mamba / DeltaNet inner projections | **no** | numerically sensitive recurrent state |
| Embeddings, lm_head | **no** | small; loss-bearing |
| Layer norms | **no** (not Linear anyway) | |
| Vision tower (VLM) | optional | if you're shipping multimodal, leave at fp16; if text-only, strip entirely |
| MTP / speculative decoding head | optional | if shipping speculative, leave; otherwise strip |

The list of names that should **not** be quantized goes into the `ignore`
field of `quantization_config`. vLLM keeps those layers at their original
dtype at load time.

---

## Implementation notes

### Memory: `llmcompressor.oneshot` vs in-place

The straightforward path — `llmcompressor.oneshot(scheme="FP8_DYNAMIC")` —
does not fit on memory-constrained boxes for large MoE models because
`llmcompressor`'s calibration shim *unfuses* MoE expert blocks into one
`nn.Linear` per expert per layer (so they're addressable with
`targets="Linear"`), while the parent module still holds the original fused
3-D tensors. Transient memory roughly doubles the MoE weight footprint.

**Workaround**: a custom in-place quantizer that

1. Loads the bf16 model once at `device_map="cpu"`.
2. Walks layers one at a time. For each layer:
   - Replaces every targeted `nn.Linear`'s `weight` in-place with an
     `float8_e4m3fn` tensor and registers a `weight_scale` parameter.
   - Quantizes any fused MoE 3-D tensors per-expert per-channel **without
     unfusing** — same fp8 shape as input plus paired `*_scale`.
3. `gc.collect()` + `empty_cache()` after every layer.
4. Aborts cleanly with a Python traceback if RSS exceeds a soft ceiling, so
   the kernel doesn't have to SIGKILL the process.

### Architecture-specific naming for vLLM

vLLM's loader for some multimodal classes (e.g. Qwen3.5-MoE's
`Qwen3_5MoeForConditionalGeneration`) wraps the LM at `self.language_model`
and rewrites incoming state-dict keys via `hf_to_vllm_mapper`. For those, the
on-disk keys must be **prefixed** with `language_model.` and (for fused MoE)
named `experts.gate_up_proj` / `experts.down_proj` plus `_weight_scale`.

Get this wrong and vLLM either rejects the checkpoint at load or silently
loads a randomly-initialized tower. Always verify with a dry run before
shipping.

### Vision tower / MTP head

For text-only shipments, strip these *before save* (set the param to an empty
tensor, then `delattr` the submodule, then `gc.collect()`). On boxes where
you can't load the full multimodal stack, this is the only way to fit.

---

## Worked example

[`runs/qwen3.6-35b-distill/recipes/fp8_dynamic.py`](../../runs/qwen3.6-35b-distill/recipes/fp8_dynamic.py)
applies this scheme to a 35 B-parameter Qwen3.5-MoE multimodal model on a
DGX Spark with 128 GiB unified memory. See
[`runs/qwen3.6-35b-distill/REPORT.md`](../../runs/qwen3.6-35b-distill/REPORT.md)
for the eval results — MMLU **−0.09 pp** vs bf16, GSM8K strict-match
**0.00 pp**.

---

## References

- compressed-tensors source: <https://github.com/vllm-project/llm-compressor> (`compressed_tensors/quantization/quant_scheme.py` for the `FP8_DYNAMIC` preset; `compressed_tensors/compressors/naive_quantized/base.py` for `FloatQuantizationCompressor`)
- vLLM compressed-tensors loader: `vllm/model_executor/layers/quantization/compressed_tensors/`
- Reference checkpoints in this format: `Qwen/Qwen3.6-35B-A3B-FP8`, `RedHatAI/*-FP8-dynamic`
