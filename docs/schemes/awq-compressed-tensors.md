# Scheme: AWQ-INT4 W4A16 (compressed-tensors)

Calibrated AWQ INT4 weight-only quantization saved in the
`compressed-tensors` `pack-quantized` format. This is still AWQ as the
quantization method, but it is not the same artifact format as AutoAWQ
GEMM.

---

## When to pick this scheme

- You need INT4 disk and memory reduction, but the model does not fit the
  AutoAWQ/GEMM loader assumptions.
- The deployment runtime is vLLM with `compressed-tensors` support.
- The model has custom architecture pieces, such as Mamba, MoE, or
  multimodal towers, that should stay dense while the large MLP/expert
  weights are quantized.
- You can run a calibration pass with `llm-compressor` instead of using
  data-free RTN.

Skip this scheme if:

- The runtime only supports AutoAWQ's `quant_method="awq"` GEMM layout.
- The target hardware or vLLM image does not include the
  `compressed-tensors` W4A16 loader path.
- You need the simplest portable AWQ upload for common transformer-only
  architectures; AutoAWQ/GEMM may be easier there.

---

## What this format means

| axis | choice |
|---|---|
| Quantization algorithm | AWQ, activation-aware calibration |
| Artifact format | `compressed-tensors` |
| Compression format | `pack-quantized` |
| **W** weight precision | INT4 |
| **A** activation precision | fp16/bf16 runtime activations (W4A16) |
| Group size | model-dependent; Nemotron Omni uses 64 |
| Zero point | symmetric scheme uses no stored zero point |
| Runtime | vLLM `compressed-tensors` loader |

The checkpoint's `config.json` identifies the runtime path with:

```json
{
  "quantization_config": {
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "quantization_status": "compressed"
  }
}
```

That is the key difference from AutoAWQ/GEMM, which uses
`quant_method="awq"` and the `qweight` / `qzeros` / `scales` tensor layout.

---

## What to quantize vs leave dense

Use the same conservative AWQ policy as other INT4 methods, but express it
through `llm-compressor` targets and ignore rules:

| component | quantize? | reason |
|---|---|---|
| Routed expert MLP up/down projections | **yes** | bulk of parameter count |
| Dense MLP projections | **yes** when architecture supports it | bulk of parameter count |
| Attention projections | usually **no** | more sensitive under INT4 |
| Mamba/recurrent mixer projections | usually **no** | state dynamics are sensitive |
| Router gates | **no** | tiny and routing-sensitive |
| Shared experts | usually **no** | always-on path, more sensitive |
| Layer 0 | **no** | early layer sensitivity |
| Embeddings, norms, lm_head | **no** | small or numerically sensitive |
| Vision/audio encoders | **no** | preserve multimodal behavior |

For custom architectures, verify the ignore regexes against both the source
module names and the names that vLLM uses after loading. Nemotron Omni needed
ignore patterns for both `language_model.backbone...` and the vLLM runtime
`model...` prefix.

---

## Implementation notes

### Avoid the default serializer memory spike

For very large multimodal models, `llmcompressor.oneshot(..., output_dir=...)`
can quantize successfully and then fail while Transformers writes shards. The
working Nemotron recipe avoids that spike:

1. Run `oneshot` with `output_dir=None` and `save_compressed=False`.
2. Compress the calibrated inner language model in memory with
   `compressed_tensors.ModelCompressor`.
3. Keep references to the compressed language-model tensors.
4. Stream those tensors plus dense multimodal tensors into bounded
   safetensors shards.

This keeps peak host memory bounded by the calibrated model plus the current
output shard, instead of building a second full serialized copy.

### Match group size to the architecture

The canonical AWQ group size of 128 is not always valid. Nemotron Omni's
expert intermediate size is 1856, so the working artifact uses group size 64
because 1856 is divisible by 64 but not by 128.

### Keep the on-disk config aligned with vLLM

The compressed artifact must include all `ignore` patterns needed by vLLM's
loader. A pattern that matched during `llm-compressor` calibration is not
automatically enough if vLLM maps module prefixes differently at runtime.

---

## Worked example

[`runs/nemotron-3-nano-omni-30b-a3b/recipes/awq_compressed_tensors.py`](../../runs/nemotron-3-nano-omni-30b-a3b/recipes/awq_compressed_tensors.py)
builds a vLLM-loadable Nemotron Omni artifact at:

```text
artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4/
```

The artifact is compressed-tensors W4A16, group size 64, symmetric INT4, with
multimodal encoders kept dense. It served successfully in vLLM with:

```bash
tools/serve_vllm_docker.sh "$PWD/artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4" \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.55 \
  --reasoning-parser nemotron_v3 \
  --served-model-name nemotron-omni-awq-ct
```

---

## References

- Original AWQ paper: <https://arxiv.org/abs/2306.00978>
- llm-compressor: <https://github.com/vllm-project/llm-compressor>
- compressed-tensors: <https://github.com/vllm-project/compressed-tensors>
- vLLM compressed-tensors loader:
  `vllm/model_executor/layers/quantization/compressed_tensors/`
