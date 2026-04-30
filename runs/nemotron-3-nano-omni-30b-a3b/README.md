# Nemotron-3-Nano-Omni-30B-A3B-Reasoning — quantization run

**Base model**: [`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16)
30B-A3B (active 3B) NemotronH Mamba2 ⨉ Transformer-attention ⨉ MoE hybrid,
multimodal (vision + audio), reasoning fine-tune. 17 safetensors shards,
~66 GB on disk. License: NVIDIA Open Model Agreement.

**Hardware**: NVIDIA DGX Spark, GB10 / SM121a, 128 GiB unified memory.

**Status**: in-progress — last updated 2026-05-01

> Driven by [`PLAN.md`](./PLAN.md). The old
> [`docs/plans/nemotron-3-nano-omni-30b-quantization.md`](../../docs/plans/nemotron-3-nano-omni-30b-quantization.md)
> is now only a pointer back here.

---

## TL;DR

| build | bits | disk | MMLU | GSM8K (strict) | ARC-C | Δ MMLU vs bf16 |
|---|---|---|---|---|---|---|
| bf16 baseline | 16 | ~66 GiB | ? | ? | ? | — |
| AWQ-INT4 W4A16 compressed-tensors (ours, multimodal) | 4 | ~22 GiB | ? | ? | ? | ? |
| NVFP4 (NVIDIA official, eval only) | 4 (NVFP4 experts) | ~21 GiB | ? | ? | ? | ? |

Current state: our AWQ artifact exists and passes vLLM smoke. Full AWQ,
NVFP4, and bf16 evals are still pending. Cite NVIDIA's FP8 / NVFP4 builds
for users who want the official paths. Numbers filled in Phase 6.

For full numbers, settings, and the head-to-head, see
[`REPORT.md`](./REPORT.md).

---

## Schemes used

- [`docs/schemes/awq-compressed-tensors.md`](../../docs/schemes/awq-compressed-tensors.md)
  — calibrated AWQ-INT4 W4A16 in `compressed-tensors` `pack-quantized`
  format. Model-specific deviations (Mamba inner Linears kept dense,
  layer-0 + shared-expert + router-gate + vision + audio + projectors all
  kept dense; only the 128 routed experts across the 23 MoE layers get
  4-bit packed) are documented in
  [`recipes/awq_compressed_tensors.py`](./recipes/awq_compressed_tensors.py)
  and in the `REPORT.md`.

FP8 and NVFP4 are not generated here — see
[Existing official builds](#existing-official-builds) below.

## Existing official builds

NVIDIA already publishes two quantized variants of this model. We don't
reproduce them; we cite/eval them so the head-to-head in `REPORT.md` covers
the realistic shipping options end-to-end.

- [`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8)
  — per-tensor E4M3, FP8 KV cache, ~32.8 GB. Generation **out of scope**
  (use NVIDIA's directly when FP8 is the target). REPORT cites NVIDIA's
  card claims for context but does not measure.
- [`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4)
  — NVFP4 experts + FP8 mamba/attention + BF16 encoders, ~20.9 GB.
  Generation **out of scope**. Eval **in scope** in Phase 4 — NVFP4 vLLM
  kernel support on SM121a is unverified, so the phase resolves it
  empirically.

## What's in this run

| path | content |
|---|---|
| [`REPORT.md`](./REPORT.md) | full quant report with deltas |
| [`PLAN.md`](./PLAN.md) | current execution plan and phase checklist |
| [`recipes/`](./recipes/) | model-specific quantizer drivers (AWQ-INT4 compressed-tensors) |
| [`results/`](./results/) | `results_*.json` + `run.log` per full eval, plus `module_inspection.txt` and quant logs |

## Architecture (from config.json)

- Outer class `NemotronH_Nano_Omni_Reasoning_V3` (multimodal wrapper);
  inner LM class `NemotronHForCausalLM` under `llm_config`;
  `model_type: nemotron_h`. `trust_remote_code=True` is required at load
  time; the artifact dir must carry the source's custom `.py` files.
- 52 hidden layers, `hybrid_override_pattern`
  `"MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"` →
  23 Mamba2 (`M`) + 23 MoE-MLP (`E`) + 6 attention (`*`).
- `hidden_size=2688`; dense `intermediate_size=1856`;
  `moe_intermediate_size=1856` per routed expert;
  `moe_shared_expert_intermediate_size=3712`;
  `n_routed_experts=128`, `num_experts_per_tok=6`, `n_shared_experts=1`.
- Attention (GQA): `num_attention_heads=32`, `num_key_value_heads=2`,
  `head_dim=128`, `rope_theta=10000`. So Q is `[2688 → 4096]`,
  K/V are `[2688 → 256]`.
- Mamba2: `mamba_num_heads=64`, `mamba_head_dim=64`, `ssm_state_size=128`,
  `conv_kernel=4`, `expand=2`, `chunk_size=128`,
  `time_step_min=0.001`, `time_step_max=0.1`, `mamba_hidden_act=silu`.
- MLP activation: `mlp_hidden_act: relu2` (squared ReLU). MoE expert MLP
  layout TBD by Phase 1 (likely `up + down` rather than `gate + up + down`
  due to relu²).
- Vision (`vision_config`): RADIOModel CRADIO v2-H,
  `vit_hidden_size=1280`, `projector_hidden_size=20480`,
  `force_image_size=512`, `patch_size=16`,
  `separate_video_embedder=true`, `max_num_patches=13312`.
- Audio (`sound_config`): Parakeet, `hidden_size=1024`,
  `num_attention_heads=8`, `num_hidden_layers=24`,
  `intermediate_size=4096`, `subsampling_factor=8`,
  `num_mel_bins=128`, `sampling_rate=16000`.
- `vocab_size=131072`, `tie_word_embeddings=false`. `torch_dtype: bfloat16`.

## Module-tree inspection (Phase 1 findings)

Live inspection of the bf16 source on CPU (`recipes/inspect_modules.py`,
3,451-line dump in [`results/module_inspection.txt`](./results/module_inspection.txt)).
The plan's config-derived numbers are confirmed; the *names* of every
module are now pinned, which is what the AWQ recipe (Phase 2) compiles
its skip policy from.

**Layer container path:** `language_model.backbone.layers` (52 entries,
`NemotronHBlock` each).  *Not* `language_model.model.layers` — the
NemotronH wrapper uses `backbone`.

**Per-layer structure:** every block has `norm` (`NemotronHRMSNorm`) plus a
polymorphic `mixer`:

| `hybrid_override_pattern` token | `mixer` class | inner Linears (dotted name) | shape |
|---|---|---|---|
| `M` (23×) Mamba2 | `NemotronHMamba2Mixer` | `mixer.in_proj` | `(10304, 2688)` |
|  |  | `mixer.out_proj` | `(2688, 4096)` |
|  |  | `mixer.conv1d` (Conv1d, kernel 4, groups 6144) | `(6144, 1, 4)` |
|  |  | `mixer.norm` (`Zamba2RMSNormGated`) | rank-1 |
| `E` (23×) MoE-MLP | `NemotronHMoE` | `mixer.gate` (`NemotronHTopkRouter`) | `(128, 2688)` |
|  |  | `mixer.experts.<j>.up_proj` for `j` in `[0..127]` | `(1856, 2688)` |
|  |  | `mixer.experts.<j>.down_proj` for `j` in `[0..127]` | `(2688, 1856)` |
|  |  | `mixer.shared_experts.up_proj` | `(3712, 2688)` |
|  |  | `mixer.shared_experts.down_proj` | `(2688, 3712)` |
| `*` (6×) Attention | `NemotronHAttention` | `mixer.q_proj` | `(4096, 2688)` |
|  |  | `mixer.k_proj` (GQA narrow) | `(256, 2688)` |
|  |  | `mixer.v_proj` (GQA narrow) | `(256, 2688)` |
|  |  | `mixer.o_proj` | `(2688, 4096)` |

Per-class top-30 from `module_inspection.txt`: `6,355 Linear`,
`2,967 NemotronHMLP` (= 23 layers × (128 routed + 1 shared) experts),
`2,967 ReLUSquaredActivation`, `52 NemotronHBlock`,
`23 NemotronHMamba2Mixer`, `23 NemotronHMoE`, `23 NemotronHTopkRouter`,
`6 NemotronHAttention`, `24 ParakeetEncoderBlock`, `2 RADIOModel`.

**MoE expert layout — UNFUSED, no `gate_proj`.** Each routed expert is a
distinct `NemotronHMLP` whose only Linears are `up_proj` and `down_proj`.
**There is no `gate_proj`** — the relu² (squared-ReLU) activation is
un-gated, so the MLP is just `down_proj(relu²(up_proj(x)))`.  Compared to
the qwen3.5-MoE 3-D fused layout (`experts.gate_up_proj` `[E, 2I, H]`),
Nemotron is per-expert 2-D, which means the AWQ recipe **does not need
the per-expert unfusing step** the qwen3 recipe carries.

Total expert-related parameters: 5,934 = 23 layers × (128 routed × 2 +
shared × 2) = 23 × (256 + 2) = 5,934. ✓

**Top-level multimodal modules:**

| dotted path | class | role |
|---|---|---|
| `language_model` | `NemotronHForCausalLM` | the LM (backbone + lm_head) |
| `language_model.backbone` | `NemotronHModel` | embeddings + 52 NemotronHBlock |
| `language_model.backbone.embeddings` | `Embedding(131072, 2688)` | input token embedding |
| `language_model.lm_head` | `Linear(131072, 2688)` | output head — **untied** despite `tie_word_embeddings=false` |
| `vision_model` | `RADIOModel` | CRADIO v2-H ViT (32 blocks, `Block`/`Attention`/`Mlp`) |
| `sound_encoder` | `SoundEncoder` (Parakeet-style) | 24 `ParakeetEncoderBlock` |
| `sound_projection` | `SoundProjection` | audio→LM bridge: `linear1` (4096←1024) + `SquaredReLU` + `linear2` (2688←4096) + `RMSNorm` |

There is no separate `vision_projector` — the RADIO output is consumed by
the LM directly via the model's `_get_visual_embeddings` path (custom
`modeling.py`).

**AWQ skip policy** (`recipes/_classify.py::should_quantize`).  Mirrors
NVIDIA's NVFP4 keep-dense policy (mamba + attention + encoders dense;
only routed experts get aggressive bits).  Quantize **iff** the tensor:

- has rank 2 or 3, AND
- ends with `.weight`, `.gate_up_proj`, `.up_proj`, `.gate_proj`, or
  `.down_proj`, AND
- does NOT end with `mixer.gate.weight` / `mlp.gate.weight` /
  `block_sparse_moe.gate.weight` (router gates), AND
- does NOT contain (case-insensitive substring): `lm_head`, `embed_tokens`,
  `embedding`, `embed`, `norm`, `layernorm`, `rmsnorm`, `vision`, `radio`,
  `vision_model`, `vision_tower`, `image_proj`, `video`, `sound`, `audio`,
  `parakeet`, `audio_encoder`, `projector`, `projection`, `mamba`, `ssm`,
  `in_proj`, `out_proj`, `dt_proj`, `conv1d`, `self_attn`, `q_proj`,
  `k_proj`, `v_proj`, `o_proj`, `shared_expert`, AND
- does NOT match `.*\.layers\.0\..*` (layer-0 conservative preservation —
  layer 0 happens to be Mamba in this model so this is doubly-redundant
  but the rule stays for portability).

The only weights this lets through are
`language_model.backbone.layers.<i>.mixer.experts.<j>.{up,down}_proj.weight`
for `i ∈ {1, 3, 6, 8, 10, 13, …}` (the 23 MoE layers, none of them
layer 0) and `j ∈ [0..127]` — i.e. **23 × 128 × 2 = 5,888 routed-expert
weights** out of 6,355 total Linears.

The classifier and its policy are unit-tested in
[`recipes/test_module_classify.py`](./recipes/test_module_classify.py)
with 32 synthetic cases + 25 real names extracted from the inspection
output (60 assertions total).  Run:

```bash
source .venv/bin/activate
pytest runs/nemotron-3-nano-omni-30b-a3b/recipes/test_module_classify.py -v
```

## Reproducing

**SRC_DIR** (filled in Phase 1): `$PWD/hf-cache/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/snapshots/d15962741057ae3a07147df504060e9f0838224e`
**NVFP4_DIR** (filled in Phase 1): `$PWD/hf-cache/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4/snapshots/889396e9cebaefdb69a469afc7bd111660f78eff`

> Note: `--cache-dir <DIR>` puts snapshots at `<DIR>/models--<org>--<name>/...`
> (no `hub/` sub-prefix, unlike when `HF_HOME` is the controlling env var).
> The paths above reflect the *actual* on-disk layout from the Phase 1
> downloads.

```bash
# 1. Pre-cache the bf16 source and the NVFP4 build (Phase 1).
#    `huggingface-cli` is deprecated; use `hf` (huggingface_hub ≥ 0.26).
HF_HUB_ENABLE_HF_TRANSFER=1 hf download nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
  --cache-dir "$PWD/hf-cache" --max-workers 8
HF_HUB_ENABLE_HF_TRANSFER=1 hf download nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 \
  --cache-dir "$PWD/hf-cache" --max-workers 8

# 2. AWQ quantize (Phase 2). Output goes under artifacts/ (gitignored).
export SRC_DIR="$PWD/hf-cache/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/snapshots/d15962741057ae3a07147df504060e9f0838224e"
export DST_DIR="$PWD/artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4"
tools/run_under_memcap.sh python runs/nemotron-3-nano-omni-30b-a3b/recipes/awq_compressed_tensors.py

# 3. Serve & smoke (Phase 2 verification).
tools/serve_vllm_docker.sh "$DST_DIR" \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.55 \
  --reasoning-parser nemotron_v3 \
  --served-model-name nemotron-omni-awq-ct

# 4. Eval (Phases 3, 4, 5).
tools/run_eval_full.sh nemotron-omni-awq-ct "$DST_DIR" \
  "$PWD/runs/nemotron-3-nano-omni-30b-a3b/results/awq_full"
```

## Publishing

The artifact directories are uploaded to Hugging Face per
[`HUGGINGFACE_PUBLISHING.md`](../../HUGGINGFACE_PUBLISHING.md).

---

## AWQ build (Phase 2 outputs)

**Artifact**: `artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4/`
**Recipe**: [`recipes/awq_compressed_tensors.py`](./recipes/awq_compressed_tensors.py) (uses [`recipes/_classify.py`](./recipes/_classify.py) for the skip policy)
**Format**: `quant_method="compressed-tensors"`, `format="pack-quantized"`, W4A16, group size 64, symmetric
**Total size**: 21.34 GiB payload / ~22G on disk across 6 shards
**Quantized tensors**: 5,888 (= 23 MoE layers × 128 experts × 2 projections — `up_proj` + `down_proj`)
**Copied dense**: 1,461 (everything else: Mamba2, attention, vision, audio, projectors, layer 0, router, lm_head, embeds, norms, shared experts)
**Quantization memory**: peak RSS ~94.75 GiB; final RSS ~92.07 GiB
**vLLM image**: `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:20260428`
**Smoke result**: `/v1/chat/completions` returned `2+2 equals 4.` with `finish_reason="stop"`

### Group-size deviation

Nemotron's `moe_intermediate_size = 1856 = 64 × 29` is not divisible by the
canonical AWQ group size of 128. Each routed-expert `down_proj` has
`in_features=1856`, so per-group quant requires `group_size ∈ {32, 64}`.
We use **`group_size = 64`** — the largest value that divides both 1856 and
2688 (the up_proj `in_features`). The working vLLM path is
`compressed-tensors` W4A16, not AutoAWQ/GEMM.

### `mlp1` postscript

Phase 1's keyword-filtered module inspection missed the top-level vision
projector `mlp1` (an `nn.Sequential` at the wrapper level — RADIO→LM
bridge, `mlp1.{1,3}.weight`). The Phase 2 preflight tally caught the
discrepancy (5,890 vs expected 5,888) **before** the full quantization ran;
`mlp1` was added to both `_classify.SKIP_SUBSTRINGS_CI` and the recipe's
`modules_to_not_convert`. Lesson for future runs: the inspect script's
keyword filter alone misses modules outside the conventional `vision_*` /
`*_projector` namings — add an explicit "top-level direct children" probe
to `inspect_modules.py`.

### Save-time memory workaround

The default `llmcompressor.oneshot(..., output_dir=..., save_compressed=True)`
path finished calibration but repeatedly died at `Writing model shards: 0%`
from a Transformers serialization memory spike. The working recipe runs
`oneshot(..., output_dir=None, save_compressed=False)`, compresses the
calibrated inner LM in memory with `ModelCompressor`, then streams compressed
LM tensors plus dense multimodal tensors into bounded safetensors shards.

### Quantization config (in artifact `config.json`)

```json
{
  "bits": 4,
  "group_size": 64,
  "quant_method": "compressed-tensors",
  "format": "pack-quantized",
  "quantization_status": "compressed",
  "config_groups": {
    "group_0": {
      "weights": {
        "num_bits": 4,
        "type": "int",
        "symmetric": true,
        "strategy": "group",
        "group_size": 64
      }
    }
  }
}
```

### Smoke test

```bash
tools/serve_vllm_docker.sh "$PWD/artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4" \
  --kv-cache-dtype fp8_e4m3 --max-model-len 8192 --gpu-memory-utilization 0.55 \
  --reasoning-parser nemotron_v3 --served-model-name nemotron-omni-awq-ct
```

**Verdict: passed.** `/v1/models` returned `nemotron-omni-awq-ct`; the
first `max_tokens=32` request produced reasoning only and stopped by length,
but the `max_tokens=128` request returned content `2+2 equals 4.` with
`finish_reason="stop"`. Full transcript:
[`results/awq_ct_smoke.txt`](./results/awq_ct_smoke.txt).

The successful serve log selected the compressed-tensors path
(`quantization=compressed-tensors`,
`Using CompressedTensorsWNA16MarlinMoEMethod`, `Using Marlin backend for
WNA16 MoE`). The earlier AutoAWQ/GEMM experiment was removed from the
tracked recipe set because it is not the working method for this run.
