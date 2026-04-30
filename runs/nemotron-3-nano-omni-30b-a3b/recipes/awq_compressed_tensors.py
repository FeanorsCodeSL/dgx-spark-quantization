#!/usr/bin/env python3
"""AWQ-INT4 quantization for Nemotron-3-Nano-Omni-30B-A3B-Reasoning via
llm-compressor, producing a `compressed-tensors` `pack-quantized` W4A16
artifact that vLLM loads through its working ungated-MoE loader path.

Why this recipe exists (Phase 2b of the plan):

  Phase 2a built an AutoAWQ-format artifact (data-free RTN) — packed
  cleanly but vLLM rejected every kernel path (`awq_marlin`, `awq`,
  `moe_wna16`) because Nemotron uses ungated relu^2 MoE and none of
  those kernels handle it. The fix is the on-disk format vLLM's
  compressed-tensors loader uses for NVIDIA's NVFP4 build (and that
  stelterlab successfully shipped for the LM-only base model).

What this recipe does differently from `awq_gemm.py`:

  - Uses `llmcompressor.modifiers.awq.AWQModifier` (data-driven per-channel
    scaling) instead of pure RTN.  Calibrates on 256 prompts of
    `open_platypus` (text-only reasoning corpus) at max_seq_len=2048.
  - Uses `llm-compressor` to calibrate and compress in memory, then streams
    the compressed state dict through this recipe's bounded sharder so the
    on-disk keys are
    `...experts.<j>.up_proj.weight_packed/.weight_scale/.weight_shape`
    (same shape as stelterlab + NVFP4) and `quantization_config.format
    == "pack-quantized"`.
  - Calibrates the inner LM (`model.language_model`) only.  The Omni
    wrapper's `forward` is image-only (line 180 of modeling.py: the
    first positional arg is `pixel_values` and the body immediately
    calls `image_flags.squeeze(-1)` with no None-guard) — a text-only
    calibration would crash there.  The inner `NemotronHForCausalLM`
    is the standard text-only LM that llm-compressor knows how to
    walk.  Re-prefixed back to `language_model.<key>` at save time so
    vLLM's multimodal `--trust-remote-code` loader still finds them.

Conservative ignore policy (matches Phase 1's `_classify.should_quantize`
exactly): only the 5,888 routed-expert MLP weights past layer 0 are
quantized.  Mamba2, attention, shared experts, vision (RADIO), audio
(Parakeet), every projector, layer-0, embeddings, lm_head, and norms all
stay dense.  This is more conservative than stelterlab (who quantize
mamba + attention + shared experts too) — the plan picks the conservative
floor on purpose.

CLI:

  # synthetic 2-layer MoE roundtrip + Linear quant; no source download
  python awq_compressed_tensors.py --selftest

  # walk the source model, apply ignore regexes, count would-quantize.
  # Aborts if count != 5888.  Run BEFORE the full quantization.
  python awq_compressed_tensors.py --dry-run

  # full quantization (1-3 hours; wrap with run_under_memcap.sh)
  python awq_compressed_tensors.py
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _classify import should_quantize  # noqa: E402


SOURCE_DIR = Path(os.environ.get("SRC_DIR", ""))
DST_DIR = Path(os.environ.get("DST_DIR", ""))
HF_HOME = os.environ.get("HF_HOME", str(Path(__file__).resolve().parents[3] / "hf-cache"))

# Nemotron's `moe_intermediate_size = 1856 = 64 * 29` is NOT divisible by 128.
# down_proj's in_features = 1856 forces group_size = 64.  vLLM's
# compressed-tensors W4A16 path supports gs in {32, 64, 128}.
GROUP_SIZE = 64

NUM_CALIB_SAMPLES = 256
MAX_SEQ_LEN = 2048
CALIB_DATASET = "open_platypus"

# Layer indices in the hybrid pattern
# "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME" — `*` = attention.
ATTN_LAYERS = [5, 12, 19, 26, 33, 42]
MOE_LAYERS = [
    1, 3, 6, 8, 10, 13, 15, 17, 20, 22, 24, 27, 29, 31,
    34, 36, 38, 40, 43, 45, 47, 49, 51,
]
assert len(ATTN_LAYERS) == 6 and len(MOE_LAYERS) == 23

# Aux files copied verbatim from source into the artifact dir.
AUX_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
)


def _rss_gib() -> float:
    return psutil.Process().memory_info().rss / 1024**3


def _log(msg: str) -> None:
    print(f"[awq-ct {time.strftime('%H:%M:%S')} RSS={_rss_gib():.2f}GiB] {msg}", flush=True)


def build_recipe(prefix: str = ""):
    """Construct the AWQModifier with config_groups + ignore + mappings.

    `prefix` is the dotted prefix to prepend to every regex (so the same
    code can target both the Omni wrapper paths and the LM-only paths).
    For LM-only calibration, prefix="" (paths are `backbone.layers.<i>.*`).
    """
    from llmcompressor.modifiers.awq import AWQModifier

    p = prefix  # shorthand

    # Conservative ignore policy.  Anything that matches one of these regexes
    # is left dense.  The recipe targets `Linear` modules, so embeddings and
    # norms are out of scope by default — but we list them explicitly for
    # parity with the plan's verification.  layer-0 is excluded by the
    # `\.layers\.0\.` pattern (Mamba layer in the hybrid pattern).
    ignore = [
        "lm_head",
        f"re:.*{p}backbone\\.embeddings$",
        f"re:.*{p}backbone\\.norm_f$",
        # router gate (NemotronHTopkRouter — registered as a Linear-shaped
        # nn.Linear-equivalent under `mixer.gate`)
        f"re:.*{p}backbone\\.layers\\.\\d+\\.mixer\\.gate$",
        # layer-0 (always Mamba in this snapshot, but ignore everything
        # under it to stay parity with Phase 2a)
        f"re:.*{p}backbone\\.layers\\.0\\..*",
        # shared expert MLP (always-active; large per-byte quality contribution)
        f"re:.*{p}backbone\\.layers\\.\\d+\\.mixer\\.shared_experts\\..*",
        # mamba2 inner Linears (in_proj, out_proj — conv1d is a Conv1d not
        # Linear, so it's already out of scope for AWQModifier targets)
        f"re:.*{p}backbone\\.layers\\.\\d+\\.mixer\\.in_proj$",
        f"re:.*{p}backbone\\.layers\\.\\d+\\.mixer\\.out_proj$",
        # transformer attention (q/k/v/o for the 6 attention layers)
        f"re:.*{p}backbone\\.layers\\.\\d+\\.mixer\\.q_proj$",
        f"re:.*{p}backbone\\.layers\\.\\d+\\.mixer\\.k_proj$",
        f"re:.*{p}backbone\\.layers\\.\\d+\\.mixer\\.v_proj$",
        f"re:.*{p}backbone\\.layers\\.\\d+\\.mixer\\.o_proj$",
        # multimodal stack (only present if calibrating the wrapper; harmless
        # when calibrating the inner LM)
        "re:^vision_model\\..*",
        "re:^sound_encoder\\..*",
        "re:^sound_projection\\..*",
        "re:^mlp1\\..*",
    ]

    # Smooth-layer mappings for AWQ scaling.  Each entry pairs a pre-mixer
    # norm with the projections that consume its output so AWQ can move
    # weight magnitude from the projections into the norm.  Two flavors:
    #
    #   - Attention layers (6 of them) — pair `norm` with q/k/v.  These
    #     are NO-OPs at runtime since q/k/v are in the ignore list (kept
    #     dense), but llmcompressor still resolves the regex match counts,
    #     which the dry-run preflight uses to verify the indices are right.
    #   - MoE layers (23 of them) — pair `norm` with every routed-expert
    #     `up_proj`.  These are the smooth mappings that actually drive
    #     AWQ scaling, since the experts ARE quantized.  Without these the
    #     finalize step crashes with ZeroDivisionError when `_error_metrics`
    #     is empty (llmcompressor 0.10 unconditionally averages).
    mappings = [
        {
            "smooth_layer": f"re:.*{p}backbone\\.layers\\.{i}\\.norm$",
            "balance_layers": [
                f"re:.*{p}backbone\\.layers\\.{i}\\.mixer\\.q_proj$",
                f"re:.*{p}backbone\\.layers\\.{i}\\.mixer\\.k_proj$",
                f"re:.*{p}backbone\\.layers\\.{i}\\.mixer\\.v_proj$",
            ],
        }
        for i in ATTN_LAYERS
    ] + [
        {
            "smooth_layer": f"re:.*{p}backbone\\.layers\\.{i}\\.norm$",
            "balance_layers": [
                f"re:.*{p}backbone\\.layers\\.{i}\\.mixer\\.experts\\.\\d+\\.up_proj$",
            ],
        }
        for i in MOE_LAYERS
    ]

    # Use `config_groups` (matches stelterlab's YAML shape) instead of
    # the `scheme="W4A16"` shorthand so we can pin group_size=64 (the
    # shorthand defaults to gs=128 which won't divide 1856).
    config_groups = {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "group_size": GROUP_SIZE,
                "strategy": "group",
                "observer": "minmax",
            },
        }
    }

    return AWQModifier(
        config_groups=config_groups,
        targets=["Linear"],
        ignore=ignore,
        mappings=mappings,
        duo_scaling=True,
        n_grid=20,
    )


def _module_path_matches_any(name: str, regexes: list[re.Pattern]) -> bool:
    return any(r.search(name) for r in regexes)


def _compile_recipe_patterns(recipe) -> tuple[list[re.Pattern], list[re.Pattern]]:
    """Return (ignore_patterns, smooth_layer_patterns) compiled from the recipe."""
    ignore_patterns: list[re.Pattern] = []
    for s in recipe.ignore:
        if s.startswith("re:"):
            ignore_patterns.append(re.compile(s[len("re:"):]))
        else:
            # Exact module-name match (e.g. "lm_head") — anchor as a regex
            # that requires the dotted name to end with `.<s>` or equal `<s>`.
            ignore_patterns.append(re.compile(rf"(?:^|\.){re.escape(s)}$"))
    smooth_patterns: list[re.Pattern] = []
    for m in recipe.mappings or []:
        sl = m.smooth_layer if hasattr(m, "smooth_layer") else m["smooth_layer"]
        if sl.startswith("re:"):
            smooth_patterns.append(re.compile(sl[len("re:"):]))
        else:
            smooth_patterns.append(re.compile(rf"^{re.escape(sl)}$"))
    return ignore_patterns, smooth_patterns


def dry_run(model) -> tuple[int, int, int, int]:
    """Walk the loaded model, apply the recipe's ignore regexes, count.

    Returns (would_quantize, would_skip_linear, attn_smooth_matches, moe_smooth_matches).

    `attn_smooth_matches` counts attention-layer norm regexes that match a
    real module (must be 6/6).  `moe_smooth_matches` counts MoE-layer norm
    regexes (must be 23/23).
    """
    import torch.nn as nn

    recipe = build_recipe(prefix="")  # LM-only paths
    ignore_pats, smooth_pats = _compile_recipe_patterns(recipe)

    # Pre-classify smooth_pats by attn vs moe for clearer reporting.
    attn_pat_set = {f"re:.*backbone\\.layers\\.{i}\\.norm$"[3:] for i in ATTN_LAYERS}
    moe_pat_set = {f"re:.*backbone\\.layers\\.{i}\\.norm$"[3:] for i in MOE_LAYERS}

    n_linear = 0
    n_quant = 0
    n_skip = 0
    attn_smooth_matched = set()
    moe_smooth_matched = set()

    quant_examples = []
    skip_examples = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            n_linear += 1
            if _module_path_matches_any(name, ignore_pats):
                n_skip += 1
                if len(skip_examples) < 8:
                    skip_examples.append(name)
            else:
                n_quant += 1
                if len(quant_examples) < 8:
                    quant_examples.append(name)
        for sp in smooth_pats:
            if sp.search(name):
                if sp.pattern in attn_pat_set:
                    attn_smooth_matched.add(sp.pattern)
                elif sp.pattern in moe_pat_set:
                    moe_smooth_matched.add(sp.pattern)

    _log(f"dry_run: Linear modules total={n_linear} would_quantize={n_quant} would_skip={n_skip}")
    _log(
        f"dry_run: smooth_layer mappings matched: "
        f"{len(attn_smooth_matched)}/{len(ATTN_LAYERS)} attention layers, "
        f"{len(moe_smooth_matched)}/{len(MOE_LAYERS)} MoE layers"
    )
    _log(f"dry_run: example would_quantize: {quant_examples[:4]}")
    _log(f"dry_run: example would_skip:    {skip_examples[:4]}")
    return n_quant, n_skip, len(attn_smooth_matched), len(moe_smooth_matched)


MAX_SHARD_BYTES = 4 * 1024**3

# Top-level prefixes of the multimodal stack — these are NOT in the LM
# calibration output and must be copied through from the source shards.
NON_LM_PREFIXES = ("vision_model.", "mlp1.", "sound_encoder.", "sound_projection.")


def _merge_lm_state_and_multimodal(
    lm_state: dict[str, torch.Tensor], src_snapshot: Path, work_dir: Path
) -> dict[str, str]:
    """Stream-build the final artifact shards from two sources.

    1. From `lm_state` (already compressed in memory): take every tensor and
       prefix its key with `language_model.` so vLLM's wrapper loader finds it
       under the right module path.  This includes both the quantized expert
       triplets (`weight_packed`/`weight_scale`/`weight_shape`) and the dense
       LM tensors (mamba, attention, shared experts, embeddings, lm_head,
       norms — everything in the ignore list).
    2. From `src_snapshot`'s original safetensors shards: take only the
       non-LM tensors (vision_model.*, mlp1.*, sound_encoder.*,
       sound_projection.*) and copy verbatim.  These are dense bf16
       (the multimodal stack stays unquantized).

    Both feeds are pushed through a single sharder that flushes a
    `_tmp_NNNNN.safetensors` file every MAX_SHARD_BYTES (4 GiB).  After
    everything is in, the tmp shards are renamed atomically to the
    canonical `model-NNNNN-of-MMMMM.safetensors`.

    Returns the final weight_map (key -> shard filename).
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    weight_map: dict[str, str] = {}
    out_idx = 1
    out_buffer: dict[str, torch.Tensor] = {}
    out_size = [0]

    def flush() -> None:
        if not out_buffer:
            return
        tmp = work_dir / f"_tmp_{out_idx:05d}.safetensors"
        save_file(out_buffer, str(tmp), metadata={"format": "pt"})
        gib = sum(t.numel() * t.element_size() for t in out_buffer.values()) / 1024**3
        _log(f"merge: wrote {tmp.name} ({len(out_buffer)} keys, {gib:.2f} GiB)")
        out_buffer.clear()
        out_size[0] = 0

    def push(key: str, tensor: torch.Tensor) -> None:
        nonlocal out_idx
        tensor = tensor.contiguous()
        nb = tensor.numel() * tensor.element_size()
        if out_size[0] + nb > MAX_SHARD_BYTES and out_buffer:
            flush()
            out_idx += 1
        out_buffer[key] = tensor
        out_size[0] += nb
        weight_map[key] = f"_tmp_{out_idx:05d}.safetensors"

    # ----- (1) Compressed LM state → re-prefixed into `language_model.*` -----
    _log(f"merge: streaming {len(lm_state)} compressed LM tensors from memory")
    n_lm_keys = 0
    for k, t in lm_state.items():
        push("language_model." + k, t)
        n_lm_keys += 1
        if n_lm_keys % 2000 == 0:
            _log(f"merge: streamed {n_lm_keys} LM keys")

    # ----- (2) Multimodal pass-through from source -----
    src_index = json.load(open(src_snapshot / "model.safetensors.index.json"))
    src_wmap: dict[str, str] = src_index["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for k, fn in src_wmap.items():
        if any(k.startswith(p) for p in NON_LM_PREFIXES):
            by_shard.setdefault(fn, []).append(k)
    src_shards = sorted(by_shard.keys())
    _log(f"merge: streaming non-LM tensors from {len(src_shards)} source shards")
    n_mm_keys = 0
    for shard_name in src_shards:
        src_path = src_snapshot / shard_name
        with safe_open(str(src_path), framework="pt") as f:
            for k in by_shard[shard_name]:
                t = f.get_tensor(k)
                push(k, t)
                n_mm_keys += 1
                del t
        _log(f"merge: read MM shard {shard_name} ({n_mm_keys} MM keys cumulative)")

    flush()
    total_shards = out_idx
    _log(f"merge: total {n_lm_keys} LM keys + {n_mm_keys} MM keys -> {total_shards} shards")

    # Atomic rename _tmp_XXXXX.safetensors -> model-XXXXX-of-NNNNN.safetensors
    rename_map: dict[str, str] = {}
    for i in range(1, total_shards + 1):
        old = work_dir / f"_tmp_{i:05d}.safetensors"
        new_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        old.rename(work_dir / new_name)
        rename_map[f"_tmp_{i:05d}.safetensors"] = new_name
    weight_map = {k: rename_map[v] for k, v in weight_map.items()}

    return weight_map


def write_artifact_files(dst_dir: Path, src_snapshot: Path, recipe_dict: dict, weight_map: dict[str, str]) -> None:
    """Write config.json, index, aux files, and pass-through .py modules."""
    # safetensors index
    total_size = sum((dst_dir / s).stat().st_size for s in set(weight_map.values()))
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    json.dump(index, open(dst_dir / "model.safetensors.index.json", "w"), indent=2, sort_keys=True)

    # Aux files (tokenizer, chat template, generation_config, etc.)
    for fname in AUX_FILES:
        src = src_snapshot / fname
        if src.exists():
            shutil.copy2(src, dst_dir / fname)
            _log(f"copied aux: {fname}")

    # Custom .py modules (modeling, configuration, processing, audio_model,
    # video_io, evs, ...) — vLLM's --trust-remote-code imports these at load.
    py_count = 0
    for src_py in sorted(src_snapshot.glob("*.py")):
        shutil.copy2(src_py, dst_dir / src_py.name)
        py_count += 1
    _log(f"copied {py_count} custom .py files")

    # config.json: source config + injected quantization_config.  Keep the
    # outer multimodal `architectures` so vLLM runs the wrapper init path.
    src_cfg = json.load(open(src_snapshot / "config.json"))
    out_cfg = dict(src_cfg)
    out_cfg["quantization_config"] = recipe_dict
    out_cfg["architectures"] = ["NemotronH_Nano_Omni_Reasoning_V3"]
    json.dump(out_cfg, open(dst_dir / "config.json", "w"), indent=2, sort_keys=True)
    _log(f"config.json written: arch={out_cfg['architectures']}, model_type={out_cfg.get('model_type')}")


def _build_compressed_tensors_config_block() -> dict:
    """Build the `quantization_config` dict that vLLM's compressed-tensors
    loader reads.  Shape mirrors the stelterlab artifact (verified via
    config.json fetch in the plan References) — but our `ignore` list is
    much longer because we keep more modules dense (mamba, attention,
    shared experts, layer-0, multimodal stack), whereas stelterlab quantizes
    those.  vLLM's loader uses this list to skip dequantization for
    Linears whose state-dict carries `.weight` instead of `.weight_packed`.
    """
    # Module-name regexes for every dense Linear in the artifact.  The artifact
    # stores HF/source paths as `language_model.backbone.*`, while vLLM builds
    # runtime modules as `language_model.model.*`.  vLLM intentionally does not
    # remap `re:` entries in compressed-tensors configs, so include both forms.
    lm_prefixes = [
        "language_model\\.backbone",
        "language_model\\.model",
        "backbone",
        "model",
    ]
    lm_ignores: list[str] = []
    for prefix in lm_prefixes:
        lm_ignores.extend(
            [
                f"re:.*{prefix}\\.layers\\.\\d+\\.mixer\\.shared_experts\\..*",
                f"re:.*{prefix}\\.layers\\.\\d+\\.mixer\\.in_proj$",
                f"re:.*{prefix}\\.layers\\.\\d+\\.mixer\\.out_proj$",
                f"re:.*{prefix}\\.layers\\.\\d+\\.mixer\\.q_proj$",
                f"re:.*{prefix}\\.layers\\.\\d+\\.mixer\\.k_proj$",
                f"re:.*{prefix}\\.layers\\.\\d+\\.mixer\\.v_proj$",
                f"re:.*{prefix}\\.layers\\.\\d+\\.mixer\\.o_proj$",
                f"re:.*{prefix}\\.layers\\.0\\..*",
            ]
        )

    ignore = [
        "lm_head",
        *lm_ignores,
        # Multimodal stack — kept dense at bf16 so multimodal serving works.
        "re:^vision_model\\..*",
        "re:^sound_encoder\\..*",
        "re:^sound_projection\\..*",
        "re:^mlp1\\..*",
    ]
    return {
        "config_groups": {
            "group_0": {
                "format": "pack-quantized",
                "input_activations": None,
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": GROUP_SIZE,
                    "num_bits": 4,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "scale_dtype": None,
                    "strategy": "group",
                    "symmetric": True,
                    "type": "int",
                    "zp_dtype": None,
                },
            }
        },
        "format": "pack-quantized",
        "global_compression_ratio": None,
        "ignore": ignore,
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "sparsity_config": {},
        "transform_config": {},
        "version": "0.13.0",
    }


def selftest() -> None:
    """Synthetic-tensor roundtrip on a tiny LlamaForCausalLM.

    Builds a 2-layer Llama (small enough for sub-second quantization) and
    runs the same llm-compressor pipeline used for the real run.  Asserts
    the saved safetensors contain:
      - `weight_packed`/`weight_scale`/`weight_shape` triplets for the
        Linear modules NOT in the ignore list (here: gate_proj/up_proj/
        down_proj/q_proj/k_proj/v_proj/o_proj for non-layer-0)
      - dense `.weight` for `lm_head` (ignored) and the layer-0 modules
        (matching the recipe's layer-0 ignore policy)

    A real PreTrainedModel (LlamaForCausalLM) is used because llm-compressor's
    pre_process step requires `model.config` and a HF processor.
    """
    from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier

    _log("selftest: building tiny LlamaForCausalLM (2 layers, hidden=128)")
    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)

    # Use a real tokenizer (any small one will do) so llm-compressor can
    # initialize a processor.  Llama's stock tokenizer needs the SentencePiece
    # files, so use the GPT-2 tokenizer which ships with transformers.
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tiny pre-tokenized calibration set so we don't hit HF Hub.
    from datasets import Dataset
    calib = Dataset.from_dict({"input_ids": [[i % 256 for i in range(16)] for _ in range(4)]})

    # Recipe: ignore lm_head and layer-0 (mirrors the real recipe).
    recipe = AWQModifier(
        config_groups={
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "group_size": 64,
                    "strategy": "group",
                    "observer": "minmax",
                },
            }
        },
        targets=["Linear"],
        ignore=["lm_head", "re:.*\\.layers\\.0\\..*"],
        mappings=[
            {
                "smooth_layer": "re:.*layers\\.1\\.input_layernorm$",
                "balance_layers": [
                    "re:.*layers\\.1\\.self_attn\\.q_proj$",
                    "re:.*layers\\.1\\.self_attn\\.k_proj$",
                    "re:.*layers\\.1\\.self_attn\\.v_proj$",
                ],
            },
        ],
        duo_scaling=True,
        n_grid=4,
    )

    model = oneshot(
        model=model,
        tokenizer=tokenizer,
        recipe=recipe,
        dataset=calib,
        num_calibration_samples=4,
        max_seq_length=16,
        pipeline="basic",
        output_dir=None,
        save_compressed=False,
    )

    from compressed_tensors import ModelCompressor

    compressor = ModelCompressor.from_pretrained_model(model)
    compressor.compress_model(model)
    keys = list(model.state_dict().keys())
    _log(f"selftest: compressed in-memory state has {len(keys)} keys; sample: {sorted(keys)[:6]}")

    # Layer-1 modules must be packed (not in ignore).
    for proj in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                 "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
        base = f"model.layers.1.{proj}"
        for suf in (".weight_packed", ".weight_scale", ".weight_shape"):
            assert (base + suf) in keys, f"selftest: missing packed key {base+suf}"
    # Layer-0 modules and lm_head must be DENSE (not packed).
    for dense_base in ("model.layers.0.self_attn.q_proj",
                       "model.layers.0.mlp.up_proj",
                       "lm_head"):
        assert (dense_base + ".weight") in keys, f"selftest: missing dense {dense_base}.weight"
        assert (dense_base + ".weight_packed") not in keys, (
            f"selftest: unexpectedly packed: {dense_base}"
        )

    _log("selftest OK")


def run_full() -> None:
    """End-to-end: load wrapper, calibrate inner LM, compress, write artifact."""
    if not str(SOURCE_DIR) or not str(DST_DIR):
        raise SystemExit("set SRC_DIR (source snapshot) and DST_DIR (output) env vars")
    src = SOURCE_DIR
    dst = DST_DIR
    assert src.exists(), f"missing SRC_DIR={src}"
    if dst.exists() and any(dst.iterdir()):
        raise RuntimeError(f"refusing to write into non-empty DST_DIR={dst}")

    # Atomic save: write into a tmp dir adjacent to dst, rename at the end.
    pid = os.getpid()
    work_dir = dst.parent / f"{dst.name}.tmp.{pid}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    _log(f"start: SRC_DIR={src} DST_DIR={dst} work={work_dir}")

    # ----- Load the Omni wrapper on CPU -----
    os.environ.setdefault("HF_HOME", HF_HOME)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _log("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(str(src), trust_remote_code=True)

    _log("loading wrapper model on CPU (bf16) — ~62 GiB RSS expected")
    wrapper = AutoModelForCausalLM.from_pretrained(
        str(src),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    _log(f"wrapper loaded: {type(wrapper).__name__} RSS={_rss_gib():.2f}GiB")

    # ----- Extract the inner LM for calibration -----
    inner = wrapper.language_model
    assert hasattr(inner, "model") or hasattr(inner, "backbone"), (
        f"unexpected inner LM structure: {type(inner).__name__}"
    )
    _log(f"inner LM: {type(inner).__name__} (calibrating this, NOT the wrapper)")

    # Free the multimodal towers before oneshot calibration. The vision/audio
    # weights are copied verbatim from `src_snapshot` later (see
    # _merge_lm_and_multimodal), so we don't need them in RAM during the save
    # spike inside oneshot. Previous run OOM-rebooted at the save step (RSS
    # peaked >100 GiB on a 128 GiB box). Target: keep peak under 90 GiB.
    pre_free_rss = _rss_gib()
    for child_name in ("vision_model", "sound_encoder", "sound_projection"):
        if hasattr(wrapper, child_name):
            delattr(wrapper, child_name)
            _log(f"freed wrapper.{child_name}")
    gc.collect()
    _log(f"freed multimodal towers: RSS {pre_free_rss:.2f} -> {_rss_gib():.2f} GiB")

    # ----- Dry-run preflight: count would_quantize against the inner LM -----
    n_quant, n_skip, attn_match, moe_match = dry_run(inner)
    if n_quant != 5888:
        raise SystemExit(
            f"dry-run preflight failed: would_quantize={n_quant} (expected 5888). "
            f"Re-check ignore regexes against module_inspection.txt before calibrating."
        )
    if attn_match != len(ATTN_LAYERS):
        raise SystemExit(
            f"smooth_layer mappings matched {attn_match}/{len(ATTN_LAYERS)} attention layers; "
            f"check the .norm path naming in NemotronHBlock."
        )
    if moe_match != len(MOE_LAYERS):
        raise SystemExit(
            f"smooth_layer mappings matched {moe_match}/{len(MOE_LAYERS)} MoE layers; "
            f"check the .norm path naming in NemotronHBlock."
        )
    _log(
        f"preflight PASS: would_quantize={n_quant}, "
        f"attn_smooth={attn_match}/{len(ATTN_LAYERS)}, "
        f"moe_smooth={moe_match}/{len(MOE_LAYERS)}"
    )

    # ----- Build the recipe -----
    recipe = build_recipe(prefix="")

    # ----- Calibration on the inner LM -----
    from llmcompressor import oneshot

    _log(f"running oneshot calibration: dataset={CALIB_DATASET} samples={NUM_CALIB_SAMPLES} maxlen={MAX_SEQ_LEN}")
    inner = oneshot(
        model=inner,
        tokenizer=tokenizer,
        recipe=recipe,
        dataset=CALIB_DATASET,
        num_calibration_samples=NUM_CALIB_SAMPLES,
        max_seq_length=MAX_SEQ_LEN,
        output_dir=None,
        save_compressed=False,
        trust_remote_code_model=True,
    )
    _log(f"oneshot complete RSS={_rss_gib():.2f}GiB")

    # Do not use `save_pretrained(save_compressed=True)` here.  On this model
    # that path reached "Writing model shards: 0%" and was OOM-killed before
    # any safetensors were written, because Transformers tried to assemble a
    # huge save payload.  Compress in-place, then stream the state dict through
    # our bounded 4 GiB sharder below.
    from compressed_tensors import ModelCompressor

    _log("compressing calibrated inner LM in memory")
    compressor = ModelCompressor.from_pretrained_model(inner)
    compressor.compress_model(inner)
    _log(f"compress_model complete RSS={_rss_gib():.2f}GiB")

    # Keep only the compressed inner LM state refs for the re-prefix copy.
    lm_state = inner.state_dict()
    _log(f"captured compressed LM state_dict refs: {len(lm_state)} keys")

    # Drop the wrapper shell before the re-prefix copy step. `lm_state` keeps
    # references to the inner tensors; deleting the modules reduces Python
    # object overhead without duplicating tensor storage.
    del wrapper, inner
    gc.collect()
    _log(f"freed wrapper/modules RSS={_rss_gib():.2f}GiB")

    # ----- Merge LM-only oneshot output with multimodal pass-through -----
    _log("merging compressed LM state with multimodal pass-through")
    weight_map = _merge_lm_state_and_multimodal(lm_state, src, work_dir)
    del lm_state
    gc.collect()

    # ----- Write config.json + aux + .py files -----
    qcfg = _build_compressed_tensors_config_block()
    write_artifact_files(work_dir, src, qcfg, weight_map)

    # ----- Atomic rename -----
    if dst.exists():
        if any(dst.iterdir()):
            raise RuntimeError(f"DST_DIR became non-empty during run: {dst}")
        dst.rmdir()
    work_dir.rename(dst)
    _log(f"renamed {work_dir} -> {dst}")

    files = sorted(dst.iterdir())
    total_bytes = sum(p.stat().st_size for p in files if p.is_file())
    _log(
        f"DONE. files={len(files)} total={total_bytes / 1024**3:.2f} GiB "
        f"final RSS={_rss_gib():.2f} GiB"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true", help="synthetic-tensor roundtrip")
    p.add_argument("--dry-run", action="store_true", help="walk source model, count would-quantize, abort")
    args = p.parse_args()

    if args.selftest:
        selftest()
        return

    if args.dry_run:
        if not str(SOURCE_DIR):
            raise SystemExit("set SRC_DIR for --dry-run")
        os.environ.setdefault("HF_HOME", HF_HOME)
        from transformers import AutoModelForCausalLM
        _log(f"--dry-run: loading wrapper from {SOURCE_DIR} on CPU (bf16)")
        wrapper = AutoModelForCausalLM.from_pretrained(
            str(SOURCE_DIR),
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        inner = wrapper.language_model
        n_quant, n_skip, attn_match, moe_match = dry_run(inner)
        if n_quant != 5888:
            raise SystemExit(f"FAIL: would_quantize={n_quant} != 5888")
        if attn_match != len(ATTN_LAYERS):
            raise SystemExit(f"FAIL: attn smooth_layer matches={attn_match} != {len(ATTN_LAYERS)}")
        if moe_match != len(MOE_LAYERS):
            raise SystemExit(f"FAIL: moe smooth_layer matches={moe_match} != {len(MOE_LAYERS)}")
        _log(
            f"--dry-run OK: would_quantize=5888, "
            f"attn_smooth={attn_match}/{len(ATTN_LAYERS)}, "
            f"moe_smooth={moe_match}/{len(MOE_LAYERS)}"
        )
        return

    run_full()


if __name__ == "__main__":
    main()
