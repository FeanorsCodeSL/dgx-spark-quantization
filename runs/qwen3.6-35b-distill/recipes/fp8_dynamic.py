#!/usr/bin/env python3
"""
FP8 W8A8 dynamic quantization of Qwen3.6-35B-A3B-Distilled (text-only).

Bypasses ``llmcompressor.oneshot`` because its ``CalibrationQwen3_5MoeSparseMoeBlock``
unfuses 256 experts into 256 ``nn.Linear`` modules per layer (40 layers) while the
parent's reference to the original fused 3D tensors stays alive for the duration of
the calibration_context — this overshoots the 121 GiB DGX Spark unified-memory
ceiling around layer 34/40.

Strategy here:
  1. Load bf16 model once (~70 GB resident).
  2. Walk the model layer-by-layer:
     - For every regular ``nn.Linear`` we want to quantize (attention projections,
       shared_expert proj's), replace ``module.weight.data`` in place with an
       ``torch.float8_e4m3fn`` tensor and register ``weight_scale`` as an
       ``nn.Parameter`` (per-output-channel, fp32). This is the canonical
       compressed-tensors FP8_DYNAMIC layout (CHANNEL strategy for weights,
       TOKEN strategy for activations — activations are dynamic so no scale
       lives on disk).
     - For each MoE layer's fused 3D experts (``experts.gate_up_proj`` /
       ``experts.down_proj``), quantize per-expert per-output-channel. We keep
       the 3D layout in memory (no unfusing of 256 modules), and at SAVE time
       we slice the 3D tensors into per-expert ``mlp.experts.<i>.{gate,up,down}_proj``
       state-dict entries — that is the canonical compressed-tensors naming
       vLLM's compressed-tensors loader expects (see
       ``compressed_tensors/utils/match.py`` lines 240-260).
  3. Force gc + empty_cache after every layer; abort if RSS > 90 GiB.
  4. Save:
       - state_dict written to ``model-XXXXX-of-YYYYY.safetensors`` shards via
         ``safetensors.torch.save_model``-style chunking (we just call our own
         small writer that re-uses HF's index format).
       - tokenizer + chat template via ``tokenizer.save_pretrained``.
       - ``config.json`` is the model's own config plus a ``quantization_config``
         block in the exact shape ``compressed_tensors.QuantizationConfig`` would
         emit (see ``compressors/model_compressors/model_compressor.py`` ``update_config``).

References (read while writing):
  - compressed_tensors/quantization/quant_scheme.py            (FP8_DYNAMIC preset)
  - compressed_tensors/quantization/lifecycle/initialize.py    (CHANNEL = (out,1))
  - compressed_tensors/compressors/naive_quantized/base.py     (FloatQuantizationCompressor)
  - compressed_tensors/compressors/model_compressors/model_compressor.py (update_config)
  - compressed_tensors/utils/match.py                          (experts.<i>.<proj> naming)
  - transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py    (Qwen3_5MoeExperts)

Output: vLLM-loadable compressed-tensors FP8_DYNAMIC checkpoint.
"""

# ---------------------------------------------------------------------------
# Environment hardening — must run BEFORE importing torch / transformers.
# ---------------------------------------------------------------------------

import os

# Configure via env vars so this script is reusable on any machine.
#   MODEL_ID  HF model id to quantize (defaults to the Qwen3.6 reasoning distill)
#   HF_CACHE  HF download/cache directory
#   SAVE_DIR  output directory for the FP8 checkpoint
MODEL_ID = os.environ.get(
    "MODEL_ID", "lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled"
)
HF_CACHE = os.environ.get("HF_CACHE", os.path.expanduser("~/.cache/huggingface"))
SAVE_DIR = os.environ.get("SAVE_DIR", "")

os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Reduce allocator fragmentation while we mutate weights in-place.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Avoid Triton autotune hangs on SM121a.
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

# ---------------------------------------------------------------------------
# Imports.
# ---------------------------------------------------------------------------

import gc
import json
import re
import sys
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
import psutil

# ---------------------------------------------------------------------------
# Constants.
# ---------------------------------------------------------------------------

# E4M3 max representable; matches torch.finfo(torch.float8_e4m3fn).max.
FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)  # 448.0
SCALE_DTYPE = torch.float32
WEIGHT_DTYPE_FP8 = torch.float8_e4m3fn
EPS = 1e-12

# Hard-stop RSS threshold. The cgroup wrapper enforces MemoryMax=112G. We use
# 105 GiB here so the script aborts cleanly with a Python traceback before the
# kernel SIGKILLs the scope. 90 GiB was too tight: the actual peak comes in
# right around layer 39-40 at ~92 GiB anon + smaller page cache, so the prior
# limit tripped one layer short of completion.
RSS_HARD_LIMIT_GIB = 105.0

# ---------------------------------------------------------------------------
# Memory snapshot helper.
# ---------------------------------------------------------------------------


def rss_gib() -> float:
    """Resident-set-size of this process in GiB. On Spark, GPU memory is in
    unified RAM so RSS already reflects total usage."""
    return psutil.Process().memory_info().rss / (1024**3)


def log_mem(tag: str) -> None:
    print(
        f"[mem] {tag:<48s}  rss={rss_gib():6.2f} GiB",
        flush=True,
    )


def assert_under_limit(tag: str) -> None:
    used = rss_gib()
    if used > RSS_HARD_LIMIT_GIB:
        raise RuntimeError(
            f"RSS {used:.2f} GiB exceeded soft limit {RSS_HARD_LIMIT_GIB} GiB at "
            f"{tag!r}; aborting before cgroup hard limit kicks in."
        )


# ---------------------------------------------------------------------------
# Quantization primitives.
# ---------------------------------------------------------------------------


@torch.no_grad()
def fp8_quantize_per_channel(
    weight: torch.Tensor,
    channel_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize ``weight`` to ``torch.float8_e4m3fn`` per output-channel.

    For an ``nn.Linear`` weight of shape ``[out_features, in_features]`` the
    output channel axis is dim 0, and the reduction axis (input features) is
    dim 1. The resulting scale has shape ``[out_features, 1]`` and dtype fp32.

    For a 3D fused expert weight of shape ``[E, out, in]`` we pass
    ``channel_dim=1`` so reduction happens over the input-feature axis and the
    scale has shape ``[E, out, 1]``.

    Math matches compressed_tensors/quantization/lifecycle/forward.py
    (``calculate_range`` for FP8 returns ±FP8_E4M3_MAX, symmetric):

        scale  = max(|w|) / fp8_max,    along reduction axis
        w_fp8  = clamp(w / scale, -fp8_max, +fp8_max).to(float8_e4m3fn)
    """
    # Reduction axis = the *input feature* axis. For Linear weight [O, I] that's 1.
    # For fused expert [E, O, I] that's 2.
    reduction_dim = channel_dim + 1
    # Compute amax in fp32 to avoid bf16 overflow on outlier channels.
    abs_max = weight.detach().abs().to(torch.float32).amax(
        dim=reduction_dim, keepdim=True
    )
    scale = (abs_max / FP8_E4M3_MAX).clamp(min=EPS).to(SCALE_DTYPE)
    # Cast the divide to fp32 too — bf16 / fp32 broadcasting works but fp32
    # gives one extra digit of headroom near the clamp edge.
    w_scaled = (weight.detach().to(torch.float32) / scale).clamp_(
        -FP8_E4M3_MAX, FP8_E4M3_MAX
    )
    w_fp8 = w_scaled.to(WEIGHT_DTYPE_FP8)
    # Drop the fp32 working buffer immediately.
    del w_scaled, abs_max
    return w_fp8, scale


@torch.no_grad()
def quantize_linear_inplace(module: nn.Linear) -> None:
    """In-place FP8 conversion of an ``nn.Linear``.

    Replaces ``module.weight`` with a fp8 ``nn.Parameter`` and adds a
    ``weight_scale`` ``nn.Parameter`` of shape ``[out_features, 1]`` (fp32).
    Bias (if any) is left in its original dtype — vLLM applies bias after the
    fp8 matmul in higher precision.
    """
    w = module.weight
    if w.dtype == WEIGHT_DTYPE_FP8:
        return  # already done
    w_fp8, scale = fp8_quantize_per_channel(w.data, channel_dim=0)

    # Replace the parameter in place. We use a fresh nn.Parameter so HF's
    # ``save_pretrained`` picks it up via the standard state_dict path.
    new_weight = nn.Parameter(w_fp8, requires_grad=False)
    module.weight = new_weight  # drops the bf16 ref

    # Scale lives as a parameter so ``state_dict()`` includes it. compressed_tensors
    # registers it as a Parameter too (see initialize.py:245).
    module.register_parameter(
        "weight_scale",
        nn.Parameter(scale, requires_grad=False),
    )

    # Tag the module so we can collect schemes later for the config.
    module._is_fp8_dynamic = True


@torch.no_grad()
def quantize_fused_experts_inplace(experts_module: nn.Module) -> None:
    """In-place FP8 conversion of ``Qwen3_5MoeExperts``.

    The module holds two ``nn.Parameter`` 3D tensors:
      - ``gate_up_proj``: ``[num_experts, 2 * intermediate, hidden]``
      - ``down_proj``:    ``[num_experts, hidden, intermediate]``

    Both are quantized per-expert per-output-channel along the input-feature
    axis (dim 2 of the 3D tensor). Result:
      - ``gate_up_proj``: fp8, shape unchanged
      - ``gate_up_proj_scale``: fp32, shape ``[num_experts, 2*intermediate, 1]``
      - ``down_proj``: fp8, shape unchanged
      - ``down_proj_scale``: fp32, shape ``[num_experts, hidden, 1]``

    NOTE: At SAVE time these 3D tensors are sliced into per-expert
    ``experts.<i>.gate_proj`` / ``up_proj`` / ``down_proj`` keys (the canonical
    compressed-tensors naming, see
    ``compressed_tensors/utils/match.py`` lines 240-260). We do NOT keep them
    fused on disk because vLLM's compressed-tensors loader expects the unfused
    layout for FP8_DYNAMIC MoE.

    Memory: we work *per-expert* on slices to limit transient buffers to one
    expert's worth (~4 MB for gate_up, ~2 MB for down) rather than materializing
    a full-tensor fp32 conversion (~2 GB) for an entire layer.
    """
    for proj_name in ("gate_up_proj", "down_proj"):
        if not hasattr(experts_module, proj_name):
            continue
        param = getattr(experts_module, proj_name)
        if param.dtype == WEIGHT_DTYPE_FP8:
            continue
        E, O, I = param.shape  # noqa: E741 -- I is intentional
        # Allocate destinations. fp8 is 1 byte/elem; scale is 4 bytes/elem.
        dst_fp8 = torch.empty(E, O, I, dtype=WEIGHT_DTYPE_FP8, device=param.device)
        dst_scale = torch.empty(E, O, 1, dtype=SCALE_DTYPE, device=param.device)
        for e in range(E):
            slc = param.data[e]  # bf16 view, shape [O, I]
            w_fp8, scale = fp8_quantize_per_channel(slc, channel_dim=0)
            dst_fp8[e].copy_(w_fp8)
            dst_scale[e].copy_(scale)
            # The fp32 working buffer inside fp8_quantize_per_channel goes out
            # of scope here, but be explicit:
            del w_fp8, scale
        # Replace the param with the fp8 version. Drop the bf16 storage.
        new_param = nn.Parameter(dst_fp8, requires_grad=False)
        setattr(experts_module, proj_name, new_param)
        # Scale registered as a Parameter so it makes it into the state_dict.
        experts_module.register_parameter(
            f"{proj_name}_scale",
            nn.Parameter(dst_scale, requires_grad=False),
        )

    experts_module._is_fp8_dynamic_fused = True


# ---------------------------------------------------------------------------
# Module classification: what to quantize / what to skip.
# ---------------------------------------------------------------------------

# Names that must NEVER be quantized:
SKIP_NAME_REGEXES = [
    re.compile(r"^lm_head$"),
    re.compile(r".*\.mlp\.gate$"),                 # MoE router (Qwen3_5MoeTopKRouter)
    re.compile(r".*shared_expert_gate$"),          # sigmoid gate
    re.compile(r".*\.linear_attn(\..*)?$"),        # Gated DeltaNet block + inner
    re.compile(r"^model\.embed_tokens$"),
    re.compile(r"^model\.norm$"),
]


def should_quantize(name: str, module: nn.Module) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    for rx in SKIP_NAME_REGEXES:
        if rx.match(name):
            return False
    # Inside any linear_attn block? Skip.
    if ".linear_attn." in name:
        return False
    # Belt-and-suspenders: skip the multimodal vision module if present.
    if name.startswith("visual.") or ".visual." in name:
        return False
    return True


# ---------------------------------------------------------------------------
# Save path: write per-expert state dict, write config.json with quant_config.
# ---------------------------------------------------------------------------


def _build_state_dict_with_unfused_experts(model: nn.Module):
    """Yield ``(name, tensor)`` pairs for the on-disk state dict.

    Output naming targets vLLM's ``Qwen3_5MoeForConditionalGeneration``:
      * Every key is prefixed with ``language_model.`` because that class
        wraps the LM at ``self.language_model = Qwen3_5MoeForCausalLM(...)``.
      * MoE experts are kept as **fused 3D tensors** named
        ``...mlp.experts.gate_up_proj`` / ``mlp.experts.down_proj`` (plus
        ``..._weight_scale``). vLLM's ``hf_to_vllm_mapper`` (qwen3_5.py:310)
        expects exactly those substrings in order to flip its ``is_fused_expert``
        path on; once flipped, it loads the 3D tensor directly into the model's
        internal ``experts.w13_weight`` / ``experts.w2_weight``.
      * Everything else passes through with the prefix.

    NOTE: function name is kept for backwards compat with the rest of the
    script; it now produces FUSED experts (the prior unfused per-expert keys
    were wrong for this vLLM version).
    """
    PREFIX = "language_model."

    fused_modules: dict[str, nn.Module] = {}
    for name, mod in model.named_modules():
        if getattr(mod, "_is_fp8_dynamic_fused", False):
            fused_modules[name] = mod

    fused_module_ids = {id(m) for m in fused_modules.values()}

    # 1) Pass through every parameter / buffer that is NOT inside a fused
    #    experts module — under the language_model. prefix.
    for pname, p in model.named_parameters(remove_duplicate=False):
        owner = _resolve_module_for_param(model, pname)
        if owner is not None and id(owner) in fused_module_ids:
            continue
        yield PREFIX + pname, p.detach()

    for bname, b in model.named_buffers(remove_duplicate=False):
        owner = _resolve_module_for_param(model, bname)
        if owner is not None and id(owner) in fused_module_ids:
            continue
        yield PREFIX + bname, b.detach()

    # 2) For each fused experts module, emit fused 3D tensors with the names
    #    vLLM's stacked-params mapper recognizes.
    for parent_name, mod in fused_modules.items():
        gate_up = mod.gate_up_proj.detach()              # [E, 2*I, H]  fp8
        gate_up_scale = mod.gate_up_proj_scale.detach()  # [E, 2*I, 1]  fp32
        down = mod.down_proj.detach()                    # [E, H, I]   fp8
        down_scale = mod.down_proj_scale.detach()        # [E, H, 1]   fp32

        intermediate = down.shape[2]
        assert gate_up.shape[1] == 2 * intermediate, (
            f"gate_up_proj second dim {gate_up.shape[1]} != 2*intermediate "
            f"{2 * intermediate}"
        )

        base = PREFIX + parent_name
        yield f"{base}.gate_up_proj", gate_up.contiguous()
        yield f"{base}.gate_up_proj_weight_scale", gate_up_scale.contiguous()
        yield f"{base}.down_proj", down.contiguous()
        yield f"{base}.down_proj_weight_scale", down_scale.contiguous()


def _resolve_module_for_param(model: nn.Module, param_path: str) -> nn.Module | None:
    """Walk a dotted path to the *parent* module. Returns None on any miss."""
    parts = param_path.split(".")
    if len(parts) <= 1:
        return None
    obj = model
    for p in parts[:-1]:
        if not hasattr(obj, p):
            return None
        obj = getattr(obj, p)
    return obj if isinstance(obj, nn.Module) else None


def _save_state_dict_sharded(
    state_dict_iter, save_dir: Path, max_shard_bytes: int = 5 * 1024**3
) -> dict:
    """Write tensors to safetensors shards of ~5 GB and produce an index.

    We use ``safetensors.torch.save_file`` per shard. The state_dict iterator
    is consumed lazily so we never hold the full state dict in memory twice.
    """
    from safetensors.torch import save_file

    save_dir.mkdir(parents=True, exist_ok=True)
    shard_idx = 1
    shards: list[dict[str, torch.Tensor]] = [{}]
    shard_sizes = [0]

    weight_map: dict[str, str] = {}

    def _flush(idx: int):
        if not shards[idx - 1]:
            return
        # filename will be re-numbered after we know the total count.
        # For now use a temp name we can rename later.
        tmp = save_dir / f"_tmp_shard_{idx:05d}.safetensors"
        save_file(shards[idx - 1], str(tmp), metadata={"format": "pt"})
        shards[idx - 1] = {}  # drop refs

    for name, t in state_dict_iter:
        # tensor sizes — float8 is 1 byte/elem.
        nbytes = t.numel() * t.element_size()
        if shard_sizes[shard_idx - 1] + nbytes > max_shard_bytes and shards[shard_idx - 1]:
            _flush(shard_idx)
            shard_idx += 1
            shards.append({})
            shard_sizes.append(0)
        # Move to CPU & contiguous before save (safetensors requires contiguous).
        t_cpu = t.detach().cpu().contiguous()
        shards[shard_idx - 1][name] = t_cpu
        shard_sizes[shard_idx - 1] += nbytes
        weight_map[name] = f"_tmp_shard_{shard_idx:05d}.safetensors"

    _flush(shard_idx)
    total_shards = shard_idx

    # Now rename shards from `_tmp_shard_NNNNN` to the canonical
    # `model-NNNNN-of-MMMMM.safetensors` and update weight_map.
    rename_map = {}
    for i in range(1, total_shards + 1):
        old = save_dir / f"_tmp_shard_{i:05d}.safetensors"
        new_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        new = save_dir / new_name
        old.rename(new)
        rename_map[f"_tmp_shard_{i:05d}.safetensors"] = new_name
    weight_map = {k: rename_map[v] for k, v in weight_map.items()}

    index = {
        "metadata": {"total_size": sum(shard_sizes)},
        "weight_map": weight_map,
    }
    with (save_dir / "model.safetensors.index.json").open("w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
    return index


def _build_quantization_config_dict(model: nn.Module) -> dict:
    """Build the ``quantization_config`` JSON in the exact shape compressed_tensors
    writes (see ``model_compressors/model_compressor.py`` ``update_config``).

    For our scheme — FP8_DYNAMIC, weights CHANNEL, activations TOKEN dynamic —
    we hard-code the structure rather than calling QuantizationConfig because:
      1. We never attached ``quantization_scheme`` attributes to the modules
         (we sidestepped the lifecycle to keep memory low).
      2. Hard-coding lets us guarantee bit-for-bit the same JSON shape vLLM
         already accepts in production FP8_DYNAMIC checkpoints.
    """
    import compressed_tensors

    # Walk the model and build the ignore list (everything that wasn't quantized
    # but COULD have been — vLLM uses this to know what stays bf16). All names
    # get the ``language_model.`` prefix because that's how vLLM's
    # ``Qwen3_5MoeForConditionalGeneration`` exposes the LM submodule.
    PREFIX = "language_model."
    skipped_linear_names: list[str] = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            if not getattr(mod, "_is_fp8_dynamic", False):
                skipped_linear_names.append(PREFIX + name)

    config_groups = {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 8,
                "type": "float",
                "strategy": "channel",
                "symmetric": True,
                "dynamic": False,
                "observer": "minmax",
            },
            "input_activations": {
                "num_bits": 8,
                "type": "float",
                "strategy": "token",
                "symmetric": True,
                "dynamic": True,
                "observer": None,
            },
            "output_activations": None,
            "format": None,
        }
    }

    return {
        "version": getattr(compressed_tensors, "__version__", "0.15.1"),
        "quant_method": "compressed-tensors",
        "sparsity_config": {},
        "transform_config": {},
        "config_groups": config_groups,
        "format": "float-quantized",
        "kv_cache_scheme": None,
        "global_compression_ratio": None,
        "ignore": sorted(set(skipped_linear_names)),
        "quantization_status": "compressed",
    }


def save_fp8_checkpoint(
    model: nn.Module,
    tokenizer,
    save_dir: str,
) -> None:
    """Atomic save: write everything to ``<save_dir>.tmp.<pid>`` first, then
    rename the directory into place. If a previous final dir exists we refuse
    rather than risk mixing stale shards with new ones.

    Pre-flight defensive checks:
      - the built ``quantization_config`` dict must round-trip through
        ``compressed_tensors.QuantizationConfig.model_validate`` so any schema
        defect surfaces before we write a checkpoint vLLM would reject.
    """
    import shutil

    final_path = Path(save_dir)
    if final_path.exists() and any(final_path.iterdir()):
        raise RuntimeError(
            f"refusing to write into non-empty {final_path}. Remove it or "
            "choose a fresh SAVE_DIR — mixing shards from a prior partial run "
            "with new ones produces silently inconsistent checkpoints."
        )

    tmp_path = final_path.with_name(f"{final_path.name}.tmp.{os.getpid()}")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True)

    try:
        print(f"[save] writing safetensors shards to {tmp_path} ...", flush=True)
        _save_state_dict_sharded(
            _build_state_dict_with_unfused_experts(model),
            tmp_path,
        )

        tokenizer.save_pretrained(tmp_path)

        # Use the SOURCE repo's full multimodal config as the base (it has
        # ``vision_config``, ``image_token_id``, etc. that vLLM's renderer
        # builder requires for ``Qwen3_5MoeForConditionalGeneration``).
        # Loading via from_pretrained gives only the inner text config when
        # ``AutoModelForCausalLM`` is used; we want the outer multimodal one.
        from huggingface_hub import hf_hub_download
        src_cfg_path = hf_hub_download(MODEL_ID, "config.json")
        with open(src_cfg_path) as f:
            cfg = json.load(f)
        cfg["quantization_config"] = _build_quantization_config_dict(model)
        # vLLM's compressed-tensors / hf_to_vllm_mapper path is implemented for
        # the multimodal class only (text-only `Qwen3_5MoeForCausalLM` isn't
        # registered in this image). `--language-model-only` at serve time
        # skips the vision tower at load.
        cfg["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]

        # Validate the quantization_config against compressed-tensors' own
        # parser before writing anything to disk that vLLM would reject.
        from compressed_tensors.quantization import QuantizationConfig
        try:
            QuantizationConfig.model_validate(cfg["quantization_config"])
        except Exception as exc:
            raise RuntimeError(
                f"built quantization_config does not validate against "
                f"compressed_tensors.QuantizationConfig: {exc}"
            ) from exc

        with (tmp_path / "config.json").open("w") as f:
            json.dump(cfg, f, indent=2, sort_keys=True)

        if getattr(model, "generation_config", None) is not None:
            try:
                model.generation_config.save_pretrained(tmp_path)
            except Exception as exc:  # pragma: no cover — non-fatal
                print(f"[save] warning: could not save generation_config: {exc}")

        # Atomic rename. Path.rename is atomic within the same filesystem.
        if final_path.exists():
            shutil.rmtree(final_path)
        tmp_path.rename(final_path)
        print(f"[save] done. listing {final_path} ...", flush=True)
    except Exception:
        # On any failure, leave the temp dir for inspection but make it
        # obvious it's not a complete checkpoint.
        bad = tmp_path.with_name(f"{tmp_path.name}.FAILED")
        try:
            tmp_path.rename(bad)
            print(
                f"[save] FAILED — partial output preserved at {bad} for inspection.",
                flush=True,
            )
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Sanity-test helpers (importable, used in __main__ block when invoked with
# --selftest).
# ---------------------------------------------------------------------------


def _selftest() -> None:
    """Synthetic-tensor sanity check. Run with: python <this file> --selftest"""
    print("[selftest] building toy module ...", flush=True)

    class FakeExperts(nn.Module):
        def __init__(self, E=4, O=16, I_=8):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.randn(E, 2 * I_, O) * 0.1)
            # down_proj: [E, hidden=O, intermediate=I_]
            self.down_proj = nn.Parameter(torch.randn(E, O, I_) * 0.1)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(32, 32, bias=False)
            self.experts = FakeExperts()

    toy = Toy().to(torch.bfloat16)

    # Snapshot originals for accuracy check.
    orig_q = toy.q_proj.weight.detach().clone().to(torch.float32)
    orig_gu = toy.experts.gate_up_proj.detach().clone().to(torch.float32)
    orig_dp = toy.experts.down_proj.detach().clone().to(torch.float32)

    pre_rss = rss_gib()

    quantize_linear_inplace(toy.q_proj)
    quantize_fused_experts_inplace(toy.experts)

    post_rss = rss_gib()

    # Dtype assertions.
    assert toy.q_proj.weight.dtype == WEIGHT_DTYPE_FP8, (
        f"weight dtype = {toy.q_proj.weight.dtype}"
    )
    assert toy.q_proj.weight_scale.dtype == SCALE_DTYPE, (
        f"weight_scale dtype = {toy.q_proj.weight_scale.dtype}"
    )
    assert toy.q_proj.weight_scale.shape == (32, 1), (
        f"weight_scale shape = {toy.q_proj.weight_scale.shape}"
    )
    assert toy.experts.gate_up_proj.dtype == WEIGHT_DTYPE_FP8
    assert toy.experts.gate_up_proj_scale.shape == (4, 16, 1), (
        f"gate_up_proj_scale shape = {toy.experts.gate_up_proj_scale.shape}"
    )
    assert toy.experts.down_proj.dtype == WEIGHT_DTYPE_FP8
    assert toy.experts.down_proj_scale.shape == (4, 16, 1), (
        f"down_proj_scale shape = {toy.experts.down_proj_scale.shape}"
    )

    # Dequantization accuracy check (3% rel error on per-channel).
    dq_q = toy.q_proj.weight.detach().to(torch.float32) * toy.q_proj.weight_scale.detach()
    rel = (dq_q - orig_q).abs() / (orig_q.abs() + 1e-6)
    assert rel.mean().item() < 0.05, (
        f"q_proj rel err mean too high: {rel.mean().item():.4f}"
    )

    dq_gu = (
        toy.experts.gate_up_proj.detach().to(torch.float32)
        * toy.experts.gate_up_proj_scale.detach()
    )
    rel_gu = (dq_gu - orig_gu).abs() / (orig_gu.abs() + 1e-6)
    assert rel_gu.mean().item() < 0.05, (
        f"gate_up rel err mean too high: {rel_gu.mean().item():.4f}"
    )

    dq_dp = (
        toy.experts.down_proj.detach().to(torch.float32)
        * toy.experts.down_proj_scale.detach()
    )
    rel_dp = (dq_dp - orig_dp).abs() / (orig_dp.abs() + 1e-6)
    assert rel_dp.mean().item() < 0.05

    # State-dict shape check via the (now FUSED + prefixed) iterator.
    sd = dict(_build_state_dict_with_unfused_experts(toy))
    PREFIX = "language_model."
    expected_keys = {
        PREFIX + "q_proj.weight",
        PREFIX + "q_proj.weight_scale",
        PREFIX + "experts.gate_up_proj",
        PREFIX + "experts.gate_up_proj_weight_scale",
        PREFIX + "experts.down_proj",
        PREFIX + "experts.down_proj_weight_scale",
    }
    missing = expected_keys - set(sd.keys())
    assert not missing, f"state dict missing keys: {missing}"

    # Validate fused 3D shapes.
    gu_w = sd[PREFIX + "experts.gate_up_proj"]
    gu_s = sd[PREFIX + "experts.gate_up_proj_weight_scale"]
    dp_w = sd[PREFIX + "experts.down_proj"]
    dp_s = sd[PREFIX + "experts.down_proj_weight_scale"]
    # Toy uses E=4, O(=hidden)=16, I_(=intermediate)=8.
    # gate_up_proj: [E, 2*I_, O] = [4, 16, 16]
    # down_proj:    [E, O, I_]   = [4, 16, 8]
    assert gu_w.shape == (4, 16, 16), f"gate_up_proj fused shape wrong: {gu_w.shape}"
    assert gu_s.shape == (4, 16, 1), f"gate_up_proj scale shape wrong: {gu_s.shape}"
    assert dp_w.shape == (4, 16, 8), f"down_proj fused shape wrong: {dp_w.shape}"
    assert dp_s.shape == (4, 16, 1), f"down_proj scale shape wrong: {dp_s.shape}"

    # Dequantize one expert's gate_up via the saved 3D tensor and check
    # round-trip accuracy against the original fp32.
    fused_back = gu_w[0].to(torch.float32) * gu_s[0].to(torch.float32)
    rel_fb = (fused_back - orig_gu[0]).abs() / (orig_gu[0].abs() + 1e-6)
    assert rel_fb.mean().item() < 0.05, "round-trip fused gate_up error too high"

    # Memory should not have *grown* — for a toy this is dominated by Python
    # overhead so we just sanity-check no major growth.
    assert post_rss <= pre_rss + 0.5, (
        f"toy quant grew RSS by {post_rss - pre_rss:.2f} GiB (unexpected)"
    )

    print(
        f"[selftest] OK. Linear weight dtype={toy.q_proj.weight.dtype}, "
        f"scale dtype={toy.q_proj.weight_scale.dtype}, "
        f"gate_up scale shape={tuple(toy.experts.gate_up_proj_scale.shape)}, "
        f"down scale shape={tuple(toy.experts.down_proj_scale.shape)}"
    )


# ---------------------------------------------------------------------------
# Main quantization driver.
# ---------------------------------------------------------------------------


def _strip_module(parent, attr_name):
    """Remove a child module and force-free its tensors immediately."""
    if hasattr(parent, attr_name):
        child = getattr(parent, attr_name)
        for p in child.parameters():
            p.data = torch.empty(0, dtype=p.dtype, device=p.device)
        delattr(parent, attr_name)
        print(f"  stripped {parent.__class__.__name__}.{attr_name}")


def main() -> None:
    if not SAVE_DIR:
        raise SystemExit(
            "set SAVE_DIR (output directory for the FP8 checkpoint) env var; "
            "optionally MODEL_ID / HF_CACHE."
        )
    log_mem("startup")

    # device_map="cpu" on DGX Spark unified memory: GPU and CPU share physical
    # RAM, so this is NOT slower than "auto" — it just stops accelerate from
    # creating a parallel "GPU" mapping that gets accounted twice (once on the
    # heap, once via the safetensors mmap page cache the GPU loader pulls in).
    # Our quantization is pure tensor math (max-abs reduce + cast); CPU is fine.
    print(f"Loading {MODEL_ID} (bf16, device_map='cpu', low_cpu_mem_usage=True) ...",
          flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    log_mem(f"after load (took {time.time() - t0:.1f}s)")
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  loaded {n_params:.1f} B params", flush=True)

    # Strip any vision / mtp modules that snuck in (defensive — text-only).
    inner = getattr(model, "model", model)
    for parent, attr in (
        (inner, "visual"),
        (inner, "vision_tower"),
        (model, "visual"),
        (model, "vision_tower"),
        (inner, "mtp"),
        (model, "mtp"),
    ):
        _strip_module(parent, attr)
    gc.collect()
    log_mem("after vision/mtp strip")
    assert_under_limit("after vision/mtp strip")

    # Identify the layer container.
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError(
            "Expected model.model.layers — this script targets Qwen3_5MoeForCausalLM."
        )
    layers: nn.ModuleList = model.model.layers
    n_layers = len(layers)
    print(f"  {n_layers} decoder layers", flush=True)

    # ------------------------------------------------------------------
    # Pass 1: Quantize "outer" Linears that aren't inside layers (just
    # in case there are any — typically only ``lm_head`` lives here and
    # we skip it). And also quantize any Linears in modules other than
    # decoder layers.
    # ------------------------------------------------------------------
    print("[quant] pass 1: top-level Linear modules ...", flush=True)
    n_quantized_outer = 0
    for name, mod in model.named_modules():
        if name.startswith("model.layers."):
            continue
        if should_quantize(name, mod):
            quantize_linear_inplace(mod)
            n_quantized_outer += 1
    print(f"  quantized {n_quantized_outer} top-level Linear modules", flush=True)

    # ------------------------------------------------------------------
    # Pass 2: Per-layer quantization. After each layer we gc + empty_cache.
    # ------------------------------------------------------------------
    print("[quant] pass 2: per-layer quantization ...", flush=True)
    for layer_idx in range(n_layers):
        layer = layers[layer_idx]
        layer_prefix = f"model.layers.{layer_idx}"

        # 2a) every Linear inside this layer.
        n_lin = 0
        for sub_name, sub_mod in layer.named_modules():
            full = f"{layer_prefix}.{sub_name}" if sub_name else layer_prefix
            if should_quantize(full, sub_mod):
                quantize_linear_inplace(sub_mod)
                n_lin += 1

        # 2b) fused experts.
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            quantize_fused_experts_inplace(layer.mlp.experts)

        # Force a clean break between layers.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log_mem(f"layer {layer_idx + 1:>2d}/{n_layers} ({n_lin} Linear quant'd)")
        assert_under_limit(f"layer {layer_idx + 1}")

    # ------------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------------
    log_mem("pre-save")
    save_fp8_checkpoint(model, tokenizer, SAVE_DIR)
    log_mem("post-save")

    print(f"\n[done] FP8 checkpoint at: {SAVE_DIR}")
    print("\nSuggested vLLM serve command:")
    print(
        f"  vllm serve {SAVE_DIR} \\\n"
        f"    --quantization compressed-tensors \\\n"
        f"    --kv-cache-dtype fp8_e4m3 \\\n"
        f"    --max-model-len 65536 \\\n"
        f"    --gpu-memory-utilization 0.85 \\\n"
        f"    --enforce-eager"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest()
        sys.exit(0)
    main()
