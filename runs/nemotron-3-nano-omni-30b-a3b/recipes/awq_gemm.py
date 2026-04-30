#!/usr/bin/env python3
"""Shard-by-shard data-free RTN quantization to AutoAWQ GEMM format.

Streams one safetensors shard at a time from the cached HF snapshot of
nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 and emits an AutoAWQ
GEMM-format checkpoint:

  - bits=4, group_size=128, version="gemm", zero_point=True
  - asymmetric per-group quant along in_features
  - 8x int4 packed into one int32 along out_features with the AWQ pack
    order [0, 4, 1, 5, 2, 6, 3, 7]
  - Nemotron MoE experts are stored UNFUSED, UNGATED in source shards
    (each routed expert is a NemotronHMLP with `up_proj` + `down_proj`
    only — relu² is un-gated). The recipe quantizes them as bare 2-D
    tensors; the qwen3 fused-3-D-split path is intentionally rejected.
  - skips Mamba2 inner Linears, attention, vision (RADIO), audio
    (Parakeet), shared expert, router gate, layer 0, embeds, lm_head,
    norms — copied through unchanged.
  - the multimodal wrapper class `NemotronH_Nano_Omni_Reasoning_V3` and
    its `auto_map` entries are preserved so vLLM with
    `--trust-remote-code` can serve the artifact for both text and
    multimodal inference (multimodal capability is preserved by keeping
    every encoder + projector dense).
"""

import json
import os
import shutil
import sys
from pathlib import Path

import psutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Local import: the AWQ skip policy lives in _classify so the recipe and
# the Phase 1 test exercise the same code path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _classify import should_quantize  # noqa: E402


# Source and destination configured via env vars — keeps the recipe portable.
SOURCE_DIR = Path(os.environ.get("SRC_DIR", ""))
DST_DIR = Path(os.environ.get("DST_DIR", ""))

MAX_SHARD_BYTES = 4 * 1024**3

BITS = 4
# Nemotron's `moe_intermediate_size = 1856 = 64 * 29` is NOT divisible by 128
# (the qwen3 default), so down_proj's in_features=1856 forces group_size=64.
# 64 works for every quantizable shape: 1856 % 64 == 0 and 2688 % 64 == 0.
# vLLM's AWQ-GEMM kernel supports gs ∈ {32, 64, 128}.
GROUP_SIZE = 64
ZERO_POINT = True
# Nemotron has 128 routed experts (vs Qwen3's 256). Used only as a defensive
# assertion: if a 3-D fused-expert tensor unexpectedly appears the recipe
# fails loud.
NUM_EXPERTS = 128

# Canonical AWQ GEMM pack order. Matches AutoAWQ's pack_intweight and
# vLLM's reverse_awq_pack_order in moe_wna16.py.
AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

# `modules_to_not_convert` is read by vLLM's loader to decide which
# weights to dequantize at load time. It MUST agree with the recipe's
# own classification (`_classify.should_quantize`) — otherwise vLLM will
# try to load a packed AWQ tensor as fp16 (or vice versa) and crash.
QUANTIZATION_CONFIG = {
    "quant_method": "awq",
    "bits": 4,
    "group_size": 64,
    "version": "gemm",
    "zero_point": True,
    "modules_to_not_convert": [
        # heads / embeddings / norms
        "lm_head", "embed_tokens", "embedding",
        "norm", "layernorm", "rmsnorm",
        # vision encoder + bridge
        "vision", "radio", "vision_model", "vision_tower", "image_proj",
        "video",
        # audio encoder + bridge
        "sound", "audio", "parakeet", "audio_encoder",
        # multimodal bridges (sound_projection.linear1/2)
        "projector", "projection",
        # vision projector (top-level nn.Sequential — RADIO→LM bridge; mlp1.{1,3}.weight)
        "mlp1",
        # mamba2 SSM block — every inner Linear/Conv1d kept dense
        "mamba", "ssm", "in_proj", "out_proj", "dt_proj", "conv1d",
        # transformer attention block — keep all 6 attention layers dense
        "self_attn", "q_proj", "k_proj", "v_proj", "o_proj",
        # always-on shared expert
        "shared_expert",
        # router gates (Nemotron uses mixer.gate; mlp.gate is precedent)
        "mixer.gate", "mlp.gate",
        # layer-0 conservative preservation
        ".layers.0.",
    ],
}


def _rss_gib() -> float:
    return psutil.Process().memory_info().rss / 1024**3


def quantize_2d(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RTN AWQ-GEMM quantize a single [out, in] weight.

    Returns (qweight, qzeros, scales) in AWQ GEMM layout:
        qweight: int32 [in, out // 8]
        qzeros : int32 [in // GROUP_SIZE, out // 8]
        scales : fp16  [in // GROUP_SIZE, out]
    """
    assert w.ndim == 2, w.shape
    out_features, in_features = w.shape
    assert in_features % GROUP_SIZE == 0, (
        f"in_features={in_features} not divisible by group_size={GROUP_SIZE}"
    )
    assert out_features % 8 == 0, (
        f"out_features={out_features} not divisible by 8"
    )
    qmax = (1 << BITS) - 1  # 15
    n_groups = in_features // GROUP_SIZE

    # Transpose to [in, out] and group along in.
    wt = w.to(torch.float32).t().contiguous()  # [in, out]
    wg = wt.view(n_groups, GROUP_SIZE, out_features)  # [G, gs, out]

    min_v = wg.min(dim=1).values  # [G, out]
    max_v = wg.max(dim=1).values  # [G, out]
    scales_f32 = (max_v - min_v) / qmax
    # Avoid division-by-zero on dead/constant groups.
    scales_safe = torch.where(scales_f32 == 0, torch.ones_like(scales_f32), scales_f32)
    scales_fp16 = scales_f32.to(torch.float16)

    # Cast scales back to fp32 (rounded through fp16) for the actual quant
    # math so the saved scales exactly reproduce dequant.
    scales_used = scales_fp16.to(torch.float32)
    scales_used_safe = torch.where(
        scales_used == 0, torch.ones_like(scales_used), scales_used
    )

    zp_f = torch.round(-min_v / scales_safe)
    zp_i = zp_f.clamp(0, qmax).to(torch.int32)  # [G, out]

    # q = round(w / scale) + zp, clipped to [0, qmax]
    q = torch.round(wg / scales_used_safe.unsqueeze(1)) + zp_i.unsqueeze(1).to(
        torch.float32
    )
    q = q.clamp(0, qmax).to(torch.int32)  # [G, gs, out]
    q = q.view(in_features, out_features)  # [in, out]

    # Pack 8 int4 columns into one int32 along out, with AWQ pack order.
    qweight_packed = _pack_int4_along_last(q)  # [in, out//8]
    qzeros_packed = _pack_int4_along_last(zp_i)  # [G, out//8]

    return qweight_packed, qzeros_packed, scales_fp16


def _pack_int4_along_last(x: torch.Tensor) -> torch.Tensor:
    """Pack 8 nibble ints along the last dim into int32 with AWQ pack order."""
    assert x.dtype in (torch.int32, torch.int64), x.dtype
    last = x.shape[-1]
    assert last % 8 == 0, last
    x = x.to(torch.int32) & 0xF
    new_last = last // 8
    out = torch.zeros(*x.shape[:-1], new_last, dtype=torch.int32)
    x_grp = x.view(*x.shape[:-1], new_last, 8)  # ..., new_last, 8
    for i in range(8):
        out |= x_grp[..., i] << (4 * AWQ_PACK_ORDER[i])
    return out


def _unpack_int4_along_last(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of _pack_int4_along_last (used by roundtrip_test)."""
    last = packed.shape[-1]
    new_last = last * 8
    out = torch.zeros(*packed.shape[:-1], new_last, dtype=torch.int32)
    out_grp = out.view(*packed.shape[:-1], last, 8)
    for i in range(8):
        out_grp[..., i] = (packed >> (4 * AWQ_PACK_ORDER[i])) & 0xF
    return out


def quantize_and_emit(name: str, t: torch.Tensor, sink) -> int:
    """Quantize one source tensor and feed (key, tensor) pairs into `sink`.

    Returns the number of output tensors emitted (always 3 for Nemotron's
    unfused 2-D experts: qweight + qzeros + scales).
    """
    # Nemotron experts are unfused 2-D — every quantizable tensor here is
    # a regular Linear weight that ends with `.weight`.  If a 3-D tensor
    # ever appears we want to fail loud, not silently mis-quantize.
    assert t.ndim == 2, (
        f"unexpected tensor rank {t.ndim} for {name}; Nemotron experts are "
        f"stored unfused as 2-D — recipe needs review"
    )

    # Strip `.weight` suffix to get the module path. Routed experts always
    # have it (`...experts.<i>.up_proj.weight`).
    base = name[: -len(".weight")] if name.endswith(".weight") else name

    qw, qz, sc = quantize_2d(t)
    sink(base + ".qweight", qw)
    sink(base + ".qzeros", qz)
    sink(base + ".scales", sc)
    return 3


def roundtrip_test():
    """Pack/unpack roundtrip + dequant sanity check on a tiny tensor."""
    torch.manual_seed(0)
    in_f, out_f = 128, 16  # one group, one packed lane
    w = torch.randn(out_f, in_f, dtype=torch.float32) * 0.1
    qw, qz, sc = quantize_2d(w)
    assert qw.shape == (in_f, out_f // 8) and qw.dtype == torch.int32
    assert qz.shape == (in_f // GROUP_SIZE, out_f // 8) and qz.dtype == torch.int32
    assert sc.shape == (in_f // GROUP_SIZE, out_f) and sc.dtype == torch.float16

    qw_int = _unpack_int4_along_last(qw)
    qz_int = _unpack_int4_along_last(qz)
    G = in_f // GROUP_SIZE
    qw_g = qw_int.view(G, GROUP_SIZE, out_f).to(torch.float32)
    deq = (qw_g - qz_int.unsqueeze(1).to(torch.float32)) * sc.unsqueeze(1).to(
        torch.float32
    )
    deq = deq.view(in_f, out_f).t()
    err = (deq - w).abs().max().item()
    assert err < 0.05, f"roundtrip max-err={err}"
    print(f"[awq] roundtrip OK (max-err={err:.4g})", flush=True)


# Aux files copied verbatim from source into the artifact dir.  All custom
# `.py` files are picked up via Path.glob('*.py') so a future-added module
# (e.g. another `*_processing.py`) is automatically included.
AUX_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
)


def main() -> None:
    if not str(SOURCE_DIR) or not str(DST_DIR):
        raise SystemExit(
            "set SRC_DIR (local source snapshot dir) and DST_DIR (output dir) env vars"
        )
    assert SOURCE_DIR.exists(), f"missing {SOURCE_DIR}"
    if DST_DIR.exists() and any(DST_DIR.iterdir()):
        raise RuntimeError(f"refusing to write into non-empty {DST_DIR}")
    DST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[awq] start RSS={_rss_gib():.2f} GiB", flush=True)

    src_index = json.load(open(SOURCE_DIR / "model.safetensors.index.json"))
    weight_map_in: dict[str, str] = src_index["weight_map"]

    by_shard: dict[str, list[str]] = {}
    for k, fn in weight_map_in.items():
        by_shard.setdefault(fn, []).append(k)
    shard_files = sorted(by_shard.keys())
    print(f"[awq] {len(shard_files)} source shards", flush=True)

    weight_map_out: dict[str, str] = {}
    out_idx = 1
    out_buffer: dict[str, torch.Tensor] = {}
    out_size = [0]

    n_quantized = 0
    n_copied = 0

    def flush(idx: int):
        if not out_buffer:
            return
        tmp = DST_DIR / f"_tmp_{idx:05d}.safetensors"
        save_file(out_buffer, str(tmp), metadata={"format": "pt"})
        gib = sum(t.numel() * t.element_size() for t in out_buffer.values()) / 1024**3
        print(
            f"[awq] wrote {tmp.name} ({len(out_buffer)} keys, {gib:.2f} GiB)",
            flush=True,
        )
        out_buffer.clear()
        out_size[0] = 0

    def sink(key: str, tensor: torch.Tensor):
        nonlocal out_idx
        tensor = tensor.contiguous()
        nb = tensor.numel() * tensor.element_size()
        if out_size[0] + nb > MAX_SHARD_BYTES and out_buffer:
            flush(out_idx)
            out_idx += 1
        out_buffer[key] = tensor
        out_size[0] += nb
        weight_map_out[key] = f"_tmp_{out_idx:05d}.safetensors"

    for src_name in shard_files:
        src_path = SOURCE_DIR / src_name
        with safe_open(str(src_path), framework="pt") as f:
            keys = list(f.keys())
            print(
                f"[awq] reading {src_name} ({len(keys)} tensors) "
                f"RSS={_rss_gib():.2f} GiB",
                flush=True,
            )
            for k in keys:
                t = f.get_tensor(k)
                if should_quantize(k, t.shape):
                    quantize_and_emit(k, t, sink)
                    n_quantized += 1
                    del t
                else:
                    sink(k, t)
                    n_copied += 1
        print(
            f"[awq] done {src_name}: nq={n_quantized}, nc={n_copied}, "
            f"RSS={_rss_gib():.2f} GiB",
            flush=True,
        )
    flush(out_idx)
    total_shards = out_idx

    # Atomic rename: _tmp_XXXXX.safetensors -> model-XXXXX-of-NNNNN.safetensors
    rename_map: dict[str, str] = {}
    for i in range(1, total_shards + 1):
        old = DST_DIR / f"_tmp_{i:05d}.safetensors"
        new_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        old.rename(DST_DIR / new_name)
        rename_map[f"_tmp_{i:05d}.safetensors"] = new_name
    weight_map_out = {k: rename_map[v] for k, v in weight_map_out.items()}

    total_size = sum(
        (DST_DIR / s).stat().st_size for s in set(weight_map_out.values())
    )
    json.dump(
        {"metadata": {"total_size": total_size}, "weight_map": weight_map_out},
        open(DST_DIR / "model.safetensors.index.json", "w"),
        indent=2,
        sort_keys=True,
    )

    # Pass-through aux files.
    for fname in AUX_FILES:
        src = SOURCE_DIR / fname
        if src.exists():
            shutil.copy2(src, DST_DIR / fname)
            print(f"[awq] copied {fname}", flush=True)

    # Pass-through every custom .py (modeling, configuration, processing,
    # audio_model, video_io, evs, ...). vLLM imports these at load time
    # via trust_remote_code.
    py_count = 0
    for src_py in sorted(SOURCE_DIR.glob("*.py")):
        shutil.copy2(src_py, DST_DIR / src_py.name)
        py_count += 1
    print(f"[awq] copied {py_count} custom .py files", flush=True)

    # config.json: source config + quantization_config. Architectures kept
    # at the multimodal wrapper class so vLLM serves the full pipeline.
    src_cfg = json.load(open(SOURCE_DIR / "config.json"))
    out_cfg = dict(src_cfg)
    out_cfg["quantization_config"] = dict(QUANTIZATION_CONFIG)
    out_cfg["architectures"] = ["NemotronH_Nano_Omni_Reasoning_V3"]
    json.dump(
        out_cfg,
        open(DST_DIR / "config.json", "w"),
        indent=2,
        sort_keys=True,
    )
    print(
        f"[awq] config.json written: arch={out_cfg['architectures']}, "
        f"model_type={out_cfg.get('model_type')}",
        flush=True,
    )

    files = sorted(DST_DIR.iterdir())
    total_bytes = sum(p.stat().st_size for p in files if p.is_file())
    print(
        f"[awq] DONE. quantized={n_quantized} tensors, copied={n_copied} tensors, "
        f"shards={total_shards}, total={total_bytes / 1024**3:.2f} GiB, "
        f"final RSS={_rss_gib():.2f} GiB",
        flush=True,
    )


if __name__ == "__main__":
    roundtrip_test()
    main()
