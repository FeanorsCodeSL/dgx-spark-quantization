#!/usr/bin/env python3
"""Shard-by-shard data-free RTN quantization to AutoAWQ GEMM format.

Streams one safetensors shard at a time from the cached HF snapshot of
Qwen3.6-35B-A3B (Qwen3.5-MoE arch) and emits an AutoAWQ GEMM-format
checkpoint that mirrors QuantTrio/Qwen3.6-35B-A3B-AWQ exactly:

  - bits=4, group_size=128, version="gemm", zero_point=True
  - asymmetric per-group quant along in_features
  - 8x int4 packed into one int32 along out_features with the AWQ pack
    order [0, 4, 1, 5, 2, 6, 3, 7] (see vLLM's
    `vllm/model_executor/layers/quantization/moe_wna16.py`
    `reverse_awq_pack_order` — packing reuses the same indices)
  - 3D fused-expert tensors (gate_up_proj, down_proj) are split into
    per-expert keys; gate_up_proj is further split into gate_proj+up_proj
    (gate = first half along out_features, up = second half).
  - skips visual / linear_attn / self_attn / shared_expert / mlp.gate
    routers / layer 0 / mtp / norms / embeds / lm_head, copying them
    through unchanged.

Output config.json is built from the source config with
`quantization_config` added (matching QuantTrio's exactly).
"""

import json
import os
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# Source and destination are configured via env vars so this script is reusable
# on any machine. Point SRC_DIR at a *local snapshot directory* (not an HF id —
# we read shards directly with safetensors).
#
# Example:
#   SRC_DIR=/path/to/hf-cache/hub/models--<org>--<name>/snapshots/<sha>/ \
#   DST_DIR=/path/to/output/<name>-awq \
#   python quantize/awq_gemm.py
SOURCE_DIR = Path(os.environ.get("SRC_DIR", ""))
DST_DIR = Path(os.environ.get("DST_DIR", ""))

MAX_SHARD_BYTES = 4 * 1024**3

BITS = 4
GROUP_SIZE = 128
ZERO_POINT = True
NUM_EXPERTS = 256

# Canonical AWQ GEMM pack order. Matches AutoAWQ's pack_intweight and
# vLLM's reverse_awq_pack_order in moe_wna16.py.
AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

QUANTIZATION_CONFIG = {
    "quant_method": "awq",
    "bits": 4,
    "group_size": 128,
    "version": "gemm",
    "zero_point": True,
    "modules_to_not_convert": [
        "visual",
        "linear_attn",
        "self_attn",
        "shared_expert",
        "mlp.gate",
        "model.layers.0.",
        "mtp",
    ],
}

LAYER0_RE = re.compile(r".*\.layers\.0\..*")
LMHEAD_RE = re.compile(r".*lm_head.*")


def should_quantize(name: str, shape: torch.Size) -> bool:
    if len(shape) not in (2, 3):
        return False
    # Qwen3.5-MoE saves fused experts as bare parameters
    # (e.g. `...experts.gate_up_proj`) WITHOUT a `.weight` suffix; regular
    # Linear weights have `.weight`. Accept both.
    if not (
        name.endswith(".weight")
        or name.endswith(".gate_up_proj")
        or name.endswith(".down_proj")
    ):
        return False
    if LMHEAD_RE.match(name):
        return False
    if "visual" in name:
        return False
    if "linear_attn" in name:
        return False
    if "self_attn" in name:
        return False
    if "shared_expert" in name:
        return False
    # Router gate (mlp.gate.weight). Don't conflate with mlp.gate_proj or
    # mlp.gate_up_proj — must end with `mlp.gate.weight` exactly.
    if name.endswith("mlp.gate.weight"):
        return False
    if LAYER0_RE.match(name):
        return False
    if "mtp" in name:
        return False
    if "norm" in name or "embed_tokens" in name or "layernorm" in name:
        return False
    return True


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
    """Pack 8 nibble ints along the last dim into int32 with AWQ pack order.

    Matches AutoRound's `export_to_awq/utils.py:247`: input lane `i` shifts to
    bit position `AWQ_PACK_ORDER[i] * 4`, i.e., input[i] -> nibble[perm[i]].
    This is the format vLLM's `convert_awq_tensor` (moe_wna16.py) expects.
    """
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
    """Inverse of _pack_int4_along_last (matches AutoRound's pack semantics)."""
    last = packed.shape[-1]
    new_last = last * 8
    out = torch.zeros(*packed.shape[:-1], new_last, dtype=torch.int32)
    out_grp = out.view(*packed.shape[:-1], last, 8)
    for i in range(8):
        out_grp[..., i] = (packed >> (4 * AWQ_PACK_ORDER[i])) & 0xF
    return out


def quantize_and_emit(name: str, t: torch.Tensor, sink) -> int:
    """Quantize one source tensor and feed (key, tensor) pairs into `sink`.

    Returns the number of output tensors emitted.
    """
    # Strip `.weight` suffix if present; Qwen3.5-MoE fused experts have no suffix.
    base = name[: -len(".weight")] if name.endswith(".weight") else name

    if t.ndim == 2:
        qw, qz, sc = quantize_2d(t)
        sink(base + ".qweight", qw)
        sink(base + ".qzeros", qz)
        sink(base + ".scales", sc)
        return 3

    # 3D fused-expert tensor: [E, out, in]
    assert t.ndim == 3
    E, out_f, in_f = t.shape
    assert E == NUM_EXPERTS, (name, t.shape)

    # Decide split. Two known shapes:
    #   gate_up_proj: out_f = 2 * moe_intermediate -> split into gate / up
    #   down_proj   : keep whole
    is_gate_up = base.endswith("gate_up_proj")
    is_down = base.endswith("down_proj")
    assert is_gate_up or is_down, f"unrecognized 3D expert tensor: {name}"

    parent = base.rsplit(".", 1)[0]  # ...mlp.experts
    emitted = 0
    if is_gate_up:
        assert out_f % 2 == 0, (name, t.shape)
        half = out_f // 2
        for e in range(E):
            we = t[e]  # [out_f, in_f]
            gate = we[:half, :].contiguous()
            up = we[half:, :].contiguous()
            qw, qz, sc = quantize_2d(gate)
            sink(f"{parent}.{e}.gate_proj.qweight", qw)
            sink(f"{parent}.{e}.gate_proj.qzeros", qz)
            sink(f"{parent}.{e}.gate_proj.scales", sc)
            qw, qz, sc = quantize_2d(up)
            sink(f"{parent}.{e}.up_proj.qweight", qw)
            sink(f"{parent}.{e}.up_proj.qzeros", qz)
            sink(f"{parent}.{e}.up_proj.scales", sc)
            emitted += 6
    else:  # down_proj
        for e in range(E):
            we = t[e].contiguous()
            qw, qz, sc = quantize_2d(we)
            sink(f"{parent}.{e}.down_proj.qweight", qw)
            sink(f"{parent}.{e}.down_proj.qzeros", qz)
            sink(f"{parent}.{e}.down_proj.scales", sc)
            emitted += 3
    return emitted


def roundtrip_test():
    """Pack/unpack roundtrip + dequant sanity check on a tiny tensor."""
    torch.manual_seed(0)
    in_f, out_f = 128, 16  # one group, one packed lane
    w = torch.randn(out_f, in_f, dtype=torch.float32) * 0.1
    qw, qz, sc = quantize_2d(w)
    assert qw.shape == (in_f, out_f // 8) and qw.dtype == torch.int32
    assert qz.shape == (in_f // GROUP_SIZE, out_f // 8) and qz.dtype == torch.int32
    assert sc.shape == (in_f // GROUP_SIZE, out_f) and sc.dtype == torch.float16

    qw_int = _unpack_int4_along_last(qw)  # [in, out]
    qz_int = _unpack_int4_along_last(qz)  # [G, out]
    G = in_f // GROUP_SIZE
    qw_g = qw_int.view(G, GROUP_SIZE, out_f).to(torch.float32)
    deq = (qw_g - qz_int.unsqueeze(1).to(torch.float32)) * sc.unsqueeze(1).to(
        torch.float32
    )
    deq = deq.view(in_f, out_f).t()  # back to [out, in]
    err = (deq - w).abs().max().item()
    # Per-group asym 4-bit step ~ range/15 ~ ~0.07 for N(0, 0.1) in 128 samples.
    assert err < 0.05, f"roundtrip max-err={err}"
    print(f"[awq] roundtrip OK (max-err={err:.4g})", flush=True)


def main() -> None:
    if not str(SOURCE_DIR) or not str(DST_DIR):
        raise SystemExit(
            "set SRC_DIR (local source snapshot dir) and DST_DIR (output dir) env vars"
        )
    assert SOURCE_DIR.exists(), f"missing {SOURCE_DIR}"
    if DST_DIR.exists() and any(DST_DIR.iterdir()):
        raise RuntimeError(f"refusing to write into non-empty {DST_DIR}")
    DST_DIR.mkdir(parents=True, exist_ok=True)

    src_index = json.load(open(SOURCE_DIR / "model.safetensors.index.json"))
    weight_map_in: dict[str, str] = src_index["weight_map"]

    # Group keys by source shard so we open each shard once.
    by_shard: dict[str, list[str]] = {}
    for k, fn in weight_map_in.items():
        by_shard.setdefault(fn, []).append(k)
    shard_files = sorted(by_shard.keys())
    print(f"[awq] {len(shard_files)} source shards", flush=True)

    weight_map_out: dict[str, str] = {}
    out_idx = 1
    out_buffer: dict[str, torch.Tensor] = {}
    out_size = [0]  # box so closures can mutate

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
                f"[awq] reading {src_name} ({len(keys)} tensors)", flush=True
            )
            for k in keys:
                t = f.get_tensor(k)
                if should_quantize(k, t.shape):
                    emitted = quantize_and_emit(k, t, sink)
                    n_quantized += 1
                    # free the bf16 source ASAP
                    del t
                else:
                    sink(k, t)
                    n_copied += 1
    flush(out_idx)
    total_shards = out_idx

    # Rename temp shards to final names.
    rename_map: dict[str, str] = {}
    for i in range(1, total_shards + 1):
        old = DST_DIR / f"_tmp_{i:05d}.safetensors"
        new_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        old.rename(DST_DIR / new_name)
        rename_map[f"_tmp_{i:05d}.safetensors"] = new_name
    weight_map_out = {k: rename_map[v] for k, v in weight_map_out.items()}

    # Index.
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
    for fname in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "processor_config.json",
    ):
        src = SOURCE_DIR / fname
        if src.exists():
            shutil.copy2(src, DST_DIR / fname)
            print(f"[awq] copied {fname}", flush=True)

    # config.json: source config + quantization_config + arch.
    src_cfg = json.load(open(SOURCE_DIR / "config.json"))
    out_cfg = dict(src_cfg)
    out_cfg["quantization_config"] = dict(QUANTIZATION_CONFIG)
    out_cfg["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
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

    # Final summary.
    files = sorted(DST_DIR.iterdir())
    total_bytes = sum(p.stat().st_size for p in files if p.is_file())
    print(
        f"[awq] DONE. quantized={n_quantized} tensors, copied={n_copied} tensors, "
        f"shards={total_shards}, total={total_bytes / 1024**3:.2f} GiB",
        flush=True,
    )


if __name__ == "__main__":
    roundtrip_test()
    main()
