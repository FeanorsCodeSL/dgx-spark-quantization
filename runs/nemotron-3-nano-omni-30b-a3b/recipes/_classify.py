"""AWQ skip policy for Nemotron-3-Nano-Omni-30B-A3B-Reasoning.

The Phase 2 recipe (`awq_gemm.py`) and the Phase 1 test
(`test_module_classify.py`) both import `should_quantize` from here so the
policy lives in exactly one place.

Policy: **only the routed-expert MLPs past layer 0 get 4-bit packed.**
Everything else stays dense:

  - Mamba2 inner Linears (in_proj / out_proj / dt_proj / conv1d / ssm /
    anything matching `mamba`) — SSM dynamics break catastrophically if
    quantized; quality collapses non-gracefully.
  - Self-attention (the 6 attention layers in the hybrid pattern) — small
    portion of params; not worth the QKVO precision risk.
  - Shared expert MLP — always-active, big quality contribution per byte.
  - Router gate (`mlp.gate.weight`) — must distinguish from `mlp.gate_proj`
    if that exists.  Endswith match.
  - Layer-0 — first layer is conservatively preserved (Qwen-MoE precedent).
  - Vision / RADIO / projector / image / video — multimodal stack stays
    fp16/bf16 so the AWQ build keeps multimodal capability.
  - Audio / Parakeet / sound — same rationale.
  - Embeddings, LM head, norms — never quantized in any AWQ recipe.

This mirrors NVIDIA's NVFP4 keep-dense policy (mamba + attention + encoders
stay un-quantized; only MoE experts get aggressive bits) — by following at
least that conservative split we avoid surprising the Nemotron architecture.
"""

from __future__ import annotations

import re
from typing import Iterable


# Names whose lowercase form contains ANY of these substrings are kept dense.
# Use lowercase substring match (`<substr> in name.lower()`).  These are
# coarse on purpose — explicit endswith / regex cases below handle the
# corner cases that the substring rules would otherwise mis-classify.
#
# Refined from Phase 1 inspection of the actual module tree:
#   - LM attention uses `mixer.q_proj/k_proj/v_proj/o_proj` (NOT
#     `self_attn.q_proj`).  The `q_proj`/`k_proj`/`v_proj`/`o_proj`
#     substrings are added.  This is safe because routed-expert weights
#     never carry those names — they're `experts.<i>.up_proj.weight` and
#     `experts.<i>.down_proj.weight`.
#   - The audio→LM bridge is named `sound_projection` (not `*projector`).
#     Added `projection` to catch it without false positives — no
#     quantizable expert tensor uses `projection` in its name.
#   - The mamba layers use `language_model.backbone.layers.<i>.mixer.*`
#     (not `*.mamba.*`); the inner Linears are caught by `in_proj`,
#     `out_proj`, `conv1d` substrings.  The `mamba` substring is kept
#     for forward-compat in case a different snapshot uses it.
SKIP_SUBSTRINGS_CI: tuple[str, ...] = (
    # heads / embeddings / norms
    "lm_head",
    "embed_tokens",
    "embedding",
    "embed",
    "norm",
    "layernorm",
    "rmsnorm",
    # vision encoder + bridge
    "vision",
    "radio",
    "vision_model",
    "vision_tower",
    "image_proj",
    "video",
    # audio encoder + bridge
    "sound",
    "audio",
    "parakeet",
    "audio_encoder",
    # multimodal bridges (sound_projection.linear1/2 + future vision_proj*)
    "projector",
    "projection",
    # vision projector (top-level nn.Sequential — RADIO→LM bridge; mlp1.{1,3}.weight)
    "mlp1",
    # mamba2 SSM block — every inner Linear / Conv1d kept dense.
    # Names actually observed: `mixer.in_proj`, `mixer.out_proj`,
    # `mixer.conv1d` (and the inner `mixer.norm` is rank-1, filtered by
    # the shape rule).
    "mamba",
    "ssm",
    "in_proj",
    "out_proj",
    "dt_proj",
    "conv1d",
    # transformer attention block — keep all 6 attention layers dense.
    # The LM uses `mixer.q_proj` etc.; the audio encoder uses
    # `self_attn.q_proj`.  `q_proj`/`k_proj`/`v_proj`/`o_proj` cover both,
    # and `self_attn` is retained for forward-compat.
    "self_attn",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    # always-on shared expert ("shared_experts" plural matches via substring)
    "shared_expert",
)

# Names that end with EXACTLY one of these are kept dense.  The router gate
# in the inspected snapshot is `mixer.gate.weight`.  Older Mistral/Mixtral
# uses `block_sparse_moe.gate.weight`; the qwen3 layout uses
# `mlp.gate.weight`.  Cover all three so this helper survives a snapshot
# rename.  CRITICAL: must use endswith — substring `gate` would also catch
# `gate_proj` weights in gated MLPs (Nemotron is un-gated relu² so this
# isn't a live concern, but the policy is robust).
SKIP_ENDSWITH: tuple[str, ...] = (
    "mixer.gate.weight",          # Nemotron-Nano-Omni layout (observed)
    "mlp.gate.weight",            # qwen3-MoE layout (precedent)
    "block_sparse_moe.gate.weight",  # mixtral layout
)

# Layer-0 conservative preservation: any tensor in the first hidden layer.
LAYER0_RE = re.compile(r".*\.layers\.0\..*")

# Acceptable suffixes for tensors we *might* quantize (anything else is an
# unusual parameter — bias, scalar, packed-table — and should be left alone).
QUANTIZABLE_SUFFIXES: tuple[str, ...] = (
    ".weight",
    ".gate_up_proj",
    ".up_proj",
    ".gate_proj",
    ".down_proj",
)


def _shape_len(shape) -> int:
    """Robust to torch.Size, tuple, list — anything iterable."""
    if hasattr(shape, "__len__"):
        return len(shape)
    return len(tuple(shape))


def should_quantize(name: str, shape) -> bool:
    """Return True if the AWQ recipe should 4-bit-pack this tensor.

    The recipe walks every parameter in the source state-dict; this helper
    picks out the ones whose 4-bit-packed footprint is worth the disk
    saving.  Non-2D / non-3D tensors are rejected outright (RTN packing only
    handles matrices and per-expert-fused 3-D experts).

    Parameters
    ----------
    name
        Dotted parameter name from `model.named_parameters()` (or the key in
        a safetensors shard).
    shape
        Tensor shape — anything len-able.  Only `len in (2, 3)` is considered
        quantizable.
    """
    if _shape_len(shape) not in (2, 3):
        return False

    if not any(name.endswith(suffix) for suffix in QUANTIZABLE_SUFFIXES):
        return False

    # End-of-name router gate: must come BEFORE substring rules so
    # `mlp.gate_proj.weight` (which contains "gate") isn't lumped in.
    for sfx in SKIP_ENDSWITH:
        if name.endswith(sfx):
            return False

    name_lower = name.lower()
    for token in SKIP_SUBSTRINGS_CI:
        if token in name_lower:
            return False

    if LAYER0_RE.match(name):
        return False

    return True


def explain(name: str, shape) -> str:
    """Return a one-line human reason for the should_quantize verdict.

    Used by the test to make failures legible.
    """
    if _shape_len(shape) not in (2, 3):
        return f"reject: shape rank={_shape_len(shape)} (must be 2 or 3)"
    if not any(name.endswith(suffix) for suffix in QUANTIZABLE_SUFFIXES):
        return "reject: name doesn't end with a quantizable suffix"
    for sfx in SKIP_ENDSWITH:
        if name.endswith(sfx):
            return f"skip: endswith({sfx!r}) — router gate"
    name_lower = name.lower()
    for token in SKIP_SUBSTRINGS_CI:
        if token in name_lower:
            return f"skip: substring({token!r}) — kept dense"
    if LAYER0_RE.match(name):
        return "skip: layer-0 conservative preservation"
    return "quantize: routed expert MLP past layer 0"


def classify_many(names_with_shapes: Iterable[tuple[str, tuple]]) -> list[tuple[str, bool, str]]:
    """Convenience: classify many (name, shape) pairs at once.  Returns
    (name, verdict, reason) tuples — useful when dumping classification
    decisions for an entire state-dict.
    """
    return [(name, should_quantize(name, shape), explain(name, shape))
            for name, shape in names_with_shapes]
