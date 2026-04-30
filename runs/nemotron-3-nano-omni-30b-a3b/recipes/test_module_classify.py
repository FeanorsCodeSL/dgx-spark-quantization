"""Test the AWQ skip policy for Nemotron-3-Nano-Omni-30B-A3B-Reasoning.

Run with:
    cd /home/sergio/git/dgx-spark-quantization
    source .venv/bin/activate
    pytest runs/nemotron-3-nano-omni-30b-a3b/recipes/test_module_classify.py -v

The test exercises `_classify.should_quantize` against a curated set of
≥ 20 example parameter names — a synthetic skeleton plus (after Phase 1's
inspection) ≥ 10 real names extracted from `module_inspection.txt`.

Expected verdict for each name is the AWQ policy:

  - True  -> 4-bit pack.  Only routed-expert MLPs past layer 0.
  - False -> keep dense.  Mamba, attention, vision, audio, projectors,
             layer 0, shared_expert, router gate, lm_head, embeds, norms.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow `from _classify import ...` whether pytest is run from repo root
# or from the recipes/ directory.
sys.path.insert(0, str(Path(__file__).parent))
import _classify  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic test set — 20+ name/expected/category triples.
# Format: (parameter_name, shape, expected_verdict, category)
# ---------------------------------------------------------------------------
SYNTHETIC_CASES: list[tuple[str, tuple, bool, str]] = [
    # ---- routed experts past layer 0 (TRUE) ------------------------------
    (
        "language_model.model.layers.5.mlp.experts.gate_up_proj",
        (128, 3712, 2688), True, "routed-expert fused-3D gate_up",
    ),
    (
        "language_model.model.layers.5.mlp.experts.down_proj",
        (128, 2688, 1856), True, "routed-expert fused-3D down",
    ),
    (
        "language_model.model.layers.7.mlp.experts.42.up_proj.weight",
        (1856, 2688), True, "routed-expert unfused up_proj",
    ),
    (
        "language_model.model.layers.10.mlp.experts.0.down_proj.weight",
        (2688, 1856), True, "routed-expert unfused down_proj",
    ),
    # ---- layer-0 routed expert: still SKIP -------------------------------
    (
        "language_model.model.layers.0.mlp.experts.gate_up_proj",
        (128, 3712, 2688), False, "layer-0 routed expert (preserved)",
    ),
    (
        "language_model.model.layers.0.mlp.experts.0.up_proj.weight",
        (1856, 2688), False, "layer-0 routed expert unfused (preserved)",
    ),
    # ---- Mamba2 inner linears: SKIP --------------------------------------
    (
        "language_model.model.layers.1.mamba.in_proj.weight",
        (4096, 2688), False, "mamba in_proj",
    ),
    (
        "language_model.model.layers.1.mamba.out_proj.weight",
        (2688, 2048), False, "mamba out_proj",
    ),
    (
        "language_model.model.layers.1.mamba.dt_proj.weight",
        (64, 256), False, "mamba dt_proj",
    ),
    (
        "language_model.model.layers.1.mamba.conv1d.weight",
        (4096, 1, 4), False, "mamba conv1d (skip on substring)",
    ),
    # ---- self-attention QKVO past layer 0: SKIP --------------------------
    (
        "language_model.model.layers.5.self_attn.q_proj.weight",
        (4096, 2688), False, "attention q_proj past layer 0",
    ),
    (
        "language_model.model.layers.5.self_attn.k_proj.weight",
        (256, 2688), False, "attention k_proj past layer 0",
    ),
    (
        "language_model.model.layers.5.self_attn.v_proj.weight",
        (256, 2688), False, "attention v_proj past layer 0",
    ),
    (
        "language_model.model.layers.5.self_attn.o_proj.weight",
        (2688, 4096), False, "attention o_proj past layer 0",
    ),
    # ---- shared_expert MLP: SKIP -----------------------------------------
    (
        "language_model.model.layers.5.mlp.shared_expert.up_proj.weight",
        (3712, 2688), False, "shared_expert up_proj",
    ),
    (
        "language_model.model.layers.5.mlp.shared_expert.down_proj.weight",
        (2688, 3712), False, "shared_expert down_proj",
    ),
    # ---- router gate vs gate_proj boundary -------------------------------
    (
        "language_model.model.layers.5.mlp.gate.weight",
        (128, 2688), False, "router gate (endswith)",
    ),
    # Hypothetical gate_proj past layer 0 must NOT be caught by the
    # router-gate rule.  (Our test asserts True here as the "no false
    # positive" direction.)  In practice Nemotron with relu² is unlikely
    # to have gate_proj, but the test guards the rule semantics.
    (
        "language_model.model.layers.5.mlp.experts.0.gate_proj.weight",
        (1856, 2688), True, "expert gate_proj must NOT match router-gate rule",
    ),
    # ---- LM head: SKIP ---------------------------------------------------
    (
        "lm_head.weight",
        (131072, 2688), False, "lm_head",
    ),
    # ---- Embedding: SKIP -------------------------------------------------
    (
        "language_model.model.embed_tokens.weight",
        (131072, 2688), False, "embed_tokens",
    ),
    # ---- Norm: SKIP ------------------------------------------------------
    (
        "language_model.model.layers.5.input_layernorm.weight",
        (2688,), False, "rank-1 norm rejected by shape",
    ),
    (
        "language_model.model.layers.5.post_attention_layernorm.weight",
        (2688,), False, "rank-1 norm rejected by shape",
    ),
    # ---- Vision tower: SKIP ----------------------------------------------
    (
        "vision_model.radio.blocks.0.attn.qkv.weight",
        (3840, 1280), False, "vision_model linear",
    ),
    (
        "vision_model.radio.blocks.0.mlp.fc1.weight",
        (5120, 1280), False, "vision_model fc1",
    ),
    # ---- Audio tower: SKIP -----------------------------------------------
    (
        "sound_model.parakeet.encoder.layers.0.self_attn.q_proj.weight",
        (1024, 1024), False, "parakeet linear",
    ),
    (
        "audio_encoder.parakeet.layers.0.feed_forward.fc1.weight",
        (4096, 1024), False, "audio encoder linear",
    ),
    # ---- Multimodal projector: SKIP --------------------------------------
    (
        "multi_modal_projector.linear_1.weight",
        (2688, 20480), False, "multimodal projector",
    ),
    (
        "vision_projector.dense.weight",
        (2688, 1280), False, "vision projector",
    ),
    # ---- Layer-0 attention: SKIP -----------------------------------------
    (
        "language_model.model.layers.0.self_attn.q_proj.weight",
        (4096, 2688), False, "layer-0 attention QKV",
    ),
    # ---- Layer-0 mamba: SKIP (both rules apply) --------------------------
    (
        "language_model.model.layers.0.mamba.in_proj.weight",
        (4096, 2688), False, "layer-0 mamba",
    ),
    # ---- Bias / non-quantizable suffix: SKIP -----------------------------
    (
        "language_model.model.layers.5.self_attn.q_proj.bias",
        (4096,), False, "bias rank-1 + non-weight suffix",
    ),
    (
        "language_model.model.layers.5.mlp.experts.42.up_proj.bias",
        (1856,), False, "expert bias non-quantizable suffix",
    ),
]


# ---------------------------------------------------------------------------
# Real-name addendum populated from `results/module_inspection.txt` (Phase 1
# Workstream C).  Names use the actual dotted paths from the bf16 snapshot:
# the LM lives at `language_model.backbone.layers.<i>.mixer.*` (NOT
# `model.layers`); MoE experts are unfused per-expert with `up_proj` +
# `down_proj` only (no `gate_proj` — relu² is un-gated); the router is
# `mixer.gate` (NOT `mlp.gate`); the audio→LM bridge is `sound_projection`
# (NOT `*projector`).
# ---------------------------------------------------------------------------
INSPECTION_CASES: list[tuple[str, tuple, bool, str]] = [
    # ---- Real routed-expert names past layer 0: QUANTIZE ----------------
    (
        "language_model.backbone.layers.1.mixer.experts.0.up_proj.weight",
        (1856, 2688), True, "real: routed expert layer-1 up_proj (FIRST MoE layer per pattern)",
    ),
    (
        "language_model.backbone.layers.1.mixer.experts.127.down_proj.weight",
        (2688, 1856), True, "real: routed expert layer-1 last-expert down_proj",
    ),
    (
        "language_model.backbone.layers.13.mixer.experts.42.up_proj.weight",
        (1856, 2688), True, "real: routed expert middle-layer middle-expert up_proj",
    ),
    (
        "language_model.backbone.layers.50.mixer.experts.0.down_proj.weight",
        (2688, 1856), True, "real: routed expert near-final-layer down_proj",
    ),
    # ---- Real Mamba2 mixer linears: SKIP --------------------------------
    (
        "language_model.backbone.layers.0.mixer.in_proj.weight",
        (10304, 2688), False, "real: layer-0 mamba in_proj (skip on layer-0 + in_proj substring)",
    ),
    (
        "language_model.backbone.layers.2.mixer.in_proj.weight",
        (10304, 2688), False, "real: mamba in_proj past layer 0 (in_proj substring)",
    ),
    (
        "language_model.backbone.layers.2.mixer.out_proj.weight",
        (2688, 4096), False, "real: mamba out_proj (out_proj substring)",
    ),
    (
        "language_model.backbone.layers.2.mixer.conv1d.weight",
        (6144, 1, 4), False, "real: mamba conv1d (conv1d substring)",
    ),
    # ---- Real LM attention (mixer.{q,k,v,o}_proj): SKIP -----------------
    (
        "language_model.backbone.layers.5.mixer.q_proj.weight",
        (4096, 2688), False, "real: LM attention q_proj (uses mixer.q_proj NOT self_attn.q_proj)",
    ),
    (
        "language_model.backbone.layers.5.mixer.k_proj.weight",
        (256, 2688), False, "real: LM attention k_proj (GQA narrow)",
    ),
    (
        "language_model.backbone.layers.5.mixer.v_proj.weight",
        (256, 2688), False, "real: LM attention v_proj (GQA narrow)",
    ),
    (
        "language_model.backbone.layers.5.mixer.o_proj.weight",
        (2688, 4096), False, "real: LM attention o_proj",
    ),
    # ---- Real router gate: SKIP via SKIP_ENDSWITH -----------------------
    (
        "language_model.backbone.layers.1.mixer.gate.weight",
        (128, 2688), False, "real: router gate uses mixer.gate.weight (NOT mlp.gate.weight)",
    ),
    # ---- Real shared_experts (plural!) MLP: SKIP ------------------------
    (
        "language_model.backbone.layers.1.mixer.shared_experts.up_proj.weight",
        (3712, 2688), False, "real: shared_experts (plural) MLP up_proj — caught by 'shared_expert' substring",
    ),
    (
        "language_model.backbone.layers.1.mixer.shared_experts.down_proj.weight",
        (2688, 3712), False, "real: shared_experts down_proj",
    ),
    # ---- Real LM head: SKIP ---------------------------------------------
    (
        "language_model.lm_head.weight",
        (131072, 2688), False, "real: lm_head (probed live; not in module_inspection.txt due to 'lm_head' filter scope)",
    ),
    # ---- Real embeddings: SKIP ------------------------------------------
    (
        "language_model.backbone.embeddings.weight",
        (131072, 2688), False, "real: backbone embeddings (caught by 'embed' substring; would be rejected by suffix rule too — Embedding params end in .weight but rank-2 with embedding-like name)",
    ),
    # ---- Real audio→LM bridge: SKIP via 'projection' substring ----------
    (
        "sound_projection.linear1.weight",
        (4096, 1024), False, "real: audio→LM bridge sound_projection.linear1 (caught by 'projection' substring)",
    ),
    (
        "sound_projection.linear2.weight",
        (2688, 4096), False, "real: audio→LM bridge sound_projection.linear2",
    ),
    # ---- Real Parakeet self-attention: SKIP via 'audio'/'sound'/'self_attn' --
    (
        "sound_encoder.encoder.layers.0.self_attn.q_proj.weight",
        (1024, 1024), False, "real: parakeet attention q_proj (caught by 'sound' AND 'self_attn' AND 'q_proj')",
    ),
    (
        "sound_encoder.encoder.layers.0.self_attn.relative_k_proj.weight",
        (1024, 1024), False, "real: parakeet relative_k_proj (caught by 'sound' substring)",
    ),
    # ---- Real Parakeet feed-forward: SKIP via 'sound' -------------------
    (
        "sound_encoder.encoder.layers.5.feed_forward1.linear1.weight",
        (4096, 1024), False, "real: parakeet feed_forward (caught by 'sound')",
    ),
    # ---- Real RADIO vision linear: SKIP via 'vision'/'radio' -----------
    (
        "vision_model.radio_model.model.blocks.0.attn.qkv.weight",
        (3840, 1280), False, "real: RADIO ViT QKV linear (caught by 'vision' AND 'radio')",
    ),
    (
        "vision_model.radio_model.model.blocks.0.mlp.fc1.weight",
        (5120, 1280), False, "real: RADIO ViT MLP fc1",
    ),
    # ---- Layer-0 routed expert (hypothetical — layer 0 IS Mamba in this
    # snapshot, so this case is synthetic-by-construction; layer 0 happens
    # to be Mamba).  Boundary check: even if layer-0 had experts they'd skip.
    (
        "language_model.backbone.layers.0.mixer.experts.0.up_proj.weight",
        (1856, 2688), False, "real-shape: hypothetical layer-0 expert (skipped by layer-0 rule)",
    ),
    # ---- Vision projector mlp1 (top-level nn.Sequential — RADIO→LM bridge):
    # SKIP via 'mlp1' substring.  Caught by the Phase 2 preflight when the
    # initial 'projector'/'projection' substrings missed it (the bare name
    # `mlp1` doesn't contain any of those tokens).  Two real Linears live
    # inside this nn.Sequential: mlp1.1 (Linear(5120, 20480)) and
    # mlp1.3 (Linear(20480, 2688)).
    (
        "mlp1.1.weight",
        (20480, 5120), False, "real: vision projector mlp1.1 (RADIO→LM bridge first Linear)",
    ),
    (
        "mlp1.3.weight",
        (2688, 20480), False, "real: vision projector mlp1.3 (RADIO→LM bridge second Linear)",
    ),
    # ---- Boundary regression check: confirm the mlp1 patch did NOT
    # accidentally skip a routed expert.  The substring `mlp1` must NOT
    # appear in any routed-expert weight name (and a real expert name
    # past layer 0 must still classify as quantize).
    (
        "language_model.backbone.layers.5.mixer.experts.10.up_proj.weight",
        (1856, 2688), True, "real: routed expert past layer 0 — must still quantize after mlp1 patch",
    ),
]


@pytest.mark.parametrize(
    "name,shape,expected,category",
    SYNTHETIC_CASES + INSPECTION_CASES,
    ids=[f"{cat}::{name}" for name, _, _, cat in SYNTHETIC_CASES + INSPECTION_CASES],
)
def test_should_quantize(name: str, shape: tuple, expected: bool, category: str) -> None:
    actual = _classify.should_quantize(name, shape)
    reason = _classify.explain(name, shape)
    assert actual == expected, (
        f"\n  category: {category}"
        f"\n  name:     {name}"
        f"\n  shape:    {shape}"
        f"\n  expected: {expected}"
        f"\n  actual:   {actual}"
        f"\n  reason:   {reason}"
    )


def test_minimum_case_count() -> None:
    total = len(SYNTHETIC_CASES) + len(INSPECTION_CASES)
    assert total >= 20, (
        f"need at least 20 test cases for the AWQ skip policy; got {total}"
    )


def test_routed_expert_quantizes_at_minimum_one_per_layer_type() -> None:
    """Smoke check: there is at least one TRUE verdict (otherwise the recipe
    will quantize nothing — useless build)."""
    verdicts = [should for _, _, should, _ in SYNTHETIC_CASES + INSPECTION_CASES]
    assert any(verdicts), "no positive cases — AWQ would skip every weight"


def test_skip_substrings_are_lowercase() -> None:
    """All skip substrings must be lowercase since we compare against
    name.lower(); a mixed-case entry would silently never match."""
    for tok in _classify.SKIP_SUBSTRINGS_CI:
        assert tok == tok.lower(), f"non-lowercase skip substring: {tok!r}"
