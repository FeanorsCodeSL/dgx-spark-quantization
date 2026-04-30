"""Module-tree inspector for Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16.

Loads the bf16 model on CPU under `trust_remote_code=True` and dumps
information sufficient to drive the AWQ recipe's skip policy (Phase 2):

  - Total Linear count + interesting-module dump (router/gate/experts/
    mamba/ssm/attention/vision/audio/projector/embed/lm_head).
  - Per-class histogram (top 30 nn.Module subclass names by count).
  - Truncated repr of the top-level model (first 200 lines) so the
    layer-container path is unambiguous.
  - Sample dump of one Mamba2, one MoE-MLP, and one attention layer
    (modules + parameters) drawn from the hybrid_override_pattern.
  - MoE expert-layout probe: parameter NAMES + SHAPES of the first
    handful of expert tensors, so we can confirm fused-3D vs unfused
    per-expert and gate+up+down vs up+down (the relu² question).

Run under cgroup guard:
    MEMORY_MAX=112G MEMORY_HIGH=100G \
      tools/run_under_memcap.sh \
      bash -lc 'source .venv/bin/activate && \
        HF_HOME="$PWD/hf-cache" \
        MODEL_ID=nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
        python runs/nemotron-3-nano-omni-30b-a3b/recipes/inspect_modules.py'

Output is plain text on stdout — pipe to `tee module_inspection.txt`.
"""

from __future__ import annotations

import os
import sys
from collections import Counter

import torch
from transformers import AutoModelForCausalLM


MODEL_ID = os.environ.get("MODEL_ID", "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16")


def main() -> None:
    print(f"[inspect] MODEL_ID={MODEL_ID}")
    print(f"[inspect] HF_HOME={os.environ.get('HF_HOME', '(default)')}")
    print(f"[inspect] torch={torch.__version__}")

    # Force `eager` attention because FlashAttention2 isn't installed in
    # this venv (and the model defaults to FA2 if available).  `eager` is
    # purely structural — we never run a forward pass, just enumerate
    # modules.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype="auto",
        device_map={"": "cpu"},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    print(f"[inspect] loaded {type(model).__name__}")

    # ------------------------------------------------------------------
    # 1. Linear count + interesting-name filter
    # ------------------------------------------------------------------
    interesting: list[tuple[str, str]] = []
    linear_count = 0
    keywords = [
        "router", "gate", "expert", "moe",
        "lm_head", "norm", "embed",
        "deltanet", "delta", "linear_attn", "linear_attention",
        "q_proj", "k_proj", "v_proj", "o_proj",
        "qkv", "self_attn",
        "mamba", "ssm", "dt", "in_proj", "out_proj", "conv1d",
        "vision", "radio", "audio", "parakeet", "sound",
        "projector", "shared_expert",
    ]
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        cls_lower = cls_name.lower()
        name_lower = name.lower()
        if cls_name == "Linear":
            linear_count += 1
        if any(k in name_lower or k in cls_lower for k in keywords):
            interesting.append((name, cls_name))

    print(f"\n[inspect] Total Linear modules: {linear_count}")

    print(f"\n[inspect] Interesting modules ({len(interesting)} matches, first 3000):")
    for name, cls_name in interesting[:3000]:
        print(f"{name} :: {cls_name}")

    print(f"\n[inspect] Last 100 interesting modules:")
    for name, cls_name in interesting[-100:]:
        print(f"{name} :: {cls_name}")

    # ------------------------------------------------------------------
    # 2. Per-class histogram
    # ------------------------------------------------------------------
    class_counter: Counter[str] = Counter()
    for _, module in model.named_modules():
        class_counter[module.__class__.__name__] += 1
    print(f"\n[inspect] Per-class histogram (top 30):")
    for cls_name, count in class_counter.most_common(30):
        print(f"  {count:>6d}  {cls_name}")

    # ------------------------------------------------------------------
    # 3. Truncated repr of the top-level model
    # ------------------------------------------------------------------
    print(f"\n[inspect] repr(model) — first 200 lines:")
    repr_lines = repr(model).split("\n")
    for line in repr_lines[:200]:
        print(line)
    if len(repr_lines) > 200:
        print(f"... ({len(repr_lines) - 200} more lines truncated)")

    # ------------------------------------------------------------------
    # 4. Find the layer container and sample one of each layer type
    # ------------------------------------------------------------------
    # Best guess: language_model.model.layers, fall back to other paths.
    candidate_paths = [
        "language_model.model.layers",
        "model.layers",
        "language_model.layers",
        "transformer.layers",
        "llm.model.layers",
    ]
    layer_container_path = None
    layer_container = None
    for path in candidate_paths:
        try:
            obj = model
            for part in path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "__len__") and len(obj) > 0:
                layer_container_path = path
                layer_container = obj
                break
        except AttributeError:
            continue

    print(f"\n[inspect] layer_container_path = {layer_container_path}")
    if layer_container is not None:
        print(f"[inspect] num layers = {len(layer_container)}")
        # Print class names for all layers
        print(f"[inspect] Layer class sequence:")
        for idx in range(len(layer_container)):
            print(f"  layer {idx:>3d}: {layer_container[idx].__class__.__name__}")

        # Sample three layer indices: try 1 (likely Mamba), 2 (likely E), and find
        # an attention layer (* in hybrid_override_pattern). The pattern from
        # config.json: "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
        pattern = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
        m_idx = pattern.find("M")
        e_idx = pattern.find("E")
        a_idx = pattern.find("*")
        print(f"\n[inspect] Sampling layers per hybrid_override_pattern (M={m_idx}, E={e_idx}, *={a_idx}):")
        for idx, label in [(m_idx, "Mamba"), (e_idx, "MoE-MLP"), (a_idx, "Attention")]:
            if idx < 0 or idx >= len(layer_container):
                print(f"  skipping {label} sample at idx={idx} (out of range)")
                continue
            print(f"\n  ---- layer {idx} ({label}, expected) ----")
            layer = layer_container[idx]
            print(f"  class = {layer.__class__.__name__}")
            print(f"  named_modules:")
            for n, m in layer.named_modules():
                if n == "":
                    continue
                cls = m.__class__.__name__
                if cls == "Linear":
                    print(f"    {n} :: Linear  in={m.in_features} out={m.out_features}")
                elif cls in ("Conv1d", "Conv2d"):
                    print(f"    {n} :: {cls}  ic={getattr(m, 'in_channels', '?')} oc={getattr(m, 'out_channels', '?')} k={getattr(m, 'kernel_size', '?')}")
                else:
                    print(f"    {n} :: {cls}")
            print(f"  named_parameters (shapes):")
            for n, p in layer.named_parameters(recurse=True):
                print(f"    {n}: {tuple(p.shape)}  dtype={p.dtype}")

    # ------------------------------------------------------------------
    # 5. MoE expert layout probe
    # ------------------------------------------------------------------
    print(f"\n[inspect] MoE expert layout probe:")
    expert_params: list[tuple[str, tuple[int, ...]]] = []
    for n, p in model.named_parameters():
        if "expert" in n.lower():
            expert_params.append((n, tuple(p.shape)))
    print(f"  total expert-related parameters: {len(expert_params)}")
    print(f"  first 12:")
    for n, shape in expert_params[:12]:
        print(f"    {n}: {shape}")
    if len(expert_params) > 12:
        print(f"  ... and {len(expert_params) - 12} more.")

    # Distinguish layouts: any 3D tensor whose name ends with 'gate_up_proj' /
    # 'up_proj' / 'down_proj' / 'gate_proj' suggests fused-3D (Qwen-MoE-style).
    # Per-expert 'experts.0.gate_proj.weight' suggests unfused 2D layout.
    fused_hits = [n for n, s in expert_params if len(s) == 3]
    unfused_hits = [n for n, s in expert_params if len(s) == 2 and ".0." in n]
    print(f"\n  fused-3D candidates: {len(fused_hits)}")
    for n in fused_hits[:6]:
        print(f"    {n}")
    print(f"  unfused per-expert (.0.) candidates: {len(unfused_hits)}")
    for n in unfused_hits[:6]:
        print(f"    {n}")

    # ------------------------------------------------------------------
    # 6. Vision + audio module class snapshot
    # ------------------------------------------------------------------
    print(f"\n[inspect] Vision/audio top-level module class names:")
    for n, m in model.named_modules():
        nl = n.lower()
        if not nl:
            continue
        if any(k in nl for k in ["vision", "radio", "audio", "parakeet", "sound", "projector"]):
            cls = m.__class__.__name__
            # Print only the depth-≤3 named hits to avoid spamming every leaf
            depth = n.count(".")
            if depth <= 3:
                print(f"  {n} :: {cls}  (depth={depth})")

    print(f"\n[inspect] Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        # Surface any error with a non-zero exit so the caller can detect it.
        # Don't suppress — Phase 1 needs real output, not a placeholder.
        import traceback
        traceback.print_exc()
        print(f"\n[inspect] FAILED: {e}", file=sys.stderr)
        sys.exit(1)
