"""4-prompt smoke probe runner for the AWQ-CT artifact.

Hits a vLLM endpoint and writes the verbatim transcript + pass/fail
verdict.  Used by Phase 2b's smoke validation step.

Usage:
    python _smoke_probes.py <served-model-name> <transcript-out-path>

Environment:
    VLLM_URL  override the OpenAI-compatible base URL
              (default: http://localhost:8000/v1)
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request


PROBES = [
    ("17 * 23 = ?  Briefly.", "391"),
    ("What is the capital of France?  One word.", "Paris"),
    ("What is the capital of Spain?  One word.", "Madrid"),
    ("What is 2+2?  Briefly.", "4"),
]


def call_chat(base: str, model: str, prompt: str, max_tokens: int = 256) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <served-model-name> <out-transcript>", file=sys.stderr)
        return 2
    served = sys.argv[1]
    out_path = sys.argv[2]
    base = os.environ.get("VLLM_URL", "http://localhost:8000/v1")

    lines: list[str] = []
    pass_count = 0
    fail_count = 0
    for prompt, expected in PROBES:
        lines.append("=" * 78)
        lines.append(f"PROMPT: {prompt}")
        lines.append(f"EXPECT contains: {expected!r}")
        try:
            t0 = time.time()
            resp = call_chat(base, served, prompt)
            dt = time.time() - t0
            content = resp["choices"][0]["message"]["content"] or ""
            reasoning = resp["choices"][0]["message"].get("reasoning_content")
            usage = resp.get("usage", {})
            lines.append(f"LATENCY: {dt:.2f}s  USAGE: {usage}")
            if reasoning:
                lines.append(f"REASONING:\n{reasoning}")
            lines.append(f"RESPONSE:\n{content}")
            ok = expected.lower() in (content + " " + (reasoning or "")).lower()
            verdict = "PASS" if ok else "FAIL"
            if ok:
                pass_count += 1
            else:
                fail_count += 1
            lines.append(f"VERDICT: {verdict}")
        except Exception as e:
            lines.append(f"ERROR: {type(e).__name__}: {e}")
            lines.append("VERDICT: FAIL (exception)")
            fail_count += 1
    lines.append("=" * 78)
    lines.append(f"SUMMARY: {pass_count}/{len(PROBES)} passed, {fail_count} failed")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"wrote {out_path} ({pass_count}/{len(PROBES)} passed)")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
