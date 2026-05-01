"""Smoke-test image, audio, and video inputs against a vLLM chat endpoint.

The probes use public sample assets from vLLM's own multimodal examples where
available. They verify that the OpenAI-compatible server accepts each media
type and returns non-empty text; the image probe additionally checks for a
known object token.

Usage:
    python3 multimodal_smoke.py <served-model-name> <out-dir>

Environment:
    VLLM_URL  override the OpenAI-compatible base URL
              (default: http://localhost:8000/v1)
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ASSETS = {
    "image": (
        "duck.jpg",
        "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/duck.jpg",
        "image/jpeg",
    ),
    "audio": (
        "voice-telephony-8khz.wav",
        "https://sample-files.com/downloads/audio/wav/voice-telephony-8khz.wav",
        "audio/wav",
    ),
    "video": (
        "sample_demo_1.mp4",
        "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4",
        "video/mp4",
    ),
}


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    req = urllib.request.Request(url, headers={"user-agent": "dgx-spark-quantization-smoke/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        path.write_bytes(resp.read())


def _data_url(path: Path, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def call_chat(base: str, model: str, content: list[dict], max_tokens: int = 512) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
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
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_text(resp: dict) -> str:
    msg = resp["choices"][0]["message"]
    return "\n".join(
        str(x)
        for x in [msg.get("reasoning_content"), msg.get("reasoning"), msg.get("content")]
        if x
    )


def run_probe(base: str, model: str, name: str, content: list[dict], expected: str | None) -> dict:
    t0 = time.time()
    try:
        resp = call_chat(base, model, content)
        elapsed = time.time() - t0
        text = extract_text(resp)
        ok = bool(text.strip())
        if expected is not None:
            ok = expected.lower() in text.lower()
        return {
            "name": name,
            "ok": ok,
            "elapsed_s": round(elapsed, 2),
            "expected": expected,
            "usage": resp.get("usage", {}),
            "text": text,
            "finish_reason": resp.get("choices", [{}])[0].get("finish_reason"),
            "raw_message": resp.get("choices", [{}])[0].get("message", {}),
        }
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        return {"name": name, "ok": False, "error": f"HTTP {e.code}: {body}"}
    except Exception as e:
        return {"name": name, "ok": False, "error": f"{type(e).__name__}: {e}"}


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <served-model-name> <out-dir>", file=sys.stderr)
        return 2

    model = sys.argv[1]
    out_dir = Path(sys.argv[2]).resolve()
    fixtures = out_dir / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)

    downloaded = {}
    for key, (filename, url, mime) in ASSETS.items():
        path = fixtures / filename
        download(url, path)
        downloaded[key] = (path, url, mime)

    base = os.environ.get("VLLM_URL", "http://localhost:8000/v1")
    image, image_url, image_mime = downloaded["image"]
    audio, audio_url, _audio_mime = downloaded["audio"]
    video, video_url, video_mime = downloaded["video"]
    probes = [
        (
            "image",
            [
                {"type": "text", "text": "What animal is in this image? Answer with one word."},
                {"type": "image_url", "image_url": {"url": _data_url(image, image_mime)}},
            ],
            "duck",
        ),
        (
            "audio",
            [
                {"type": "text", "text": "Briefly describe this audio clip."},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64.b64encode(audio.read_bytes()).decode("ascii"),
                        "format": "wav",
                    },
                },
            ],
            None,
        ),
        (
            "video",
            [
                {"type": "text", "text": "Briefly describe what is happening in this video."},
                {"type": "video_url", "video_url": {"url": _data_url(video, video_mime)}},
            ],
            None,
        ),
    ]

    results = []
    for name, content, expected in probes:
        results.append(run_probe(base, model, name, content, expected))

    summary = {
        "model": model,
        "base_url": base,
        "fixtures": {
            "image": str(image),
            "audio": str(audio),
            "video": str(video),
        },
        "source_urls": {
            "image": image_url,
            "audio": audio_url,
            "video": video_url,
        },
        "results": results,
        "passed": sum(1 for r in results if r["ok"]),
        "total": len(results),
    }
    (out_dir / "multimodal_smoke.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [f"# Multimodal smoke: {summary['passed']}/{summary['total']} passed", ""]
    for result in results:
        lines.append(f"## {result['name']}: {'PASS' if result['ok'] else 'FAIL'}")
        if "elapsed_s" in result:
            lines.append(f"- latency: {result['elapsed_s']}s")
        if result.get("expected"):
            lines.append(f"- expected substring: `{result['expected']}`")
        if result.get("usage"):
            lines.append(f"- usage: `{json.dumps(result['usage'], sort_keys=True)}`")
        if result.get("error"):
            lines.append(f"- error: `{result['error']}`")
        if result.get("text"):
            lines.append("")
            lines.append("```text")
            lines.append(result["text"].strip())
            lines.append("```")
        lines.append("")
    (out_dir / "multimodal_smoke.md").write_text("\n".join(lines))
    print(f"wrote {out_dir / 'multimodal_smoke.md'} ({summary['passed']}/{summary['total']} passed)")
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
