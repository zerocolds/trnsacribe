#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple adapter layer for LLM backends used in the project.

Provides a small stable API:
  generate_text(backend, model, prompt, **opts) -> str
  generate_json(backend, model, prompt, **opts) -> dict

Supported backends: 'ollama' (HTTP), 'openai' (HTTP), 'vllm' (HTTP), 'local' (alias to ollama)

This file intentionally uses only requests so it can run in minimal envs.
If you prefer Python vllm client, replace the 'vllm' HTTP branch with client calls.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

import requests


def _extract_first_json(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    s = s.strip()
    # remove triple backticks
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start : i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    try:
                        return json.loads(
                            candidate.encode("utf-8", "ignore").decode(
                                "utf-8", "ignore"
                            )
                        )
                    except Exception:
                        return None
    return None


def generate_text(
    backend: str,
    model: str,
    prompt: str,
    timeout: int = 300,
    **opts,
) -> str:
    """Return plain text response from the chosen backend.

    opts may contain backend-specific options (num_ctx, temperature...).
    """
    be = (backend or "local").lower()
    if be in ("local", "ollama"):
        url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {},
        }
        # copy known opts
        for k in ("num_ctx", "num_keep", "temperature"):
            if k in opts:
                payload["options"][k] = opts[k]
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        # Ollama usually returns field 'response'
        return data.get("response", "") or ""

    if be == "vllm":
        # Expecting a small HTTP server in front of vLLM that accepts same shape
        url = os.getenv("VLLM_URL", "http://localhost:8000/generate")
        payload = {"model": model, "prompt": prompt, "options": opts}
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        # try common shapes
        if isinstance(data, dict):
            if "text" in data:
                return data.get("text", "") or ""
            if "outputs" in data and data.get("outputs"):
                out0 = data["outputs"][0]
                if isinstance(out0, dict) and "text" in out0:
                    return out0.get("text", "") or ""
                if isinstance(out0, str):
                    return out0
        # fallback to raw text
        return r.text or ""

    if be == "openai":
        api_key = opts.get("api_key") or os.getenv("OPENAI_API_KEY")
        base_url = opts.get("base_url") or os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set for openai backend")
        url = f"{base_url}/responses"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "input": [
                {"role": "system", "content": opts.get("system", "")},
                {"role": "user", "content": prompt},
            ],
        }
        r = requests.post(url, headers=headers, json=body, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        out = data.get("output", {})
        text = out.get("text") or (
            data.get("choices", [{}])[0].get("message", {}).get("content")
        )
        if not text:
            return json.dumps(data)
        return text

    raise RuntimeError(f"Unsupported backend: {backend}")


def generate_json(
    backend: str, model: str, prompt: str, timeout: int = 300, **opts
) -> Dict[str, Any]:
    txt = generate_text(
        backend=backend, model=model, prompt=prompt, timeout=timeout, **opts
    )
    # try direct loads first
    try:
        return json.loads(txt)
    except Exception:
        # try to extract first JSON object
        js = _extract_first_json(txt)
        if js is not None:
            return js
    # last resort: if backend returned a dict already
    if isinstance(txt, dict):
        return txt
    # fallback empty structure
    return {}
