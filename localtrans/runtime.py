#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runtime helpers for deciding whether we run locally (Mac/CPU) or on a GPU server.

Usage:
  from localtrans import runtime
  IS_GPU = runtime.IS_GPU
  BACKEND = runtime.BACKEND  # 'vllm'|'ollama'|'openai'|'local'

Environment overrides:
  USE_GPU=1        -> force GPU mode
  USE_VLLM=1       -> prefer vLLM backend when GPU is available
  USE_OLLAMA=1     -> prefer Ollama
  USE_OPENAI=1     -> prefer OpenAI
"""
import os
import shutil
from pathlib import Path


def is_gpu_target() -> bool:
    """Return True if we should consider this runtime a GPU/server target.

    Priority:
      - USE_GPU env var (1/true/yes)
      - presence of nvidia-smi
      - torch.cuda.is_available()
    """
    v = os.environ.get("USE_GPU", "").lower()
    if v in ("1", "true", "yes"):
        return True

    if shutil.which("nvidia-smi"):
        return True

    try:
        import torch

        if torch.cuda.is_available():
            return True
    except Exception:
        pass

    return False


# Evaluate once on import
IS_GPU = is_gpu_target()


def choose_backend() -> str:
    """Heuristic choice of LLM backend. Respects environment variables.

    Returns one of: 'vllm', 'ollama', 'openai', 'local'
    """
    # Explicit user preference
    if os.environ.get("USE_VLLM", "").lower() in ("1", "true", "yes"):
        return "vllm"
    if os.environ.get("USE_OLLAMA", "").lower() in ("1", "true", "yes"):
        return "ollama"
    if os.environ.get("USE_OPENAI", "").lower() in ("1", "true", "yes"):
        return "openai"

    # If we have GPU, prefer vLLM (user can override with USE_OLLAMA/OPENAI)
    if IS_GPU:
        return "vllm"

    # default for local dev
    return "local"


# Derived defaults
BACKEND = choose_backend()
MODEL_DEVICE = "cuda" if IS_GPU else "cpu"
# reasonable default model path placeholders
MODEL_PATH = (
    os.environ.get("MODEL_PATH", "/mnt/models")
    if IS_GPU
    else os.environ.get("MODEL_PATH", "./models")
)


def log_env():
    return {
        "IS_GPU": IS_GPU,
        "BACKEND": BACKEND,
        "MODEL_DEVICE": MODEL_DEVICE,
        "MODEL_PATH": MODEL_PATH,
    }
