#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI REST API for aggregate_intervue pipeline.
Supports endpoints for: transcribe, diarize, summarize, assign_roles.
"""
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    Request,
    HTTPException,
    status,
    Depends,
)
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pathlib import Path
import requests
import shutil
import tempfile
import os
import subprocess
import sys
import traceback


WHISPER_MODEL_URL = (
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin"
)
WHISPER_MODEL_PATH = Path("/model/ggml-large-v3.bin")


def download_whisper_model():
    WHISPER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not WHISPER_MODEL_PATH.exists():
        print(f"Downloading Whisper large-v3 to {WHISPER_MODEL_PATH} ...")
        r = requests.get(WHISPER_MODEL_URL, stream=True)
        r.raise_for_status()
        with open(WHISPER_MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Whisper model downloaded.")
    else:
        print("Whisper model already present.")


# --- Download LLM model from HF ---
def download_hf_model(model_name: str, target_dir: Path = Path("/model/llm")):
    from huggingface_hub import snapshot_download

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading HF model {model_name} to {target_dir} ...")
    snapshot_download(
        repo_id=model_name, local_dir=target_dir, local_dir_use_symlinks=False
    )
    print("HF model downloaded.")


@app.on_event("startup")
def startup_event():
    download_whisper_model()


# --- /pipeline: full process ---
@app.post("/download_llm_model")
async def download_llm_model(
    model_name: str = Form(...), auth: bool = Depends(check_auth)
):
    """Download LLM model from HuggingFace by name (repo_id)."""
    try:
        download_hf_model(model_name)
        return {"status": "ok", "model": model_name}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# --- Simple Bearer token auth ---
API_TOKEN = os.environ.get("API_TOKEN", "changeme")
auth_scheme = HTTPBearer()


def check_auth(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API token"
        )
    return True


# --- Utility: run subprocess and capture output ---
def run_subprocess(cmd, cwd=None):
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        return proc.returncode, proc.stdout
    except Exception as e:
        return 1, str(e)


# --- /transcribe ---
@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    lang: str = Form("ru"),
    whisper_bin: str = Form(None),
    whisper_model: str = Form(None),
    threads: int = Form(4),
    jobs: int = Form(1),
    auth: bool = Depends(check_auth),
):
    """Transcribe audio file using whisper.cpp pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / audio.filename
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
        out_dir = Path(tmpdir) / "out"
        out_dir.mkdir()
        # Call transcribe_local_whispercpp.py
        cmd = [
            sys.executable,
            "localtrans/transcribe_local_whispercpp.py",
            "--audio-root",
            str(tmpdir),
            "--out-root",
            str(out_dir),
            "--whisper-bin",
            whisper_bin
            or os.environ.get("WHISPER_BIN", "../whisper.cpp/build/bin/whisper-cli"),
            "--model",
            whisper_model
            or os.environ.get(
                "WHISPER_MODEL", "../whisper.cpp/models/ggml-large-v3.bin"
            ),
            "--lang",
            lang,
            "--threads",
            str(threads),
            "--jobs",
            str(jobs),
        ]
        rc, out = run_subprocess(cmd)
        if rc != 0:
            return JSONResponse(status_code=500, content={"error": out})
        # Return path to segments.json
        segs = list(out_dir.glob("*.segments.json"))
        if not segs:
            return JSONResponse(
                status_code=500, content={"error": "No segments.json produced"}
            )
        return {"segments_json": segs[0].read_text(encoding="utf-8")}


# --- /diarize ---
@app.post("/diarize")
async def diarize(
    audio: UploadFile = File(...),
    hf_token: str = Form(...),
    device: str = Form("auto"),
    jobs: int = Form(1),
    auth: bool = Depends(check_auth),
):
    """Diarize audio file using pyannote pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / audio.filename
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
        out_dir = Path(tmpdir) / "rttm"
        out_dir.mkdir()
        # Call diarize_pyannote.py
        cmd = [
            sys.executable,
            "localtrans/diarize_pyannote.py",
            "--src",
            str(tmpdir),
            "--dst",
            str(out_dir),
            "--hf-token",
            hf_token,
            "--device",
            device,
            "--jobs",
            str(jobs),
        ]
        rc, out = run_subprocess(cmd)
        if rc != 0:
            return JSONResponse(status_code=500, content={"error": out})
        rttms = list(out_dir.glob("*.rttm"))
        if not rttms:
            return JSONResponse(status_code=500, content={"error": "No RTTM produced"})
        return {"rttm": rttms[0].read_text(encoding="utf-8")}


# --- /summarize ---
@app.post("/summarize")
async def summarize(
    spk_json: UploadFile = File(...),
    backend: str = Form("ollama"),
    jobs: int = Form(1),
    auth: bool = Depends(check_auth),
):
    """Summarize .spk.json file using summarize_structured.py."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spk_path = Path(tmpdir) / spk_json.filename
        with open(spk_path, "wb") as f:
            shutil.copyfileobj(spk_json.file, f)
        out_dir = Path(tmpdir) / "summaries"
        out_dir.mkdir()
        # Call summarize_structured.py
        cmd = [
            sys.executable,
            "localtrans/summarize_structured.py",
            "--src",
            str(tmpdir),
            "--out",
            str(out_dir),
            "--backend",
            backend,
            "--jobs",
            str(jobs),
        ]
        rc, out = run_subprocess(cmd)
        if rc != 0:
            return JSONResponse(status_code=500, content={"error": out})
        mds = list(out_dir.glob("*.spk.summary.md"))
        if not mds:
            return JSONResponse(
                status_code=500, content={"error": "No summary produced"}
            )
        return {"summary_md": mds[0].read_text(encoding="utf-8")}


# --- /assign_roles ---
@app.post("/assign_roles")
async def assign_roles(
    spk_json: UploadFile = File(...),
    labels: str = Form("Менеджер,Клиент,Саппорт,Другое"),
    mode: str = Form("local"),
    auth: bool = Depends(check_auth),
):
    """Assign roles to speakers using assign_roles_with_ollama.py."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spk_path = Path(tmpdir) / spk_json.filename
        with open(spk_path, "wb") as f:
            shutil.copyfileobj(spk_json.file, f)
        out_dir = Path(tmpdir) / "roles"
        out_dir.mkdir()
        # Call assign_roles_with_ollama.py
        cmd = [
            sys.executable,
            "localtrans/assign_roles_with_ollama.py",
            "--src",
            str(tmpdir),
            "--out",
            str(out_dir),
            "--labels",
            labels,
            "--mode",
            mode,
        ]


# --- /pipeline: full process ---
@app.post("/pipeline")
async def pipeline(
    file: UploadFile = File(...),
    hf_token: str = Form(...),
    backend: str = Form("ollama"),
    labels: str = Form("Менеджер,Клиент,Саппорт,Другое"),
    lang: str = Form("ru"),
    device: str = Form("auto"),
    auth: bool = Depends(check_auth),
):
    """Full pipeline: convert to wav, transcribe, diarize, assign roles, return final JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Save input file
        orig_path = Path(tmpdir) / file.filename
        with open(orig_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        # 2. Convert to wav (if not wav)
        if not orig_path.suffix.lower() == ".wav":
            wav_path = Path(tmpdir) / (orig_path.stem + ".wav")
            cmd = ["ffmpeg", "-y", "-i", str(orig_path), str(wav_path)]
            rc, out = run_subprocess(cmd)
            if rc != 0:
                return JSONResponse(
                    status_code=500, content={"error": "ffmpeg: " + out}
                )
        else:
            wav_path = orig_path
        # 3. Transcribe
        trans_out = Path(tmpdir) / "trans_out"
        trans_out.mkdir()
        cmd = [
            sys.executable,
            "localtrans/transcribe_local_whispercpp.py",
            "--audio-root",
            str(tmpdir),
            "--out-root",
            str(trans_out),
            "--whisper-bin",
            os.environ.get("WHISPER_BIN", "../whisper.cpp/build/bin/whisper-cli"),
            "--model",
            os.environ.get("WHISPER_MODEL", "../whisper.cpp/models/ggml-large-v3.bin"),
            "--lang",
            lang,
            "--threads",
            "4",
            "--jobs",
            "1",
        ]
        rc, out = run_subprocess(cmd)
        if rc != 0:
            return JSONResponse(
                status_code=500, content={"error": "transcribe: " + out}
            )
        segs = list(trans_out.glob("*.segments.json"))
        if not segs:
            return JSONResponse(status_code=500, content={"error": "No segments.json"})
        # 4. Diarize
        diar_out = Path(tmpdir) / "rttm"
        diar_out.mkdir()
        cmd = [
            sys.executable,
            "localtrans/diarize_pyannote.py",
            "--src",
            str(tmpdir),
            "--dst",
            str(diar_out),
            "--hf-token",
            hf_token,
            "--device",
            device,
            "--jobs",
            "1",
        ]
        rc, out = run_subprocess(cmd)
        if rc != 0:
            return JSONResponse(status_code=500, content={"error": "diarize: " + out})
        rttms = list(diar_out.glob("*.rttm"))
        if not rttms:
            return JSONResponse(status_code=500, content={"error": "No RTTM"})
        # 5. Merge diarization and transcript
        merge_out = Path(tmpdir) / "merged"
        merge_out.mkdir()
        cmd = [
            sys.executable,
            "localtrans/merge_diar_any.py",
            "--transcripts",
            str(trans_out),
            "--rttm",
            str(diar_out),
            "--out",
            str(merge_out),
        ]
        rc, out = run_subprocess(cmd)
        if rc != 0:
            return JSONResponse(status_code=500, content={"error": "merge: " + out})
        spk_jsons = list(merge_out.glob("*.spk.json"))
        if not spk_jsons:
            return JSONResponse(
                status_code=500, content={"error": "No .spk.json after merge"}
            )
        # 6. Assign roles
        roles_out = Path(tmpdir) / "roles"
        roles_out.mkdir()
        cmd = [
            sys.executable,
            "localtrans/assign_roles_with_ollama.py",
            "--src",
            str(merge_out),
            "--out",
            str(roles_out),
            "--labels",
            labels,
            "--mode",
            backend,
        ]
        rc, out = run_subprocess(cmd)
        if rc != 0:
            return JSONResponse(
                status_code=500, content={"error": "assign_roles: " + out}
            )
        roles_jsons = list(roles_out.glob("*.roles.json"))
        # 7. Return all results
        return {
            "segments_json": segs[0].read_text(encoding="utf-8") if segs else None,
            "rttm": rttms[0].read_text(encoding="utf-8") if rttms else None,
            "spk_json": spk_jsons[0].read_text(encoding="utf-8") if spk_jsons else None,
            "roles_json": (
                roles_jsons[0].read_text(encoding="utf-8") if roles_jsons else None
            ),
        }
        rc, out = run_subprocess(cmd)
        if rc != 0:
            return JSONResponse(status_code=500, content={"error": out})
        roles = list(out_dir.glob("*.roles.json"))
        if not roles:
            return JSONResponse(
                status_code=500, content={"error": "No roles.json produced"}
            )
        return {"roles_json": roles[0].read_text(encoding="utf-8")}


# --- Health check ---
@app.get("/")
def root():
    return {
        "status": "ok",
        "endpoints": ["/transcribe", "/diarize", "/summarize", "/assign_roles"],
    }
