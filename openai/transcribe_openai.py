#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, sys, json, time, io, hashlib, tempfile, shutil
from pathlib import Path
from typing import Optional, Iterable, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from openai import OpenAI

try:
    import httpx
except Exception:
    httpx = None

AUDIO_EXTS = {".m4a", ".wav", ".mp3", ".flac", ".ogg", ".aac", ".webm", ".mp4"}

# ---------- helpers ----------


def ascii_name(stem: str, default: str = "audio") -> str:
    import unicodedata, re

    s = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
    return s or default


def iter_audio(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            yield p


def now() -> str:
    return time.strftime("%H:%M:%S")


def sha1_file(path: Path, bufsize: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_manifest(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"files": {}}  # sha1 -> {"out_base": "...", "size": int, "src": "path"}


def save_manifest(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def make_ascii_shadow(orig_path: Path) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="oai_upload_"))
    shadow = tmpdir / f"{ascii_name(orig_path.stem)}{orig_path.suffix.lower()}"
    try:
        os.link(orig_path, shadow)  # быстрый hardlink
    except Exception:
        shutil.copy2(orig_path, shadow)  # если другая ФС
    return shadow


class ProgressFile(io.BufferedReader):
    def __init__(self, path: Path, bar: Optional[tqdm] = None):
        self._f = open(path, "rb")
        super().__init__(self._f)
        self._len = os.fstat(self._f.fileno()).st_size
        self._bar = bar

    def __len__(self):
        return self._len

    def read(self, amt: int = -1) -> bytes:
        chunk = super().read(amt)
        if self._bar and chunk:
            self._bar.update(len(chunk))
        return chunk

    def reset(self):
        try:
            self.seek(0)
            if self._bar:
                self._bar.reset()
        except Exception:
            pass

    def close(self):
        try:
            if self._bar:
                self._bar.close()
        finally:
            super().close()


# ---------- core ----------


def transcribe_one(
    client: OpenAI,
    path: Path,
    out_dir: Path,
    model: str,
    language: Optional[str],
    timeout_s: int,
    retries: int,
    verbose: bool,
    http1: bool,
    job_index: int,
    total_jobs: int,
    manifest: Dict[str, Any],
    manifest_path: Path,
    on_duplicate: str,  # "skip" | "symlink" | "copy"
    max_upload_mb: Optional[float],
):
    rel = str(path)
    size = path.stat().st_size
    size_mb = size / (1024 * 1024)

    # size gate (например, вы уже нарезали на части <= 24MB и хотите игнорировать "полные" дубликаты)
    if max_upload_mb is not None and size_mb > max_upload_mb:
        if verbose:
            print(
                f"[{now()}] ⏭  skip by size: {rel} ({size_mb:.1f} MB > {max_upload_mb} MB)"
            )
        return True, "skip-size"

    # имена выходников
    out_base = out_dir / path.with_suffix("").name
    out_raw = out_base.with_suffix(".raw.json")
    out_seg = out_base.with_suffix(".segments.json")

    # посчитать sha1 для дедупа
    file_sha1 = sha1_file(path)

    # если этот файл уже обработан ранее (по хэшу) — действуем по политике
    prev = manifest["files"].get(file_sha1)
    if prev:
        prev_base = Path(prev["out_base"])
        prev_seg = prev_base.with_suffix(".segments.json")
        prev_raw = prev_base.with_suffix(".raw.json")

        # сами выходники на месте?
        if prev_seg.exists():
            if on_duplicate == "skip":
                if verbose:
                    print(f"[{now()}] ⏭  duplicate(by sha1): {rel} -> {prev_seg.name}")
                return True, "dup-skip"
            elif on_duplicate == "symlink":
                out_seg.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if out_seg.exists():
                        out_seg.unlink()
                    out_seg.symlink_to(prev_seg)
                    if prev_raw.exists():
                        if out_raw.exists():
                            out_raw.unlink()
                        out_raw.symlink_to(prev_raw)
                    if verbose:
                        print(f"[{now()}] 🔗 duplicate → symlink: {rel}")
                    return True, "dup-symlink"
                except OSError:
                    # если ФС не поддерживает симлинки — падаем в copy
                    on_duplicate = "copy"
            if on_duplicate == "copy":
                out_seg.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(prev_seg, out_seg)
                if prev_raw.exists():
                    shutil.copy2(prev_raw, out_raw)
                if verbose:
                    print(f"[{now()}] 📄 duplicate → copy: {rel}")
                return True, "dup-copy"

    # если конкретный out уже есть — не платим повторно
    if out_seg.exists():
        if verbose:
            print(f"[{now()}] ⏭  already exists: {out_seg}")
        # всё равно запишем в манифест связь sha1 -> out_base
        manifest["files"][file_sha1] = {
            "out_base": str(out_base),
            "size": size,
            "src": rel,
        }
        save_manifest(manifest_path, manifest)
        return True, "skip-exists"

    # обычная загрузка
    if verbose:
        print(f"[{now()}] ▶ [{job_index}/{total_jobs}] {rel}  ({size_mb:.1f} MB)")

    want_verbose = model == "whisper-1"
    response_format = "verbose_json" if want_verbose else "json"
    ts_gran = ["segment", "word"] if want_verbose else None

    shadow = make_ascii_shadow(path)

    upbar = None
    pf = None
    try:
        upbar = tqdm(
            total=size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"UPLOAD {Path(shadow).name}",
            leave=False,
            position=1,
        )
        pf = ProgressFile(shadow, upbar)

        attempt = 0
        while True:
            try:
                if verbose:
                    print(f"[{now()}] ⇡  upload → {model}  (timeout={timeout_s}s)")
                resp = client.audio.transcriptions.create(
                    model=model,
                    file=pf,
                    language=language or None,
                    response_format=response_format,
                    temperature=0,
                    timestamp_granularities=ts_gran,
                    timeout=timeout_s,
                )
                if verbose:
                    print(f"[{now()}] ⌛ decoding response")

                try:
                    raw = resp.model_dump() if hasattr(resp, "model_dump") else resp
                except Exception:
                    raw = json.loads(str(resp))

                out_seg.parent.mkdir(parents=True, exist_ok=True)
                out_raw.write_text(
                    json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
                )

                # нормализация сегментов
                segments = []
                if want_verbose:
                    for s in raw.get("segments", []):
                        st = float(s.get("start", 0.0))
                        en = float(s.get("end", st))
                        txt = s.get("text", "")
                        segments.append({"start": st, "end": en, "text": txt})
                else:
                    txt = raw.get("text", "")
                    segments.append({"start": 0.0, "end": 0.0, "text": txt})

                out_seg.write_text(
                    json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
                )

                # обновить манифест (чтобы дубликаты дальше не улетали)
                manifest["files"][file_sha1] = {
                    "out_base": str(out_base),
                    "size": size,
                    "src": rel,
                }
                save_manifest(manifest_path, manifest)

                if verbose:
                    print(
                        f"[{now()}] ✅ {rel} → {out_seg.name}  (segments={len(segments)})"
                    )
                return True, "ok"
            except Exception as e:
                attempt += 1
                if verbose:
                    print(f"[{now()}] ❌ attempt {attempt}/{retries}: {e}")
                if attempt >= max(1, retries):
                    raise
                pf.reset()
                time.sleep(min(60, 2**attempt))
    finally:
        try:
            if pf:
                pf.close()
        except Exception:
            pass
        try:
            if shadow.exists():
                tmpdir = shadow.parent
                shadow.unlink(missing_ok=True)
                try:
                    tmpdir.rmdir()
                except Exception:
                    pass
        except Exception:
            pass


# ---------- main ----------


def main():
    ap = argparse.ArgumentParser(
        description="Transcribe folder with OpenAI, with hash-based deduplication."
    )
    ap.add_argument(
        "--src", required=True, type=Path, help="Folder with audio (m4a/wav/mp3/mp4/…)"
    )
    ap.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Where to write *.raw.json / *.segments.json",
    )
    ap.add_argument(
        "--model", default="whisper-1", help="OpenAI model (default whisper-1)"
    )
    ap.add_argument("--language", default=None, help="Language hint, e.g. 'ru'")
    ap.add_argument(
        "--timeout", type=int, default=1200, help="Per-file timeout seconds"
    )
    ap.add_argument("--retries", type=int, default=3, help="Retries on errors")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel uploads")
    ap.add_argument("--verbose", action="store_true", help="Verbose logs")
    ap.add_argument("--http1", action="store_true", help="Force HTTP/1.1 (disable h2)")
    ap.add_argument(
        "--skip-exists",
        action="store_true",
        help="(kept for compatibility; dedupe covers it)",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to manifest JSON (default: <out>/.manifest.json)",
    )
    ap.add_argument(
        "--on-duplicate",
        choices=["skip", "symlink", "copy"],
        default="skip",
        help="What to do when a file with the same content was already processed",
    )
    ap.add_argument(
        "--max-upload-mb",
        type=float,
        default=None,
        help="Skip files larger than this (e.g., 24 for OpenAI upload limit when you have chunks)",
    )
    args = ap.parse_args()

    src = args.src.resolve()
    out = args.out.resolve()
    files = list(iter_audio(src))
    if not files:
        print(f"[INFO] no audio in {src}")
        return

    manifest_path = (args.manifest or (out / ".manifest.json")).resolve()
    manifest = load_manifest(manifest_path)

    http_client = None
    if args.http1 and httpx is not None:
        http_client = httpx.Client(http2=False, timeout=args.timeout)
    client = OpenAI(timeout=args.timeout, http_client=http_client)

    pbar = tqdm(total=len(files), desc="Transcribe", unit="file")
    ok = err = 0

    if args.jobs <= 1:
        for idx, path in enumerate(files, 1):
            try:
                success, _ = transcribe_one(
                    client,
                    path,
                    out,
                    args.model,
                    args.language,
                    args.timeout,
                    args.retries,
                    args.verbose,
                    args.http1,
                    idx,
                    len(files),
                    manifest,
                    manifest_path,
                    args.on_duplicate,
                    args.max_upload_mb,
                )
                ok += 1 if success else 0
            except Exception as e:
                err += 1
                print(f"[ERR] {path}: {e}", file=sys.stderr)
            finally:
                pbar.update(1)
    else:

        def worker(i_path_tuple):
            i, path = i_path_tuple
            try:
                return (
                    transcribe_one(
                        client,
                        path,
                        out,
                        args.model,
                        args.language,
                        args.timeout,
                        args.retries,
                        args.verbose,
                        args.http1,
                        i,
                        len(files),
                        manifest,
                        manifest_path,
                        args.on_duplicate,
                        args.max_upload_mb,
                    )[0],
                    path,
                )
            except Exception as e:
                return False, (path, e)

        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(worker, (i + 1, p)): p for i, p in enumerate(files)}
            for fut in as_completed(futs):
                res, info = fut.result()
                pbar.update(1)
                if res:
                    ok += 1
                else:
                    err += 1
                    if isinstance(info, tuple):
                        path, e = info
                        print(f"[ERR] {path}: {e}", file=sys.stderr)
                    else:
                        print(f"[ERR] {info}", file=sys.stderr)

    pbar.close()
    print(f"[SUMMARY] ok={ok}, err={err}")


if __name__ == "__main__":
    main()
