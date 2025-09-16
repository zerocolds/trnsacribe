#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import math
import re
import subprocess
import sys
import unicodedata
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

VIDEO_EXTS_DEFAULT = {
    ".mp4",
    ".mov",
    ".m4v",
    ".mkv",
    ".avi",
    ".webm",
    ".mts",
    ".m2ts",
    ".ts",
    ".flv",
    # иногда прилетает «чистое аудио» — пусть тоже обрабатывается
    ".m4a",
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".aac",
}

# -------------------- utils --------------------


def run_cmd(cmd: list[str]) -> Tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def ffprobe_json(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    code, out = run_cmd(cmd)
    if code != 0:
        raise RuntimeError(f"ffprobe failed:\n{out[-4000:]}")
    return json.loads(out)


def get_duration_sec(path: Path) -> float:
    info = ffprobe_json(path)
    dur = info.get("format", {}).get("duration")
    if dur:
        try:
            return float(dur)
        except Exception:
            pass
    for s in info.get("streams", []):
        if s.get("codec_type") == "audio" and "duration" in s:
            try:
                return float(s["duration"])
            except Exception:
                pass
    return 0.0


def ascii_slug(text: str, salt: str) -> str:
    s = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
    h = hashlib.md5(salt.encode("utf-8")).hexdigest()[:8]
    base = s or "file"
    if len(base) > 64:
        base = base[:64].rstrip("_-.")
    return f"{base}__{h}"


@dataclass
class PartInfo:
    path: str  # относительный путь от dst_root
    duration: float
    offset: float


@dataclass
class ManifestItem:
    src_rel: str
    base_out_rel: str  # путь без расширения (относительно dst_root)
    codec: str
    container: str
    sample_rate: int
    channels: int
    bitrate: str
    chunked: bool
    size_bytes: int
    duration: float
    parts: List[PartInfo]


# -------------------- encoders --------------------


def encode_local_wav(
    input_path: Path,
    output_path: Path,
    sr: int,
    ch: int,
    loudnorm: bool,
    audio_stream: Optional[int],
    overwrite: bool,
) -> Tuple[bool, str]:
    """Локальный режим для whisper.cpp: WAV PCM16 16k mono, без резки."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return True, "skip"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y" if overwrite else "-n",
        "-analyzeduration",
        "200M",
        "-probesize",
        "200M",
        "-fflags",
        "+genpts",
        "-err_detect",
        "ignore_err",
        "-i",
        str(input_path),
        "-vn",
    ]
    if audio_stream is not None:
        cmd += ["-map", f"0:a:{audio_stream}"]
    if loudnorm:
        cmd += ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"]

    cmd += ["-ac", str(ch), "-ar", str(sr), "-c:a", "pcm_s16le", str(output_path)]
    code, out = run_cmd(cmd)
    if code == 0:
        return True, "ok"
    return False, out[-4000:]


def encode_openai_m4a(
    input_path: Path,
    output_path: Path,
    sr: int,
    ch: int,
    bitrate: str,
    loudnorm: bool,
    audio_stream: Optional[int],
    overwrite: bool,
) -> Tuple[bool, str]:
    """Режим под OpenAI API: AAC (m4a) для совместимости и экономии веса."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return True, "skip"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y" if overwrite else "-n",
        "-analyzeduration",
        "200M",
        "-probesize",
        "200M",
        "-fflags",
        "+genpts",
        "-err_detect",
        "ignore_err",
        "-i",
        str(input_path),
        "-vn",
    ]
    if audio_stream is not None:
        cmd += ["-map", f"0:a:{audio_stream}"]
    if loudnorm:
        cmd += ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"]

    cmd += [
        "-ac",
        str(ch),
        "-ar",
        str(sr),
        "-c:a",
        "aac",
        "-b:a",
        bitrate,
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    code, out = run_cmd(cmd)
    if code == 0:
        return True, "ok"
    return False, out[-4000:]


def segment_by_time(
    input_file: Path, base_out_noext: Path, segment_time: int, ext: str, overwrite: bool
) -> Tuple[List[Path], Optional[str]]:
    """
    Резка по времени (секундам). Работает и для WAV (PCM), и для M4A.
    """
    pattern = str(base_out_noext) + f".part_%03d{ext}"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y" if overwrite else "-n",
        "-i",
        str(input_file),
        "-f",
        "segment",
        "-segment_time",
        str(segment_time),
        "-c",
        "copy",
        "-reset_timestamps",
        "1",
        pattern,
    ]
    code, out = run_cmd(cmd)
    if code != 0:
        return ([], out[-4000:])
    parts = sorted(base_out_noext.parent.glob(base_out_noext.name + f".part_*{ext}"))
    return (parts, None)


def compute_segment_time_for_size(
    actual_size_bytes: int, duration_sec: float, size_limit_bytes: int
) -> int:
    if actual_size_bytes <= 0 or duration_sec <= 0:
        return 1200
    ratio = size_limit_bytes / float(actual_size_bytes)
    approx = int(max(60, math.floor(duration_sec * ratio * 0.95)))
    return min(approx, int(max(60, duration_sec)))


# -------------------- pipeline --------------------


def rel_output_path(
    src_dir: Path, file: Path, dst_dir: Path, out_ext: str, ascii_names: bool
) -> Path:
    rel = file.relative_to(src_dir)
    if ascii_names:
        slug = ascii_slug(rel.stem, str(rel))
        rel = rel.with_name(slug).with_suffix(out_ext)
    else:
        rel = rel.with_suffix(out_ext)
    return dst_dir / rel


def task_convert_one(
    inp: Path,
    src_dir: Path,
    dst_dir: Path,
    mode: str,
    sr: int,
    ch: int,
    bitrate: str,
    loudnorm: bool,
    audio_stream: Optional[int],
    overwrite: bool,
    jobs_chunk_seconds: Optional[int],
    size_limit_mb: float,
    ascii_names: bool,
) -> Tuple[str, Path, str]:
    try:
        if mode == "local":
            out_ext = ".wav"
        else:
            out_ext = ".m4a"

        out_path = rel_output_path(src_dir, inp, dst_dir, out_ext, ascii_names)
        base_noext = out_path.with_suffix("")

        # 1) кодирование
        if mode == "local":
            ok, msg = encode_local_wav(
                inp, out_path, sr, ch, loudnorm, audio_stream, overwrite
            )
        else:
            ok, msg = encode_openai_m4a(
                inp, out_path, sr, ch, bitrate, loudnorm, audio_stream, overwrite
            )

        if not ok:
            return ("err", inp, f"encode failed:\n{msg}")

        # 2) если файл пуст/бит
        if not out_path.exists() or out_path.stat().st_size == 0:
            return ("err", inp, "no/empty output after encode")

        dur = get_duration_sec(out_path)
        size_bytes = out_path.stat().st_size

        # 3) локально: резка выключена, если не задан --chunk-seconds
        if mode == "local":
            if jobs_chunk_seconds and jobs_chunk_seconds > 0:
                parts, seg_err = segment_by_time(
                    out_path, base_noext, jobs_chunk_seconds, out_ext, overwrite
                )
                if seg_err:
                    return ("err", inp, f"segment failed:\n{seg_err}")
                if not parts:
                    return ("err", inp, "segment produced no parts")

                parts_info: List[PartInfo] = []
                offset = 0.0
                total_bytes = 0
                for p in parts:
                    d = get_duration_sec(p)
                    parts_info.append(
                        PartInfo(
                            path=str(p.relative_to(dst_dir)),
                            duration=float(d),
                            offset=float(offset),
                        )
                    )
                    offset += float(d)
                    total_bytes += p.stat().st_size

                man = ManifestItem(
                    src_rel=str(inp.relative_to(src_dir)),
                    base_out_rel=str(base_noext.relative_to(dst_dir)),
                    codec="pcm_s16le",
                    container="wav",
                    sample_rate=sr,
                    channels=ch,
                    bitrate="",
                    chunked=True,
                    size_bytes=total_bytes,
                    duration=dur,
                    parts=parts_info,
                )
                base_noext.with_suffix(".manifest.json").write_text(
                    json.dumps(asdict(man), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                # удаляем общий WAV, если порезали (оставляем только части)
                try:
                    out_path.unlink()
                except Exception:
                    pass

                return ("ok", inp, f"{len(parts)} parts (local)")
            else:
                man = ManifestItem(
                    src_rel=str(inp.relative_to(src_dir)),
                    base_out_rel=str(base_noext.relative_to(dst_dir)),
                    codec="pcm_s16le",
                    container="wav",
                    sample_rate=sr,
                    channels=ch,
                    bitrate="",
                    chunked=False,
                    size_bytes=size_bytes,
                    duration=dur,
                    parts=[],
                )
                base_noext.with_suffix(".manifest.json").write_text(
                    json.dumps(asdict(man), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return ("ok", inp, str(out_path))

        # 4) режим openai: резка по лимиту размера (как раньше)
        size_limit_bytes = int(size_limit_mb * 1024 * 1024)
        if size_bytes <= size_limit_bytes:
            man = ManifestItem(
                src_rel=str(inp.relative_to(src_dir)),
                base_out_rel=str(base_noext.relative_to(dst_dir)),
                codec="aac",
                container="m4a",
                sample_rate=sr,
                channels=ch,
                bitrate=bitrate,
                chunked=False,
                size_bytes=size_bytes,
                duration=dur,
                parts=[],
            )
            base_noext.with_suffix(".manifest.json").write_text(
                json.dumps(asdict(man), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return ("ok", inp, str(out_path))

        seg_time = compute_segment_time_for_size(size_bytes, dur, size_limit_bytes)
        parts, seg_err = segment_by_time(
            out_path, base_noext, seg_time, out_ext, overwrite
        )
        if seg_err:
            return ("err", inp, f"segment failed:\n{seg_err}")
        if not parts:
            return ("err", inp, "segment produced no parts")

        parts_info: List[PartInfo] = []
        offset = 0.0
        total_bytes = 0
        for p in parts:
            d = get_duration_sec(p)
            parts_info.append(
                PartInfo(
                    path=str(p.relative_to(dst_dir)),
                    duration=float(d),
                    offset=float(offset),
                )
            )
            offset += float(d)
            total_bytes += p.stat().st_size

        man = ManifestItem(
            src_rel=str(inp.relative_to(src_dir)),
            base_out_rel=str(base_noext.relative_to(dst_dir)),
            codec="aac",
            container="m4a",
            sample_rate=sr,
            channels=ch,
            bitrate=bitrate,
            chunked=True,
            size_bytes=total_bytes,
            duration=dur,
            parts=parts_info,
        )
        base_noext.with_suffix(".manifest.json").write_text(
            json.dumps(asdict(man), ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # удаляем большой m4a, если порезали
        try:
            out_path.unlink()
        except Exception:
            pass

        return ("ok", inp, f"{len(parts)} parts (openai)")
    except Exception as e:
        return ("err", inp, str(e))


# -------------------- CLI --------------------


def collect_inputs(src_dir: Path, exts: set[str]) -> List[Path]:
    return sorted(
        [p for p in src_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    )


def main():
    ap = argparse.ArgumentParser(
        description="Конвертация медиа в аудио. --mode local → WAV для whisper.cpp; --mode openai → M4A + резка по размеру."
    )
    ap.add_argument(
        "--src", required=True, type=Path, help="Папка с видео/аудио (рекурсивно)"
    )
    ap.add_argument("--dst", required=True, type=Path, help="Папка для аудио-вывода")
    ap.add_argument(
        "--exts",
        nargs="*",
        default=None,
        help="Расширения входа (по умолчанию популярные)",
    )
    ap.add_argument(
        "--mode",
        choices=["local", "openai"],
        default="local",
        help="Режим: local (WAV) или openai (M4A+резка)",
    )
    ap.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Частота дискретизации, Гц (16k для whisper — ок)",
    )
    ap.add_argument("--channels", type=int, default=1, help="Каналы (1=mono)")
    ap.add_argument("--bitrate", default="96k", help="[openai] Битрейт AAC")
    ap.add_argument("--loudnorm", action="store_true", help="EBU R128 нормализация")
    ap.add_argument(
        "--audio-stream", type=int, default=None, help="Индекс аудиодорожки, 0=первая"
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="Перезаписывать существующее"
    )
    ap.add_argument("--jobs", type=int, default=1, help="Параллельные задания")
    ap.add_argument(
        "--size-limit-mb", type=float, default=24.0, help="[openai] Лимит размера (MB)"
    )
    ap.add_argument("--ascii-names", action="store_true", help="ASCII-имена + хэш")
    ap.add_argument(
        "--chunk-seconds",
        type=int,
        default=0,
        help="[local] Резать по N сек (0=не резать)",
    )
    args = ap.parse_args()

    # runtime info
    try:
        from localtrans.runtime import (
            IS_GPU,
            BACKEND,
            MODEL_DEVICE,
            MODEL_PATH,
            log_env,
        )
    except Exception:
        IS_GPU = os.environ.get("USE_GPU", "").lower() in ("1", "true", "yes")
        BACKEND = os.environ.get("USE_BACKEND", "local")
        MODEL_DEVICE = "cuda" if IS_GPU else "cpu"
        MODEL_PATH = os.environ.get("MODEL_PATH", "./models")

        def log_env():
            return {
                "IS_GPU": IS_GPU,
                "BACKEND": BACKEND,
                "MODEL_DEVICE": MODEL_DEVICE,
                "MODEL_PATH": MODEL_PATH,
            }

    if os.environ.get("VERBOSE", "").lower() in ("1", "true", "yes"):
        print(f"[ENV] {log_env()}")

    src_dir = args.src.resolve()
    dst_dir = args.dst.resolve()
    if not src_dir.exists():
        print(f"[ERR] Source not found: {src_dir}", file=sys.stderr)
        sys.exit(2)

    exts = set(
        e.lower() if e.startswith(".") else "." + e.lower()
        for e in (args.exts or VIDEO_EXTS_DEFAULT)
    )
    inputs = collect_inputs(src_dir, exts)
    if not inputs:
        print(f"[INFO] No inputs in {src_dir} with exts: {sorted(exts)}")
        return

    print(f"[INFO] Mode: {args.mode}")
    if args.mode == "local":
        print(
            f"[INFO] Output: WAV PCM16 @ {args.sr} Hz, {args.channels} ch (no split; --chunk-seconds={args.chunk_seconds or 0})"
        )

    else:
        print(
            f"[INFO] Output: M4A AAC {args.bitrate} @ {args.sr} Hz, {args.channels} ch; size-limit={args.size_limit_mb} MB"
        )

    print(f"[INFO] Files: {len(inputs)}")
    print(f"[INFO] Out root: {dst_dir}")

    ok = skipped = failed = 0
    tasks = {}
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for inp in inputs:
            fut = ex.submit(
                task_convert_one,
                inp,
                src_dir,
                dst_dir,
                args.mode,
                args.sr,
                args.channels,
                args.bitrate,
                args.loudnorm,
                args.audio_stream,
                args.overwrite,
                args.chunk_seconds,
                args.size_limit_mb,
                args.ascii_names,
            )
            tasks[fut] = inp

        for i, fut in enumerate(as_completed(tasks), 1):
            status, inp, msg = fut.result()
            prefix = f"[{i}/{len(tasks)}]"
            if status == "ok":
                ok += 1
                print(f"{prefix} OK  {inp} → {msg}")
            elif status == "skip":
                skipped += 1
                print(f"{prefix} SKIP {inp} ({msg})")
            else:
                failed += 1
                print(f"{prefix} ERR {inp}\n{msg}\n", file=sys.stderr)

    # Соберём индекс
    all_mans: List[dict] = []
    for man in dst_dir.rglob("*.manifest.json"):
        try:
            all_mans.append(json.loads(man.read_text(encoding="utf-8")))
        except Exception:
            pass
    if all_mans:
        (dst_dir / "_manifest_index.json").write_text(
            json.dumps(all_mans, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"\n[SUMMARY] done={ok}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
