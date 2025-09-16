#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import re
import shutil
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
}

# ---------- утилиты ----------


def run_cmd(cmd: list[str]) -> Tuple[int, str]:
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    return proc.returncode, proc.stdout


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
    if dur is not None:
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
    """
    ASCII-имя + короткий хэш от 'salt' (обычно относительный путь).
    Гарантирует уникальность и убирает коллизии.
    """
    s = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
    h = hashlib.md5(salt.encode("utf-8")).hexdigest()[:8]
    base = s or "file"
    if len(base) > 64:
        base = base[:64].rstrip("_-.")
    return f"{base}__{h}"


def clean_component(comp: str, strip_spaces: bool) -> str:
    # убираем ведущие/хвостовые пробелы и сжимаем внутренние последовательности пробелов
    if strip_spaces:
        comp = comp.strip()
    return comp


@dataclass
class PartInfo:
    path: str  # путь к части (относительно dst_root), с расширением .m4a
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


# ---------- кодеки/резка ----------


def encode_audio_primary(
    input_path: Path,
    output_path: Path,
    sr: int,
    ch: int,
    bitrate: str,
    loudnorm: bool,
    audio_stream: Optional[int],
    overwrite: bool,
) -> Tuple[bool, str]:
    """
    Основная попытка: AAC (m4a) с нормализацией по желанию.
    """
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


def encode_audio_fallback(
    input_path: Path,
    output_path: Path,
    sr: int,
    ch: int,
    bitrate: str,
    audio_stream: Optional[int],
    overwrite: bool,
) -> Tuple[bool, str]:
    """
    Фоллбэк для битых контейнеров: агрессивные опции часов/таймстампов.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return True, "skip"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y" if overwrite else "-n",
        "-analyzeduration",
        "400M",
        "-probesize",
        "400M",
        "-fflags",
        "+genpts+igndts",
        "-use_wallclock_as_timestamps",
        "1",
        "-err_detect",
        "ignore_err",
        "-i",
        str(input_path),
        "-vn",
    ]
    if audio_stream is not None:
        cmd += ["-map", f"0:a:{audio_stream}"]
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
        return True, "ok-fallback"
    return False, out[-4000:]


def segment_m4a_by_time(
    input_m4a: Path, base_out_noext: Path, segment_time: int, overwrite: bool
) -> Tuple[List[Path], Optional[str]]:
    """
    Режем без рекодирования (-c copy), сбрасываем таймстампы в сегментах.
    """
    pattern = str(base_out_noext) + ".part_%03d.m4a"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y" if overwrite else "-n",
        "-i",
        str(input_m4a),
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
    parts = sorted(base_out_noext.parent.glob(base_out_noext.name + ".part_*.m4a"))
    return (parts, None)


def compute_segment_time(
    actual_size_bytes: int, duration_sec: float, size_limit_bytes: int
) -> int:
    """
    Прикидываем длину сегмента, чтобы средний кусок был <= size_limit_bytes.
    Добавляем запас ~5%.
    """
    if actual_size_bytes <= 0 or duration_sec <= 0:
        return 1200  # 20 мин
    ratio = size_limit_bytes / float(actual_size_bytes)
    approx = int(max(60, math.floor(duration_sec * ratio * 0.95)))
    return min(approx, int(max(60, duration_sec)))


# ---------- нормализация путей ----------


def normalize_rel(src_dir: Path, file: Path, strip_dir_spaces: bool) -> Path:
    """
    Нормализуем относительный путь: режем ведущие/хвостовые пробелы у КАЖДОГО компонента.
    (структуру директорий сохраняем)
    """
    rel = file.relative_to(src_dir)
    parts = [clean_component(p, strip_dir_spaces) for p in rel.parts]
    return Path(*parts)


def rel_output_path(
    src_dir: Path,
    file: Path,
    dst_dir: Path,
    ascii_names: bool,
    strip_dir_spaces: bool,
) -> Path:
    rel = normalize_rel(src_dir, file, strip_dir_spaces)
    if ascii_names:
        salt = str(rel)
        slug = ascii_slug(rel.stem, salt)
        rel = rel.with_name(slug).with_suffix(".m4a")
    else:
        rel = rel.with_suffix(".m4a")
    return dst_dir / rel


# ---------- пайплайн одного файла ----------


def task_convert_one(
    inp: Path,
    src_dir: Path,
    dst_dir: Path,
    sr: int,
    ch: int,
    bitrate: str,
    loudnorm: bool,
    audio_stream: Optional[int],
    overwrite: bool,
    size_limit_mb: float,
    ascii_names: bool,
    strip_dir_spaces: bool,
    force_segment_sec: Optional[int],
) -> Tuple[str, Path, str]:
    try:
        out_m4a = rel_output_path(src_dir, inp, dst_dir, ascii_names, strip_dir_spaces)
        print(f"→ SRC: {inp}\n  DST: {out_m4a}")

        ok, msg = encode_audio_primary(
            inp, out_m4a, sr, ch, bitrate, loudnorm, audio_stream, overwrite
        )
        if not ok:
            print(f"  [WARN] primary encode failed, fallback...\n  {msg}")
            ok2, msg2 = encode_audio_fallback(
                inp, out_m4a, sr, ch, bitrate, audio_stream, overwrite
            )
            if not ok2:
                return ("err", inp, f"encode failed:\n{msg}\n--- fallback ---\n{msg2}")

        # размер/длительность
        size_limit_bytes = int(size_limit_mb * 1024 * 1024)
        try:
            actual_size = out_m4a.stat().st_size
        except FileNotFoundError:
            return ("err", inp, "no output after encode")

        if actual_size == 0:
            return ("err", inp, "empty output file")

        try:
            dur = get_duration_sec(out_m4a)
        except Exception as e:
            return ("err", inp, f"ffprobe on output failed:\n{e}")

        base_noext = out_m4a.with_suffix("")

        # Решение о разбиении: либо принудительно, либо по размеру
        need_chunk = False
        seg_time = None
        if force_segment_sec and force_segment_sec > 0:
            need_chunk = True
            seg_time = int(max(30, force_segment_sec))
        elif actual_size > size_limit_bytes:
            need_chunk = True
            seg_time = compute_segment_time(actual_size, dur, size_limit_bytes)

        if not need_chunk:
            manifest = ManifestItem(
                src_rel=str(normalize_rel(src_dir, inp, strip_dir_spaces)),
                base_out_rel=str(base_noext.relative_to(dst_dir)),
                codec="aac",
                container="m4a",
                sample_rate=sr,
                channels=ch,
                bitrate=bitrate,
                chunked=False,
                size_bytes=actual_size,
                duration=dur,
                parts=[],
            )
            man_path = base_noext.with_suffix(".manifest.json")
            man_path.parent.mkdir(parents=True, exist_ok=True)
            man_path.write_text(
                json.dumps(asdict(manifest), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return ("ok", inp, str(out_m4a))

        # иначе режем на части
        parts, seg_err = segment_m4a_by_time(out_m4a, base_noext, seg_time, overwrite)
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
                    path=str(p.relative_to(dst_dir)),  # с расширением
                    duration=float(d),
                    offset=float(offset),
                )
            )
            offset += float(d)
            total_bytes += p.stat().st_size

        manifest = ManifestItem(
            src_rel=str(normalize_rel(src_dir, inp, strip_dir_spaces)),
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
        man_path = base_noext.with_suffix(".manifest.json")
        man_path.write_text(
            json.dumps(asdict(manifest), ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # удаляем крупный цельный m4a (чтобы не путать пайплайн)
        try:
            out_m4a.unlink()
        except Exception:
            pass

        return ("ok", inp, f"{len(parts)} parts (≈{seg_time}s each)")
    except FileNotFoundError as e:
        return ("err", inp, f"ffmpeg/ffprobe not found? {e}")
    except Exception as e:
        return ("err", inp, f"{e}")


# ---------- CLI ----------


def collect_inputs(src_dir: Path, exts: set[str]) -> List[Path]:
    return sorted(
        [p for p in src_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    )


def main():
    ap = argparse.ArgumentParser(
        description="Extract audio → m4a (AAC, 16kHz mono). Если > size-limit (24MB) — разрезает на части и пишет манифест."
    )
    ap.add_argument(
        "--src", required=True, type=Path, help="Папка с видео (рекурсивно)"
    )
    ap.add_argument("--dst", required=True, type=Path, help="Папка для аудио (m4a)")
    ap.add_argument(
        "--exts",
        nargs="*",
        default=None,
        help="Расширения видео. По умолчанию — популярные.",
    )
    ap.add_argument("--sr", type=int, default=16000, help="Частота дискретизации, Гц")
    ap.add_argument("--channels", type=int, default=1, help="Число каналов (1=mono)")
    ap.add_argument("--bitrate", default="96k", help="Битрейт AAC (напр. 96k)")
    ap.add_argument(
        "--loudnorm",
        action="store_true",
        help="Включить EBU R128 нормализацию (ffmpeg loudnorm)",
    )
    ap.add_argument(
        "--audio-stream", type=int, default=None, help="Индекс аудиодорожки (0=первая)"
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="Перезаписывать существующие файлы"
    )
    ap.add_argument("--jobs", type=int, default=1, help="Параллельные задания")
    ap.add_argument(
        "--size-limit-mb", type=float, default=24.0, help="Лимит размера части (MB)"
    )
    ap.add_argument(
        "--ascii-names",
        action="store_true",
        help="ASCII-имена выходных файлов + хэш (рекомендуется)",
    )
    ap.add_argument(
        "--no-ascii-names",
        action="store_true",
        help="Отключить ASCII-имена (перебивает --ascii-names)",
    )
    ap.add_argument(
        "--strip-dir-spaces",
        action="store_true",
        help="Срезать ведущие/хвостовые пробелы у директорий (вкл. по умолчанию)",
    )
    ap.add_argument(
        "--no-strip-dir-spaces",
        action="store_true",
        help="Не трогать пробелы в именах директорий",
    )
    ap.add_argument(
        "--force-segment-sec",
        type=int,
        default=None,
        help="Принудительная длина части (сек). Если задано — режем всегда.",
    )
    args = ap.parse_args()

    # дефолты с учётом взаимоисключающих флагов
    ascii_names = True
    if args.no_ascii_names:
        ascii_names = False
    elif args.ascii_names:
        ascii_names = True

    strip_dir_spaces = True
    if args.no_strip_dir_spaces:
        strip_dir_spaces = False
    elif args.strip_dir_spaces:
        strip_dir_spaces = True

    src_dir = args.src.resolve()
    dst_dir = args.dst.resolve()
    if not src_dir.exists():
        print(f"[ERR] Source folder not found: {src_dir}", file=sys.stderr)
        sys.exit(2)

    exts = set(
        e.lower() if e.startswith(".") else "." + e.lower()
        for e in (args.exts or VIDEO_EXTS_DEFAULT)
    )
    inputs = collect_inputs(src_dir, exts)
    if not inputs:
        print(f"[INFO] No input videos found in {src_dir} with exts: {sorted(exts)}")
        return

    print(
        f"[INFO] Found {len(inputs)} files → m4a @ {args.sr} Hz, {args.channels} ch, aac {args.bitrate}, "
        f"size-limit={args.size_limit_mb} MB, ascii_names={ascii_names}, strip_dir_spaces={strip_dir_spaces}, "
        f"force_segment_sec={args.force_segment_sec}"
    )
    print(f"[INFO] Output root: {dst_dir}")

    ok = skipped = failed = 0
    tasks = {}
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for inp in inputs:
            tasks[
                ex.submit(
                    task_convert_one,
                    inp,
                    src_dir,
                    dst_dir,
                    args.sr,
                    args.channels,
                    args.bitrate,
                    args.loudnorm,
                    args.audio_stream,
                    args.overwrite,
                    args.size_limit_mb,
                    ascii_names,
                    strip_dir_spaces,
                    args.force_segment_sec,
                )
            ] = inp

        for fut in as_completed(tasks):
            status, inp, msg = fut.result()
            if status == "ok":
                ok += 1
                print(f"[OK] {inp} → {msg}")
            elif status == "skip":
                skipped += 1
                print(f"[SKIP] {inp} ({msg})")
            else:
                failed += 1
                print(f"[ERR] {inp}\n{msg}\n", file=sys.stderr)

    # индекс всех манифестов
    all_manifests: List[dict] = []
    for man in dst_dir.rglob("*.manifest.json"):
        try:
            data = json.loads(man.read_text(encoding="utf-8"))
            all_manifests.append(data)
        except Exception:
            pass
    if all_manifests:
        (dst_dir / "_manifest_index.json").write_text(
            json.dumps(all_manifests, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"\n[SUMMARY] done={ok}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
