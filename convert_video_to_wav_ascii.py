#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple, List, Set

# Какие видео считаем входом
VIDEO_EXTS_DEFAULT: Set[str] = {
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

# ---------------------- Транслитерация и очистка путей ----------------------

_RU2LAT = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "yo",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "kh",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "shch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}


def _transliterate_ru(text: str) -> str:
    """Грубая транслитерация ru→lat без внешних библиотек, с сохранением регистра первой буквы."""
    out = []
    for ch in text:
        base = _RU2LAT.get(ch.lower())
        if not base:
            out.append(ch)
            continue
        if ch.isupper():
            # Первую латинскую букву делаем заглавной (Zh → Zh)
            base = base[:1].upper() + base[1:]
        out.append(base)
    return "".join(out)


_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_component(name: str) -> str:
    # Транслит → пробелы в "_" → выкинуть лишнее → сжать "_"
    s = _transliterate_ru(name)
    s = s.replace(" ", "_")
    s = _SAFE_RE.sub("_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("._-")
    return s or "item"


def ascii_rel_path(src_root: Path, p: Path, add_hash: bool) -> Path:
    """Зеркалим структуру, транслитерируя каждый компонент. На файле — опционально добавляем короткий хэш."""
    rel = p.relative_to(src_root)
    parts = list(rel.parts)
    # каталоги
    clean_dirs = [sanitize_component(x) for x in parts[:-1]]
    # файл
    stem = p.stem
    ext = p.suffix  # исходное расширение, но нам всё равно потом станет .wav
    clean_stem = sanitize_component(stem)
    if add_hash:
        salt = rel.as_posix()
        h = hashlib.md5(salt.encode("utf-8")).hexdigest()[:8]
        clean_stem = f"{clean_stem}__{h}"
    clean_name = clean_stem + ext
    return Path(*clean_dirs, clean_name)


# ---------------------- ffmpeg ----------------------


def run_ffmpeg_to_wav(
    input_path: Path,
    output_path: Path,
    sr: int,
    ch: int,
    loudnorm: bool,
    audio_stream: int | None,
    overwrite: bool,
) -> Tuple[str, Path, str]:
    """
    Декодируем аудио из видео в WAV PCM16, с устойчивыми флагами чтения «битых» контейнеров.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return ("skip", input_path, "exists")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y" if overwrite else "-n",
        # чуть агрессивнее анализ для кривых mp4
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
        "pcm_s16le",
        str(output_path),
    ]

    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
    except FileNotFoundError:
        return (
            "err",
            input_path,
            "ffmpeg not found. Install with: brew install ffmpeg",
        )

    if proc.returncode == 0:
        return ("ok", input_path, "")
    return ("err", input_path, proc.stdout[-4000:])


# ---------------------- сбор входов/выходов ----------------------


def collect_inputs(src_dir: Path, exts: Set[str]) -> List[Path]:
    return sorted(
        [p for p in src_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    )


def rel_output_wav(src_dir: Path, file: Path, dst_dir: Path, add_hash: bool) -> Path:
    # Берём транслитерированный путь, меняем расширение на .wav
    clean_rel = ascii_rel_path(src_dir, file, add_hash=add_hash)
    clean_rel = clean_rel.with_suffix(".wav")
    return (dst_dir / clean_rel).resolve()


# ---------------------- CLI ----------------------


def main():
    ap = argparse.ArgumentParser(
        description="Convert videos to WAV (PCM16). Output paths are transliterated (ru→lat), sanitized, spaces→'_'."
    )
    ap.add_argument(
        "--src", required=True, type=Path, help="Папка с видео (рекурсивно)"
    )
    ap.add_argument("--dst", required=True, type=Path, help="Папка для WAV")
    ap.add_argument(
        "--exts",
        nargs="*",
        default=None,
        help="Список расширений видео (через пробел). По умолчанию популярные.",
    )
    ap.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Частота дискретизации, Гц (по умолчанию 16000)",
    )
    ap.add_argument("--channels", type=int, default=1, help="Число каналов (1=mono)")
    ap.add_argument(
        "--loudnorm",
        action="store_true",
        help="Включить EBU R128 нормализацию (ffmpeg loudnorm)",
    )
    ap.add_argument(
        "--audio-stream", type=int, default=None, help="Индекс аудиодорожки (0=первая)"
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="Перезаписывать существующие WAV"
    )
    ap.add_argument(
        "--jobs", type=int, default=2, help="Параллельные задания (файлов одновременно)"
    )
    ap.add_argument(
        "--no-hash",
        action="store_true",
        help="Не добавлять короткий хэш-суффикс к имени файла",
    )
    args = ap.parse_args()

    src_dir: Path = args.src.resolve()
    dst_dir: Path = args.dst.resolve()
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

    add_hash = not args.no_hash
    print(f"[INFO] Found {len(inputs)} files.")
    print(
        f"[INFO] WAV: {args.sr} Hz, {args.channels} ch, loudnorm={'on' if args.loudnorm else 'off'}"
    )
    print(f"[INFO] Output root: {dst_dir}")
    print(
        f"[INFO] Filenames: transliterated + sanitized + hash_suffix={'on' if add_hash else 'off'}"
    )

    ok = skipped = failed = 0
    tasks = {}

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for inp in inputs:
            outp = rel_output_wav(src_dir, inp, dst_dir, add_hash=add_hash)
            # Лог соответствия исходного → очищенного пути
            print(f"→ SRC: {inp}\n  DST: {outp}")
            fut = ex.submit(
                run_ffmpeg_to_wav,
                inp,
                outp,
                args.sr,
                args.channels,
                args.loudnorm,
                args.audio_stream,
                args.overwrite,
            )
            tasks[fut] = inp

        for fut in as_completed(tasks):
            status, inp, msg = fut.result()
            if status == "ok":
                ok += 1
                print(f"[OK] {inp}")
            elif status == "skip":
                skipped += 1
                print(f"[SKIP] {inp} ({msg})")
            else:
                failed += 1
                print(f"[ERR] {inp}\n{msg}\n", file=sys.stderr)

    print(f"\n[SUMMARY] done={ok}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
