#!/usr/bin/env python3
import argparse, subprocess, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

AUDIO_EXTS = {".wav"}  # держим просто и надежно: уже подготовленные WAV 16k mono

FORMAT_FLAG = {
    "srt": "-osrt",
    "vtt": "-ovtt",
    "txt": "-otxt",
    "json": "-oj",
}


def run_whisper(
    bin_path: Path,
    model_path: Path,
    inp: Path,
    out_base: Path,
    lang: str | None,
    threads: int,
    fmt: str,
    overwrite: bool,
):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    # whisper.cpp сам добавит расширение (.srt/.vtt/...)
    cmd = [
        str(bin_path),
        "-m",
        str(model_path),
        "-f",
        str(inp),
        FORMAT_FLAG[fmt],
        "-t",
        str(max(1, threads)),
        "-of",
        str(out_base),
    ]
    if lang and lang.lower() != "auto":
        cmd += ["-l", lang]

    # overwrite: whisper.cpp не имеет -y, поэтому просто перезапишет
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        ok = proc.returncode == 0
        return ("ok" if ok else "err", inp, "" if ok else proc.stdout[-4000:])
    except FileNotFoundError:
        return ("err", inp, f"Не найден бинарь whisper-cli: {bin_path}")


def collect_inputs(src_dir: Path) -> list[Path]:
    return sorted(
        [
            p
            for p in src_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS
        ]
    )


def rel_out_base(src_dir: Path, file: Path, dst_dir: Path) -> Path:
    rel = file.relative_to(src_dir)
    return (dst_dir / rel).with_suffix("")  # whisper.cpp сам добавит нужное расширение


def main():
    ap = argparse.ArgumentParser(
        description="Batch transcription via whisper.cpp (Core ML)."
    )
    ap.add_argument("--src", required=True, type=Path, help="Папка с WAV (рекурсивно)")
    ap.add_argument(
        "--dst", required=True, type=Path, help="Папка для субтитров/текста"
    )
    ap.add_argument(
        "--whisper-bin",
        type=Path,
        default=Path("whisper-cli"),
        help="Путь к whisper-cli (по умолчанию ищется в PATH)",
    )
    ap.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Путь к ggml-модели (.bin), напр. ggml-base.bin",
    )
    ap.add_argument(
        "--lang", default="auto", help="Язык (например 'ru', 'en' или 'auto')"
    )
    ap.add_argument(
        "--format", choices=FORMAT_FLAG.keys(), default="srt", help="Формат вывода"
    )
    ap.add_argument(
        "--threads", type=int, default=4, help="Потоки CPU для препроцессинга/декодера"
    )
    ap.add_argument(
        "--jobs", type=int, default=1, help="Параллельные файлы (осторожно с памятью)"
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Перезаписывать результаты (если уже есть)",
    )
    args = ap.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()
    inputs = collect_inputs(src)
    if not inputs:
        print(f"[INFO] Нет входных WAV в {src}")
        return

    print(
        f"[INFO] Найдено файлов: {len(inputs)}. Формат: {args.format}. Вывод в: {dst}"
    )
    tasks, ok, fail = {}, 0, 0
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for inp in inputs:
            out_base = rel_out_base(src, inp, dst)
            fut = ex.submit(
                run_whisper,
                args.whisper_bin,
                args.model,
                inp,
                out_base,
                args.lang,
                args.threads,
                args.format,
                args.overwrite,
            )
            tasks[fut] = inp

        for fut in as_completed(tasks):
            status, inp, msg = fut.result()
            if status == "ok":
                ok += 1
                print(f"[OK] {inp}")
            else:
                fail += 1
                print(f"[ERR] {inp}\n{msg}\n")

    print(f"\n[SUMMARY] ok={ok}, failed={fail}")


if __name__ == "__main__":
    main()
