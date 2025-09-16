#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, re, sys, hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ВАЖНО: нужны пакеты
#   pip install "pyannote.audio>=3.1" torch torchaudio
#   (и доступ к модели: https://huggingface.co/pyannote/speaker-diarization-3.1)
from pyannote.audio import Pipeline
import torch

AUDIO_EXT = {
    ".wav",
    ".mp3",
    ".m4a",
    ".mp4",
    ".mkv",
    ".mov",
    ".aac",
    ".ogg",
    ".flac",
    ".webm",
    ".m4v",
}


def slug_uri(rel_path: Path) -> str:
    """
    Делаем безопасный URI для RTTM: латиница/цифры/подчёркивания.
    Добавляем короткий хэш для уникальности.
    """
    s = str(rel_path).replace(os.sep, "/")
    base = re.sub(r"[^A-Za-z0-9]+", "_", s)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    uri = f"{base.strip('_')}__{h}"
    return uri or f"item__{h}"


def write_rttm(annotation, uri: str, out_path: Path):
    """
    Пишем простой RTTM. Формат:
    SPEAKER <uri> 1 <start> <dur> <NA> <NA> <speaker> <NA> <NA>
    """
    lines = []
    for segment, _, label in annotation.itertracks(yield_label=True):
        start = float(segment.start)
        end = float(segment.end)
        dur = max(0.0, end - start)
        lines.append(
            f"SPEAKER {uri} 1 {start:.3f} {dur:.3f} <NA> <NA> {label} <NA> <NA>"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def infer_args_for_pipeline(min_speakers: int, max_speakers: int, num_speakers: int):
    """
    Возвращаем словарь аргументов для вызова pipeline(...).
    Если задан num_speakers — используем его,
    иначе — min/max при наличии.
    """
    args = {}
    if num_speakers is not None:
        args["num_speakers"] = int(num_speakers)
    else:
        if min_speakers is not None:
            args["min_speakers"] = int(min_speakers)
        if max_speakers is not None:
            args["max_speakers"] = int(max_speakers)
    return args


def process_one(
    audio_path: Path,
    src_root: Path,
    dst_root: Path,
    hf_token: str,
    use_mps: bool,
    min_speakers: int,
    max_speakers: int,
    num_speakers: int,
):
    rel = audio_path.relative_to(src_root)
    uri = slug_uri(rel.with_suffix(""))  # без расширения
    out_path = (dst_root / rel).with_suffix(".rttm")

    # инициализация пайплайна внутри процесса (важно для fork)
    device = "mps" if use_mps and torch.backends.mps.is_available() else "cpu"
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    try:
        pipeline.to(device)
    except Exception:
        # у некоторых сборок .to() на Pipeline отсутствует — тогда ок, CPU/MPS выберется изнутри
        pass

    # прогон
    kwargs = infer_args_for_pipeline(min_speakers, max_speakers, num_speakers)
    # pyannote ожидает словарь с ключом "audio" или путь
    result = pipeline({"audio": str(audio_path)}, **kwargs)
    annotation = result.get("annotation", result)  # совместимость

    write_rttm(annotation, uri, out_path)
    return {"uri": uri, "src_rel": str(rel), "rttm": str(out_path)}


def main():
    ap = argparse.ArgumentParser(
        description="Pyannote diarization → RTTM (безопасные URI, маппинг uri→путь)"
    )
    ap.add_argument("--src", required=True, type=Path, help="Папка с аудио/видео")
    ap.add_argument(
        "--dst", required=True, type=Path, help="Папка для RTTM (зеркальная структура)"
    )
    ap.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
        help="Токен HuggingFace (обязателен для pyannote/speaker-diarization-3.1)",
    )
    ap.add_argument(
        "--use-mps", action="store_true", help="Использовать GPU (Apple Silicon, MPS)"
    )
    ap.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Минимум спикеров (если известен диапазон)",
    )
    ap.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Максимум спикеров (если известен диапазон)",
    )
    ap.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Жёстко задать количество спикеров",
    )
    ap.add_argument(
        "--jobs", type=int, default=1, help="Параллелизм (на MPS безопаснее 1-2)"
    )
    ap.add_argument(
        "--skip-exists", action="store_true", help="Пропускать, если RTTM уже есть"
    )
    args = ap.parse_args()

    if not args.hf_token:
        print(
            "[ERR] Нужен --hf-token или переменная окружения HF_TOKEN/HUGGINGFACE_TOKEN",
            file=sys.stderr,
        )
        sys.exit(2)

    src = args.src.resolve()
    dst = args.dst.resolve()
    files = [p for p in src.rglob("*") if p.suffix.lower() in AUDIO_EXT]
    if not files:
        print(f"[INFO] Не нашёл входных аудио/видео в {src}")
        return

    uri_map_path = dst / "_uri_map.json"
    uri_map = []
    ok = err = 0

    if args.jobs <= 1:
        for p in files:
            rel = p.relative_to(src)
            rttm_out = (dst / rel).with_suffix(".rttm")
            if args.skip_exists and rttm_out.exists():
                ok += 1
                continue
            try:
                rec = process_one(
                    p,
                    src,
                    dst,
                    args.hf_token,
                    args.use_mps,
                    args.min_speakers,
                    args.max_speakers,
                    args.num_speakers,
                )
                uri_map.append(rec)
                ok += 1
                print(f"[OK] {rel} -> {rec['uri']}")
            except Exception as e:
                err += 1
                print(f"[ERR] {rel}: {e}", file=sys.stderr)
    else:
        # ВНИМАНИЕ: на MPS много процессов могут мешать друг другу. Тестируй 1-2.
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {}
            for p in files:
                rel = p.relative_to(src)
                rttm_out = (dst / rel).with_suffix(".rttm")
                if args.skip_exists and rttm_out.exists():
                    ok += 1
                    continue
                fut = ex.submit(
                    process_one,
                    p,
                    src,
                    dst,
                    args.hf_token,
                    args.use_mps,
                    args.min_speakers,
                    args.max_speakers,
                    args.num_speakers,
                )
                futs[fut] = rel
            for fut in as_completed(futs):
                rel = futs[fut]
                try:
                    rec = fut.result()
                    uri_map.append(rec)
                    ok += 1
                    print(f"[OK] {rel} -> {rec['uri']}")
                except Exception as e:
                    err += 1
                    print(f"[ERR] {rel}: {e}", file=sys.stderr)

    # сохраняем маппинг
    dst.mkdir(parents=True, exist_ok=True)
    uri_map_sorted = sorted(uri_map, key=lambda r: r["src_rel"])
    uri_map_path.write_text(
        json.dumps(uri_map_sorted, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[SUMMARY] ok={ok}, err={err}")
    print(f"[MAP] {uri_map_path}")


if __name__ == "__main__":
    main()
