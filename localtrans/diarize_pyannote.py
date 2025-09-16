#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch speaker diarization (pyannote 3.1) → RTTM

- Безопасный и стабильный RTTM-URI по относительному пути.
- Поддержка устройств: auto | mps | cuda | cpu.
- Управление числом спикеров: --num-speakers или --min/--max.
- Переопределение гиперпараметров пайплайна: --params-json.
- Skip/overwrite семантика, .meta.json рядом с RTTM.

Примеры:
  python diarize_pyannote.py --src ./wav16k --dst ./rttm --use-mps
  python diarize_pyannote.py --src ./wav16k --dst ./rttm --num-speakers 3
  python diarize_pyannote.py --src ./wav16k --dst ./rttm --min-speakers 2 --max-speakers 5
  python diarize_pyannote.py --src ./wav16k --dst ./rttm --params-json ./pyannote_params.json
"""

from __future__ import annotations
import os
import argparse
import json
import os
import re
import sys
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# torch/pyannote
import torch
from pyannote.audio import Pipeline

# По умолчанию работаем с готовыми WAV 16k mono (чисто и надёжно).
# Если хочешь шире — добавь сюда другие контейнеры, но лучше конвертить заранее.
AUDIO_EXTS = {".wav"}


def make_safe_uri(src_root: Path, inp: Path) -> str:
    """
    Безопасный URI для RTTM:
      <relpath without ext>, '/' → '__', пробелы → '_', прочие небезопасные → '_', + короткий хэш от относительного пути.
    Это защищает от пробелов/кириллицы и гарантирует стабильное сопоставление с транскриптами.
    """
    rel = inp.relative_to(src_root)
    stem = rel.as_posix().rsplit(".", 1)[0]  # без расширения
    s = stem.replace("/", "__")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-zА-Яа-я._-]+", "_", s)
    h = hashlib.sha1(rel.as_posix().encode("utf-8")).hexdigest()[:8]
    return f"{s}__{h}"


def collect_inputs(src_dir: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in src_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS
        ]
    )


def rel_out_path(src_dir: Path, file: Path, dst_dir: Path) -> Path:
    rel = file.relative_to(src_dir)
    return (dst_dir / rel).with_suffix(".rttm")


def pick_device(pref: str) -> str:
    pref = (pref or "auto").lower()
    if pref == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "cpu":
        return "cpu"
    # auto
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_pipeline(hf_token: str, device: str, params_json: Optional[Path]) -> Pipeline:
    try:
        pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
    except Exception as e:
        raise RuntimeError(
            f"Не удалось загрузить pyannote/speaker-diarization-3.1. "
            f"Проверь HF_TOKEN и доступ к модели (прими условия на HF). "
            f"Оригинальная ошибка: {e}"
        )

    # Переопределение гиперпараметров, если нужно (опционально)
    if params_json is not None:
        try:
            hp = json.loads(Path(params_json).read_text(encoding="utf-8"))
            # pyannote>=3.1: pipeline.instantiate(hp) допустим.
            pipe = pipe.instantiate(hp)  # type: ignore[attr-defined]
        except Exception as e:
            raise RuntimeError(f"Ошибка чтения/применения --params-json: {e}")

    # Переводим на выбранное устройство
    dev = pick_device(device)
    if dev == "mps":
        # На macOS иногда полезно:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        pipe.to(torch.device("mps"))
    elif dev == "cuda":
        pipe.to(torch.device("cuda"))
    else:
        pipe.to(torch.device("cpu"))

    print(f"[INFO] Device: {dev}")
    return pipe


def run_diarization(
    pipe: Pipeline,
    inp: Path,
    out_path: Path,
    num_speakers: Optional[int],
    min_spk: Optional[int],
    max_spk: Optional[int],
    src_root: Path,
    overwrite: bool,
) -> Tuple[str, Path, str]:
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not overwrite:
            return ("skip", inp, "exists")

        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = int(num_speakers)
        else:
            if min_spk is not None:
                kwargs["min_speakers"] = int(min_spk)
            if max_spk is not None:
                kwargs["max_speakers"] = int(max_spk)

        # Запуск диаризации
        ann = pipe(str(inp), **kwargs)

        # Стабильный URI
        ann.uri = make_safe_uri(src_root, inp)

        # RTTM
        with open(out_path, "w", encoding="utf-8") as f:
            ann.write_rttm(f)

        # Немного метаданных рядом
        meta = {
            "src_rel": str(inp.relative_to(src_root)),
            "rttm_rel": str(out_path),
            "uri": ann.uri,
            "params": {
                "num_speakers": num_speakers,
                "min_speakers": min_spk,
                "max_speakers": max_spk,
            },
        }
        (out_path.with_suffix(".meta.json")).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return ("ok", inp, "")
    except Exception as e:
        return ("err", inp, str(e))


def main():
    ap = argparse.ArgumentParser(
        description="Batch speaker diarization via pyannote.audio (3.1) → RTTM"
    )
    ap.add_argument(
        "--src", required=True, type=Path, help="Папка с WAV 16k mono (рекурсивно)"
    )
    ap.add_argument("--dst", required=True, type=Path, help="Папка для RTTM")
    ap.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN"),
        help="HuggingFace token (или переменная окружения HF_TOKEN)",
    )

    grp_n = ap.add_mutually_exclusive_group()
    grp_n.add_argument(
        "--num-speakers", type=int, default=None, help="Точное число спикеров"
    )
    grp_rng = ap.add_argument_group("speaker range (если число неизвестно)")
    grp_rng.add_argument(
        "--min-speakers", type=int, default=None, help="Минимум спикеров"
    )
    grp_rng.add_argument(
        "--max-speakers", type=int, default=None, help="Максимум спикеров"
    )

    ap.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Устройство вычислений (по умолчанию auto)",
    )
    ap.add_argument(
        "--params-json",
        type=Path,
        default=None,
        help="JSON с гиперпараметрами пайплайна (опционально)",
    )
    ap.add_argument(
        "--use-mps", action="store_true", help="(устар.) Эквивалент --device mps"
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="Перезаписывать существующие RTTM"
    )
    ap.add_argument(
        "--jobs", type=int, default=1, help="Параллельные файлы (на MPS/CUDA лучше 1-2)"
    )
    args = ap.parse_args()

    if not args.hf_token:
        print(
            "[ERR] Не задан HF token. Передай --hf-token или переменную окружения HF_TOKEN",
            file=sys.stderr,
        )
        sys.exit(2)

    # runtime
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

    # Совместимость со старым ключом --use-mps
    device = "mps" if args.use_mps else args.device
    # если auto — предпочитаем MODEL_DEVICE из runtime
    if device == "auto":
        device = MODEL_DEVICE

    src = args.src.resolve()
    dst = args.dst.resolve()
    inputs = collect_inputs(src)
    if not inputs:
        print(f"[INFO] Нет входных WAV в {src}")
        return

    print(f"[INFO] Найдено файлов: {len(inputs)}. Вывод RTTM: {dst}")
    try:
        pipe = build_pipeline(args.hf_token, device, args.params_json)
    except Exception as e:
        print(f"[ERR] {e}", file=sys.stderr)
        sys.exit(2)

    ok = skipped = failed = 0
    tasks = {}
    # На GPU/MPS держим небольшой параллелизм, чтобы не переполнить память
    max_workers = max(1, args.jobs)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for inp in inputs:
            outp = rel_out_path(src, inp, dst)
            fut = ex.submit(
                run_diarization,
                pipe,
                inp,
                outp,
                args.num_speakers,
                args.min_speakers,
                args.max_speakers,
                src,
                args.overwrite,
            )
            tasks[fut] = (inp, outp)

        for fut in as_completed(tasks):
            status, inp, msg = fut.result()
            if status == "ok":
                ok += 1
                print(f"[OK] {inp}")
            elif status == "skip":
                skipped += 1
                print(f"[SKIP] {inp} (exists)")
            else:
                failed += 1
                print(f"[ERR] {inp}\n{msg}\n", file=sys.stderr)

    print(f"\n[SUMMARY] ok={ok}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
