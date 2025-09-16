#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import datetime as dt
import srt

# ---------- парсер сегментов (устойчивый к разным JSON-форматам) ----------


def _time_parse_any(v):
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", ".")
    m = re.match(r"^(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2}(?:\.\d+)?)$", s)
    if m:
        return int(m.group("h")) * 3600 + int(m.group("m")) * 60 + float(m.group("s"))
    m = re.match(r"^(?P<m>\d{1,2}):(?P<s>\d{2}(?:\.\d+)?)$", s)
    if m:
        return int(m.group("m")) * 60 + float(m.group("s"))
    try:
        return float(s)
    except Exception:
        return 0.0


def _join_words(words):
    parts = []
    for w in words:
        val = w.get("word") or w.get("value") or ""
        parts.append(str(val))
    return "".join(parts).strip()


def _segments_from_list(lst):
    out = []
    for it in lst:
        if not isinstance(it, dict):
            return None

        text = it.get("text") or it.get("content") or it.get("line") or it.get("value")
        if not text and isinstance(it.get("alternatives"), list) and it["alternatives"]:
            text = it["alternatives"][0].get("transcript")
        if not text and isinstance(it.get("words"), list):
            text = _join_words(it["words"])
        if not isinstance(text, str):
            text = str(text or "").strip()

        have = False
        start = end = None

        if "start" in it and "end" in it:
            start, end = _time_parse_any(it["start"]), _time_parse_any(it["end"])
            have = True
        elif "t0" in it and "t1" in it:
            start, end = float(it["t0"]) / 100.0, float(it["t1"]) / 100.0
            have = True
        elif "from" in it and "to" in it:
            start, end = _time_parse_any(it["from"]), _time_parse_any(it["to"])
            have = True
        elif isinstance(it.get("timestamps"), dict):
            ts = it["timestamps"]
            start, end = _time_parse_any(ts.get("from")), _time_parse_any(ts.get("to"))
            have = True
        elif isinstance(it.get("offsets"), dict):
            of = it["offsets"]
            if "from" in of and "to" in of:
                start, end = float(of["from"]) / 1000.0, float(of["to"]) / 1000.0
                have = True
        elif "offset" in it and "duration" in it:
            start = _time_parse_any(it["offset"])
            end = start + _time_parse_any(it["duration"])
            have = True
        elif isinstance(it.get("timestamp"), str) and "-->" in it["timestamp"]:
            a, b = [x.strip() for x in it["timestamp"].split("-->", 1)]
            start, end = _time_parse_any(a), _time_parse_any(b)
            have = True
        elif isinstance(it.get("timestamp"), dict):
            ts = it["timestamp"]
            if ("start" in ts and "end" in ts) or ("from" in ts and "to" in ts):
                s = ts.get("start", ts.get("from"))
                e = ts.get("end", ts.get("to"))
                start, end = _time_parse_any(s), _time_parse_any(e)
                have = True
        elif isinstance(it.get("words"), list) and it["words"]:
            ws = it["words"]
            starts = [
                w.get("start") or w.get("from")
                for w in ws
                if (w.get("start") is not None or w.get("from") is not None)
            ]
            ends = [
                w.get("end") or w.get("to")
                for w in ws
                if (w.get("end") is not None or w.get("to") is not None)
            ]
            if starts and ends:
                start, end = _time_parse_any(starts[0]), _time_parse_any(ends[-1])
                have = True

        if not have:
            return None

        out.append(
            {
                "start": float(start or 0.0),
                "end": float(end or (start or 0.0)),
                "text": text,
            }
        )
    return out


def _find_segments_any(obj):
    if isinstance(obj, list):
        segs = _segments_from_list(obj)
        if segs is not None:
            return segs
        return None
    if isinstance(obj, dict):
        for key in (
            "segments",
            "result",
            "data",
            "items",
            "transcription",
            "monologues",
            "utterances",
            "lines",
        ):
            if key in obj:
                cand = _find_segments_any(obj[key])
                if cand:
                    return cand
        for v in obj.values():
            cand = _find_segments_any(v)
            if cand:
                return cand
    return None


def parse_json_segments(path: Path):
    raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    # 1) цельный JSON
    try:
        obj = json.loads(raw)
        segs = _find_segments_any(obj)
        if segs:
            return segs
    except Exception:
        pass
    # 2) JSON Lines
    segs = []
    ok = True
    for line in raw.splitlines():
        t = line.strip()
        if not t:
            continue
        try:
            it = json.loads(t)
        except Exception:
            ok = False
            break
        one = _segments_from_list([it])
        if not one:
            ok = False
            break
        segs.extend(one)
    if ok and segs:
        return segs
    raise ValueError(f"Не удалось извлечь сегменты из {path}")


def parse_srt_segments(path: Path):
    items = list(srt.parse(path.read_text(encoding="utf-8")))
    segs = []
    for it in items:
        segs.append(
            {
                "start": it.start.total_seconds(),
                "end": it.end.total_seconds(),
                "text": srt.make_legal_content(it.content),
            }
        )
    return segs


def write_segments_json(path_noext: Path, segments):
    out = path_noext.with_suffix(".segments.json")
    out.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


# ---------- запуск whisper.cpp ----------


def run_whisper_cli(
    whisper_bin: Path,
    model_path: Path,
    lang: str,
    wav_path: Path,
    out_prefix: Path,
    threads: int,
    extra_args: list[str],
):
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    # Просим JSON и SRT. out_prefix — общий префикс (whisper.cpp сам добавит суффиксы)
    cmd = [
        str(whisper_bin),
        "-m",
        str(model_path),
        "-l",
        lang,
        "-f",
        str(wav_path),
        "-t",
        str(threads),
        "-of",
        str(out_prefix),
        "-oj",  # JSON (достаточно; если хочешь "full", поменяй на -ojf)
        "-osrt",  # сразу SRT на диск
    ]
    if extra_args:
        cmd += extra_args
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    return proc.returncode, proc.stdout


# ---------- пайплайн одного файла ----------


def process_one_file(
    wav: Path,
    audio_root: Path,
    out_root: Path,
    whisper_bin: Path,
    model_path: Path,
    lang: str,
    threads: int,
    overwrite: bool,
    dry_run: bool,
    extra_args: list[str],
):
    rel = wav.relative_to(audio_root)
    out_prefix = (out_root / rel).with_suffix("")  # общий префикс
    raw_json = out_prefix.with_suffix(".json")  # что положит whisper.cpp
    srt_path = out_prefix.with_suffix(".srt")
    std_raw = out_prefix.with_suffix(
        ".raw.json"
    )  # куда скопируем/переименуем «сырой» JSON
    seg_json = out_prefix.with_suffix(".segments.json")

    if seg_json.exists() and not overwrite:
        return "skip", wav, "segments exist"

    if dry_run:
        return "ok", wav, "(dry-run)"

    # Если у нас уже есть валидный .raw.json (или .json) — попробуем просто распарсить → segments.
    candidate_json = None
    if std_raw.exists():
        candidate_json = std_raw
    elif raw_json.exists():
        candidate_json = raw_json

    if candidate_json and not overwrite:
        try:
            segs = parse_json_segments(candidate_json)
            write_segments_json(out_prefix, segs)
            # Если источник был *.json, дублируем в *.raw.json для единообразия
            if candidate_json == raw_json and not std_raw.exists():
                std_raw.write_text(
                    raw_json.read_text(encoding="utf-8", errors="ignore"),
                    encoding="utf-8",
                )
            return "ok", wav, "parsed cached json"
        except Exception:
            # Фолбэк на SRT, если есть
            if srt_path.exists():
                try:
                    segs = parse_srt_segments(srt_path)
                    write_segments_json(out_prefix, segs)
                    return "ok", wav, "parsed cached srt"
                except Exception:
                    pass
            # Пойдём транскрибировать заново
            pass

    # Запускаем whisper.cpp
    rc, out = run_whisper_cli(
        whisper_bin, model_path, lang, wav, out_prefix, threads, extra_args
    )
    if rc != 0:
        return "err", wav, out[-4000:]

    # После прогона должны появиться .json и .srt
    if raw_json.exists() and not std_raw.exists():
        std_raw.write_text(
            raw_json.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8"
        )

    # Парсим сегменты: сначала JSON, если не вышло — SRT
    try:
        segs = parse_json_segments(std_raw if std_raw.exists() else raw_json)
    except Exception:
        if srt_path.exists():
            segs = parse_srt_segments(srt_path)
        else:
            return "err", wav, "whisper finished, but no parseable json/srt"

    write_segments_json(out_prefix, segs)
    return "ok", wav, f"{len(segs)} segments"


# ---------- режим repair: доразложить уже готовые raw.json ----------


def repair_out_root(out_root: Path, overwrite: bool):
    ok = err = 0
    for raw in out_root.rglob("*.raw.json"):
        base = raw.with_suffix("")
        seg = base.with_suffix(".segments.json")
        srt = base.with_suffix(".srt")
        if seg.exists() and not overwrite:
            continue
        try:
            segs = parse_json_segments(raw)
        except Exception:
            if srt.exists():
                try:
                    segs = parse_srt_segments(srt)
                except Exception as e:
                    err += 1
                    print(f"[ERR] {raw}: {e}")
                    continue
            else:
                err += 1
                print(f"[ERR] {raw}: no parseable json and no srt")
                continue
        write_segments_json(base, segs)
        ok += 1
        print(f"[OK] repaired {raw} -> {seg}")
    print(f"[SUMMARY] repair ok={ok}, err={err}")


# ---------- CLI ----------


def collect_wavs(root: Path):
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"]
    )


def main():
    ap = argparse.ArgumentParser(
        description="Local transcription via whisper.cpp → *.raw.json + *.segments.json (+*.srt)"
    )
    ap.add_argument(
        "--audio-root",
        required=True,
        type=Path,
        help="Папка с WAV (16k mono рекомендовано)",
    )
    ap.add_argument(
        "--out-root",
        required=True,
        type=Path,
        help="Куда писать результаты (зеркалит структуру)",
    )
    ap.add_argument(
        "--whisper-bin",
        required=True,
        type=Path,
        help="Путь к whisper-cli (whisper.cpp)",
    )
    ap.add_argument("--model", required=True, type=Path, help="Путь к ggml/gguf модели")
    ap.add_argument("--lang", default="ru", help="Код языка (например, ru, en)")
    ap.add_argument(
        "--threads", type=int, default=6, help="Потоки для whisper.cpp (-t)"
    )
    ap.add_argument("--jobs", type=int, default=3, help="Сколько файлов в параллель")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Перезаписывать *.segments.json и форсить повтор",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Показывать, что бы сделали, но не запускать",
    )
    ap.add_argument(
        "--mode",
        choices=["transcribe", "repair"],
        default="transcribe",
        help="transcribe: прогнать аудио; repair: допарсить уже готовые raw.json",
    )
    ap.add_argument(
        "--extra", nargs="*", default=[], help="Доп. флаги для whisper.cpp (как есть)"
    )

    args = ap.parse_args()

    # runtime info (optional)
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

    audio_root = args.audio_root.resolve()
    out_root = args.out_root.resolve()
    whisper_bin = args.whisper_bin.resolve()
    model_path = args.model.resolve()

    if args.mode == "repair":
        repair_out_root(out_root, args.overwrite)
        return

    wavs = collect_wavs(audio_root)
    if not wavs:
        print(f"[INFO] Нет WAV в {audio_root}")
        return

    print(f"[INFO] Files: {len(wavs)} → out: {out_root}")
    ok = skip = err = 0
    tasks = {}
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for w in wavs:
            fut = ex.submit(
                process_one_file,
                w,
                audio_root,
                out_root,
                whisper_bin,
                model_path,
                args.lang,
                args.threads,
                args.overwrite,
                args.dry_run,
                args.extra,
            )
            tasks[fut] = w

        for fut in as_completed(tasks):
            status, inp, msg = fut.result()
            if status == "ok":
                ok += 1
                print(f"[OK] {inp} -> {msg}")
            elif status == "skip":
                skip += 1
                print(f"[SKIP] {inp} ({msg})")
            else:
                err += 1
                print(f"[ERR] {inp}\n{msg}\n", file=sys.stderr)

    print(f"[SUMMARY] ok={ok}, skip={skip}, err={err}")


if __name__ == "__main__":
    main()
