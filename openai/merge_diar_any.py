#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
import srt
import datetime as dt


# ---------- ВРЕМЯ / ПАРСИНГ ----------


def _time_parse_any(v):
    """float или 'HH:MM:SS(.mmm|,mmm)' -> секунды float."""
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", ".")  # важное: запятая -> точка
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
    """Склейка массива слов [{word|value, start/end}, ...] в строку."""
    parts = []
    for w in words:
        val = w.get("word") or w.get("value") or ""
        parts.append(str(val))
    return "".join(parts).strip()


def _segments_from_list(lst):
    """
    Пытается распарсить список словарей -> сегменты {start,end,text}.
    Поддерживает разные схемы: start/end, t0/t1, from/to, timestamps{}, offsets{}, offset+duration, words[].
    """
    out = []
    for it in lst:
        if not isinstance(it, dict):
            return None

        # текст
        text = it.get("text") or it.get("content") or it.get("line") or it.get("value")
        if not text and isinstance(it.get("alternatives"), list) and it["alternatives"]:
            text = it["alternatives"][0].get("transcript")
        if not text and isinstance(it.get("words"), list):
            text = _join_words(it["words"])
        if not isinstance(text, str):
            text = str(text or "").strip()

        have = False
        start = end = None

        # 1) классика start/end
        if "start" in it and "end" in it:
            start, end = _time_parse_any(it["start"]), _time_parse_any(it["end"])
            have = True

        # 2) t0/t1 (шаг 10 мс от whisper.cpp)
        elif "t0" in it and "t1" in it:
            start, end = float(it["t0"]) / 100.0, float(it["t1"]) / 100.0
            have = True

        # 3) from/to на верхнем уровне
        elif "from" in it and "to" in it:
            start, end = _time_parse_any(it["from"]), _time_parse_any(it["to"])
            have = True

        # 4) timestamps: {from:"HH:MM:SS,mmm", to:"..."}  ← твой случай
        elif isinstance(it.get("timestamps"), dict):
            ts = it["timestamps"]
            s = ts.get("from")
            e = ts.get("to")
            start, end = _time_parse_any(s), _time_parse_any(e)
            have = True

        # 5) offsets: {from:<ms>, to:<ms>}  (миллисекунды)
        elif isinstance(it.get("offsets"), dict):
            of = it["offsets"]
            if "from" in of and "to" in of:
                start, end = float(of["from"]) / 1000.0, float(of["to"]) / 1000.0
                have = True

        # 6) offset + duration (в секундах или строках-временах)
        elif "offset" in it and "duration" in it:
            start = _time_parse_any(it["offset"])
            end = start + _time_parse_any(it["duration"])
            have = True

        # 7) timestamp строкой: "00:00:01.000 --> 00:00:03.500"
        elif isinstance(it.get("timestamp"), str) and "-->" in it["timestamp"]:
            a, b = [x.strip() for x in it["timestamp"].split("-->", 1)]
            start, end = _time_parse_any(a), _time_parse_any(b)
            have = True

        # 8) timestamp как dict: {"start":...,"end":...} или {"from":...,"to":...}
        elif isinstance(it.get("timestamp"), dict):
            ts = it["timestamp"]
            if ("start" in ts and "end" in ts) or ("from" in ts and "to" in ts):
                s = ts.get("start", ts.get("from"))
                e = ts.get("end", ts.get("to"))
                start, end = _time_parse_any(s), _time_parse_any(e)
                have = True

        # 9) words[] с таймкодами → берем min/max
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
            return None  # этот список не сегменты

        out.append(
            {
                "start": float(start or 0.0),
                "end": float(end or (start or 0.0)),
                "text": text,
            }
        )
    return out


def _find_segments_any(obj):
    """Рекурсивно ищет сегменты в произвольной JSON-структуре."""
    if isinstance(obj, list):
        segs = _segments_from_list(obj)
        if segs is not None:
            return segs
        return None
    if isinstance(obj, dict):
        # частые контейнеры
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
        # если не нашли — обходим все значения
        for v in obj.values():
            cand = _find_segments_any(v)
            if cand:
                return cand
    return None


# ---------- RTTM / СПИКЕРЫ ----------


def load_rttm(rttm_path: Path):
    """Чтение RTTM в список (start, end, spk)."""
    segs = []
    if not rttm_path.exists():
        return segs
    for line in rttm_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        if p[0] != "SPEAKER":
            continue
        start = float(p[3])
        dur = float(p[4])
        spk = p[7]
        segs.append((start, start + dur, spk))
    return segs


def choose_spk(rttm_segs, tmid):
    """Возвращает спикера, покрывающего середину сегмента, или ближайшего."""
    for s, e, spk in rttm_segs:
        if s <= tmid <= e:
            return spk
    if rttm_segs:
        s, e, spk = min(rttm_segs, key=lambda se: abs(((se[0] + se[1]) / 2) - tmid))
        return spk
    return "spk??"


def norm_spk_tag(spk):
    m = re.search(r"(\d+)", spk)
    return f"SPK{int(m.group(1)):02d}" if m else spk.upper()


# ---------- ПАРСЕР ИСТОЧНИКОВ ----------


def parse_srt(path: Path):
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


def parse_json_whisper(path: Path):
    raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    # 1) обычный JSON
    try:
        obj = json.loads(raw)
        segs = _find_segments_any(obj)
        if segs:
            return segs
    except Exception:
        pass
    # 2) JSON Lines (каждая строка — отдельный объект)
    segs = []
    ok = True
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            it = json.loads(line)
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
    # 3) не распознали
    raise ValueError(f"Не найдено подходящее поле с сегментами в {path}")


# ---------- ВЫВОД ----------


def compose_srt(segments):
    subs = []
    for i, seg in enumerate(segments, 1):
        subs.append(
            srt.Subtitle(
                index=i,
                start=dt.timedelta(seconds=seg["start"]),
                end=dt.timedelta(seconds=seg["end"]),
                content=seg["text"],
            )
        )
    return srt.compose(subs)


def process_one(tr_path: Path, base_rel: Path, rttm_root: Path, out_root: Path):
    """Обработка одного файла транскрипта + соответствующего RTTM."""
    # подобрать rttm по относительному пути
    rttm_path = (rttm_root / base_rel).with_suffix(".rttm")
    rttm = load_rttm(rttm_path)

    # распарсить транскрипт с фоллбеком .json -> .srt
    segs = None
    if tr_path.suffix.lower() == ".srt":
        segs = parse_srt(tr_path)
    elif tr_path.suffix.lower() == ".json":
        try:
            segs = parse_json_whisper(tr_path)
        except Exception:
            srt_alt = tr_path.with_suffix(".srt")
            if srt_alt.exists():
                segs = parse_srt(srt_alt)
            else:
                raise
    else:
        # неизвестное расширение — попробуем .srt рядом
        srt_alt = tr_path.with_suffix(".srt")
        if srt_alt.exists():
            segs = parse_srt(srt_alt)
        else:
            return False, f"Пропускаю {tr_path} (не .srt/.json)"

    if not segs:
        return False, f"Пустые сегменты: {tr_path}"

    # назначить спикеров
    out_segs = []
    for seg in segs:
        mid = (seg["start"] + seg["end"]) / 2.0
        spk = norm_spk_tag(choose_spk(rttm, mid))
        out_segs.append(
            {**seg, "speaker": spk, "text": f"{spk}: {seg['text']}".strip()}
        )

    # сохранить .spk.srt / .spk.txt / .spk.json
    out_base = (out_root / base_rel).with_suffix("")  # без расширения
    out_base.parent.mkdir(parents=True, exist_ok=True)

    (out_base.with_suffix(".spk.srt")).write_text(
        compose_srt(out_segs), encoding="utf-8"
    )
    (out_base.with_suffix(".spk.txt")).write_text(
        "\n".join(s["text"] for s in out_segs), encoding="utf-8"
    )
    (out_base.with_suffix(".spk.json")).write_text(
        json.dumps(out_segs, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return True, str(out_base)


def main():
    ap = argparse.ArgumentParser(
        description="Склейка транскрипта (SRT/JSON от whisper.cpp) и RTTM (pyannote) → .spk.srt/.spk.txt/.spk.json"
    )
    ap.add_argument(
        "--transcripts",
        required=True,
        type=Path,
        help="Корень с транскриптами (SRT или JSON)",
    )
    ap.add_argument(
        "--rttm",
        required=True,
        type=Path,
        help="Корень с RTTM (та же структура подпапок)",
    )
    ap.add_argument("--out", required=True, type=Path, help="Куда писать результаты")
    args = ap.parse_args()

    tr_root = args.transcripts.resolve()
    rttm_root = args.rttm.resolve()
    out_root = args.out.resolve()

    tr_files = [
        p
        for p in tr_root.rglob("*")
        if p.is_file() and p.suffix.lower() in (".srt", ".json")
    ]
    if not tr_files:
        print(f"[INFO] Нет входных .srt или .json в {tr_root}")
        return

    ok = err = 0
    for p in tr_files:
        base_rel = p.relative_to(tr_root)
        try:
            done, msg = process_one(p, base_rel, rttm_root, out_root)
        except Exception as e:
            done, msg = False, f"Ошибка: {e}"
        if done:
            ok += 1
            print(f"[OK] {p} -> {msg}.spk.*")
        else:
            err += 1
            print(f"[ERR] {p}: {msg}")
    print(f"[SUMMARY] ok={ok}, err={err}")


if __name__ == "__main__":
    main()
