#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, sys
from pathlib import Path
from typing import List, Dict, Any, Tuple


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_safe(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    # если по этому пути висит симлинк — уберём, чтобы не получить ELOOP
    try:
        if path.is_symlink():
            path.unlink()
    except Exception:
        try:
            path.unlink()
        except Exception:
            pass
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def valid_segments_json(path: Path) -> bool:
    try:
        data = read_json(path)
        return isinstance(data, list) and all(
            isinstance(x, dict) and "text" in x and "start" in x and "end" in x
            for x in data
        )
    except Exception:
        return False


def load_segments(path: Path) -> List[Dict[str, Any]]:
    data = read_json(path)
    if isinstance(data, list):
        # формат уже нормализован (наши скрипты так и пишут)
        return [
            {
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "text": s.get("text", ""),
            }
            for s in data
        ]
    # если вдруг пришёл verbose_json целиком — пробуем вынуть "segments"
    if isinstance(data, dict) and "segments" in data:
        segs = []
        for s in data["segments"]:
            segs.append(
                {
                    "start": float(s.get("start", 0.0)),
                    "end": float(s.get("end", 0.0)),
                    "text": s.get("text", ""),
                }
            )
        return segs
    raise ValueError(f"Не распознан формат сегментов в {path}")


def combine_from_manifest(
    manifest_path: Path,
    audio_root: Path,
    trans_root: Path,
    out_root: Path,
    skip_exists: bool = False,
) -> Tuple[bool, str]:
    """
    Возвращает (ok, message).
    """
    try:
        man = read_json(manifest_path)
    except Exception as e:
        return False, f"Не удалось прочитать манифест: {e}"

    # Поддержка обоих вариантов наших манифестов:
    # 1) { base_out_rel: "sub/dir/file", chunked: bool, parts: [{path, duration, offset}, ...] }
    # 2) (fallback) если base_out_rel нет — берём имя манифеста.
    if "base_out_rel" in man and man["base_out_rel"]:
        base_rel = Path(man["base_out_rel"])
    else:
        # имя манифеста совпадает с базой аудио: <audio_root>/<...>/<name>.manifest.json
        base_rel = manifest_path.relative_to(audio_root).with_suffix("")

    chunked = bool(man.get("chunked", False))
    out_seg = (out_root / base_rel).with_suffix(".segments.json")

    # skip, если уже есть валидный объединённый файл
    if skip_exists and out_seg.exists() and valid_segments_json(out_seg):
        return True, f"SKIP (exists): {out_seg}"

    if not chunked:
        # Нечего комбинировать — просто убедимся, что транскрипт базового файла есть.
        src_seg = (trans_root / base_rel).with_suffix(".segments.json")
        if not src_seg.exists():
            return False, f"missing {src_seg}"
        if not valid_segments_json(src_seg):
            return False, f"invalid JSON {src_seg}"
        # скопируем содержимое как «комбинированный»
        segs = load_segments(src_seg)
        write_json_safe(out_seg, segs)
        return True, f"COPIED base → {out_seg}"

    # chunked=true
    parts = man.get("parts") or []
    if not parts:
        return False, "chunked=true, но parts пустой"

    # собираем части в порядке offset (на всякий случай сортируем)
    parts_sorted = sorted(parts, key=lambda p: float(p.get("offset", 0.0)))
    combined: List[Dict[str, Any]] = []
    for i, p in enumerate(parts_sorted, 1):
        rel_part = Path(p["path"])  # путь части ОТНОСИТЕЛЬНО audio_root
        offset = float(p.get("offset", 0.0))
        seg_path = (trans_root / rel_part).with_suffix(".segments.json")
        if not seg_path.exists():
            return False, f"missing {seg_path}"
        try:
            segs = load_segments(seg_path)
        except Exception as e:
            return False, f"bad {seg_path}: {e}"
        # сместим таймкоды на offset
        for s in segs:
            combined.append(
                {
                    "start": float(s["start"]) + offset,
                    "end": float(s["end"]) + offset,
                    "text": s.get("text", ""),
                }
            )

    # финальная сортировка по start (на случай «плавающих» offset’ов)
    combined.sort(key=lambda s: (s["start"], s["end"]))
    write_json_safe(out_seg, combined)
    return True, f"COMBINED {len(parts_sorted)} parts → {out_seg}"


def main():
    ap = argparse.ArgumentParser(
        description="Склейка *.segments.json по манифестам из audio-root → единые *.segments.json в out-root (по умолчанию trans-root)."
    )
    ap.add_argument(
        "--audio-root",
        required=True,
        type=Path,
        help="Где лежат .manifest.json (папка аудио из конвертора)",
    )
    ap.add_argument(
        "--trans-root",
        required=True,
        type=Path,
        help="Где лежат частичные *.segments.json (из транскрибера)",
    )
    ap.add_argument(
        "--out",
        default=None,
        type=Path,
        help="Куда писать объединённые *.segments.json (по умолчанию = --trans-root)",
    )
    ap.add_argument(
        "--skip-exists",
        action="store_true",
        help="Пропускать файлы, если объединённый результат уже есть и валиден",
    )
    args = ap.parse_args()

    audio_root = args.audio_root.resolve()
    trans_root = args.trans_root.resolve()
    out_root = (args.out or args.trans_root).resolve()

    manifests = sorted(audio_root.rglob("*.manifest.json"))
    if not manifests:
        print(f"[INFO] Не найдено .manifest.json в {audio_root}")
        return

    ok = err = 0
    for m in manifests:
        good, msg = combine_from_manifest(
            m, audio_root, trans_root, out_root, args.skip_exists
        )
        if good:
            ok += 1
            print(f"[OK] {msg}")
        else:
            err += 1
            print(f"[ERR] {m}: {msg}", file=sys.stderr)

    print(f"[SUMMARY] ok={ok}, err={err}")


if __name__ == "__main__":
    main()
