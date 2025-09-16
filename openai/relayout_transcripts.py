#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------- helpers ----------------

SEG_EXT = ".segments.json"
RAW_EXT = ".raw.json"


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def is_valid_segments(p: Path) -> bool:
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return isinstance(data, list) and all(
            isinstance(x, dict) and "text" in x and "start" in x and "end" in x
            for x in data
        )
    except Exception:
        return False


def base_from_segments_name(name: str) -> Optional[str]:
    """'file.segments.json' -> 'file' | None если не подходит."""
    return name[: -len(SEG_EXT)] if name.endswith(SEG_EXT) else None


def segments_name_from_base(base: str) -> str:
    return base + SEG_EXT


def raw_name_from_base(base: str) -> str:
    return base + RAW_EXT


def move_pair(src_seg: Path, dst_seg: Path) -> Tuple[bool, str]:
    """
    Перемещает seg и соседний raw (если есть) в целевую папку.
    Сопоставление делаем по БАЗЕ имени (до суффикса '.segments.json' / '.raw.json').
    """
    try:
        dst_seg.parent.mkdir(parents=True, exist_ok=True)

        # Уже на месте?
        try:
            if src_seg.resolve() == dst_seg.resolve():
                return True, f"already in place: {dst_seg}"
        except Exception:
            pass

        # Вычислим базу
        src_base = base_from_segments_name(src_seg.name)
        dst_base = base_from_segments_name(dst_seg.name)
        if not src_base or not dst_base:
            return False, f"bad name mapping: {src_seg.name} -> {dst_seg.name}"

        # Двигаем *.segments.json
        shutil.move(str(src_seg), str(dst_seg))

        # Попробуем двинуть парный *.raw.json (в той же исходной директории)
        src_raw = src_seg.with_name(raw_name_from_base(src_base))
        dst_raw = dst_seg.with_name(raw_name_from_base(dst_base))

        if src_raw.exists():
            shutil.move(str(src_raw), str(dst_raw))

        return True, f"MOVED → {dst_seg}"
    except Exception as e:
        return False, f"move error: {e}"


# ---------------- поиск кандидатов ----------------


def _glob_here_and_rglob(root: Path, pattern: str) -> List[Path]:
    # сначала в корне, если пусто — рекурсивно
    res = list(root.glob(pattern))
    if not res:
        res = list(root.rglob(pattern))
    # отсортируем по времени (свежие первыми)
    try:
        res.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception:
        pass
    return res


def find_flat_candidates_by_base(trans_root: Path, base: str) -> List[Path]:
    """
    Ищем файлы:
      <base>.segments.json
      <base>__*.segments.json
    и фильтруем на валидность.
    """
    cands = []
    cands += _glob_here_and_rglob(trans_root, segments_name_from_base(base))
    cands += _glob_here_and_rglob(trans_root, base + "__*" + SEG_EXT)
    # валидные и уникальные
    uniq, seen = [], set()
    for p in cands:
        if p.is_file() and is_valid_segments(p):
            s = str(p)
            if s not in seen:
                uniq.append(p)
                seen.add(s)
    return uniq


def pick_best(cands: List[Path]) -> Optional[Path]:
    if not cands:
        return None
    # уже отсортировано по mtime у нас, но на всякий случай
    try:
        return sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    except Exception:
        return cands[0]


# ---------------- логика перекладки по манифесту ----------------


def relayout_one_manifest(
    manifest: Path, audio_root: Path, trans_root: Path, dry_run: bool
) -> List[str]:
    logs: List[str] = []

    try:
        man = read_json(manifest)
    except Exception as e:
        return [f"[ERR] {manifest}: read fail: {e}"]

    # куда хотим сложить внутри trans_root
    if man.get("base_out_rel"):
        base_rel = Path(man["base_out_rel"])
    else:
        base_rel = manifest.relative_to(audio_root).with_suffix(
            ""
        )  # без .manifest.json

    # целевой «цельный» сегменты файл
    dst_base_seg = (trans_root / base_rel).with_suffix(SEG_EXT)

    # если уже лежит и валиден — ок
    if dst_base_seg.exists() and is_valid_segments(dst_base_seg):
        logs.append(f"[OK] already nested: {dst_base_seg}")

    chunked = bool(man.get("chunked", False))
    parts = man.get("parts") or []

    # ---------- не чанкнутый ----------
    if not chunked:
        if dst_base_seg.exists() and is_valid_segments(dst_base_seg):
            return logs

        # база имени (без суффикса)
        dst_base_name = base_from_segments_name(dst_base_seg.name)
        if not dst_base_name:
            logs.append(f"[ERR] bad dst name: {dst_base_seg.name}")
            return logs

        cands = find_flat_candidates_by_base(trans_root, dst_base_name)
        best = pick_best(cands)

        if not best:
            logs.append(f"[MISS] {dst_base_seg} (no flat candidate)")
            return logs

        if dry_run:
            logs.append(f"[DRY-RUN] would move {best} -> {dst_base_seg}")
        else:
            ok, msg = move_pair(best, dst_base_seg)
            logs.append(("[OK] " if ok else "[ERR] ") + msg)
        return logs

    # ---------- чанкнутый ----------
    if not parts:
        logs.append(f"[ERR] {manifest}: chunked=true but parts empty")
        return logs

    found_any_part = False

    for p in parts:
        rel_part = Path(p["path"])  # относительный путь от audio_root (без расширений)
        dst_part_seg = (trans_root / rel_part).with_suffix(SEG_EXT)

        if dst_part_seg.exists() and is_valid_segments(dst_part_seg):
            logs.append(f"[OK] already nested: {dst_part_seg}")
            found_any_part = True
            continue

        part_base_name = base_from_segments_name(dst_part_seg.name)
        if not part_base_name:
            logs.append(f"[ERR] bad dst part name: {dst_part_seg.name}")
            continue

        cands = find_flat_candidates_by_base(trans_root, part_base_name)
        best = pick_best(cands)

        if not best:
            logs.append(f"[MISS] {dst_part_seg} (no flat candidate)")
            continue

        found_any_part = True
        if dry_run:
            logs.append(f"[DRY-RUN] would move {best} -> {dst_part_seg}")
        else:
            ok, msg = move_pair(best, dst_part_seg)
            logs.append(("[OK] " if ok else "[ERR] ") + msg)

    # Если частей не нашли — попробуем «цельный» вариант
    if not found_any_part:
        if not (dst_base_seg.exists() and is_valid_segments(dst_base_seg)):
            dst_base_name = base_from_segments_name(dst_base_seg.name)
            if not dst_base_name:
                logs.append(f"[ERR] bad dst name: {dst_base_seg.name}")
                return logs
            cands = find_flat_candidates_by_base(trans_root, dst_base_name)
            best = pick_best(cands)
            if best:
                logs.append(
                    f"[HINT] parts missing, using combined candidate for {dst_base_seg.name}: {best.name}"
                )
                if dry_run:
                    logs.append(f"[DRY-RUN] would move {best} -> {dst_base_seg}")
                else:
                    ok, msg = move_pair(best, dst_base_seg)
                    logs.append(("[OK] " if ok else "[ERR] ") + msg)
            else:
                logs.append(
                    f"[MISS] no parts and no combined candidate for {dst_base_seg}"
                )

    return logs


def main():
    ap = argparse.ArgumentParser(
        description="Разложить *.segments.json/*.raw.json из корня trans_root по подпапкам согласно *.manifest.json (из audio_root)."
    )
    ap.add_argument(
        "--audio-root",
        required=True,
        type=Path,
        help="Корень аудио/манифестов (из конвертора)",
    )
    ap.add_argument(
        "--trans-root",
        required=True,
        type=Path,
        help="Где лежат результаты транскрибации (сейчас в «плоском» виде)",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Только план без перемещений"
    )
    args = ap.parse_args()

    audio_root = args.audio_root.resolve()
    trans_root = args.trans_root.resolve()

    mans = sorted(audio_root.rglob("*.manifest.json"))
    if not mans:
        print(f"[INFO] no manifests in {audio_root}")
        return

    total_ok = total_err = total_miss = 0
    for m in mans:
        logs = relayout_one_manifest(m, audio_root, trans_root, args.dry_run)
        for line in logs:
            print(line)
            if line.startswith("[OK]"):
                total_ok += 1
            elif line.startswith("[ERR]"):
                total_err += 1
            elif line.startswith("[MISS]"):
                total_miss += 1

    print(f"[SUMMARY] ok={total_ok}, err={total_err}, missing={total_miss}")


if __name__ == "__main__":
    main()
