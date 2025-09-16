#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, hashlib, re, shutil
from pathlib import Path
import os

try:
    from localtrans.runtime import IS_GPU, BACKEND, MODEL_DEVICE, MODEL_PATH, log_env
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


# --------- простая транслитерация кириллицы в латиницу для путей ----------
_T = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
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
    "х": "h",
    "ц": "c",
    "ч": "ch",
    "ш": "sh",
    "щ": "sch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}


def translit(s: str) -> str:
    out = []
    for ch in s:
        lo = ch.lower()
        if lo in _T:
            t = _T[lo]
            out.append(t if ch.islower() else t.upper())
        else:
            out.append(ch)
    return "".join(out)


def ascii_slug(text: str) -> str:
    # оставляем "/" для подпапок, остальное чистим
    s = translit(text)
    s = s.replace(" ", "_")
    s = re.sub(r"[^0-9A-Za-z._/-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def ascii_slug_with_hash(stem: str, rel_for_hash: str, maxlen: int = 64) -> str:
    base = ascii_slug(stem)
    if len(base) > maxlen:
        base = base[:maxlen].rstrip("_.-")
    h = hashlib.md5(rel_for_hash.encode("utf-8")).hexdigest()[:8]
    return f"{base}__{h}"


def expected_ascii_rel(rel_spk_json: Path) -> Path:
    """Строим относительный ASCII-путь для summary по нашему правилу."""
    # ascii для директорий
    ascii_dirs = [ascii_slug(p) for p in rel_spk_json.parts[:-1]]
    # имя файла
    stem = Path(rel_spk_json.name).stem  # без .spk.json
    rel_for_hash = str(
        rel_spk_json.with_suffix("")
    )  # исходный относительный путь без .spk.json
    ascii_file = ascii_slug_with_hash(stem, rel_for_hash) + ".spk.summary.md"
    return Path(*ascii_dirs) / ascii_file


def original_rel(rel_spk_json: Path) -> Path:
    """Строим относительный путь для 'старого' варианта (без ASCII, без hash)."""
    return rel_spk_json.with_suffix(".spk.summary.md")


def fuzzy_candidates(root: Path, ascii_rel: Path, orig_rel: Path) -> list[Path]:
    """Фаззи-поиск: точные пути + попытка найти файл без hash, но с тем же stem."""
    cands = []
    for rel in (ascii_rel, orig_rel):
        p = root / rel
        if p.exists():
            cands.append(p)

    # если не нашли — пробуем «stem без hash»
    if not cands:
        stem = ascii_rel.name.rsplit(".spk.summary.md", 1)[0]
        stem_nohash = re.sub(r"__[0-9a-f]{8}$", "", stem)
        glob_pat = stem_nohash + "*.spk.summary.md"
        maybe = sorted((root / ascii_rel.parent).glob(glob_pat))
        cands.extend(maybe)
    return cands


def main():
    ap = argparse.ArgumentParser(
        description="Проверка (и опц. починка) соответствия итоговых summaries после смены схемы путей."
    )
    ap.add_argument(
        "--with-speakers", required=True, type=Path, help="Корень с *.spk.json"
    )
    ap.add_argument(
        "--out-roots",
        nargs="*",
        default=["./summaries_ru", "./summaries"],
        help="Где искать готовые *.spk.summary.md (поиск слева направо)",
    )
    ap.add_argument(
        "--primary-out",
        type=Path,
        default=Path("./summaries_ru"),
        help="Куда приводить всё к единому виду (ASCII+hash)",
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--copy",
        action="store_true",
        help="Копировать найденные старые файлы в primary-out",
    )
    mode.add_argument(
        "--move",
        action="store_true",
        help="Перемещать найденные старые файлы в primary-out",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Показывать, не изменяя файлы"
    )
    ap.add_argument("--verbose", action="store_true", help="Болтать больше")
    args = ap.parse_args()

    ws_root = args.with_speakers.resolve()
    out_roots = [Path(p).resolve() for p in args.out_roots]
    primary = args.primary_out.resolve()

    spk_jsons = sorted(ws_root.rglob("*.spk.json"))
    if not spk_jsons:
        print(f"[INFO] Не найдено *.spk.json в {ws_root}")
        return

    ok_ascii = ok_orig = fixed = missing = 0

    for j in spk_jsons:
        rel = j.relative_to(ws_root)
        ascii_rel = expected_ascii_rel(rel)
        orig_rel = original_rel(rel)

        found = None
        found_root = None

        # ищем по всем корням
        for root in out_roots:
            cands = fuzzy_candidates(root, ascii_rel, orig_rel)
            if cands:
                found = cands[0]
                found_root = root
                break

        if found is None:
            print(f"MISSING → {primary / ascii_rel}")
            missing += 1
            continue

        # уже на месте как ASCII?
        if found == (primary / ascii_rel):
            if args.verbose:
                print(f"OK(ascii) {found}")
            ok_ascii += 1
            continue

        # лежит где-то ещё (или в старом имени)
        if args.copy or args.move:
            dst = primary / ascii_rel
            if args.verbose:
                print(f"{'COPY' if args.copy else 'MOVE'} {found} -> {dst}")
            if not args.dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if args.copy:
                    shutil.copy2(str(found), str(dst))
                else:
                    shutil.move(str(found), str(dst))
            print(f"FIXED  → {dst}")
            fixed += 1
        else:
            # просто констатируем, что файл есть, но не по финальному пути
            if (found_root / ascii_rel).exists():
                if args.verbose:
                    print(f"OK(ascii@{found_root}) {found}")
                ok_ascii += 1
            else:
                if args.verbose:
                    print(f"OK(orig) {found}")
                ok_orig += 1

    print(
        f"\n[SUMMARY] OK(ascii)={ok_ascii}, OK(orig)={ok_orig}, FIXED={fixed}, MISSING={missing}"
    )


if __name__ == "__main__":
    main()
