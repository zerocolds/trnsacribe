#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from datetime import datetime

import requests

# -------------------- Настройки по умолчанию --------------------

DEF_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:14b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

PROMPT_SYSTEM = """Ты — строгий аналитик-выписчик из стенограммы.
Тебе даны сегменты разговора с полями: index, start, end (сек), speaker, text.
Твоя задача — извлечь ТОЛЬКО факты, явно присутствующие в тексте.

Сформируй JSON с полями:
{
  "discussed": [{"point": "<краткая формулировка>", "segments": [индексы]}],
  "decisions": [{"point": "<что договорились/решили>", "segments": [индексы]}],
  "product_issues": [{"point": "<проблема/риск продукта>", "segments": [индексы]}],
  "user_feedback": [{"point": "<мнение/ожидания/оценка пользователя>", "segments": [индексы]}]
}

Правила:
- НИЧЕГО НЕ ПРИДУМЫВАТЬ. Только то, что прямо сказано в сегментах.
- Если информации не хватает — верни пустой список для соответствующего поля.
- Объедини дублирующиеся формулировки.
- "segments" — массив номеров сегментов, в которых есть ПРЯМОЕ подтверждение тезиса (может быть несколько).
- Формулировки делай короткими, деловыми.
- Язык ответа: русский.
Выведи ТОЛЬКО JSON, без пояснений.
"""

# -------------------- Утилиты --------------------


def secs_to_tc(s: float) -> str:
    s = max(0, int(round(s)))
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def load_spk_json(path: Path):
    """
    Ожидает .spk.json: список {start,end,text,speaker}.
    Допускает dict с ключом "segments".
    """
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        segs = obj
    elif isinstance(obj, dict) and isinstance(obj.get("segments"), list):
        segs = obj["segments"]
    else:
        raise ValueError(f"Не распознан формат {path} (жду list или dict.segments)")
    norm = []
    for i, s in enumerate(segs):
        try:
            start = float(s.get("start", 0.0))
            end = float(s.get("end", start))
        except Exception:
            start = 0.0
            end = 0.0
        text = str(s.get("text", "")).strip()
        spk = str(s.get("speaker", "")).strip() or "SPK??"
        if not text:
            continue
        norm.append(
            {"index": i, "start": start, "end": end, "speaker": spk, "text": text}
        )
    return norm


def call_ollama(
    model: str,
    prompt: str,
    num_ctx: int,
    num_keep: int,
    num_predict: int,
    force_json: bool,
    temperature: float = 0.0,
    retries: int = 3,
    http_timeout: int = 1200,
):
    """
    Вызов Ollama /api/generate с опциями контекста, лимита генерации и строгим JSON.
    """
    url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "num_keep": num_keep,
            "num_predict": num_predict,
            "temperature": temperature,
        },
    }
    if force_json:
        payload["format"] = "json"

    last_err = None
    for _ in range(retries):
        try:
            r = requests.post(url, json=payload, timeout=http_timeout)
            r.raise_for_status()
            data = r.json()
            return data.get("response", "")
        except Exception as e:
            last_err = e
            time.sleep(1.5)
    raise RuntimeError(f"Ollama request failed: {last_err}")


def build_prompt(segments):
    """Собираем user-промпт для одного чанка сегментов."""
    lines = []
    for s in segments:
        lines.append(
            f"[{s['index']}] {secs_to_tc(s['start'])}-{secs_to_tc(s['end'])} {s['speaker']}: {s['text']}\n"
        )
    user = (
        "Ниже приведены сегменты:\n"
        + "".join(lines)
        + "\nСформируй JSON согласно инструкции."
    )
    return PROMPT_SYSTEM + "\n" + user


def safe_json_parse(txt: str):
    """
    Робастный разбор JSON из ответа модели.
    Поддерживает:
    - format=json (чистый JSON);
    - JSON внутри ```json ... ``` или просто с префиксом/суффиксом текста.
    """
    txt = txt.strip()

    # Вариант с кодовыми блоками ```json ... ```
    fence = re.search(r"```json\s*(\{.*?\})\s*```", txt, re.S | re.I)
    if fence:
        return json.loads(fence.group(1))

    # Попытка прямого парса
    try:
        return json.loads(txt)
    except Exception:
        pass

    # Вырезаем минимальный JSON от первой '{' до последней '}'
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        raw = m.group(0)
        return json.loads(raw)

    # Ничего не нашли
    raise ValueError("Ответ модели не похож на JSON")


def to_markdown(summary, segments, src_rel, model_tag):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    def indices_to_tcs(idxs):
        tcs, seen = [], set()
        for idx in idxs or []:
            try:
                idx = int(idx)
            except Exception:
                continue
            if 0 <= idx < len(segments):
                tc = secs_to_tc(segments[idx]["start"])
                if tc not in seen:
                    seen.add(tc)
                    tcs.append(tc)
        return tcs

    def section_to_md(title, items):
        if not items:
            return f"## {title}\n\n_нет явных данных_\n"
        out = [f"## {title}\n"]
        for it in items:
            point = (it.get("point") or "").strip()
            tcs = indices_to_tcs(it.get("segments", []))
            if not point:
                continue
            tc_str = (
                ", ".join(f"[{tc}](#tc{tc.replace(':','')})" for tc in tcs)
                if tcs
                else ""
            )
            out.append(f"- {point}{(' — ' + tc_str) if tc_str else ''}")
        return "\n".join(out) + "\n"

    speakers = sorted({s["speaker"] for s in segments})

    anchors = []
    for s in segments:
        tc = secs_to_tc(s["start"])
        anchors.append(
            f'<a id="tc{tc.replace(":","")}"></a>\n\n### ⏱ {tc}\n\n`{s["speaker"]}` — {s["text"]}\n'
        )

    md = []
    md.append("---")
    md.append(f'source: "{src_rel}"')
    md.append(f"created: {now}")
    md.append(f"model: {model_tag}")
    md.append("tags: [summary, transcript]")
    md.append("---\n")
    md.append("# Итоговая сводка\n")
    md.append("**Говорящие:** " + ", ".join(speakers) + "\n")
    md.append(section_to_md("Обсуждали", summary.get("discussed", [])))
    md.append(section_to_md("Договорённости", summary.get("decisions", [])))
    md.append(
        section_to_md("Проблемы продукта / риски", summary.get("product_issues", []))
    )
    md.append(
        section_to_md("Мнение / реакция пользователя", summary.get("user_feedback", []))
    )
    md.append("\n---\n")
    md.append("## Основания по таймкодам\n")
    md.extend(anchors)
    return "\n".join(md)


def split_segments_by_chars(segments, max_chars: int):
    """
    Режем вход на чанки по ~символьному бюджету, сохраняя целостность сегментов.
    Учитываем небольшой оверхед метаданных на сегмент.
    """
    chunks, cur, cur_len = [], [], 0
    for s in segments:
        # грубая оценка "веса" строки с метаданными и текстом
        line_len = len(s["text"]) + 64
        if cur and cur_len + line_len > max_chars:
            chunks.append(cur)
            cur, cur_len = [], 0
        cur.append(s)
        cur_len += line_len
    if cur:
        chunks.append(cur)
    return chunks


def merge_json_points(objs):
    """
    Дедупликация пунктов и объединение ссылок на сегменты по разделам.
    """
    keys = ("discussed", "decisions", "product_issues", "user_feedback")
    merged = {k: [] for k in keys}

    def norm(t: str) -> str:
        return re.sub(r"\s+", " ", (t or "").strip().lower())

    for k in keys:
        pool = {}
        for obj in objs:
            for it in obj.get(k) or []:
                p = (it.get("point") or "").strip()
                if not p:
                    continue
                key = norm(p)
                pool.setdefault(key, {"point": p, "segments": set()})
                for idx in it.get("segments") or []:
                    try:
                        pool[key]["segments"].add(int(idx))
                    except Exception:
                        pass
        merged[k] = [
            {"point": v["point"], "segments": sorted(v["segments"])}
            for v in pool.values()
        ]
    return merged


# -------------------- Основной процесс --------------------


def process_file(
    path: Path,
    out_root: Path,
    src_root: Path,
    model: str,
    num_ctx: int,
    num_keep: int,
    num_predict: int,
    force_json: bool,
    max_chars: int,
    temperature: float,
    retries: int,
    http_timeout: int,
    skip_exists: bool,
):
    """
    Обработка одного *.spk.json: чанкинг -> Ollama -> мердж -> Markdown.
    """
    # Итоговый путь MD заранее (нужен для пропуска)
    rel = path.relative_to(src_root)
    out_path = (out_root / rel).with_suffix(".summary.md")
    if skip_exists and out_path.exists():
        return True, f"{out_path} (skip)"

    # Загружаем сегменты
    segments = load_spk_json(path)
    if not segments:
        return False, f"Пустые сегменты: {path}"

    # Разбиваем на чанки (map)
    chunks = split_segments_by_chars(segments, max_chars=max_chars)
    partials = []
    for ch in chunks:
        prompt = build_prompt(ch)
        resp = call_ollama(
            model=model,
            prompt=prompt,
            num_ctx=num_ctx,
            num_keep=num_keep,
            num_predict=num_predict,
            force_json=force_json,
            temperature=temperature,
            retries=retries,
            http_timeout=http_timeout,
        )
        part = safe_json_parse(resp)
        partials.append(part)

    # Объединяем (reduce)
    data = merge_json_points(partials)

    # Markdown → файл
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md = to_markdown(data, segments, src_rel=str(rel), model_tag=model)
    out_path.write_text(md, encoding="utf-8")
    return True, str(out_path)


def main():
    ap = argparse.ArgumentParser(
        description="Саммаризация *.spk.json → Markdown (Obsidian) с таймкодами-основаниями. Запуск через Ollama HTTP API."
    )
    ap.add_argument(
        "--src",
        required=True,
        type=Path,
        help="Корень с *.spk.json (структура сохраняется)",
    )
    ap.add_argument("--out", required=True, type=Path, help="Куда класть *.summary.md")
    ap.add_argument(
        "--model", default=DEF_MODEL, help=f"Модель Ollama (по умолчанию {DEF_MODEL})"
    )

    # Контекст и генерация
    ap.add_argument(
        "--num-ctx", type=int, default=8192, help="Окно контекста модели (tokens)"
    )
    ap.add_argument(
        "--num-keep", type=int, default=64, help="Сколько токенов держать закреплёнными"
    )
    ap.add_argument(
        "--num-predict", type=int, default=600, help="Максимум токенов ответа"
    )
    ap.add_argument(
        "--max-chars",
        type=int,
        default=14000,
        help="Макс. символов на один чанк (защита контекста)",
    )
    ap.add_argument(
        "--temperature", type=float, default=0.0, help="Температура генерации"
    )
    ap.add_argument(
        "--force-json",
        action="store_true",
        help="Просить строго JSON (payload.format=json)",
    )

    # Надёжность
    ap.add_argument("--retries", type=int, default=3, help="Повторы при сетевом сбое")
    ap.add_argument("--http-timeout", type=int, default=1200, help="HTTP timeout, сек")
    ap.add_argument(
        "--skip-exists",
        action="store_true",
        help="Если *.summary.md уже существует — пропустить файл",
    )

    args = ap.parse_args()

    src = args.src.resolve()
    out_root = args.out.resolve()

    # Собираем список входных файлов
    files = [p for p in src.rglob("*.spk.json")]
    if not files:
        files = [p for p in src.rglob("*.json")]  # fallback, если без .spk
    if not files:
        print(f"[INFO] Не нашёл входных .spk.json/.json в {src}")
        return

    ok = err = 0
    for p in files:
        try:
            done, msg = process_file(
                path=p,
                out_root=out_root,
                src_root=src,
                model=args.model,
                num_ctx=args.num_ctx,
                num_keep=args.num_keep,
                num_predict=args.num_predict,
                force_json=args.force_json,
                max_chars=args.max_chars,
                temperature=args.temperature,
                retries=args.retries,
                http_timeout=args.http_timeout,
                skip_exists=args.skip_exists,
            )
        except Exception as e:
            done, msg = False, f"Ошибка: {e}"
        if done:
            ok += 1
            print(f"[OK] {p} -> {msg}")
        else:
            err += 1
            print(f"[ERR] {p}: {msg}", file=sys.stderr)

    print(f"\n[SUMMARY] ok={ok}, err={err}")


if __name__ == "__main__":
    main()
