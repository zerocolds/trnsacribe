#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Суммаризация .spk.json (стенограммы с таймкодами и спикерами)
→ структурированный Markdown на русском для Obsidian.

Особенности:
- Чанкование диалога по символам с перекрытием.
- Флаг --prompt-budget: авто-усечение чанка под лимит контекста модели.
- Двухпроходная схема: кусочные JSON-выводы → кодовым мерджем в финальный отчёт.
- Поддержка backend=ollama (локально). Можно добавить openai при желании.
- Строгий формат ответа от модели: JSON (мы извлекаем первую валидную фиг.скобку).
- В итоговом MD: темы, договорённости, проблемы, пожелания, возражения,
  рекомендации по обработке возражений, оценка продавца (по Якубе) + таймкоды.

Пример:
python summarize_spk_json.py \
  --src ./with_speakers \
  --out ./summaries_ru \
  --backend ollama \
  --ollama-model qwen2.5:14b-instruct-q6_K \
  --num-ctx 16384 \
  --prompt-budget 14000 \
  --chunk-chars 9000 \
  --chunk-overlap 800 \
  --jobs 2 \
  --skip-exists
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except Exception as e:
    print("[ERR] Требуется пакет 'requests' (pip install requests)", file=sys.stderr)
    raise

# ------------------------- Константы / подсказки -------------------------

SUPPORTED_EXT = (".spk.json",)

SYSTEM_PROMPT_RU = """Ты аналитик переговоров. Твоя задача — кратко и структурированно извлечь смысл из фрагмента русскоязычной стенограммы (только аудио; это могут быть видео-встречи, но доступна лишь звуковая дорожка). Говорящие размечены как SPKxx. Всегда оперируй фактами из текста — ничего не выдумывай. Обязательно сохраняй таймкоды источников.
Отвечай строго в формате JSON согласно схеме ниже.
"""

INSTR_SCHEMA_RU = r"""
СХЕМА ВЫХОДА (строго JSON, без пояснений и форматирования вне JSON):

{
  "topics": [  // О ЧЁМ ГОВОРИЛИ — короткие тезисы
    {"point": "тезис", "evidence": [{"t":"MM:SS","speaker":"SPK01","quote":"цитата"}]}
  ],
  "agreements": [ // О ЧЁМ ДОГОВОРИЛИСЬ — конкретные договорённости/след.шаги
    {"point": "договорённость", "evidence": [{"t":"MM:SS","speaker":"SPK02","quote":"..."}]}
  ],
  "product_issues": [ // ПРОБЛЕМЫ ПРОДУКТА/СЕРВИСА
    {"point": "проблема", "evidence": [{"t":"MM:SS","speaker":"SPK..","quote":"..."}]}
  ],
  "customer_wishes": [ // ПОЖЕЛАНИЯ КЛИЕНТА
    {"point": "пожелание", "evidence": [{"t":"MM:SS","speaker":"SPK..","quote":"..."}]}
  ],
  "objections": [ // СПИСОК ВОЗРАЖЕНИЙ КЛИЕНТА (дословно/сжато)
    {"point": "возражение", "evidence": [{"t":"MM:SS","speaker":"SPK..","quote":"..."}]}
  ],
  "objection_tips": [ // РЕКОМЕНДАЦИИ ПО РАБОТЕ С ВОЗРАЖЕНИЯМИ (конкретные, краткие)
    "совет 1",
    "совет 2"
  ],
"seller_scaling": { // Оценка качества звонка по стадиям разговора и эмоц. составляющей (0..2 каждый)
  // ШКАЛА: 0 = не выполнено/плохо; 1 = частично/средне; 2 = полностью/отлично

  "object": 0,                  // Объект: корректность фокуса разговора (значимость темы/объекта для клиента)
  "contact_entry": 0,           // Вход в контакт: представление (имя и компания)
  "call_purpose": 0,            // Цель звонка: ясно озвучена причина звонка (заявка, визит в ОП и т.п.)
  "agenda_setting": 0,          // Программирование: перехват инициативы, плавный переход ("чтобы сэкономить время, задам несколько вопросов...")
  "needs_discovery": 0,         // Выявление потребностей: задаёт вопросы по срокам, бюджету, параметрам, важным критериям
  "motivation_identified": 0,   // Мотив: выявлена личная причина/ситуация клиента, подталкивающая к покупке
  "value_presentation_meeting": 0, // Презентация/ценность/встреча: опирается на потребности, выгоды озвучены корректно, сформировано желание приехать
  "urgency_creation": 0,        // Создание срочности: дефицит/ограниченность (осталось X квартир, цена держится недолго и т.п.)
  "objections_handling": 0,     // Возражения: выявляет и снимает возражения
  "closing_next_step": 0,       // Закрытие и следующий шаг: назначена контрольная точка/договорённость (встреча, бронь, выслать материалы)
  "use_of_name": 0,             // Обращение по имени: не реже 2 раз (приветствие и завершение)
  "expertise_confidence": 0,    // Экспертиза: уверенность, владение инфо (прайс, сроки, адрес ОП), не путается
  "initiative_control": 0,      // Инициатива: удерживает ведение разговора, фокусирует на решении вопроса клиента
  "active_listening_alignment": 0, // Активное слушание/присоединение: интерес, перефраз, подчеркивание выгод ("это тоже экономия...")
  "politeness": 0,              // Вежливость: без грубости/сарказма, вопросы не как допрос, не давит
  "client_satisfaction": 0,     // Удовлетворённость клиента: нет выраженного недовольства качеством работы
  "lexicon_clarity": 0,         // Лексика: не отвлекается, без длинных пауз, простые понятные формулировки
  "no_interruptions": 0,        // Не перебивает: не перебивает, не забегает вперёд, не «предугадывает» ответы
  "asked_if_already_bought": 0, // Уже купил: уточнил, не купил ли клиент у другого (если купил и спросил — засчитываем)
  "positive_attitude": 0,       // Позитивный настрой: «внутренняя улыбка», нацеленность на клиента

  "comment": "1–2 предложения с пояснением по ключевым сильным/слабым моментам"
}

}

ПРАВИЛА:
- Верни ТОЛЬКО JSON по схеме выше, без markdown, без подсказок и преамбул.
- "evidence" обязательно с таймкодами MM:SS (округляй до ближайшей секунды).
- Если чего-то нет в фрагменте — верни пустой список/минимальные поля.
- Цитаты короткие (до ~120 символов), без перефразирования, русские кавычки не требуются.
- Баллы (1..5) — консервативные, на основе доступного фрагмента.
"""

CHUNK_PROMPT_TMPL = """{system}

{schema}

Ниже фрагмент диалога. Формат строк: "[HH:MM:SS] SPKxx: текст".
Проанализируй ТОЛЬКО этот фрагмент.

<dialog>
{dialog_text}
</dialog>

Верни ТОЛЬКО JSON строго по схеме.
"""

REDUCE_PROMPT_TMPL = """{system}

Тебе даны несколько частичных JSON-результатов по разным фрагментам одной встречи.
Слей их в один итог (объедини дубликаты по смыслу, суммируй evidence, средние оценки).
Сохраняй ту же схему JSON, что и раньше. Таймкоды оставляй как есть.

<parts>
{parts_json}
</parts>

Верни ТОЛЬКО ОДИН JSON.
"""

# ------------------------- Утилиты -------------------------


def mmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    m = seconds // 60
    s = seconds % 60
    return f"{m:02d}:{s:02d}"


def approx_tokens(s: str) -> int:
    # очень грубая оценка токенов. Для кириллицы обычно 0.8..1.3 символа/токен.
    return max(1, int(len(s) / 1.1))


def fit_chunk_to_budget(
    system_txt: str, instr_txt: str, chunk_txt: str, budget_tokens: int
) -> str:
    """Обрезает chunk_txt, чтобы суммарно уложиться в budget_tokens."""
    used = approx_tokens(system_txt) + approx_tokens(instr_txt)
    free = max(512, budget_tokens - used)  # минимально оставим запас
    max_chars = int(free * 1.0)  # консервативно 1 токен ~= 1 символ
    if len(chunk_txt) <= max_chars:
        return chunk_txt
    cut = chunk_txt[:max_chars]
    # постараемся обрезать по границе
    for sep in ["\n\n", "\n", ". ", "! ", "? ", "; "]:
        i = cut.rfind(sep)
        if i > max_chars * 0.6:
            cut = cut[: i + 1]
            break
    return cut + "\n\n[...обрезано для соблюдения лимита контекста...]"


def safe_json_loads(s: str) -> Optional[dict]:
    """Пытается вытащить первый JSON-объект { ... } из ответа модели."""
    # убрать возможные ```json ... ```
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    # найти первую фигурную скобку и парсить балансом
    start = s.find("{")
    if start == -1:
        return None
    # баланс скобок
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start : i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    # попробуем ещё: убрать хвосты с BOM/непечат.
                    try:
                        return json.loads(
                            candidate.encode("utf-8", "ignore").decode(
                                "utf-8", "ignore"
                            )
                        )
                    except Exception:
                        return None
    return None


def normalize_point(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[\"'`«»“”„]", "", t)
    return t


# ------------------------- Загрузка сегментов -------------------------


@dataclass
class Seg:
    start: float
    end: float
    speaker: str
    text: str


def load_spk_json(path: Path) -> List[Seg]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    segs: List[Seg] = []
    for it in raw:
        start = float(it.get("start", 0.0))
        end = float(it.get("end", start))
        speaker = str(it.get("speaker") or it.get("spk") or "")
        text = str(it.get("text") or it.get("content") or "")
        if not text:
            continue
        segs.append(Seg(start, end, speaker, text))
    return segs


def format_dialog_lines(segs: List[Seg]) -> List[str]:
    lines = []
    for s in segs:
        lines.append(f"[{mmss(s.start)}] {s.speaker or 'SPK??'}: {s.text}".strip())
    return lines


# ------------------------- Чанкование -------------------------


def chunk_by_chars(lines: List[str], chunk_chars: int, overlap_chars: int) -> List[str]:
    """Возвращает список текстовых чанков (многострочных)."""
    if not lines:
        return []
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0
    i = 0

    while i < len(lines):
        line = lines[i]
        ln = len(line) + 1  # + newline
        if buf_len + ln <= chunk_chars or not buf:
            buf.append(line)
            buf_len += ln
            i += 1
            continue

        # завершили чанк
        chunks.append("\n".join(buf))
        if overlap_chars > 0:
            # вернёмся назад по символам
            back_chars = 0
            j = len(buf) - 1
            tail: List[str] = []
            while j >= 0 and back_chars < overlap_chars:
                tail.insert(0, buf[j])
                back_chars += len(buf[j]) + 1
                j -= 1
            buf = tail[:]  # перекрытие
            buf_len = sum(len(x) + 1 for x in buf)
        else:
            buf = []
            buf_len = 0

    if buf:
        chunks.append("\n".join(buf))
    return chunks


# ------------------------- Вызов LLM -------------------------


def call_ollama_generate(
    model: str,
    prompt: str,
    num_ctx: int = 4096,
    num_keep: int = 64,
    temperature: float = 0.1,
    timeout: int = 600,
) -> str:
    try:
        from localtrans.llm_adapters import generate_text

        return generate_text(
            backend=os.getenv("USE_BACKEND") or "ollama",
            model=model,
            prompt=prompt,
            timeout=timeout,
            num_ctx=num_ctx,
            num_keep=num_keep,
            temperature=temperature,
        )
    except Exception:
        # fallback to direct HTTP to Ollama like before
        url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": int(num_ctx),
                "num_keep": int(num_keep),
                "temperature": float(temperature),
            },
        }
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")


# ------------------------- Агрегация chunk-JSON -------------------------


def merge_chunk_json(items: List[dict]) -> dict:
    # Инициализация агрегатов
    def empty():
        return {
            "topics": [],
            "agreements": [],
            "product_issues": [],
            "customer_wishes": [],
            "objections": [],
            "objection_tips": [],
            "seller_score": {
                "structure": 0,
                "needs": 0,
                "objections": 0,
                "closing": 0,
                "rapport": 0,
                "comment": "",
            },
        }

    agg = empty()

    def merge_points(dst_list, src_list):
        # объединяем по смыслу (нормализованный ключ)
        seen: Dict[str, Dict[str, Any]] = {}
        for it in dst_list:
            key = normalize_point(it.get("point", ""))
            if key:
                seen[key] = it

        for it in src_list or []:
            point = it.get("point", "")
            key = normalize_point(point)
            if not key:
                continue
            if key not in seen:
                seen[key] = {"point": point, "evidence": []}
            # evidence
            ev_src = it.get("evidence") or []
            ev_dst = seen[key].setdefault("evidence", [])
            # простое объединение с защитой от дублей по (t, speaker, quote)
            existed = {(e.get("t"), e.get("speaker"), e.get("quote")) for e in ev_dst}
            for e in ev_src:
                tup = (e.get("t"), e.get("speaker"), e.get("quote"))
                if tup not in existed:
                    ev_dst.append(
                        {
                            "t": e.get("t"),
                            "speaker": e.get("speaker"),
                            "quote": e.get("quote"),
                        }
                    )
                    existed.add(tup)
        # вернуть отсортированный список (по алфавиту точки)
        return sorted(seen.values(), key=lambda x: x.get("point", ""))

    # коллекция для усреднения оценок
    cnt = 0
    sum_scores = {
        "structure": 0,
        "needs": 0,
        "objections": 0,
        "closing": 0,
        "rapport": 0,
    }
    comments: List[str] = []

    for it in items:
        agg["topics"] = merge_points(agg["topics"], it.get("topics") or [])
        agg["agreements"] = merge_points(agg["agreements"], it.get("agreements") or [])
        agg["product_issues"] = merge_points(
            agg["product_issues"], it.get("product_issues") or []
        )
        agg["customer_wishes"] = merge_points(
            agg["customer_wishes"], it.get("customer_wishes") or []
        )
        agg["objections"] = merge_points(agg["objections"], it.get("objections") or [])
        # советы — как простые строки, объединяем с дедупом
        tips_src = [
            str(x).strip() for x in (it.get("objection_tips") or []) if str(x).strip()
        ]
        existing = set(agg["objection_tips"])
        for tip in tips_src:
            if tip not in existing:
                agg["objection_tips"].append(tip)
                existing.add(tip)
        # оценки
        s = it.get("seller_score") or {}
        try:
            sum_scores["structure"] += int(s.get("structure") or 0)
            sum_scores["needs"] += int(s.get("needs") or 0)
            sum_scores["objections"] += int(s.get("objections") or 0)
            sum_scores["closing"] += int(s.get("closing") or 0)
            sum_scores["rapport"] += int(s.get("rapport") or 0)
            cmt = str(s.get("comment") or "").strip()
            if cmt:
                comments.append(cmt)
            cnt += 1
        except Exception:
            pass

    if cnt > 0:
        avg = {k: int(round(v / cnt)) for k, v in sum_scores.items()}
    else:
        avg = {k: 0 for k in sum_scores.keys()}

    agg["seller_score"].update(avg)
    if comments:
        # возьмём 1-2 кратких комментария
        agg["seller_score"]["comment"] = " / ".join(comments[:2])

    return agg


def render_markdown(agg: dict, title: str) -> str:
    def bullet_points(section_name: str, items: List[dict]) -> str:
        if not items:
            return f"### {section_name}\n—\n"
        out = [f"### {section_name}"]
        for it in items:
            point = it.get("point", "").strip()
            ev = it.get("evidence") or []
            if point:
                out.append(f"- **{point}**")
                # таймкоды
                if ev:
                    ev_lines = []
                    for e in ev[:6]:  # максимум 6 ссылок, чтобы не раздувать
                        t = e.get("t") or ""
                        sp = e.get("speaker") or ""
                        q = e.get("quote") or ""
                        ev_lines.append(f"  - [{t}] {sp}: {q}")
                    out.extend(ev_lines)
        out.append("")  # пустая строка
        return "\n".join(out)

    md = [f"# {title}", ""]
    md.append(bullet_points("О чём говорили", agg.get("topics") or []))
    md.append(bullet_points("Договорённости", agg.get("agreements") or []))
    md.append(bullet_points("Проблемы продукта", agg.get("product_issues") or []))
    md.append(bullet_points("Пожелания клиента", agg.get("customer_wishes") or []))
    md.append(bullet_points("Возражения клиента", agg.get("objections") or []))

    tips = agg.get("objection_tips") or []
    md.append("### Рекомендации по работе с возражениями")
    if tips:
        for t in tips[:10]:
            md.append(f"- {t}")
    else:
        md.append("—")
    md.append("")

    # Новый ключ в данных
    s = agg.get("seller_scaling") or {}

    # Отображаемые названия для удобства
    labels = {
        "object": "Объект (значимость темы/объекта для клиента)",
        "contact_entry": "Вход в контакт (представление)",
        "call_purpose": "Цель звонка",
        "agenda_setting": "Программирование (плавный переход)",
        "needs_discovery": "Выявление потребностей",
        "motivation_identified": "Мотив (личная причина клиента)",
        "value_presentation_meeting": "Презентация/ценность/встреча",
        "urgency_creation": "Создание срочности",
        "objections_handling": "Работа с возражениями",
        "closing_next_step": "Закрытие и следующий шаг",
        "use_of_name": "Обращение по имени",
        "expertise_confidence": "Экспертиза и уверенность",
        "initiative_control": "Инициатива и контроль разговора",
        "active_listening_alignment": "Активное слушание/присоединение",
        "politeness": "Вежливость",
        "client_satisfaction": "Удовлетворённость клиента",
        "lexicon_clarity": "Лексика и ясность речи",
        "no_interruptions": "Не перебивает",
        "asked_if_already_bought": "Уточнил, не купил ли клиент у другого",
        "positive_attitude": "Позитивный настрой",
    }

    md.append(
        "### Оценка продавца (Оценка качества звонка по стадиям разговора и эмоц. составляющей)"
    )
    md.append("_Шкала: 0 = нет/плохо, 1 = частично/средне, 2 = отлично_")

    # Автоматически формируем вывод
    for key, label in labels.items():
        score = s.get(key, 0)
        md.append(f"- {label}: **{score} / 2**")

    # Добавляем комментарий, если он есть
    cmt = s.get("comment", "").strip()
    if cmt:
        md.append(f"- **Комментарий:** {cmt}")

    # Итоговый markdown-текст
    markdown_output = "\n".join(md)


# ------------------------- Основной процесс для одного файла -------------------------


def process_file(
    in_path: Path,
    base_rel: Path,
    out_root: Path,
    backend: str,
    model: str,
    num_ctx: int,
    num_keep: int,
    chunk_chars: int,
    chunk_overlap: int,
    prompt_budget: Optional[int],
    temperature: float,
    timeout: int,
    skip_exists: bool,
) -> Tuple[bool, str]:
    out_path = (out_root / base_rel).with_suffix(".spk.summary.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if skip_exists and out_path.exists():
        return True, f"skip (exists): {out_path}"

    # 1) загрузить сегменты
    segs = load_spk_json(in_path)
    if not segs:
        return False, "пустые сегменты"

    # 2) подготовить строки диалога и чанки
    lines = format_dialog_lines(segs)
    chunks = chunk_by_chars(lines, chunk_chars=chunk_chars, overlap_chars=chunk_overlap)
    if not chunks:
        return False, "пустые чанки"

    # 3) прогнать каждый чанк через модель → собрать частичные JSON
    parts_json: List[dict] = []
    for idx, ch in enumerate(chunks, 1):
        system = SYSTEM_PROMPT_RU
        schema = INSTR_SCHEMA_RU
        dialog_text = ch

        if prompt_budget:
            # применим авто-усечение под бюджет
            dialog_text = fit_chunk_to_budget(
                system, schema, dialog_text, prompt_budget
            )

        prompt = CHUNK_PROMPT_TMPL.format(
            system=system, schema=schema, dialog_text=dialog_text
        )

        # вызов LLM
        if backend == "ollama":
            raw = call_ollama_generate(
                model=model,
                prompt=prompt,
                num_ctx=num_ctx,
                num_keep=num_keep,
                temperature=temperature,
                timeout=timeout,
            )
        else:
            return False, f"backend '{backend}' не поддержан в этом скрипте"

        js = safe_json_loads(raw)
        if not js:
            # если не распарсили — запишем пустышку, чтобы не ронять пайплайн
            js = {
                "topics": [],
                "agreements": [],
                "product_issues": [],
                "customer_wishes": [],
                "objections": [],
                "objection_tips": [],
                "seller_score": {
                    "structure": 0,
                    "needs": 0,
                    "objections": 0,
                    "closing": 0,
                    "rapport": 0,
                    "comment": "",
                },
            }
        parts_json.append(js)

    # 4) reduce-слияние (кодом; опционально можно дополнительно прогнать через LLM)
    agg = merge_chunk_json(parts_json)

    # 5) рендер финального Markdown
    title = base_rel.as_posix()
    md = render_markdown(agg, title=title)
    out_path.write_text(md, encoding="utf-8")
    return True, str(out_path)


# ------------------------- CLI -------------------------


def collect_inputs(src_root: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in src_root.rglob("*")
            if p.is_file() and p.name.endswith(SUPPORTED_EXT)
        ]
    )


def main():
    ap = argparse.ArgumentParser(
        description="Суммаризация .spk.json → Markdown (RU). Поддержка Ollama и авто-бюджета промпта."
    )
    ap.add_argument("--src", required=True, type=Path, help="Папка с *.spk.json")
    ap.add_argument(
        "--out", required=True, type=Path, help="Куда писать *.spk.summary.md"
    )
    ap.add_argument(
        "--backend", default="ollama", choices=["ollama"], help="LLM-бэкенд"
    )
    ap.add_argument(
        "--ollama-model",
        dest="model",
        default="qwen2.5:14b-instruct-q6_K",
        help="Имя модели Ollama",
    )
    ap.add_argument("--num-ctx", type=int, default=16384, help="Контекст окна модели")
    ap.add_argument(
        "--num-keep",
        type=int,
        default=64,
        help="Сколько токенов сохранять между вызовами (Ollama)",
    )
    ap.add_argument("--temperature", type=float, default=0.1, help="Температура")
    ap.add_argument(
        "--timeout", type=int, default=600, help="HTTP таймаут запроса к модели, сек"
    )
    ap.add_argument(
        "--chunk-chars", type=int, default=9000, help="Целевой размер чанка в символах"
    )
    ap.add_argument(
        "--chunk-overlap", type=int, default=800, help="Перекрытие чанков, символы"
    )
    ap.add_argument(
        "--prompt-budget",
        type=int,
        default=None,
        help="Максимум токенов на ИНПУТ (инструкции+контент). Если задан — чанк авто-уменьшится под бюджет.",
    )
    ap.add_argument("--jobs", type=int, default=2, help="Параллельные файлы")
    ap.add_argument(
        "--skip-exists", action="store_true", help="Пропускать уже готовые *.summary.md"
    )
    args = ap.parse_args()

    src_root = args.src.resolve()
    out_root = args.out.resolve()
    files = collect_inputs(src_root)
    if not files:
        print(f"[INFO] Нет входных *.spk.json в {src_root}")
        return

    print(
        f"[INFO] Files: {len(files)} | backend={args.backend} model={args.model} "
        f"| num_ctx={args.num_ctx} | chunk={args.chunk_chars}/{args.chunk_overlap}chars "
        f"| prompt_budget={args.prompt_budget or '-'}"
    )
    ok = err = 0

    def _one(p: Path):
        base_rel = p.relative_to(src_root)
        try:
            done, msg = process_file(
                in_path=p,
                base_rel=base_rel,
                out_root=out_root,
                backend=args.backend,
                model=args.model,
                num_ctx=args.num_ctx,
                num_keep=args.num_keep,
                chunk_chars=args.chunk_chars,
                chunk_overlap=args.chunk_overlap,
                prompt_budget=args.prompt_budget,
                temperature=args.temperature,
                timeout=args.timeout,
                skip_exists=args.skip_exists,
            )
        except Exception as e:
            done, msg = False, f"Ошибка: {e}"
        return (done, p, msg)

    with cf.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = [ex.submit(_one, p) for p in files]
        for i, fut in enumerate(cf.as_completed(futs), 1):
            done, p, msg = fut.result()
            prefix = f"[{i}/{len(files)}]"
            if done:
                ok += 1
                print(f"{prefix} [OK] {p} -> {msg}")
            else:
                err += 1
                print(f"{prefix} [ERR] {p}: {msg}", file=sys.stderr)

    print(f"[SUMMARY] ok={ok}, err={err}")


if __name__ == "__main__":
    main()
