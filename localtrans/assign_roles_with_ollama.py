#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json, re, math, time
from pathlib import Path
from collections import defaultdict
import datetime as dt
import srt
import requests

# runtime detection (IS_GPU, BACKEND, MODEL_DEVICE, MODEL_PATH)
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


# ---------- utils ----------


def compose_srt(segments):
    subs = []
    for i, seg in enumerate(segments, 1):
        subs.append(
            srt.Subtitle(
                index=i,
                start=dt.timedelta(seconds=float(seg["start"])),
                end=dt.timedelta(seconds=float(seg["end"])),
                content=str(seg["text"]),
            )
        )
    return srt.compose(subs)


def human_sec(s):
    s = float(s or 0)
    h = int(s // 3600)
    s -= h * 3600
    m = int(s // 60)
    s -= m * 60
    return f"{h:02d}:{m:02d}:{int(s):02d}"


def take_samples(segments, spk, max_utts=20, max_chars=1200):
    """Выбрать равномерную выборку коротких реплик данного спикера."""
    own = [seg for seg in segments if seg.get("speaker") == spk]
    if not own:
        return []
    # сортируем по времени
    own.sort(key=lambda x: x["start"])
    # тонкая равномерная подвыборка
    step = max(1, len(own) // max_utts)
    picked = own[::step][:max_utts]

    # ограничим общий объём
    out = []
    total = 0
    for p in picked:
        t = re.sub(r"\s+", " ", str(p.get("text", "")))
        # убрать префикс "SPKxx:"
        t = re.sub(r"^[A-Z]{3}\d{2}:\s*", "", t).strip()
        if not t:
            continue
        out.append({"t": human_sec(p["start"]), "text": t})
        total += len(t)
        if total >= max_chars:
            break
    return out


def summarize_stats(segments):
    stats = defaultdict(
        lambda: {"dur": 0.0, "turns": 0, "first": None, "last": None, "chars": 0}
    )
    for seg in segments:
        spk = seg.get("speaker", "SPK??")
        dur = float(seg.get("end", 0)) - float(seg.get("start", 0))
        txt = str(seg.get("text", ""))
        # срежем префикс если уже есть
        txt = re.sub(r"^[A-Z]{3}\d{2}:\s*", "", txt).strip()
        st = stats[spk]
        st["dur"] += max(0.0, dur)
        st["turns"] += 1
        st["first"] = (
            float(seg["start"])
            if st["first"] is None
            else min(st["first"], float(seg["start"]))
        )
        st["last"] = (
            float(seg["end"])
            if st["last"] is None
            else max(st["last"], float(seg["end"]))
        )
        st["chars"] += len(txt)
    # добавим долю говорения
    total_dur = sum(v["dur"] for v in stats.values()) or 1.0
    for v in stats.values():
        v["talk_share"] = v["dur"] / total_dur
    return dict(stats)


# ---------- LLM callers ----------

ROLE_SYS_PROMPT_RU = (
    "Ты помогаешь определить роли участников звонка по их репликам. "
    "Возможные роли ограничены заранее. Возвращай строго JSON."
)


def build_user_prompt_ru(labels, file_rel, stats, samples):
    # компактный и строгий промпт
    lines = []
    lines.append("Задача: для каждого спикера (SPKxx) назначить одну роль из списка.")
    lines.append(f"Список допустимых ролей: {', '.join(labels)}.")
    lines.append("Возвращай строгий JSON формата:")
    lines.append(
        '{"mapping": {"SPK01":"Клиент","SPK02":"Менеджер", ...}, '
        '"confidence": {"SPK01":0.82,"SPK02":0.71,...}, '
        '"reasons": {"SPK01":"Ключевые фразы...","SPK02":"..."}}'
    )
    lines.append("Без пояснений вне JSON.")
    lines.append("")
    lines.append(f"Файл: {file_rel}")
    lines.append("Статистика по спикерам:")
    for spk, st in stats.items():
        lines.append(
            f"- {spk}: dur={st['dur']:.1f}s share={st['talk_share']*100:.1f}% "
            f"turns={st['turns']} first={human_sec(st['first'])} last={human_sec(st['last'])} chars={st['chars']}"
        )
    lines.append("")
    lines.append("Примеры реплик по каждому спикеру:")
    for spk, arr in samples.items():
        lines.append(f"{spk}:")
        for it in arr[:6]:
            txt = it["text"]
            # урежем очень длинные
            if len(txt) > 200:
                txt = txt[:200] + "…"
            lines.append(f"  [{it['t']}] {txt}")
    return "\n".join(lines)


try:
    from localtrans.llm_adapters import generate_json
except Exception:

    def generate_json(backend, model, prompt, **opts):
        # minimal fallback: call local ollama like before
        url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        payload = {
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"num_ctx": int(opts.get("num_ctx", 8192))},
        }
        r = requests.post(url, json=payload, timeout=opts.get("timeout", 180))
        r.raise_for_status()
        data = r.json()
        return json.loads(data.get("response", "{}"))


def call_openai_json(
    model, prompt, api_key=None, base_url="https://api.openai.com/v1", timeout=180
):
    """Используем /responses с JSON-инструкцией — ответом должен быть JSON."""
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY не задан")

    url = f"{base_url}/responses"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "input": [
            {"role": "system", "content": ROLE_SYS_PROMPT_RU},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    # text in data["output"]["text"] OR choices[0].message?
    # /responses returns consolidated output:
    out = data.get("output", {})
    text = out.get("text")
    if not text:
        # fallback older shape
        text = data.get("choices", [{}])[0].get("message", {}).get("content")
    if not text:
        raise RuntimeError(f"Empty OpenAI response: {data}")
    return json.loads(text)


# ---------- heuristic fallback ----------


def heuristic_roles(labels, stats, samples):
    # простая эвристика: самый «говорящий» = Менеджер, следующий = Клиент, остальные = Другое
    # (если таких меток нет — кладём в первую и последнюю из списка)
    lab_default = list(labels)
    if not lab_default:
        lab_default = [
            "Менеджер",
            "Клиент",
            "Другое",
            "Саппорт",
            "Эксперт",
            "Модератор",
            "Продажи",
            "Техник",
            "Проектный менеджер",
        ]

    spks = sorted(
        stats.keys(),
        key=lambda s: (stats[s]["talk_share"], -stats[s]["first"]),
        reverse=True,
    )
    mapping = {}
    conf = {}
    rsn = {}

    def take_label(i):
        if i < len(lab_default):
            return lab_default[i]
        return lab_default[-1]

    for i, spk in enumerate(spks):
        role = take_label(i)
        mapping[spk] = role
        conf[spk] = 0.55 if i == 0 else (0.5 if i == 1 else 0.4)
        ex = samples.get(spk, [])
        rsn[spk] = (
            f"Эвристика по доле речи ({stats[spk]['talk_share']*100:.1f}%), порядку входа и количеству реплик. Примеры: "
            + " | ".join(x["text"][:80] for x in ex[:3])
        )
    return {"mapping": mapping, "confidence": conf, "reasons": rsn}


# ---------- main pipeline ----------


def load_spk_json(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    # ожидаем список сегментов [{start,end,text,speaker}]
    if isinstance(data, list):
        return data
    raise ValueError(f"{path} не является списком сегментов")


def rewrite_with_roles(segments, rolemap):
    out = []
    for seg in segments:
        spk = seg.get("speaker", "SPK??")
        role = rolemap.get(spk, spk)
        # удалим старый префикс и добавим новый
        txt = str(seg.get("text", ""))
        txt = re.sub(r"^[A-Z]{3}\d{2}:\s*", "", txt).strip()
        new_txt = f"{role} ({spk}): {txt}".strip()
        out.append({**seg, "text": new_txt})
    return out


def process_file(
    in_path: Path,
    base_rel: Path,
    out_root: Path,
    labels,
    mode,
    model,
    num_ctx,
    timeout,
    skip_exists,
):
    out_base = (out_root / base_rel).with_suffix("")  # без .json
    out_base.parent.mkdir(parents=True, exist_ok=True)

    roles_json_path = out_base.with_suffix(".roles.json")
    spk_roles_json = out_base.with_suffix(".spk.roles.json")
    spk_roles_txt = out_base.with_suffix(".spk.roles.txt")
    spk_roles_srt = out_base.with_suffix(".spk.roles.srt")

    if (
        skip_exists
        and roles_json_path.exists()
        and spk_roles_json.exists()
        and spk_roles_txt.exists()
        and spk_roles_srt.exists()
    ):
        return ("skip", in_path, "exists")

    segments = load_spk_json(in_path)
    stats = summarize_stats(segments)
    spk_list = sorted(stats.keys())

    # собрать примеры
    samples = {
        spk: take_samples(segments, spk, max_utts=18, max_chars=1200)
        for spk in spk_list
    }

    # LLM prompt
    prompt = build_user_prompt_ru(labels, str(base_rel), stats, samples)

    # Вызов модели
    llm_ok = True
    try:
        if mode == "openai":
            # call OpenAI via adapter
            res = generate_json(
                backend="openai", model=model, prompt=prompt, timeout=timeout
            )
        else:
            # default local/ollama or vllm depending on runtime.BACKEND
            res = generate_json(
                backend=BACKEND,
                model=model,
                prompt=prompt,
                num_ctx=num_ctx,
                timeout=timeout,
            )
    except Exception as e:
        llm_ok = False
        res = heuristic_roles(labels, stats, samples)
        try:
            res["_error"] = f"LLM fail: {e}"
        except Exception:
            # ensure res is dict-like
            res = {**(res or {}), "_error": f"LLM fail: {e}"}

    # нормализуем роль-неймы (чтобы не улетели за список)
    allowed = set(labels)
    mapping = {spk: (res.get("mapping", {}).get(spk) or "Другое") for spk in spk_list}
    for spk in mapping:
        if mapping[spk] not in allowed:
            mapping[spk] = "Другое" if "Другое" in allowed else list(allowed)[-1]

    # пишем roles.json
    roles_payload = {
        "file": str(base_rel),
        "labels": labels,
        "mode": mode,
        "model": model,
        "llm_success": llm_ok,
        "result": {
            "mapping": mapping,
            "confidence": res.get("confidence", {}),
            "reasons": res.get("reasons", {}),
        },
        "stats": stats,
    }
    roles_json_path.write_text(
        json.dumps(roles_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # перепишем транскрипт с ролями
    seg2 = rewrite_with_roles(segments, mapping)
    spk_roles_json.write_text(
        json.dumps(seg2, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    spk_roles_txt.write_text("\n".join(s["text"] for s in seg2), encoding="utf-8")
    spk_roles_srt.write_text(compose_srt(seg2), encoding="utf-8")

    return ("ok", in_path, f"roles → {roles_json_path.name}")


def collect_inputs(src_dir: Path):
    return sorted([p for p in src_dir.rglob("*.spk.json") if p.is_file()])


def main():
    ap = argparse.ArgumentParser(
        description="Назначение ролей спикерам по *.spk.json + переименование реплик"
    )
    ap.add_argument(
        "--src",
        required=True,
        type=Path,
        help="Папка с *.spk.json (после merge_diar_any.py)",
    )
    ap.add_argument(
        "--out", required=True, type=Path, help="Папка для результата (та же структура)"
    )
    ap.add_argument(
        "--labels",
        default="Менеджер,Клиент,Саппорт,Другое",
        help="Список ролей через запятую",
    )
    ap.add_argument(
        "--mode",
        choices=["local", "openai"],
        default="local",
        help="local=Ollama, openai=OpenAI API",
    )
    ap.add_argument(
        "--model", default="gpt-oss:20b", help="Имя модели (для Ollama или OpenAI)"
    )
    ap.add_argument("--num-ctx", type=int, default=8192, help="num_ctx для Ollama")
    ap.add_argument("--timeout", type=int, default=180, help="HTTP таймаут, сек")
    ap.add_argument(
        "--skip-exists", action="store_true", help="Пропускать уже готовые файлы"
    )
    args = ap.parse_args()

    src = args.src.resolve()
    out = args.out.resolve()
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]

    files = collect_inputs(src)
    if not files:
        print(f"[INFO] Нет входных *.spk.json в {src}")
        return

    ok = skip = err = 0
    for p in files:
        base_rel = p.relative_to(src)
        try:
            st, _, msg = process_file(
                in_path=p,
                base_rel=base_rel,
                out_root=out,
                labels=labels,
                mode=args.mode,
                model=args.model,
                num_ctx=args.num_ctx,
                timeout=args.timeout,
                skip_exists=args.skip_exists,
            )
            if st == "ok":
                ok += 1
                print(f"[OK] {base_rel} → {msg}")
            elif st == "skip":
                skip += 1
                print(f"[SKIP] {base_rel} ({msg})")
            else:
                err += 1
                print(f"[ERR] {base_rel}: {msg}")
        except Exception as e:
            err += 1
            print(f"[ERR] {base_rel}: {e}")

    print(f"[SUMMARY] ok={ok}, skip={skip}, err={err}")


if __name__ == "__main__":
    main()
