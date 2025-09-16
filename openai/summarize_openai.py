#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, os, re, sys, time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from openai import OpenAI

SCHEMA_SUM = {
    "name": "meeting_summary_schema",
    "schema": {
        "type": "object",
        "properties": {
            "discussed": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "point": {"type": "string"},
                        "segments": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["point", "segments"],
                    "additionalProperties": False,
                },
            },
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "point": {"type": "string"},
                        "segments": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["point", "segments"],
                    "additionalProperties": False,
                },
            },
            "product_issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "point": {"type": "string"},
                        "segments": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["point", "segments"],
                    "additionalProperties": False,
                },
            },
            "user_feedback": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "point": {"type": "string"},
                        "segments": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["point", "segments"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["discussed", "decisions", "product_issues", "user_feedback"],
        "additionalProperties": False,
    },
    "strict": True,
}

PROMPT_DEV = (
    "Ты аналитик: извлеки ТОЛЬКО факты из сегментов беседы. "
    "Не выдумывай. На каждый тезис приложи индексы сегментов-оснований."
)


def secs_to_tc(s: float) -> str:
    s = max(0, int(round(float(s))))
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def load_segments(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        segs = data
    elif isinstance(data, dict) and isinstance(data.get("segments"), list):
        segs = data["segments"]
    else:
        raise ValueError("Ожидаю list или dict.segments")
    out = []
    for i, s in enumerate(segs):
        out.append(
            {
                "index": i,
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "speaker": s.get("speaker") or "SPK??",
                "role": s.get("role"),
                "text": (s.get("text") or "").strip(),
            }
        )
    return out


def build_chunk_prompt(chunk: List[Dict[str, Any]]) -> str:
    lines = []
    for s in chunk:
        role = f"{s['role']} " if s.get("role") else ""
        lines.append(
            f"[{s['index']}] {secs_to_tc(s['start'])}-{secs_to_tc(s['end'])} {s['speaker']} {role}: {s['text']}"
        )
    return "Ниже сегменты:\n" + "\n".join(lines) + "\nСформируй JSON по схеме."


def split_by_chars(segs, max_chars=16000):
    chunks = []
    cur = []
    cur_len = 0
    for s in segs:
        L = len(s["text"]) + 64
        if cur and cur_len + L > max_chars:
            chunks.append(cur)
            cur = []
            cur_len = 0
        cur.append(s)
        cur_len += L
    if cur:
        chunks.append(cur)
    return chunks


def merge_points(objs: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ("discussed", "decisions", "product_issues", "user_feedback")
    merged = {k: [] for k in keys}

    def norm(t):
        return re.sub(r"\s+", " ", (t or "").strip().lower())

    for k in keys:
        pool = {}
        for o in objs:
            for it in o.get(k, []) or []:
                p = (it.get("point") or "").strip()
                if not p:
                    continue
                key = norm(p)
                pool.setdefault(key, {"point": p, "segments": set()})
                for idx in it.get("segments", []) or []:
                    try:
                        pool[key]["segments"].add(int(idx))
                    except:
                        pass
        merged[k] = [
            {"point": v["point"], "segments": sorted(v["segments"])}
            for v in pool.values()
        ]
    return merged


def render_md(
    summary: Dict[str, Any], segs: List[Dict[str, Any]], src_rel: str, model: str
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    def idx_to_tcs(arr):
        out = []
        seen = set()
        for i in arr or []:
            if 0 <= int(i) < len(segs):
                tc = secs_to_tc(segs[int(i)]["start"])
                if tc not in seen:
                    seen.add(tc)
                    out.append(tc)
        return out

    def section(name, items):
        if not items:
            return f"## {name}\n\n_нет явных данных_\n"
        lines = [f"## {name}\n"]
        for it in items:
            tcs = ", ".join(
                f"[{tc}](#tc{tc.replace(':','')})"
                for tc in idx_to_tcs(it.get("segments"))
            )
            lines.append(f"- {it['point']}{(' — ' + tcs) if tcs else ''}")
        return "\n".join(lines) + "\n"

    speakers = sorted({s["speaker"] for s in segs})
    anchors = []
    for s in segs:
        tc = secs_to_tc(s["start"])
        role = s.get("role") or ""
        anchors.append(
            f'<a id="tc{tc.replace(":","")}"></a>\n\n### ⏱ {tc}\n\n`{s["speaker"]}` {role and f"({role})"} — {s["text"]}\n'
        )

    parts = [
        "---",
        f'source: "{src_rel}"',
        f"created: {now}",
        f"model: {model}",
        "tags: [summary, transcript]",
        "---\n",
        "# Итоговая сводка\n",
        "**Говорящие:** " + ", ".join(speakers) + "\n",
        section("Обсуждали", summary.get("discussed", [])),
        section("Договорённости", summary.get("decisions", [])),
        section("Проблемы продукта / риски", summary.get("product_issues", [])),
        section("Мнение / реакция пользователя", summary.get("user_feedback", [])),
        "\n---\n## Основания по таймкодам\n",
        *anchors,
    ]
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(
        description="Саммаризация *.spk(.roles).json → Markdown с таймкодами (OpenAI Responses API)"
    )
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default="o4-mini")
    ap.add_argument("--max-chars", type=int, default=16000)
    ap.add_argument("--skip-exists", action="store_true")
    args = ap.parse_args()

    client = OpenAI()
    files = [p for p in args.src.rglob("*.spk.roles.json")]
    if not files:
        files = [p for p in args.src.rglob("*.spk.json")]

    ok = err = 0
    for p in files:
        rel = p.relative_to(args.src)
        out_path = (args.out / rel).with_suffix(".summary.md")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.skip_exists and out_path.exists():
            ok += 1
            continue
        try:
            segs = load_segments(p)
            chunks = split_by_chars(segs, args.max_chars)
            partials = []
            for ch in chunks:
                prompt = build_chunk_prompt(ch)
                resp = client.responses.create(
                    model=args.model,
                    input=[
                        {"role": "developer", "content": PROMPT_DEV},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_schema", "json_schema": SCHEMA_SUM},
                )
                part = (
                    resp.output_parsed
                    if hasattr(resp, "output_parsed")
                    else json.loads(resp.output[0].content[0].text)
                )
                partials.append(part)
            merged = merge_points(partials)
            md = render_md(merged, segs, str(rel), args.model)
            out_path.write_text(md, encoding="utf-8")
            ok += 1
        except Exception as e:
            err += 1
            print(f"[ERR] {p}: {e}", file=sys.stderr)
    print(f"[SUMMARY] ok={ok}, err={err}")


if __name__ == "__main__":
    main()
