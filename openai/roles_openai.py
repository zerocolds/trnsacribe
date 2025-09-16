#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, os, sys, math
from pathlib import Path
from collections import defaultdict
from openai import OpenAI

SCHEMA = {
    "name": "roles_schema",
    "schema": {
        "type": "object",
        "properties": {
            "roles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {"type": "string"},
                        "role": {
                            "type": "string",
                            "description": "Краткая роль: Менеджер, Клиент, Инженер, Аналитик, Руководитель и т.п.",
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "evidence_segments": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                    },
                    "required": ["speaker", "role", "confidence", "evidence_segments"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["roles"],
        "additionalProperties": False,
    },
    "strict": True,
}

PROMPT_DEV = (
    "Ты классифицируешь роли участников встречи по их репликам. "
    "Отвечай только по тексту, без догадок. Если роли неясны — ставь 'Неопределено'. "
    "Верни JSON по схеме."
)


def load_spk_json(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, list), "Ожидаю список сегментов"
    return data


def build_profile(segments, max_chars=4000):
    by_spk = defaultdict(list)
    for i, s in enumerate(segments):
        by_spk[s.get("speaker", "SPK??")].append((i, s.get("text", "")))
    # формируем компактные профили
    profiles = {}
    for spk, items in by_spk.items():
        acc, used = [], []
        total = 0
        for idx, txt in items:
            line = f"[{idx}] {txt.strip()}\n"
            if total + len(line) > max_chars:
                break
            acc.append(line)
            used.append(idx)
            total += len(line)
        profiles[spk] = {"sample": "".join(acc), "indices": used}
    return profiles


def main():
    ap = argparse.ArgumentParser(
        description="Назначение ролей спикерам (по *.spk.json) через OpenAI Responses API"
    )
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default="o4-mini")  # разумный баланс цены/качества
    ap.add_argument("--skip-exists", action="store_true")
    args = ap.parse_args()

    client = OpenAI()
    files = [p for p in args.src.rglob("*.spk.json")]
    ok = err = 0
    for p in files:
        rel = p.relative_to(args.src)
        out_base = (args.out / rel).with_suffix("")
        out_base.parent.mkdir(parents=True, exist_ok=True)
        map_path = out_base.with_suffix(".roles.map.json")
        spk_roles_path = out_base.with_suffix(".spk.roles.json")
        if args.skip_exists and spk_roles_path.exists():
            ok += 1
            continue

        try:
            segs = load_spk_json(p)
            profiles = build_profile(segs, max_chars=6000)
            # готовим пользовательский контент
            parts = ["Определи роли участников по их примерам реплик:\n"]
            for spk, prof in profiles.items():
                parts.append(f"=== {spk} ===\n{prof['sample']}\n")
            user_text = "\n".join(parts)

            resp = client.responses.create(
                model=args.model,
                input=[
                    {"role": "developer", "content": PROMPT_DEV},
                    {"role": "user", "content": user_text},
                ],
                response_format={"type": "json_schema", "json_schema": SCHEMA},
            )
            out = (
                resp.output_parsed
                if hasattr(resp, "output_parsed")
                else json.loads(resp.output[0].content[0].text)
            )
            roles = {r["speaker"]: r for r in out.get("roles", [])}

            # применяем роли к сегментам
            for i, s in enumerate(segs):
                spk = s.get("speaker", "SPK??")
                s["role"] = roles.get(spk, {}).get("role", "Неопределено")

            map_path.write_text(
                json.dumps(roles, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            spk_roles_path.write_text(
                json.dumps(segs, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            ok += 1
        except Exception as e:
            err += 1
            print(f"[ERR] {p}: {e}", file=sys.stderr)
    print(f"[SUMMARY] ok={ok}, err={err}")


if __name__ == "__main__":
    main()
