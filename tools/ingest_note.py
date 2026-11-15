#!/usr/bin/env python3
"""Кладёт текстовый файл в базу знаний RAG.

Режет на чанки (1500 символов, перекрытие 150) и шлёт каждый в POST /ingest
с metadata_source. Нужен, когда MCP-клиент недоступен, но RAG-API жив.

Запуск:  python3 ingest_note.py <файл> [metadata_source]
"""
import json
import sys
import urllib.request

API = "http://127.0.0.1:8080"
CHUNK = 1500
OVERLAP = 150


def split_chunks(text: str) -> list[str]:
    if len(text) <= CHUNK:
        return [text]
    out = []
    start = 0
    while start < len(text):
        out.append(text[start:start + CHUNK])
        start += CHUNK - OVERLAP
    return out


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = sys.argv[1]
    source = sys.argv[2] if len(sys.argv) > 2 else "tick:note"

    with open(path, encoding="utf-8") as f:
        text = f.read()

    parts = split_chunks(text)
    total = len(parts)
    print(f"файл: {path}  символов: {len(text)}  чанков: {total}")

    for i, part in enumerate(parts, 1):
        payload = json.dumps({
            "text": part,
            "metadata_source": source,
            "chunk_index": i,
            "chunk_total": total,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{API}/ingest",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read())
        print(f"  [{i}/{total}] {result.get('status')} {result.get('id', '')}")

    print(f"готово: {total} чанков -> {source}")


if __name__ == "__main__":
    main()
