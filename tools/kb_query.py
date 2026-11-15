#!/usr/bin/env python3
"""Поиск по базе знаний RAG без MCP-клиента.

Запуск:  python3 kb_query.py "текст запроса" [top_k]
"""
import json
import sys
import urllib.request

API = "http://127.0.0.1:8080"


def post(path: str, payload: dict, timeout: int = 120):
    req = urllib.request.Request(
        API + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    data = post("/search", {"query": query, "top_k": top_k})
    for i, r in enumerate(data.get("results", []), 1):
        print(f"--- {i}. source={r.get('source')} "
              f"chunk={r.get('chunk_index')}/{r.get('chunk_total')}")
        print(r.get("text", "")[:700])
        print()


if __name__ == "__main__":
    main()
