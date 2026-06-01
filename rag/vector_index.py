# -*- coding: utf-8 -*-
"""A lightweight local vector-like index based on token cosine similarity."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _is_cjk(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    ascii_buffer: list[str] = []
    cjk_buffer: list[str] = []

    def flush_ascii() -> None:
        if ascii_buffer:
            tokens.append("".join(ascii_buffer))
            ascii_buffer.clear()

    def flush_cjk() -> None:
        if not cjk_buffer:
            return
        chars = cjk_buffer[:]
        tokens.extend(chars)
        tokens.extend("".join(chars[i : i + 2]) for i in range(len(chars) - 1))
        tokens.extend("".join(chars[i : i + 3]) for i in range(len(chars) - 2))
        cjk_buffer.clear()

    for char in (text or "").lower():
        if _is_cjk(char):
            flush_ascii()
            cjk_buffer.append(char)
        elif char.isalnum():
            flush_cjk()
            ascii_buffer.append(char)
        else:
            flush_ascii()
            flush_cjk()

    flush_ascii()
    flush_cjk()
    return [token for token in tokens if token]


def _vectorize(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(weight * right.get(token, 0) for token, weight in left.items())
    left_norm = math.sqrt(sum(weight * weight for weight in left.values()))
    right_norm = math.sqrt(sum(weight * weight for weight in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


@dataclass(frozen=True)
class SearchResult:
    score: float
    text: str
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"score": self.score, "text": self.text, "meta": self.meta}


class LocalVectorIndex:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.documents: list[dict[str, Any]] = []

    def load(self) -> None:
        if not self.path.is_file():
            self.documents = []
            return
        with self.path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
        self.documents = list(raw.get("documents", []))

    def save(self) -> None:
        payload = {"documents": self.documents}
        with self.path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    def add(self, texts: list[str], metas: list[dict[str, Any]]) -> None:
        for text, meta in zip(texts, metas):
            clean = (text or "").strip()
            if not clean:
                continue
            self.documents.append({"text": clean, "meta": meta})

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        query_vector = _vectorize(query)
        if not query_vector:
            return []

        scored: list[SearchResult] = []
        for document in self.documents:
            doc_text = str(document.get("text", ""))
            score = _cosine(query_vector, _vectorize(doc_text))
            if score > 0:
                scored.append(
                    SearchResult(
                        score=score,
                        text=doc_text,
                        meta=dict(document.get("meta", {})),
                    )
                )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[: max(0, top_k)]

    def clear(self) -> None:
        self.documents = []
        if self.path.exists():
            self.path.unlink()

