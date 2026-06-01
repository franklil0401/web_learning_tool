# -*- coding: utf-8 -*-
"""Split course text into retrieval-friendly chunks."""

from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"([。！？；.!?;\n])", text)
    sentences: list[str] = []
    for index in range(0, len(parts), 2):
        body = parts[index].strip()
        punct = parts[index + 1] if index + 1 < len(parts) else ""
        sentence = (body + punct).strip()
        if sentence:
            sentences.append(sentence)
    return sentences or [text]


def _fixed_chunks(text: str, max_chars: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    step = max(1, max_chars - max(0, overlap))
    start = 0
    while start < len(text):
        piece = text[start : start + max_chars].strip()
        if piece:
            chunks.append(piece)
        if start + max_chars >= len(text):
            break
        start += step
    return chunks


def chunk_text(text: str, max_chars: int = 420, overlap: int = 60) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""
    for sentence in _split_sentences(text):
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_fixed_chunks(sentence, max_chars, overlap))
            continue

        candidate = f"{current}{sentence}" if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
        prefix = current[-overlap:] if overlap > 0 and current else ""
        current = f"{prefix}{sentence}" if prefix else sentence

    if current:
        chunks.append(current)
    return chunks

