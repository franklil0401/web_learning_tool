# -*- coding: utf-8 -*-
"""Split long text into overlapping chunks for retrieval."""

from __future__ import annotations


def chunk_text(
    text: str,
    max_chars: int = 400,
    overlap: int = 80,
) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks
