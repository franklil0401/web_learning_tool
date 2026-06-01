# -*- coding: utf-8 -*-
"""Append speech-to-text lines to JSONL for audit and index rebuild."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal

Source = Literal["teacher", "student", "system"]


class TranscriptStore:
    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.data_dir = Path(data_dir or Path(__file__).resolve().parent.parent / "data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.data_dir / "transcripts.jsonl"

    def append(self, text: str, source: Source = "teacher") -> str:
        text = (text or "").strip()
        if not text:
            return ""
        rid = str(uuid.uuid4())
        row = {
            "id": rid,
            "ts": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "text": text,
        }
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return rid

    def iter_rows(self) -> Iterator[dict]:
        if not self._path.is_file():
            return
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def load_texts_by_source(
        self, source: Source | None = None
    ) -> list[tuple[str, str, str]]:
        out: list[tuple[str, str, str]] = []
        for row in self.iter_rows():
            if source is not None and row.get("source") != source:
                continue
            out.append((row["id"], row["source"], row["text"]))
        return out
    
    def clear(self) -> None:
        """Clear all transcript history"""
        if self._path.is_file():
            self._path.unlink()
        # Create empty file to maintain structure
        self._path.touch()
