# -*- coding: utf-8 -*-
"""Persistent classroom transcript storage."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class TranscriptRecord:
    id: str
    source: str
    text: str
    created_at: str

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "source": self.source,
            "text": self.text,
            "created_at": self.created_at,
        }


class TranscriptStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, text: str, source: str = "teacher") -> TranscriptRecord | None:
        clean = (text or "").strip()
        if not clean:
            return None

        record = TranscriptRecord(
            id=str(uuid.uuid4()),
            source=source,
            text=clean,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        return record

    def iter_records(self, source: str | None = None) -> Iterator[TranscriptRecord]:
        if not self.path.is_file():
            return
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                if source is not None and raw.get("source") != source:
                    continue
                yield TranscriptRecord(
                    id=str(raw["id"]),
                    source=str(raw["source"]),
                    text=str(raw["text"]),
                    created_at=str(raw["created_at"]),
                )

    def recent(self, limit: int = 5) -> list[TranscriptRecord]:
        rows = list(self.iter_records())
        return rows[-max(0, limit) :]

    def count(self) -> int:
        return sum(1 for _ in self.iter_records())

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
        self.path.touch()

