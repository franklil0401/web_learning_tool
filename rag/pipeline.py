# -*- coding: utf-8 -*-
"""Course content ingestion and retrieval pipeline."""

from __future__ import annotations

from typing import Any

from app.config import Settings
from rag.chunking import chunk_text
from rag.transcript_store import TranscriptRecord, TranscriptStore
from rag.vector_index import LocalVectorIndex


class CourseRAGPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.store = TranscriptStore(self.settings.transcript_path)
        self.index = LocalVectorIndex(self.settings.index_path)
        self.index.load()

    def add_teacher_text(self, text: str) -> TranscriptRecord | None:
        record = self.store.append(text, source="teacher")
        if record is None:
            return None
        self._index_record(record)
        self.index.save()
        return record

    def save_student_question(self, question: str) -> TranscriptRecord | None:
        return self.store.append(question, source="student")

    def _index_record(self, record: TranscriptRecord) -> None:
        chunks = chunk_text(
            record.text,
            max_chars=self.settings.chunk_max_chars,
            overlap=self.settings.chunk_overlap,
        )
        metas: list[dict[str, Any]] = [
            {
                "record_id": record.id,
                "source": record.source,
                "chunk_index": index,
                "created_at": record.created_at,
            }
            for index, _ in enumerate(chunks)
        ]
        self.index.add(chunks, metas)

    def rebuild_index(self) -> int:
        self.index.clear()
        count = 0
        for record in self.store.iter_records(source="teacher"):
            before = len(self.index.documents)
            self._index_record(record)
            count += len(self.index.documents) - before
        self.index.save()
        return count

    def retrieve(self, question: str, top_k: int | None = None) -> list[dict[str, Any]]:
        results = self.index.search(question, top_k or self.settings.retrieve_top_k)
        return [
            result.to_dict()
            for result in results
            if result.score >= self.settings.min_similarity
        ]

    def build_context(self, references: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for index, item in enumerate(references, start=1):
            lines.append(f"[课堂片段 {index}] {item['text']}")
        return "\n".join(lines)

    def recent_records(self, limit: int = 5) -> list[TranscriptRecord]:
        return self.store.recent(limit)

    def clear_history(self) -> None:
        self.store.clear()
        self.index.clear()
        self.index.save()
