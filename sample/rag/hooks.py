# -*- coding: utf-8 -*-
"""Called in speech recognition callback to write teacher/student text to storage and (optionally) incrementally build vector index."""


from __future__ import annotations

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from .pipeline import RAGPipeline

        _pipeline = RAGPipeline()
    return _pipeline


def on_teacher_text(text: str, index_immediately: bool = True) -> None:
    p = get_pipeline()
    if index_immediately:
        p.add_utterance_and_index(text, "teacher")
    else:
        p.save_utterance(text, "teacher")


def on_student_text(text: str, index_immediately: bool = False) -> None:
    p = get_pipeline()
    if index_immediately:
        p.add_utterance_and_index(text, "student")
    else:
        p.save_utterance(text, "student")
