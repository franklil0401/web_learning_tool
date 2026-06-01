# -*- coding: utf-8 -*-
from .hooks import get_pipeline, on_student_text, on_teacher_text
from .pipeline import RAGPipeline
from .transcript_store import TranscriptStore

__all__ = [
    "TranscriptStore",
    "RAGPipeline",
    "get_pipeline",
    "on_teacher_text",
    "on_student_text",
]


def __getattr__(name: str):
    if name == "aliyun_dashscope":
        from . import aliyun_dashscope as m

        return m
    raise AttributeError(name)
