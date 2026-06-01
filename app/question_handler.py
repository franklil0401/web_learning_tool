# -*- coding: utf-8 -*-
"""Student question handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.llm import AnswerGenerator
from rag.pipeline import CourseRAGPipeline


@dataclass(frozen=True)
class QuestionAnswer:
    answer: str
    references: list[dict[str, Any]]
    used_llm: bool = False
    warning: str = ""


class StudentQuestionHandler:
    def __init__(
        self,
        pipeline: CourseRAGPipeline | None = None,
        answer_generator: AnswerGenerator | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.pipeline = pipeline or CourseRAGPipeline(self.settings)
        self.answer_generator = answer_generator or AnswerGenerator(self.settings)

    def ask(self, question: str) -> QuestionAnswer:
        clean_question = (question or "").strip()
        if not clean_question:
            return QuestionAnswer(answer="请输入有效问题。", references=[])

        self.pipeline.save_student_question(clean_question)
        references = self.pipeline.retrieve(clean_question)
        if not references:
            return QuestionAnswer(
                answer="当前课堂内容不足，暂时无法根据已记录课程回答这个问题。",
                references=[],
            )

        context = self.pipeline.build_context(references)
        generated = self.answer_generator.generate(clean_question, references, context)
        return QuestionAnswer(
            answer=generated.text,
            references=references,
            used_llm=generated.used_llm,
            warning=generated.warning,
        )

