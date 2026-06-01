# -*- coding: utf-8 -*-
"""Answer generation with LLM providers and a local fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import Settings


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    used_llm: bool = False
    warning: str = ""


class AnswerGenerator:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()

    def generate(
        self,
        question: str,
        references: list[dict[str, Any]],
        context: str,
    ) -> GeneratedAnswer:
        if not references:
            return GeneratedAnswer(text="当前课堂内容不足，暂时无法根据已记录课程回答这个问题。")

        provider = self._resolve_provider()
        if provider == "doubao":
            try:
                return GeneratedAnswer(
                    text=self._doubao_answer(question, context),
                    used_llm=True,
                )
            except Exception as exc:
                return GeneratedAnswer(
                    text=self._local_answer(references),
                    used_llm=False,
                    warning=f"豆包回答失败，已使用本地兜底：{exc}",
                )

        if provider == "openai":
            try:
                return GeneratedAnswer(
                    text=self._openai_answer(question, context),
                    used_llm=True,
                )
            except Exception as exc:
                return GeneratedAnswer(
                    text=self._local_answer(references),
                    used_llm=False,
                    warning=f"OpenAI 回答失败，已使用本地兜底：{exc}",
                )

        if provider == "dashscope":
            try:
                return GeneratedAnswer(
                    text=self._dashscope_answer(question, context),
                    used_llm=True,
                )
            except Exception as exc:
                return GeneratedAnswer(
                    text=self._local_answer(references),
                    used_llm=False,
                    warning=f"通义千问回答失败，已使用本地兜底：{exc}",
                )

        if provider == "deepseek":
            try:
                return GeneratedAnswer(
                    text=self._deepseek_answer(question, context),
                    used_llm=True,
                )
            except Exception as exc:
                return GeneratedAnswer(
                    text=self._local_answer(references),
                    used_llm=False,
                    warning=f"DeepSeek 回答失败，已使用本地兜底：{exc}",
                )

        return GeneratedAnswer(text=self._local_answer(references), used_llm=False)

    def _resolve_provider(self) -> str:
        provider = self.settings.llm_provider
        if provider == "auto":
            if self.settings.ark_api_key:
                return "doubao"
            if self.settings.openai_api_key:
                return "openai"
            if self.settings.deepseek_api_key:
                return "deepseek"
            if self.settings.dashscope_api_key:
                return "dashscope"
            return "local"
        if provider == "local" and self.settings.use_llm:
            if self.settings.ark_api_key:
                return "doubao"
            if self.settings.openai_api_key:
                return "openai"
            if self.settings.deepseek_api_key:
                return "deepseek"
            if self.settings.dashscope_api_key:
                return "dashscope"
        return provider

    def _openai_answer(self, question: str, context: str) -> str:
        if not self.settings.openai_api_key:
            raise RuntimeError("缺少 OPENAI_API_KEY")

        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("缺少 openai 依赖，请先执行：pip install openai") from exc

        kwargs = {"api_key": self.settings.openai_api_key}
        if self.settings.openai_base_url:
            kwargs["base_url"] = self.settings.openai_base_url
        client = OpenAI(**kwargs)

        system = (
            "你是学生上课时使用的实时答疑助手。以下是重要规则：\n"
            "1. 优先依据提供的课堂片段回答问题，尽量从片段中推断和综合出答案。\n"
            "2. 课堂片段来自实时语音识别，可能存在识别错误（如同音字错误）、碎片化、口语化等问题，你需要理解上下文语义并自动纠正。\n"
            "3. 即使片段看起来不完整，也应尽力从中提取有用信息，综合多个片段给出合理的推断和解释。\n"
            "4. 只有当片段与问题完全不相关时，才说'当前课堂内容不足'。不要因为片段碎片化或存在识别错误就拒绝回答。\n"
            "5. 回答要像助教一样，把碎片信息组织成连贯易懂的解释，不要只是复述原文。"
        )
        user = f"课堂片段：\n{context}\n\n学生问题：{question}"

        if hasattr(client, "responses"):
            response = client.responses.create(
                model=self.settings.openai_answer_model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            text = getattr(response, "output_text", "")
            if text:
                return text.strip()

        response = client.chat.completions.create(
            model=self.settings.openai_answer_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content.strip()

    def _dashscope_answer(self, question: str, context: str) -> str:
        if not self.settings.dashscope_api_key:
            raise RuntimeError("缺少 DASHSCOPE_API_KEY")

        import dashscope
        from dashscope import Generation

        dashscope.api_key = self.settings.dashscope_api_key
        messages = [
            {
                "role": "system",
                "content": (
                    "你是学生上课时使用的实时答疑助手。"
                    "优先依据提供的课堂片段回答，尽量从碎片化、口语化的片段中推断和综合。"
                    "课堂片段来自实时语音识别，可能存在同音字错误，需要理解上下文语义自动纠正。"
                    "即使片段不完整，也应尽力提取有用信息。"
                    "只有片段与问题完全不相关时才说内容不足。"
                    "回答要像助教一样把碎片信息组织成连贯易懂的解释。"
                ),
            },
            {
                "role": "user",
                "content": f"课堂片段：\n{context}\n\n学生问题：{question}",
            },
        ]
        response = Generation.call(
            model=self.settings.dashscope_model,
            messages=messages,
            result_format="message",
        )
        if getattr(response, "status_code", None) != 200:
            raise RuntimeError(getattr(response, "message", str(response)))
        return response.output["choices"][0]["message"]["content"].strip()

    def _doubao_answer(self, question: str, context: str) -> str:
        if not self.settings.ark_api_key:
            raise RuntimeError("缺少 ARK_API_KEY")

        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("缺少 openai 依赖，请先执行：pip install openai") from exc

        client = OpenAI(
            api_key=self.settings.ark_api_key,
            base_url=self.settings.ark_base_url,
        )
        response = client.chat.completions.create(
            model=self.settings.doubao_answer_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是学生上课时使用的实时答疑助手。以下是重要规则：\n"
                        "1. 优先依据提供的课堂片段回答问题，尽量从片段中推断和综合出答案。\n"
                        "2. 课堂片段来自实时语音识别，可能存在识别错误（如同音字错误）、碎片化、口语化等问题，你需要理解上下文语义并自动纠正。\n"
                        "3. 即使片段看起来不完整，也应尽力从中提取有用信息，综合多个片段给出合理的推断和解释。\n"
                        "4. 只有当片段与问题完全不相关时，才说'当前课堂内容不足'。不要因为片段碎片化或存在识别错误就拒绝回答。\n"
                        "5. 回答要像助教一样，把碎片信息组织成连贯易懂的解释，不要只是复述原文。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"课堂片段：\n{context}\n\n学生问题：{question}",
                },
            ],
            stream=False,
        )
        return response.choices[0].message.content.strip()

    def _deepseek_answer(self, question: str, context: str) -> str:
        if not self.settings.deepseek_api_key:
            raise RuntimeError("缺少 DEEPSEEK_API_KEY")

        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("缺少 openai 依赖，请先执行：pip install openai") from exc

        client = OpenAI(
            api_key=self.settings.deepseek_api_key,
            base_url=self.settings.deepseek_base_url,
        )
        response = client.chat.completions.create(
            model=self.settings.deepseek_answer_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是学生上课时使用的实时答疑助手。"
                        "优先依据提供的课堂片段回答，尽量从碎片化、口语化的片段中推断和综合。"
                        "课堂片段来自实时语音识别，可能存在同音字错误，需要理解上下文语义自动纠正。"
                        "即使片段不完整，也应尽力提取有用信息。"
                        "只有片段与问题完全不相关时才说内容不足。"
                        "回答要像助教一样把碎片信息组织成连贯易懂的解释。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"课堂片段：\n{context}\n\n学生问题：{question}",
                },
            ],
            stream=False,
        )
        return response.choices[0].message.content.strip()

    def _local_answer(self, references: list[dict[str, Any]]) -> str:
        snippets = []
        seen: set[str] = set()
        for item in references[:2]:
            text = str(item.get("text", "")).strip()
            if text and text not in seen:
                snippets.append(text)
                seen.add(text)

        joined = "；".join(snippets)
        return (
            "我先根据已记录的课堂片段给你一个保守回答："
            f"{joined}。"
            "当前未启用云端大模型，所以这里只做基于检索内容的简要整理。"
        )
