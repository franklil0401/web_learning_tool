# -*- coding: utf-8 -*-
"""Audio transcription providers."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    provider: str
    warning: str = ""


class AudioTranscriber:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()

    @property
    def provider(self) -> str:
        provider = self.settings.transcription_provider
        if provider == "auto":
            if self._has_doubao_streaming_credentials():
                return "doubao_streaming"
            if self.settings.ark_api_key:
                return "doubao"
            return "openai" if self.settings.openai_api_key else "vosk"
        return provider

    def should_use_cloud(self) -> bool:
        return self.provider in {"openai", "doubao", "doubao_streaming"}

    def should_use_realtime_stream(self) -> bool:
        return self.provider == "doubao_streaming"

    def missing_realtime_stream_settings(self) -> list[str]:
        missing = []
        if not self.settings.doubao_asr_app_key:
            missing.append("TA_DOUBAO_ASR_APP_KEY")
        if not self.settings.doubao_asr_access_key:
            missing.append("TA_DOUBAO_ASR_ACCESS_KEY")
        if not self.settings.doubao_asr_resource_id:
            missing.append("TA_DOUBAO_ASR_RESOURCE_ID")
        if not self.settings.doubao_asr_endpoint:
            missing.append("TA_DOUBAO_ASR_ENDPOINT")
        return missing

    def _has_doubao_streaming_credentials(self) -> bool:
        return (
            bool(self.settings.doubao_asr_app_key)
            and bool(self.settings.doubao_asr_access_key)
            and bool(self.settings.doubao_asr_resource_id)
        )

    def transcribe_wav(self, wav_path: str | Path) -> TranscriptionResult:
        provider = self.provider
        if provider == "openai":
            return self._transcribe_openai(Path(wav_path))
        if provider == "doubao":
            return self._transcribe_doubao(Path(wav_path))
        if provider == "doubao_streaming":
            raise RuntimeError("豆包流式 ASR 需要直接接收实时 PCM 音频，不能使用 WAV 片段转写。")
        raise RuntimeError(f"不支持的云端转写提供方：{provider}")

    def _transcribe_openai(self, wav_path: Path) -> TranscriptionResult:
        if not self.settings.openai_api_key:
            raise RuntimeError("缺少 OPENAI_API_KEY，无法使用 OpenAI 音频转写。")

        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("缺少 openai 依赖，请先执行：pip install openai") from exc

        kwargs = {"api_key": self.settings.openai_api_key}
        if self.settings.openai_base_url:
            kwargs["base_url"] = self.settings.openai_base_url
        client = OpenAI(**kwargs)

        with wav_path.open("rb") as audio_file:
            response = client.audio.transcriptions.create(
                model=self.settings.openai_transcribe_model,
                file=audio_file,
                response_format="text",
                prompt=self.settings.openai_transcribe_prompt or None,
            )

        text = response if isinstance(response, str) else getattr(response, "text", "")
        return TranscriptionResult(text=(text or "").strip(), provider="openai")

    def _transcribe_doubao(self, wav_path: Path) -> TranscriptionResult:
        if not self.settings.ark_api_key:
            raise RuntimeError("缺少 ARK_API_KEY，无法使用豆包音频转写。")

        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("缺少 openai 依赖，请先执行：pip install openai") from exc

        client = OpenAI(
            api_key=self.settings.ark_api_key,
            base_url=self.settings.ark_base_url,
        )
        encoded_audio = base64.b64encode(wav_path.read_bytes()).decode("utf-8")
        response = client.chat.completions.create(
            model=self.settings.doubao_transcribe_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": encoded_audio,
                                "format": "wav",
                            },
                        },
                        {
                            "type": "text",
                            "text": "请把这段课堂音频准确转写成中文文字，只输出转写结果。",
                        },
                    ],
                }
            ],
        )
        return TranscriptionResult(
            text=response.choices[0].message.content.strip(),
            provider="doubao",
        )
