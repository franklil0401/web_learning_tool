# -*- coding: utf-8 -*-
"""Central configuration for the tutoring assistant."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_TRANSCRIBE_PROMPT = (
    "\u8fd9\u662f\u4e00\u6bb5\u4e2d\u6587\u8bfe\u5802\u6216\u7f51\u8bfe\u8bb2\u89e3\u97f3\u9891\uff0c"
    "\u8bf7\u8f6c\u5199\u6210\u901a\u987a\u3001\u51c6\u786e\u7684\u4e2d\u6587\u8bfe\u5802\u8bb0\u5f55\u3002"
)


def _bool_from_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _load_dotenv_if_available(root: Path) -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(root / ".env", override=False)


def is_vosk_model_dir(path: str | Path) -> bool:
    model_path = Path(path)
    return (
        model_path.is_dir()
        and (model_path / "am").is_dir()
        and (model_path / "conf").is_dir()
        and (model_path / "graph").is_dir()
    )


def _default_speech_model_path(root: Path) -> str:
    bundled = root / "models" / "vosk-model-small-cn-0.22"
    if is_vosk_model_dir(bundled):
        return str(bundled)
    return ""


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    transcript_path: Path
    index_path: Path
    chunk_max_chars: int = 800
    chunk_overlap: int = 120
    retrieve_top_k: int = 5
    min_similarity: float = 0.05
    use_llm: bool = False
    llm_provider: str = "auto"
    dashscope_model: str = "qwen-plus"
    dashscope_api_key: str = ""
    ark_api_key: str = ""
    ark_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    doubao_answer_model: str = "doubao-seed-2-0-lite-260428"
    doubao_transcribe_model: str = "doubao-seed-2-0-lite-260428"
    doubao_asr_endpoint: str = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel"
    doubao_asr_app_key: str = ""
    doubao_asr_access_key: str = ""
    doubao_asr_resource_id: str = "volc.bigasr.sauc.duration"
    doubao_asr_model_name: str = "bigmodel"
    doubao_asr_language: str = "zh-CN"
    doubao_asr_user_id: str = "teaching-assistant"
    doubao_asr_enable_nonstream: bool = True
    doubao_asr_show_utterances: bool = True
    doubao_asr_end_window_size: int = 800
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_answer_model: str = "deepseek-v4-flash"
    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_answer_model: str = "gpt-5.4-mini"
    openai_transcribe_model: str = "gpt-4o-mini-transcribe"
    openai_transcribe_prompt: str = DEFAULT_TRANSCRIBE_PROMPT
    transcription_provider: str = "auto"
    speech_model_path: str = ""
    audio_source: str = "system"
    audio_sample_rate: int = 16000
    audio_block_seconds: float = 1.0
    audio_chunk_seconds: float = 8.0
    audio_streaming: bool = True
    audio_silence_seconds: float = 0.9
    audio_max_segment_seconds: float = 12.0
    audio_min_segment_seconds: float = 1.2
    audio_transcribe_queue_size: int = 3
    audio_min_rms: float = 0.006
    keep_audio_chunks: bool = False

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "Settings":
        root = Path(project_root or Path(__file__).resolve().parent.parent)
        _load_dotenv_if_available(root)
        data_dir = Path(os.environ.get("TA_DATA_DIR", root / "data"))
        env_speech_model_path = os.environ.get("TA_SPEECH_MODEL_PATH", "").strip()
        speech_model_path = (
            env_speech_model_path
            if env_speech_model_path and is_vosk_model_dir(env_speech_model_path)
            else _default_speech_model_path(root)
        )
        return cls(
            project_root=root,
            data_dir=data_dir,
            transcript_path=data_dir / "transcripts.jsonl",
            index_path=data_dir / "vector_index.json",
            chunk_max_chars=int(os.environ.get("TA_CHUNK_MAX_CHARS", "800")),
            chunk_overlap=int(os.environ.get("TA_CHUNK_OVERLAP", "120")),
            retrieve_top_k=int(os.environ.get("TA_RETRIEVE_TOP_K", "5")),
            min_similarity=float(os.environ.get("TA_MIN_SIMILARITY", "0.05")),
            use_llm=_bool_from_env("TA_USE_LLM", False),
            llm_provider=os.environ.get("TA_LLM_PROVIDER", "auto").strip().lower(),
            dashscope_model=os.environ.get("TA_DASHSCOPE_MODEL", "qwen-plus").strip(),
            dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "").strip(),
            ark_api_key=(
                os.environ.get("ARK_API_KEY", "").strip()
                or os.environ.get("DOUBAO_API_KEY", "").strip()
            ),
            ark_base_url=os.environ.get(
                "ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"
            ).strip(),
            doubao_answer_model=os.environ.get(
                "TA_DOUBAO_ANSWER_MODEL", "doubao-seed-2-0-lite-260428"
            ).strip(),
            doubao_transcribe_model=os.environ.get(
                "TA_DOUBAO_TRANSCRIBE_MODEL", "doubao-seed-2-0-lite-260428"
            ).strip(),
            doubao_asr_endpoint=os.environ.get(
                "TA_DOUBAO_ASR_ENDPOINT",
                "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel",
            ).strip(),
            doubao_asr_app_key=(
                os.environ.get("TA_DOUBAO_ASR_APP_KEY", "").strip()
                or os.environ.get("TA_DOUBAO_ASR_APP_ID", "").strip()
            ),
            doubao_asr_access_key=(
                os.environ.get("TA_DOUBAO_ASR_ACCESS_KEY", "").strip()
                or os.environ.get("TA_DOUBAO_ASR_API_KEY", "").strip()
            ),
            doubao_asr_resource_id=os.environ.get(
                "TA_DOUBAO_ASR_RESOURCE_ID", "volc.bigasr.sauc.duration"
            ).strip(),
            doubao_asr_model_name=os.environ.get(
                "TA_DOUBAO_ASR_MODEL_NAME", "bigmodel"
            ).strip(),
            doubao_asr_language=os.environ.get("TA_DOUBAO_ASR_LANGUAGE", "zh-CN").strip(),
            doubao_asr_user_id=os.environ.get(
                "TA_DOUBAO_ASR_USER_ID", "teaching-assistant"
            ).strip(),
            doubao_asr_enable_nonstream=_bool_from_env(
                "TA_DOUBAO_ASR_ENABLE_NONSTREAM", True
            ),
            doubao_asr_show_utterances=_bool_from_env(
                "TA_DOUBAO_ASR_SHOW_UTTERANCES", True
            ),
            doubao_asr_end_window_size=int(
                os.environ.get("TA_DOUBAO_ASR_END_WINDOW_SIZE", "800")
            ),
            deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY", "").strip(),
            deepseek_base_url=os.environ.get(
                "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
            ).strip(),
            deepseek_answer_model=os.environ.get(
                "TA_DEEPSEEK_ANSWER_MODEL", "deepseek-v4-flash"
            ).strip(),
            openai_api_key=os.environ.get("OPENAI_API_KEY", "").strip(),
            openai_base_url=os.environ.get("OPENAI_BASE_URL", "").strip(),
            openai_answer_model=os.environ.get("TA_OPENAI_ANSWER_MODEL", "gpt-5.4-mini").strip(),
            openai_transcribe_model=os.environ.get(
                "TA_OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe"
            ).strip(),
            openai_transcribe_prompt=os.environ.get(
                "TA_OPENAI_TRANSCRIBE_PROMPT", DEFAULT_TRANSCRIBE_PROMPT
            ).strip(),
            transcription_provider=os.environ.get(
                "TA_TRANSCRIPTION_PROVIDER", "auto"
            ).strip().lower(),
            speech_model_path=speech_model_path,
            audio_source=os.environ.get("TA_AUDIO_SOURCE", "system").strip().lower(),
            audio_sample_rate=int(os.environ.get("TA_AUDIO_SAMPLE_RATE", "16000")),
            audio_block_seconds=float(os.environ.get("TA_AUDIO_BLOCK_SECONDS", "1.0")),
            audio_chunk_seconds=float(os.environ.get("TA_AUDIO_CHUNK_SECONDS", "8.0")),
            audio_streaming=_bool_from_env("TA_AUDIO_STREAMING", True),
            audio_silence_seconds=float(os.environ.get("TA_AUDIO_SILENCE_SECONDS", "0.9")),
            audio_max_segment_seconds=float(
                os.environ.get("TA_AUDIO_MAX_SEGMENT_SECONDS", "12.0")
            ),
            audio_min_segment_seconds=float(
                os.environ.get("TA_AUDIO_MIN_SEGMENT_SECONDS", "1.2")
            ),
            audio_transcribe_queue_size=int(
                os.environ.get("TA_AUDIO_TRANSCRIBE_QUEUE_SIZE", "3")
            ),
            audio_min_rms=float(os.environ.get("TA_AUDIO_MIN_RMS", "0.006")),
            keep_audio_chunks=_bool_from_env("TA_KEEP_AUDIO_CHUNKS", False),
        )

    @classmethod
    def for_data_dir(cls, data_dir: str | Path) -> "Settings":
        root = Path(data_dir).resolve().parent
        data_path = Path(data_dir)
        return cls(
            project_root=root,
            data_dir=data_path,
            transcript_path=data_path / "transcripts.jsonl",
            index_path=data_path / "vector_index.json",
        )
