# -*- coding: utf-8 -*-

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from app.audio_listener import AudioListener
from app.config import Settings, is_vosk_model_dir
from app.doubao_streaming_asr import (
    COMPRESSION_GZIP,
    MESSAGE_TYPE_AUDIO_ONLY_REQUEST,
    MESSAGE_TYPE_FULL_SERVER_RESPONSE,
    NEG_SEQUENCE,
    SERIALIZATION_JSON,
    build_audio_only_request,
    build_packet,
    parse_response,
)
from app.main import TutorConsole
from app.question_handler import StudentQuestionHandler
from app.transcriber import AudioTranscriber
from rag.chunking import chunk_text
from rag.pipeline import CourseRAGPipeline


class PipelineTestCase(unittest.TestCase):
    def _settings(self, data_dir: Path) -> Settings:
        return Settings.for_data_dir(data_dir)

    def test_chunk_text_splits_long_content(self) -> None:
        text = "第一句话。" * 100
        chunks = chunk_text(text, max_chars=60, overlap=10)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 70 for chunk in chunks))

    def test_add_retrieve_answer_and_clear(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = self._settings(Path(temp_dir))
            pipeline = CourseRAGPipeline(settings)
            handler = StudentQuestionHandler(pipeline=pipeline, settings=settings)

            pipeline.add_teacher_text(
                "牛顿第二定律说明，物体的加速度与合外力成正比，与质量成反比。"
            )
            pipeline.add_teacher_text(
                "课堂上还提到，光合作用主要发生在植物叶绿体中。"
            )

            references = pipeline.retrieve("力和加速度是什么关系？")
            self.assertTrue(references)
            self.assertIn("加速度", references[0]["text"])

            result = handler.ask("力和加速度是什么关系？")
            self.assertTrue(result.references)
            self.assertIn("加速度", result.answer)

            pipeline.clear_history()
            self.assertEqual(pipeline.retrieve("力和加速度是什么关系？"), [])

    def test_audio_listener_requires_model_or_cloud_transcriber(self) -> None:
        listener = AudioListener(lambda text: None, Settings.for_data_dir("unused"))
        ok, message = listener.start()
        self.assertFalse(ok)
        self.assertIn("OPENAI_API_KEY", message)

    def test_vosk_model_dir_validator_rejects_plain_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertFalse(is_vosk_model_dir(temp_dir))

    def test_transcriber_auto_uses_vosk_without_openai_key(self) -> None:
        settings = Settings.for_data_dir("unused")
        transcriber = AudioTranscriber(settings)
        self.assertEqual(transcriber.provider, "vosk")
        self.assertFalse(transcriber.should_use_cloud())

    def test_auto_answer_provider_prefers_deepseek_when_configured(self) -> None:
        settings = Settings.for_data_dir("unused")
        settings = settings.__class__(**{**settings.__dict__, "deepseek_api_key": "test"})
        handler = StudentQuestionHandler(
            pipeline=CourseRAGPipeline(settings),
            settings=settings,
        )
        self.assertEqual(handler.answer_generator._resolve_provider(), "deepseek")

    def test_auto_provider_prefers_doubao_when_ark_key_configured(self) -> None:
        settings = Settings.for_data_dir("unused")
        settings = settings.__class__(**{**settings.__dict__, "ark_api_key": "test"})
        transcriber = AudioTranscriber(settings)
        handler = StudentQuestionHandler(
            pipeline=CourseRAGPipeline(settings),
            settings=settings,
        )
        self.assertEqual(transcriber.provider, "doubao")
        self.assertTrue(transcriber.should_use_cloud())
        self.assertEqual(handler.answer_generator._resolve_provider(), "doubao")

    def test_auto_transcriber_prefers_doubao_streaming_when_configured(self) -> None:
        settings = Settings.for_data_dir("unused")
        settings = settings.__class__(
            **{
                **settings.__dict__,
                "doubao_asr_app_key": "app-key",
                "doubao_asr_access_key": "access-key",
                "doubao_asr_resource_id": "resource-id",
            }
        )
        transcriber = AudioTranscriber(settings)
        self.assertEqual(transcriber.provider, "doubao_streaming")
        self.assertTrue(transcriber.should_use_realtime_stream())

    def test_doubao_streaming_audio_packet_marks_final_request(self) -> None:
        packet = build_audio_only_request(b"abc", is_last=True)
        self.assertEqual(packet[1] >> 4, MESSAGE_TYPE_AUDIO_ONLY_REQUEST)
        self.assertEqual(packet[1] & 0x0F, NEG_SEQUENCE)

    def test_doubao_streaming_response_parser_reads_json_payload(self) -> None:
        packet = build_packet(
            message_type=MESSAGE_TYPE_FULL_SERVER_RESPONSE,
            flags=0,
            payload=b'{"result":{"text":"hello","definite":true}}',
            serialization=SERIALIZATION_JSON,
            compression=COMPRESSION_GZIP,
        )
        response = parse_response(packet)
        self.assertEqual(response.payload["result"]["text"], "hello")

    def test_cloud_audio_streaming_is_enabled_by_default(self) -> None:
        settings = Settings.for_data_dir("unused")
        settings = settings.__class__(**{**settings.__dict__, "ark_api_key": "test"})
        self.assertTrue(settings.audio_streaming)
        self.assertEqual(settings.audio_silence_seconds, 0.9)

    def test_console_can_toggle_streaming_mode(self) -> None:
        console = TutorConsole()
        console.settings = Settings.for_data_dir("unused")
        console.audio_listener = console._build_audio_listener()

        with patch("builtins.print"):
            console._set_streaming("off")
            self.assertFalse(console.settings.audio_streaming)

            console._set_streaming("on")
            self.assertTrue(console.settings.audio_streaming)


if __name__ == "__main__":
    unittest.main()
