# -*- coding: utf-8 -*-
"""Doubao big-model streaming ASR client.

The Volcengine endpoint uses a compact binary WebSocket protocol. This module
keeps the protocol framing isolated from the audio capture code.
"""

from __future__ import annotations

import gzip
import json
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from app.config import Settings

# 抑制 websockets 内部 keepalive 断连时的 traceback（主动关闭时属于正常行为）
logging.getLogger("websockets").setLevel(logging.CRITICAL)


PROTOCOL_VERSION = 0x1
HEADER_SIZE_WORDS = 0x1

MESSAGE_TYPE_FULL_CLIENT_REQUEST = 0x1
MESSAGE_TYPE_AUDIO_ONLY_REQUEST = 0x2
MESSAGE_TYPE_FULL_SERVER_RESPONSE = 0x9
MESSAGE_TYPE_SERVER_ERROR = 0xF

NO_SEQUENCE = 0x0
POS_SEQUENCE = 0x1
NEG_SEQUENCE = 0x2

SERIALIZATION_NONE = 0x0
SERIALIZATION_JSON = 0x1

COMPRESSION_NONE = 0x0
COMPRESSION_GZIP = 0x1


@dataclass(frozen=True)
class AsrResponse:
    payload: dict[str, Any]
    message_type: int
    sequence: int | None = None


def build_full_client_request(settings: Settings) -> bytes:
    payload = {
        "user": {"uid": settings.doubao_asr_user_id},
        "audio": {
            "format": "pcm",
            "codec": "raw",
            "rate": settings.audio_sample_rate,
            "bits": 16,
            "channel": 1,
            "language": settings.doubao_asr_language,
        },
        "request": {
            "model_name": settings.doubao_asr_model_name,
            "enable_punc": True,
            "enable_itn": True,
            "enable_ddc": False,
            "enable_nonstream": settings.doubao_asr_enable_nonstream,
            "show_utterances": settings.doubao_asr_show_utterances,
            "result_type": "single",
            "vad_segment": True,
            "end_window_size": settings.doubao_asr_end_window_size,
        },
    }
    return build_packet(
        message_type=MESSAGE_TYPE_FULL_CLIENT_REQUEST,
        flags=NO_SEQUENCE,
        payload=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        serialization=SERIALIZATION_JSON,
        compression=COMPRESSION_GZIP,
    )


def build_audio_only_request(pcm: bytes, is_last: bool = False) -> bytes:
    return build_packet(
        message_type=MESSAGE_TYPE_AUDIO_ONLY_REQUEST,
        flags=NEG_SEQUENCE if is_last else NO_SEQUENCE,
        payload=pcm,
        serialization=SERIALIZATION_NONE,
        compression=COMPRESSION_GZIP,
    )


def build_packet(
    message_type: int,
    flags: int,
    payload: bytes,
    serialization: int,
    compression: int,
) -> bytes:
    if compression == COMPRESSION_GZIP:
        payload = gzip.compress(payload)
    header = bytes(
        [
            (PROTOCOL_VERSION << 4) | HEADER_SIZE_WORDS,
            (message_type << 4) | flags,
            (serialization << 4) | compression,
            0x00,
        ]
    )
    return header + len(payload).to_bytes(4, "big", signed=False) + payload


def parse_response(frame: bytes | str) -> AsrResponse:
    if isinstance(frame, str):
        text = frame.strip()
        if not text:
            return AsrResponse(payload={}, message_type=MESSAGE_TYPE_FULL_SERVER_RESPONSE)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = {"raw": text}
        return AsrResponse(payload=payload, message_type=MESSAGE_TYPE_FULL_SERVER_RESPONSE)
    if len(frame) < 8:
        raise RuntimeError("豆包 ASR 返回包过短。")

    header_size = (frame[0] & 0x0F) * 4
    message_type = frame[1] >> 4
    flags = frame[1] & 0x0F
    serialization = frame[2] >> 4
    compression = frame[2] & 0x0F
    offset = header_size
    sequence: int | None = None

    if message_type == MESSAGE_TYPE_SERVER_ERROR:
        if len(frame) < offset + 8:
            raise RuntimeError("豆包 ASR 错误包格式不完整。")
        code = int.from_bytes(frame[offset : offset + 4], "big", signed=False)
        offset += 4
        payload_size = int.from_bytes(frame[offset : offset + 4], "big", signed=False)
        offset += 4
        payload = frame[offset : offset + payload_size]
        text = payload.decode("utf-8", errors="replace")
        raise RuntimeError(f"豆包 ASR 服务返回错误 {code}：{text}")

    # 根据 flags 判断是否有 sequence number
    # flags: 0x0 = NO_SEQUENCE, 0x1 = POS_SEQUENCE, 0x2 = NEG_SEQUENCE
    has_sequence = flags in {POS_SEQUENCE, NEG_SEQUENCE}
    if has_sequence:
        if len(frame) < offset + 8:
            raise RuntimeError("豆包 ASR 返回包缺少 sequence/payload size。")
        sequence = int.from_bytes(frame[offset : offset + 4], "big", signed=True)
        offset += 4
        payload_size = int.from_bytes(frame[offset : offset + 4], "big", signed=False)
        offset += 4
    else:
        if len(frame) < offset + 4:
            raise RuntimeError("豆包 ASR 返回包缺少 payload size。")
        payload_size = int.from_bytes(frame[offset : offset + 4], "big", signed=False)
        offset += 4

    payload = frame[offset : offset + payload_size]
    if compression == COMPRESSION_GZIP:
        payload = gzip.decompress(payload)
    if serialization == SERIALIZATION_JSON or payload[:1] in {b"{", b"["}:
        return AsrResponse(
            payload=json.loads(payload.decode("utf-8")),
            message_type=message_type,
            sequence=sequence,
        )
    return AsrResponse(
        payload={"raw": payload.decode("utf-8", errors="replace")},
        message_type=message_type,
        sequence=sequence,
    )


class DoubaoStreamingASRClient:
    def __init__(
        self,
        settings: Settings,
        on_final_text: Callable[[str], None],
        on_partial_text: Callable[[str], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self.settings = settings
        self.on_final_text = on_final_text
        self.on_partial_text = on_partial_text
        self.on_status = on_status
        self._connection = None
        self._receive_thread: threading.Thread | None = None
        self._closed = threading.Event()
        self._send_lock = threading.Lock()
        self._seen_finals: set[tuple[str, int | None, int | None]] = set()
        self._last_partial: str = ""
        self._errors: "queue.Queue[Exception]" = queue.Queue()

    def start(self) -> None:
        missing = self._missing_settings()
        if missing:
            raise RuntimeError("缺少豆包流式 ASR 配置：" + "、".join(missing))

        try:
            from websockets.sync.client import connect
        except Exception as exc:
            raise RuntimeError("缺少 websockets 依赖，请先执行：pip install websockets") from exc

        connect_id = str(uuid.uuid4())
        headers = {
            "X-Api-App-Key": self.settings.doubao_asr_app_key,
            "X-Api-Access-Key": self.settings.doubao_asr_access_key,
            "X-Api-Resource-Id": self.settings.doubao_asr_resource_id,
            "X-Api-Connect-Id": connect_id,
            "X-Api-Request-Id": connect_id,
            "X-Api-Sequence": "-1",
        }
        self._connection = connect(
            self.settings.doubao_asr_endpoint,
            additional_headers=headers,
            open_timeout=10,
            ping_interval=20,
            ping_timeout=10,
            max_size=None,
            proxy=None,
        )
        self._connection.send(build_full_client_request(self.settings))
        self._receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._receive_thread.start()
        self._emit_status("豆包流式 ASR 已连接。")

    def send_pcm(self, pcm: bytes) -> None:
        if not pcm:
            return
        self._raise_pending_error()
        self._send_packet(build_audio_only_request(pcm, is_last=False))

    def finish(self) -> None:
        if self._connection is None:
            return
        try:
            self._send_packet(build_audio_only_request(b"", is_last=True))
            time.sleep(0.8)
        except Exception as exc:
            self._emit_status(f"发送结束包失败：{exc}")
        finally:
            self.close()

    def close(self) -> None:
        self._closed.set()
        connection = self._connection
        self._connection = None
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
        if self._receive_thread is not None:
            self._receive_thread.join(timeout=2)

    def _missing_settings(self) -> list[str]:
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

    def _send_packet(self, packet: bytes) -> None:
        if self._connection is None:
            raise RuntimeError("豆包流式 ASR 尚未连接。")
        with self._send_lock:
            self._connection.send(packet)

    def _receive_loop(self) -> None:
        while not self._closed.is_set():
            try:
                if self._connection is None:
                    return
                frame = self._connection.recv(timeout=0.2)
            except TimeoutError:
                continue
            except Exception as exc:
                if not self._closed.is_set():
                    self._errors.put(exc)
                    self._emit_status(f"豆包流式 ASR 接收失败：{exc}")
                return

            try:
                response = parse_response(frame)
                self._handle_response(response.payload)
            except Exception as exc:
                # 记录原始帧内容方便调试
                try:
                    if isinstance(frame, bytes):
                        debug_info = f" (二进制帧, {len(frame)} 字节, 首字节: 0x{frame[0]:02X})"
                    else:
                        debug_info = f" (文本帧: {frame[:200]!r})"
                except Exception:
                    debug_info = ""
                self._errors.put(exc)
                self._emit_status(f"豆包流式 ASR 解析失败：{exc}{debug_info}")

    def _handle_response(self, payload: dict[str, Any]) -> None:
        if not payload:
            return
        code = payload.get("code")
        if code not in {None, 0, 1000}:
            message = payload.get("message") or payload.get("msg") or str(payload)
            raise RuntimeError(f"豆包 ASR 返回错误 {code}：{message}")

        result = payload.get("result") or payload.get("payload") or payload
        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict):
                    self._handle_result_item(item)
            return
        if isinstance(result, dict):
            self._handle_result_item(result)

    def _handle_result_item(self, result: dict[str, Any]) -> None:
        utterances = result.get("utterances") or []
        emitted_final = False
        if isinstance(utterances, list):
            for utterance in utterances:
                if not isinstance(utterance, dict):
                    continue
                text = self._clean_text(utterance.get("text"))
                if not text:
                    continue
                is_final = bool(
                    utterance.get("definite")
                    or utterance.get("is_final")
                    or utterance.get("final")
                )
                if is_final:
                    self._emit_final(
                        text,
                        utterance.get("start_time") or utterance.get("start"),
                        utterance.get("end_time") or utterance.get("end"),
                    )
                    emitted_final = True
                elif self.on_partial_text and text != self._last_partial:
                    self._last_partial = text
                    self.on_partial_text(text)

        text = self._clean_text(result.get("text"))
        if not text:
            return
        is_final_result = bool(
            result.get("definite")
            or result.get("is_final")
            or result.get("final")
            or result.get("last")
        )
        if is_final_result and not emitted_final:
            self._emit_final(text, result.get("start_time"), result.get("end_time"))
        elif self.on_partial_text and text != self._last_partial:
            self._last_partial = text
            self.on_partial_text(text)

    def _emit_final(self, text: str, start_time: int | None, end_time: int | None) -> None:
        key = (text, start_time, end_time)
        if key in self._seen_finals:
            return
        self._seen_finals.add(key)
        self.on_final_text(text)

    def _clean_text(self, value: Any) -> str:
        if not isinstance(value, str):
            return ""
        return value.strip()

    def _emit_status(self, message: str) -> None:
        if self.on_status:
            self.on_status(message)

    def _raise_pending_error(self) -> None:
        try:
            error = self._errors.get_nowait()
        except queue.Empty:
            return
        raise error
