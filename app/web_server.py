# -*- coding: utf-8 -*-
"""Flask web server providing REST API + SSE for the tutoring assistant."""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from dataclasses import replace
from typing import Generator

from flask import Flask, Response, jsonify, request, send_from_directory

from app.audio_listener import AudioListener
from app.config import Settings
from app.question_handler import StudentQuestionHandler
from rag.pipeline import CourseRAGPipeline


# ── SSE event bus ──────────────────────────────────────────────────────────

class SSEEventBus:
    """Fan-out SSE event bus: one producer, multiple SSE consumers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._consumers: dict[str, queue.Queue] = {}

    def publish(self, event_type: str, data: dict) -> None:
        payload = json.dumps({"type": event_type, **data}, ensure_ascii=False)
        with self._lock:
            for q in self._consumers.values():
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    pass  # drop if client is slow

    def subscribe(self) -> tuple[str, queue.Queue]:
        q: queue.Queue[str] = queue.Queue(maxsize=200)
        cid = uuid.uuid4().hex[:8]
        with self._lock:
            self._consumers[cid] = q
        return cid, q

    def unsubscribe(self, cid: str) -> None:
        with self._lock:
            self._consumers.pop(cid, None)


# ── App factory ────────────────────────────────────────────────────────────

def create_app(settings: Settings | None = None) -> Flask:
    settings = settings or Settings.from_env()
    # static folder is <project_root>/static/
    static_dir = str(settings.project_root / "static")
    app = Flask(__name__, static_folder=static_dir, static_url_path="/static")

    # ensure static dir exists
    import os
    os.makedirs(static_dir, exist_ok=True)

    pipeline = CourseRAGPipeline(settings)
    bus = SSEEventBus()

    # ── callbacks ──────────────────────────────────────────────────────
    def on_teacher_text(text: str) -> None:
        record = pipeline.add_teacher_text(text)
        if record is not None:
            bus.publish("record", {"text": record.text, "source": "teacher"})

    def on_status(message: str) -> None:
        bus.publish("status", {"message": message})

    # ── shared state ───────────────────────────────────────────────────
    state = {
        "listener": AudioListener(
            on_final_text=on_teacher_text,
            settings=settings,
            on_status=on_status,
        ),
        "settings": settings,
    }

    def _rebuild_listener() -> None:
        state["listener"] = AudioListener(
            on_final_text=on_teacher_text,
            settings=state["settings"],
            on_status=on_status,
        )

    # ── routes ──────────────────────────────────────────────────────────

    @app.get("/")
    def index():
        from pathlib import Path as P
        html_path = P(static_dir) / "index.html"
        return html_path.read_text(encoding="utf-8")

    @app.post("/api/start")
    def api_start():
        listener: AudioListener = state["listener"]
        ok, message = listener.start()
        return jsonify({"ok": ok, "message": message})

    @app.post("/api/stop")
    def api_stop():
        listener: AudioListener = state["listener"]
        ok, message = listener.stop()
        error = listener.last_error or ""
        return jsonify({"ok": ok, "message": message, "error": error})

    @app.post("/api/note")
    def api_note():
        body = request.get_json(silent=True) or {}
        text = body.get("text", "").strip()
        if not text:
            return jsonify({"ok": False, "message": "内容不能为空"}), 400
        record = pipeline.add_teacher_text(text)
        if record is None:
            return jsonify({"ok": False, "message": "添加失败"}), 400
        return jsonify({"ok": True, "message": "已加入课堂记录", "text": record.text})

    @app.get("/api/recent")
    def api_recent():
        limit = request.args.get("limit", 20, type=int)
        records = pipeline.recent_records(limit=limit)
        items = [
            {"source": r.source, "text": r.text, "created_at": r.created_at}
            for r in records
        ]
        return jsonify({"records": items})

    @app.post("/api/clear")
    def api_clear():
        pipeline.clear_history()
        bus.publish("status", {"message": "课堂历史已清空"})
        return jsonify({"ok": True, "message": "本节课历史已清空"})

    @app.post("/api/ask")
    def api_ask():
        body = request.get_json(silent=True) or {}
        question = body.get("question", "").strip()
        if not question:
            return jsonify({"ok": False, "message": "问题不能为空"}), 400

        handler = StudentQuestionHandler(pipeline=pipeline, settings=state["settings"])
        result = handler.ask(question)
        refs = []
        if result.references:
            for item in result.references:
                refs.append({"text": item["text"], "score": round(item["score"], 3)})
        return jsonify({
            "ok": True,
            "answer": result.answer,
            "references": refs,
            "warning": result.warning,
        })

    @app.get("/api/status")
    def api_status():
        listener: AudioListener = state["listener"]
        s = state["settings"]
        provider = listener.transcriber.provider
        provider_name = {
            "doubao": "豆包音频理解",
            "doubao_streaming": "豆包流式 ASR",
            "openai": "OpenAI 云端转写",
            "vosk": "本地 Vosk",
        }.get(provider, provider)
        answer_name = {
            "doubao": "豆包", "openai": "OpenAI",
            "deepseek": "DeepSeek", "dashscope": "通义千问",
        }.get(s.llm_provider, s.llm_provider)
        source_labels = {"system": "电脑播放声音", "microphone": "麦克风输入", "mixed": "混合模式"}
        return jsonify({
            "listening": listener.is_running,
            "audio_source": source_labels.get(s.audio_source, s.audio_source),
            "transcriber": provider_name,
            "answer_provider": answer_name,
        })

    @app.post("/api/source")
    def api_source():
        body = request.get_json(silent=True) or {}
        source = body.get("source", "").strip()
        if source not in {"system", "microphone", "mixed"}:
            return jsonify({"ok": False, "message": "参数需为 system、microphone 或 mixed"}), 400
        if state["listener"].is_running:
            state["listener"].stop()
        state["settings"] = replace(state["settings"], audio_source=source)
        _rebuild_listener()
        labels = {"system": "电脑播放声音", "microphone": "麦克风输入", "mixed": "混合模式"}
        return jsonify({"ok": True, "message": f"已切换为：{labels[source]}"})

    @app.get("/api/stream")
    def api_stream():
        cid, q = bus.subscribe()

        def generate() -> Generator[str, None, None]:
            try:
                while True:
                    try:
                        payload = q.get(timeout=30)
                        yield f"data: {payload}\n\n"
                    except queue.Empty:
                        # heartbeat
                        yield f": keepalive\n\n"
            finally:
                bus.unsubscribe(cid)

        return Response(generate(), mimetype="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })

    return app
