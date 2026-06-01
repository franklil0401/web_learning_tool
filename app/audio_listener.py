# -*- coding: utf-8 -*-
"""Live audio listener for computer playback or microphone input."""

from __future__ import annotations

from collections import deque
import json
import queue
import tempfile
import threading
import time
import uuid
import warnings
import wave
from pathlib import Path
from typing import Callable

from app.config import Settings, is_vosk_model_dir
from app.transcriber import AudioTranscriber


class AudioListener:
    # ── 短句合并参数 ──
    _MERGE_MIN_CHARS = 80       # 积累到 80 字以上才作为一条入库
    _MERGE_MAX_CHARS = 500      # 超过 500 字强制入库（避免太长）
    _MERGE_FLUSH_SECONDS = 5.0 # 超过 5 秒没新句子也强制入库

    def __init__(
        self,
        on_final_text: Callable[[str], None],
        settings: Settings | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self.on_final_text = on_final_text
        self.on_status = on_status
        self.settings = settings or Settings.from_env()
        self.transcriber = AudioTranscriber(self.settings)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_error = ""
        self._last_partial_text = ""
        self._last_partial_time = 0.0

        # 短句合并缓冲
        self._merge_buffer: list[str] = []
        self._merge_chars = 0
        self._merge_last_time = 0.0
        self._merge_lock = threading.Lock()
        self._merge_timer: threading.Timer | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def last_error(self) -> str:
        return self._last_error

    def start(self) -> tuple[bool, str]:
        if self.is_running:
            return False, "实时听课已经在运行。"

        if not self.transcriber.should_use_cloud():
            if not self.settings.speech_model_path:
                return False, "未配置语音识别模型路径，也未配置 ARK_API_KEY 或 OPENAI_API_KEY。"
            if not is_vosk_model_dir(self.settings.speech_model_path):
                return False, (
                    "语音识别模型目录不正确，请选择包含 am、conf、graph 的模型目录："
                    f"{self.settings.speech_model_path}"
                )
        elif self.transcriber.should_use_realtime_stream():
            missing = self.transcriber.missing_realtime_stream_settings()
            if missing:
                return False, "缺少豆包流式 ASR 配置：" + "、".join(missing)

        self._stop_event.clear()
        self._last_error = ""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        source_names = {
            "system": "电脑播放声音",
            "microphone": "麦克风输入",
            "mixed": "混合模式（系统+麦克风）",
        }
        source_name = source_names.get(self.settings.audio_source, self.settings.audio_source)
        mode_name = (
            "WebSocket 真流式"
            if self.transcriber.should_use_realtime_stream()
            else "流式片段"
            if self.transcriber.should_use_cloud() and self.settings.audio_streaming
            else "固定片段"
            if self.transcriber.should_use_cloud()
            else "流式识别"
        )
        return True, f"实时听课已启动：{source_name} -> {self._transcriber_name()}（{mode_name}）。"

    def stop(self) -> tuple[bool, str]:
        if not self.is_running:
            return False, "实时听课没有运行。"
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._flush_merge_buffer(reason="停止听课")
        return True, "实时听课已停止。"

    def _emit_status(self, message: str) -> None:
        if self.on_status:
            self.on_status(message)

    # ── 短句合并逻辑 ──────────────────────────────────────────────────

    def _on_final_text_merged(self, text: str) -> None:
        """替代直接调 on_final_text 的入口：先做短句合并，再入库。"""
        text = text.strip()
        if not text:
            return

        with self._merge_lock:
            self._merge_buffer.append(text)
            self._merge_chars += len(text)
            self._merge_last_time = time.monotonic()

            # 取消之前的定时器，重新计时
            if self._merge_timer is not None:
                self._merge_timer.cancel()
            self._merge_timer = threading.Timer(
                self._MERGE_FLUSH_SECONDS, self._flush_merge_buffer, kwargs={"reason": "超时"}
            )
            self._merge_timer.daemon = True
            self._merge_timer.start()

            # 积累够了就立刻入库
            if self._merge_chars >= self._MERGE_MAX_CHARS:
                self._do_flush(reason="长度足够")

    def _flush_merge_buffer(self, reason: str = "") -> None:
        """定时器或外部调用时刷新缓冲区。"""
        with self._merge_lock:
            self._do_flush(reason=reason)

    def _do_flush(self, reason: str = "") -> None:
        """内部方法：必须在 _merge_lock 内调用。"""
        if self._merge_timer is not None:
            self._merge_timer.cancel()
            self._merge_timer = None
        if not self._merge_buffer:
            return

        merged = "".join(self._merge_buffer)
        self._merge_buffer = []
        self._merge_chars = 0

        # 通知入库
        self.on_final_text(merged)

    # ── end 短句合并 ──────────────────────────────────────────────────

    def _transcriber_name(self) -> str:
        provider = self.transcriber.provider
        if provider == "doubao":
            return "豆包音频理解"
        if provider == "openai":
            return "OpenAI 云端转写"
        if provider == "doubao_streaming":
            return "豆包流式 ASR"
        if provider == "vosk":
            return "本地 Vosk"
        return provider

    def _run(self) -> None:
        if self.settings.audio_source == "microphone":
            self._run_microphone_listener()
            return
        if self.settings.audio_source == "mixed":
            self._run_mixed_listener()
            return
        self._run_system_audio_listener()

    def _load_recognizer(self):
        try:
            import vosk
        except Exception as exc:
            raise RuntimeError(f"缺少语音识别依赖 vosk，请先执行 pip install vosk：{exc}") from exc

        model = vosk.Model(self.settings.speech_model_path)
        return vosk.KaldiRecognizer(model, self.settings.audio_sample_rate)

    def _accept_audio_bytes(self, recognizer, data: bytes) -> None:
        if not data:
            return
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "").strip()
            if text:
                self._on_final_text_merged(text)

    def _run_mixed_listener(self) -> None:
        """同时录制系统音频 loopback + 麦克风，混合成一路发给 ASR。

        适用于腾讯会议 + 耳机场景：耳机导致 system loopback 无法录到会议声音，
        但麦克风能收到自己的声音，loopback 能收到系统声音。混合后两端都能识别。
        """
        try:
            import soundcard as sc
            import sounddevice as sd
            import numpy as np
            try:
                from soundcard.mediafoundation import SoundcardRuntimeWarning
            except Exception:
                SoundcardRuntimeWarning = RuntimeWarning
        except Exception as exc:
            self._last_error = (
                "缺少混合模式所需的依赖。请执行：pip install soundcard sounddevice numpy。"
                f" 原始错误：{exc}"
            )
            self._emit_status(self._last_error)
            return

        try:
            speaker = sc.default_speaker()
            loopback = sc.get_microphone(id=str(speaker.name), include_loopback=True)

            block_seconds = (
                min(self.settings.audio_block_seconds, 0.2)
                if self.transcriber.should_use_realtime_stream()
                else self.settings.audio_block_seconds
            )
            block_frames = max(1600, int(self.settings.audio_sample_rate * block_seconds))
            warnings.filterwarnings(
                "ignore",
                message="data discontinuity in recording",
                category=SoundcardRuntimeWarning,
            )

            # 麦克风音频队列
            mic_queue: "queue.Queue[object]" = queue.Queue(maxsize=20)

            def mic_callback(indata, frames, time_info, status) -> None:
                if status:
                    pass  # 忽略溢出等
                try:
                    mic_queue.put_nowait(indata.copy())
                except queue.Full:
                    pass  # 丢弃旧数据

            blocksize = max(1600, int(self.settings.audio_sample_rate * 0.2))

            if self.transcriber.should_use_realtime_stream():
                self._emit_status("混合模式：同时录制系统声音 + 麦克风（WebSocket 真流式）。")
                with loopback.recorder(samplerate=self.settings.audio_sample_rate) as recorder, \
                     sd.InputStream(
                        samplerate=self.settings.audio_sample_rate,
                        blocksize=blocksize,
                        channels=1,
                        dtype="float32",
                        callback=mic_callback,
                    ):
                    self._record_realtime_stream_for_cloud(
                        read_block=lambda: self._read_mixed_block(recorder, mic_queue, block_frames),
                    )
                return

            if self.transcriber.should_use_cloud():
                if self.settings.audio_streaming:
                    self._emit_status("混合模式：同时录制系统声音 + 麦克风（流式片段）。")
                else:
                    self._emit_status("混合模式：同时录制系统声音 + 麦克风（固定片段）。")
                with loopback.recorder(samplerate=self.settings.audio_sample_rate) as recorder, \
                     sd.InputStream(
                        samplerate=self.settings.audio_sample_rate,
                        blocksize=blocksize,
                        channels=1,
                        dtype="float32",
                        callback=mic_callback,
                    ):
                    self._record_float_blocks_for_cloud(
                        read_block=lambda: self._read_mixed_block(recorder, mic_queue, block_frames),
                    )
                return

            # 本地 Vosk 模式
            recognizer = self._load_recognizer()
            with loopback.recorder(samplerate=self.settings.audio_sample_rate) as recorder, \
                 sd.InputStream(
                    samplerate=self.settings.audio_sample_rate,
                    blocksize=blocksize,
                    channels=1,
                    dtype="float32",
                    callback=mic_callback,
                ):
                while not self._stop_event.is_set():
                    audio = self._read_mixed_block(recorder, mic_queue, block_frames)
                    pcm = self._float_audio_to_pcm_bytes(audio)
                    self._accept_audio_bytes(recognizer, pcm)
        except Exception as exc:
            self._last_error = f"混合模式监听失败：{exc}"
            self._emit_status(self._last_error)

    def _read_mixed_block(self, recorder, mic_queue: "queue.Queue[object]", block_frames: int):
        """从系统 loopback 和麦克风队列各读一帧，混合成一路 mono float32。"""
        import numpy as np

        # 读系统音频（阻塞，保证时间同步）
        sys_audio = recorder.record(numframes=block_frames)
        sys_mono = self._to_mono_float_array(sys_audio)

        # 从队列取最近的麦克风数据（非阻塞，取不到就用静音）
        mic_frames: list[object] = []
        while True:
            try:
                mic_frames.append(mic_queue.get_nowait())
            except queue.Empty:
                break

        if mic_frames:
            mic_audio = np.concatenate(mic_frames, axis=0)
            mic_mono = self._to_mono_float_array(mic_audio)
            # 对齐长度：取较短的那个，或者裁剪/填充
            target_len = len(sys_mono)
            if len(mic_mono) < target_len:
                # 填充静音
                mic_mono = np.pad(mic_mono, (0, target_len - len(mic_mono)))
            elif len(mic_mono) > target_len:
                mic_mono = mic_mono[:target_len]
            # 混合：两路等权重相加，限幅
            mixed = np.clip(sys_mono + mic_mono, -1.0, 1.0)
            return mixed
        else:
            # 没有麦克风数据，直接用系统音频
            return sys_mono

    def _run_system_audio_listener(self) -> None:
        try:
            import soundcard as sc
            try:
                from soundcard.mediafoundation import SoundcardRuntimeWarning
            except Exception:
                SoundcardRuntimeWarning = RuntimeWarning
        except Exception as exc:
            self._last_error = (
                "缺少系统声音捕获依赖。请执行：pip install soundcard numpy vosk。"
                f" 原始错误：{exc}"
            )
            return

        try:
            speaker = sc.default_speaker()
            loopback = sc.get_microphone(id=str(speaker.name), include_loopback=True)
            block_seconds = (
                min(self.settings.audio_block_seconds, 0.2)
                if self.transcriber.should_use_realtime_stream()
                else self.settings.audio_block_seconds
            )
            block_frames = max(1600, int(self.settings.audio_sample_rate * block_seconds))
            warnings.filterwarnings(
                "ignore",
                message="data discontinuity in recording",
                category=SoundcardRuntimeWarning,
            )

            if self.transcriber.should_use_realtime_stream():
                self._emit_status("豆包 WebSocket 真流式识别模式：音频会边录边发送。")
                with loopback.recorder(samplerate=self.settings.audio_sample_rate) as recorder:
                    self._record_realtime_stream_for_cloud(
                        read_block=lambda: recorder.record(numframes=block_frames),
                    )
                return

            if self.transcriber.should_use_cloud():
                if self.settings.audio_streaming:
                    self._emit_status("云端流式片段模式：检测到一句话后立即后台转写。")
                else:
                    self._emit_status("云端固定片段模式：正在按音频片段送入模型。")
                with loopback.recorder(samplerate=self.settings.audio_sample_rate) as recorder:
                    self._record_float_blocks_for_cloud(
                        read_block=lambda: recorder.record(numframes=block_frames),
                    )
                return

            recognizer = self._load_recognizer()
            with loopback.recorder(samplerate=self.settings.audio_sample_rate) as recorder:
                while not self._stop_event.is_set():
                    audio = recorder.record(numframes=block_frames)
                    pcm = self._float_audio_to_pcm_bytes(audio)
                    self._accept_audio_bytes(recognizer, pcm)
        except Exception as exc:
            self._last_error = f"电脑播放声音监听失败：{exc}"
            self._emit_status(self._last_error)

    def _run_microphone_listener(self) -> None:
        try:
            import sounddevice as sd
        except Exception as exc:
            self._last_error = (
                "缺少麦克风监听依赖。请执行：pip install sounddevice vosk。"
                f" 原始错误：{exc}"
            )
            return

        try:
            if self.transcriber.should_use_realtime_stream():
                audio_queue: "queue.Queue[object]" = queue.Queue()

                def float_stream_callback(indata, frames, time_info, status) -> None:
                    if status:
                        self._emit_status(f"输入设备状态：{status}")
                    audio_queue.put(indata.copy())

                blocksize = max(1600, int(self.settings.audio_sample_rate * 0.2))
                with sd.InputStream(
                    samplerate=self.settings.audio_sample_rate,
                    blocksize=blocksize,
                    channels=1,
                    dtype="float32",
                    callback=float_stream_callback,
                ):
                    self._record_realtime_stream_for_cloud(
                        read_block=lambda: audio_queue.get(timeout=0.2),
                    )
                return

            if self.transcriber.should_use_cloud():
                audio_queue: "queue.Queue[object]" = queue.Queue()

                def float_callback(indata, frames, time_info, status) -> None:
                    if status:
                        self._emit_status(f"输入设备状态：{status}")
                    audio_queue.put(indata.copy())

                with sd.InputStream(
                    samplerate=self.settings.audio_sample_rate,
                    channels=1,
                    dtype="float32",
                    callback=float_callback,
                ):
                    self._record_float_blocks_for_cloud(
                        read_block=lambda: audio_queue.get(timeout=0.2),
                    )
                return

            audio_queue: "queue.Queue[bytes]" = queue.Queue()

            def raw_callback(indata, frames, time_info, status) -> None:
                if status:
                    self._emit_status(f"输入设备状态：{status}")
                audio_queue.put(bytes(indata))

            recognizer = self._load_recognizer()
            with sd.RawInputStream(
                samplerate=self.settings.audio_sample_rate,
                blocksize=8000,
                dtype="int16",
                channels=1,
                callback=raw_callback,
            ):
                while not self._stop_event.is_set():
                    try:
                        data = audio_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    self._accept_audio_bytes(recognizer, data)
        except Exception as exc:
            self._last_error = f"麦克风监听失败：{exc}"
            self._emit_status(self._last_error)

    def _record_realtime_stream_for_cloud(self, read_block: Callable[[], object]) -> None:
        try:
            from app.doubao_streaming_asr import DoubaoStreamingASRClient
        except Exception as exc:
            self._last_error = "缺少豆包流式 ASR 运行依赖。请执行：pip install websockets"
            self._emit_status(f"{self._last_error}。原始错误：{exc}")
            return

        client = DoubaoStreamingASRClient(
            settings=self.settings,
            on_final_text=self._on_final_text_merged,
            on_partial_text=self._on_streaming_partial_text,
            on_status=self._emit_status,
        )
        try:
            client.start()
            while not self._stop_event.is_set():
                try:
                    audio = read_block()
                except queue.Empty:
                    continue
                if audio is None or len(audio) == 0:
                    continue
                client.send_pcm(self._float_audio_to_pcm_bytes(audio))
        except Exception as exc:
            self._last_error = f"豆包流式 ASR 监听失败：{exc}"
            self._emit_status(self._last_error)
        finally:
            client.finish()

    def _on_streaming_partial_text(self, text: str) -> None:
        if not text:
            return
        import time
        now = time.monotonic()
        # 去重：相同内容不重复打印；限频：即使内容变化也至少间隔 1.5 秒
        if text == self._last_partial_text:
            return
        if now - self._last_partial_time < 1.5:
            return
        self._last_partial_text = text
        self._last_partial_time = now
        self._emit_status(f"实时识别中：{text[:60]}")

    def _record_float_blocks_for_cloud(self, read_block: Callable[[], object]) -> None:
        if self.settings.audio_streaming:
            self._record_streaming_segments_for_cloud(read_block)
            return
        self._record_fixed_chunks_for_cloud(read_block)

    def _record_fixed_chunks_for_cloud(self, read_block: Callable[[], object]) -> None:
        import numpy as np

        chunk_frames = max(
            self.settings.audio_sample_rate,
            int(self.settings.audio_sample_rate * self.settings.audio_chunk_seconds),
        )
        buffers: list[object] = []
        frame_count = 0

        while not self._stop_event.is_set():
            try:
                audio = read_block()
            except queue.Empty:
                continue
            if audio is None or len(audio) == 0:
                continue
            mono = self._to_mono_float_array(audio)
            buffers.append(mono)
            frame_count += len(mono)
            if frame_count < chunk_frames:
                continue

            chunk = np.concatenate(buffers)
            buffers = []
            frame_count = 0
            if self._rms(chunk) < self.settings.audio_min_rms:
                continue
            self._transcribe_float_chunk(chunk)

    def _record_streaming_segments_for_cloud(self, read_block: Callable[[], object]) -> None:
        import numpy as np

        sample_rate = self.settings.audio_sample_rate
        silence_limit_frames = max(1, int(sample_rate * self.settings.audio_silence_seconds))
        max_segment_frames = max(sample_rate, int(sample_rate * self.settings.audio_max_segment_seconds))
        min_segment_frames = max(1, int(sample_rate * self.settings.audio_min_segment_seconds))
        pre_roll_limit_frames = max(0, int(sample_rate * 0.25))

        segment_queue: "queue.Queue[object]" = queue.Queue(
            maxsize=max(1, self.settings.audio_transcribe_queue_size)
        )
        drain_event = threading.Event()
        worker = threading.Thread(
            target=self._cloud_transcribe_worker,
            args=(segment_queue, drain_event),
            daemon=True,
        )
        worker.start()

        pre_roll = deque()
        pre_roll_frames = 0
        segment_buffers: list[object] = []
        segment_frames = 0
        silence_frames = 0
        in_speech = False

        def append_pre_roll(mono) -> None:
            nonlocal pre_roll_frames
            if pre_roll_limit_frames <= 0:
                return
            pre_roll.append(mono)
            pre_roll_frames += len(mono)
            while pre_roll and pre_roll_frames > pre_roll_limit_frames:
                removed = pre_roll.popleft()
                pre_roll_frames -= len(removed)

        def clear_pre_roll() -> None:
            nonlocal pre_roll_frames
            pre_roll.clear()
            pre_roll_frames = 0

        def reset_segment() -> None:
            nonlocal segment_buffers, segment_frames, silence_frames, in_speech
            segment_buffers = []
            segment_frames = 0
            silence_frames = 0
            in_speech = False

        def submit_segment(reason: str) -> None:
            nonlocal segment_buffers
            if not segment_buffers:
                reset_segment()
                return

            chunk = np.concatenate(segment_buffers)
            reset_segment()
            if len(chunk) < min_segment_frames:
                return
            if self._rms(chunk) < self.settings.audio_min_rms * 0.45:
                return
            self._enqueue_cloud_segment(segment_queue, chunk, reason)

        try:
            while not self._stop_event.is_set():
                try:
                    audio = read_block()
                except queue.Empty:
                    continue
                if audio is None or len(audio) == 0:
                    continue

                mono = self._to_mono_float_array(audio)
                block_is_speech = self._rms(mono) >= self.settings.audio_min_rms

                if block_is_speech:
                    if not in_speech:
                        segment_buffers = list(pre_roll)
                        segment_frames = sum(len(item) for item in segment_buffers)
                        clear_pre_roll()
                        in_speech = True
                    segment_buffers.append(mono)
                    segment_frames += len(mono)
                    silence_frames = 0
                elif in_speech:
                    segment_buffers.append(mono)
                    segment_frames += len(mono)
                    silence_frames += len(mono)
                else:
                    append_pre_roll(mono)

                if not in_speech:
                    continue
                if silence_frames >= silence_limit_frames:
                    submit_segment("静音结束")
                elif segment_frames >= max_segment_frames:
                    submit_segment("达到最长片段")

            if in_speech:
                submit_segment("停止监听前最后一段")
        finally:
            drain_event.set()
            self._stop_cloud_transcribe_worker(segment_queue, worker)

    def _enqueue_cloud_segment(
        self,
        segment_queue: "queue.Queue[object]",
        audio,
        reason: str,
    ) -> None:
        duration = len(audio) / max(1, self.settings.audio_sample_rate)
        try:
            segment_queue.put_nowait(audio)
        except queue.Full:
            self._emit_status(f"转写队列已满，跳过 {duration:.1f}s 课堂片段。")
            return
        self._emit_status(f"捕获 {duration:.1f}s 课堂片段（{reason}），后台转写中。")

    def _cloud_transcribe_worker(
        self,
        segment_queue: "queue.Queue[object]",
        drain_event: threading.Event,
    ) -> None:
        while True:
            try:
                audio = segment_queue.get(timeout=0.2)
            except queue.Empty:
                if drain_event.is_set():
                    return
                continue

            try:
                if audio is None:
                    return
                self._transcribe_float_chunk(audio)
            finally:
                segment_queue.task_done()

    def _stop_cloud_transcribe_worker(
        self,
        segment_queue: "queue.Queue[object]",
        worker: threading.Thread,
    ) -> None:
        deadline = time.time() + 3
        while True:
            try:
                segment_queue.put(None, timeout=0.2)
                break
            except queue.Full:
                try:
                    segment_queue.get_nowait()
                    segment_queue.task_done()
                    self._emit_status("停止监听时转写队列仍较忙，已跳过一个待转写片段。")
                except queue.Empty:
                    pass
                if not worker.is_alive() or time.time() > deadline:
                    self._emit_status("转写线程仍在处理最后的音频片段，已放到后台结束。")
                    break
        worker.join(timeout=10)

    def _transcribe_float_chunk(self, audio) -> None:
        wav_path = self._write_wav_file(audio)
        try:
            started_at = time.time()
            result = self.transcriber.transcribe_wav(wav_path)
            elapsed = time.time() - started_at
            if result.text:
                self._emit_status(f"转写完成（{elapsed:.1f}s）：{result.text[:60]}")
                self._on_final_text_merged(result.text)
        except Exception as exc:
            self._last_error = f"云端音频转写失败：{exc}"
            self._emit_status(self._last_error)
        finally:
            if not self.settings.keep_audio_chunks:
                try:
                    wav_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def _write_wav_file(self, audio) -> Path:
        pcm = self._float_audio_to_pcm_bytes(audio)
        target_dir = self.settings.data_dir / "audio_chunks"
        target_dir.mkdir(parents=True, exist_ok=True)
        if self.settings.keep_audio_chunks:
            wav_path = target_dir / f"chunk-{uuid.uuid4().hex}.wav"
        else:
            handle = tempfile.NamedTemporaryFile(
                suffix=".wav",
                dir=target_dir,
                delete=False,
            )
            wav_path = Path(handle.name)
            handle.close()

        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.settings.audio_sample_rate)
            wav.writeframes(pcm)
        return wav_path

    def _float_audio_to_pcm_bytes(self, audio) -> bytes:
        import numpy as np

        mono = self._to_mono_float_array(audio)
        return (np.clip(mono, -1.0, 1.0) * 32767).astype("<i2").tobytes()

    def _to_mono_float_array(self, audio):
        import numpy as np

        array = np.asarray(audio, dtype="float32")
        if getattr(array, "ndim", 1) > 1:
            array = array.mean(axis=1)
        return array

    def _rms(self, audio) -> float:
        import numpy as np

        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio))))
