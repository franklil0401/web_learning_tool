# -*- coding: utf-8 -*-
"""Student-friendly live classroom console."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.audio_listener import AudioListener
from app.config import Settings
from app.question_handler import StudentQuestionHandler
from rag.pipeline import CourseRAGPipeline


class TutorConsole:
    def __init__(self) -> None:
        self.settings = Settings.from_env()
        self.pipeline = CourseRAGPipeline(self.settings)
        self.question_handler = StudentQuestionHandler(
            pipeline=self.pipeline,
            settings=self.settings,
        )
        self.audio_listener = self._build_audio_listener()

    def run(self) -> None:
        self._print_welcome()
        while True:
            try:
                line = input("\n你问 > ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                self._exit()
                break

            if not line:
                continue
            if line.startswith("/"):
                if self._handle_command(line):
                    break
                continue
            self._ask_question(line)

    def _print_welcome(self) -> None:
        print("学生听课实时问答助手")
        print("=" * 34)
        print("使用方式：后台听课，前台直接输入问题。")
        print("常用命令：/start 开始听课，/stop 停止，/note 手动记一段，/recent 最近记录，/clear 清空，/exit 退出。")
        self._print_status()

    def _print_status(self) -> None:
        running = "运行中" if self.audio_listener.is_running else "未启动"
        source = "电脑播放声音" if self.settings.audio_source == "system" else "麦克风输入"
        provider = self.audio_listener.transcriber.provider
        if provider == "doubao":
            transcriber = "豆包音频理解"
        elif provider == "doubao_streaming":
            transcriber = "豆包流式 ASR"
        elif provider == "openai":
            transcriber = "OpenAI 云端转写"
        else:
            transcriber = "本地 Vosk"
        if self.audio_listener.transcriber.should_use_realtime_stream():
            mode = "WebSocket 真流式"
        elif self.audio_listener.transcriber.should_use_cloud():
            mode = "流式片段" if self.settings.audio_streaming else "固定片段"
        else:
            mode = "流式识别"
        answer = self.settings.llm_provider
        if answer == "auto":
            if self.settings.ark_api_key:
                answer = "豆包"
            elif self.settings.openai_api_key:
                answer = "OpenAI"
            elif self.settings.deepseek_api_key:
                answer = "DeepSeek"
            else:
                answer = "本地兜底"
        else:
            answer = {
                "doubao": "豆包",
                "openai": "OpenAI",
                "deepseek": "DeepSeek",
                "dashscope": "通义千问",
                "local": "本地兜底",
            }.get(answer, answer)
        print(f"状态：听课 {running} | 来源 {source} | 转写 {transcriber} | 模式 {mode} | 回答 {answer}")

    def _build_audio_listener(self) -> AudioListener:
        return AudioListener(
            on_final_text=self._on_teacher_audio_text,
            settings=self.settings,
            on_status=self._on_listener_status,
        )

    def _handle_command(self, line: str) -> bool:
        command, _, arg = line.partition(" ")
        command = command.lower()
        arg = arg.strip()

        if command in {"/start", "/s"}:
            self._start_listening()
        elif command in {"/stop", "/pause"}:
            self._stop_listening()
        elif command in {"/note", "/n"}:
            self._add_teacher_text(arg)
        elif command in {"/recent", "/r"}:
            self._show_recent_records()
        elif command == "/clear":
            self._clear_history()
        elif command == "/status":
            self._print_status()
        elif command == "/source":
            self._set_source(arg)
        elif command == "/stream":
            self._set_streaming(arg)
        elif command == "/help":
            self._print_help()
        elif command in {"/exit", "/quit", "/q"}:
            self._exit()
            return True
        else:
            print("未知命令。输入 /help 查看可用命令。")
        return False

    def _print_help(self) -> None:
        print("命令：")
        print("/start              开始监听 Bilibili/电脑声音或麦克风")
        print("/stop               停止监听")
        print("/note 课堂内容       手动加入一段老师讲课内容")
        print("/recent             查看最近课堂记录")
        print("/source system      使用电脑播放声音")
        print("/source microphone  使用麦克风输入")
        print("/source mixed       混合模式（同时录系统+麦克风，适合腾讯会议）")
        print("/stream on          云端转写使用流式片段模式")
        print("/stream off         云端转写退回固定片段模式")
        print("/clear              清空本节课历史")
        print("/status             查看当前配置")
        print("/exit               退出")
        print("没有斜杠的输入都会当作学生问题。")

    def _start_listening(self) -> None:
        ok, message = self.audio_listener.start()
        print(message)
        if not ok:
            self._print_realtime_help()

    def _stop_listening(self) -> None:
        _ok, message = self.audio_listener.stop()
        print(message)
        if self.audio_listener.last_error:
            print(self.audio_listener.last_error)

    def _add_teacher_text(self, text: str) -> None:
        if not text:
            print("用法：/note 老师刚才讲的内容")
            return
        record = self.pipeline.add_teacher_text(text)
        if record is None:
            print("未添加内容：输入为空。")
            return
        print("已加入课堂记录。")

    def _ask_question(self, question: str) -> None:
        print("正在检索并生成回答...", end="", flush=True)
        result = self.question_handler.ask(question)
        print("\r" + " " * 20 + "\r", end="", flush=True)  # 清除提示
        if result.warning:
            print(result.warning)
        print("\n答：")
        print(result.answer)
        if result.references:
            print("\n参考课堂片段：")
            for index, item in enumerate(result.references, start=1):
                print(f"{index}. {item['text']}（相关度 {item['score']:.2f}）")

    def _show_recent_records(self) -> None:
        records = self.pipeline.recent_records(limit=8)
        if not records:
            print("暂无课堂记录。")
            return
        print("最近课堂记录：")
        for index, record in enumerate(records, start=1):
            source = "老师" if record.source == "teacher" else "学生"
            print(f"{index}. [{source}] {record.text}")

    def _clear_history(self) -> None:
        confirm = input("确定清空本节课历史吗？输入 y 确认：").strip().lower()
        if confirm != "y":
            print("已取消清空。")
            return
        self.pipeline.clear_history()
        print("本节课历史已清空。")

    def _set_source(self, source: str) -> None:
        labels = {"system": "电脑播放声音", "microphone": "麦克风输入", "mixed": "混合模式"}
        if source not in labels:
            print("用法：/source system 或 /source microphone 或 /source mixed")
            return
        if self.audio_listener.is_running:
            self.audio_listener.stop()
        self.settings = replace(self.settings, audio_source=source)
        self.audio_listener = self._build_audio_listener()
        print("已切换为：" + labels[source])

    def _set_streaming(self, value: str) -> None:
        if value not in {"on", "off"}:
            print("用法：/stream on 或 /stream off")
            return
        was_running = self.audio_listener.is_running
        if was_running:
            self.audio_listener.stop()
        enabled = value == "on"
        self.settings = replace(self.settings, audio_streaming=enabled)
        self.audio_listener = self._build_audio_listener()
        print("已切换为：" + ("流式片段模式" if enabled else "固定片段模式"))
        if was_running:
            self._start_listening()

    def _print_realtime_help(self) -> None:
        print("实时听课配置提示：")
        print("1. 推荐设置 ARK_API_KEY 或 OPENAI_API_KEY，这样会用云端模型做音频转写和回答。")
        print("2. 没有云端 Key 时，会退回本地 Vosk，准确率较弱。")
        print("3. 默认监听电脑播放声音，适合 Bilibili；也可用 /source microphone 切到麦克风。")

    def _exit(self) -> None:
        if self.audio_listener.is_running:
            self.audio_listener.stop()
        print("已退出。")

    def _on_teacher_audio_text(self, text: str) -> None:
        record = self.pipeline.add_teacher_text(text)
        if record is not None:
            print(f"\n[课堂已记录] {record.text}")

    def _on_listener_status(self, message: str) -> None:
        print(f"\n[听课状态] {message}")


def main() -> None:
    TutorConsole().run()


if __name__ == "__main__":
    main()
