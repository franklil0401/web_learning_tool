# -*- coding: utf-8 -*-
"""Launch the web UI for the tutoring assistant."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 抑制 torchvision 警告（本项目只用文本 embedding，不用图像功能）
warnings.filterwarnings("ignore", message=".*torchvision.*")
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

from app.web_server import create_app


def main() -> None:
    app = create_app()
    port = 5001
    print("学生听课实时问答助手 — Web 版")
    print("=" * 40)
    print(f"打开浏览器访问：http://127.0.0.1:{port}")
    print("按 Ctrl+C 退出")
    print()
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
