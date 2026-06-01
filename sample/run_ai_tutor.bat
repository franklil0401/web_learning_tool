@echo off

rem 一键运行 AI 直播课程辅导系统

echo ==============================
echo AI 直播课程辅导系统启动脚本
echo ==============================

echo 正在检查 Python 环境...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误：未找到 Python 环境，请先安装 Python 3.8+
    pause
    exit /b 1
)

echo 正在检查依赖项...
pip list | findstr "faiss-cpu" >nul 2>&1
if %errorlevel% neq 0 (
    echo 正在安装依赖项...
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if %errorlevel% neq 0 (
        echo 错误：依赖项安装失败
        pause
        exit /b 1
    )
)

echo 依赖项检查完成，正在启动系统...
echo.
echo 系统启动后，请按照以下步骤操作：
echo 1. 选择选项 1 开始监听直播课程
2. 播放直播课程内容，系统会自动识别
3. 选择选项 3 输入学生问题
4. 系统会基于课程内容生成回答
5. 选择选项 2 停止监听
6. 选择选项 4 退出系统
echo.

python ai_tutor_system.py

pause