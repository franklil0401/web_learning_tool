# 一键运行 AI 直播课程辅导系统

Write-Host "==============================="
Write-Host "AI 直播课程辅导系统启动脚本"
Write-Host "==============================="

# 检查 Python 环境
Write-Host "正在检查 Python 环境..."
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python 版本: $pythonVersion"
} catch {
    Write-Host "错误：未找到 Python 环境，请先安装 Python 3.8+" -ForegroundColor Red
    pause
    exit 1
}

# 检查依赖项
Write-Host "正在检查依赖项..."
try {
    $faissInstalled = pip list | Select-String "faiss-cpu"
    if (-not $faissInstalled) {
        Write-Host "正在安装依赖项..."
        pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
        if ($LASTEXITCODE -ne 0) {
            Write-Host "错误：依赖项安装失败" -ForegroundColor Red
            pause
            exit 1
        }
    } else {
        Write-Host "依赖项已安装"
    }
} catch {
    Write-Host "错误：检查依赖项时出错: $($_.Exception.Message)" -ForegroundColor Red
    pause
    exit 1
}

# 启动系统
Write-Host "依赖项检查完成，正在启动系统..."
Write-Host ""
Write-Host "系统启动后，请按照以下步骤操作："
Write-Host "1. 选择选项 1 开始监听直播课程"
Write-Host "2. 播放直播课程内容，系统会自动识别"
Write-Host "3. 选择选项 3 输入学生问题"
Write-Host "4. 系统会基于课程内容生成回答"
Write-Host "5. 选择选项 2 停止监听"
Write-Host "6. 选择选项 4 退出系统"
Write-Host ""

# 运行系统
python ai_tutor_system.py

# 暂停
pause