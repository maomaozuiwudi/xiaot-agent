@echo off
chcp 65001 >nul
title 小t Agent 安装器

echo ============================================
echo   🐱 小t Agent — 一键安装
echo   小红书内容工坊 AI 助手
echo ============================================
echo.

:: 检查 Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 没找到 Python，请先安装 Python 3.10+
    echo    下载: https://www.python.org/downloads/
    pause
    exit /b 1
)

:: 下载
echo 📥 下载小t...
set ZIP_URL=https://github.com/maomaozuiwudi/xiaot-agent/archive/refs/heads/main.zip
set ZIP_FILE=%TEMP%\xiaot-agent.zip
set EXTRACT_DIR=%USERPROFILE%\Desktop

curl.exe -L -o "%ZIP_FILE%" "%ZIP_URL%" --progress-bar
if %errorlevel% neq 0 (
    echo ❌ 下载失败，请检查网络
    pause
    exit /b 1
)

:: 解压
echo 📦 解压中...
powershell -Command "Expand-Archive -Path '%ZIP_FILE%' -DestinationPath '%EXTRACT_DIR%' -Force"
if exist "%EXTRACT_DIR%\xiaot-agent-main" (
    move /y "%EXTRACT_DIR%\xiaot-agent-main" "%EXTRACT_DIR%\xiaot-agent" >nul 2>&1
)
echo ✅ 下载完成：%USERPROFILE%\Desktop\xiaot-agent

:: 安装
echo 🔧 安装依赖...
cd /d "%USERPROFILE%\Desktop\xiaot-agent"
pip install -e . --quiet
if %errorlevel% neq 0 (
    echo ❌ 安装依赖失败
    pause
    exit /b 1
)

:: 安装浏览器引擎
echo 🌐 安装浏览器引擎...
playwright install chromium --quiet

:: 检查config
if not exist config.yaml (
    if exist config.example.yaml (
        copy config.example.yaml config.yaml >nul
    )
)

echo.
echo ============================================
echo   ✅ 安装完成！
echo.
echo   现在可以运行：  xiaot
echo   或者：          cd Desktop\xiaot-agent ^&^& python main.py
echo ============================================
echo.

:: 启动
echo 🚀 启动小t...
cd /d "%USERPROFILE%\Desktop\xiaot-agent"
xiaot
pause
