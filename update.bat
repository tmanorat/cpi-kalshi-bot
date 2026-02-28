@echo off
:: ================================================================
:: update.bat — Fast update: sync .py files and restart (no rebuild)
:: ================================================================
:: Use when only Python files changed (requirements.txt unchanged).
:: Requires Git for Windows with bash in PATH, or WSL.
:: Usage: update.bat ROOT@DROPLET_IP
:: ================================================================
setlocal

if "%~1"=="" (
    echo Usage: update.bat ROOT@DROPLET_IP
    exit /b 1
)

bash "%~dp0update.sh" %*
