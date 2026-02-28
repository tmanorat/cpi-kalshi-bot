@echo off
:: ================================================================
:: logs.bat — Tail live logs from running containers
:: ================================================================
:: Requires Git for Windows with bash in PATH, or WSL.
:: Usage: logs.bat ROOT@DROPLET_IP [dashboard^|bot^|all]
:: ================================================================
setlocal

if "%~1"=="" (
    echo Usage: logs.bat ROOT@DROPLET_IP [dashboard^|bot^|all]
    exit /b 1
)

bash "%~dp0logs.sh" %*
