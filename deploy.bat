@echo off
:: ================================================================
:: deploy.bat — Full deploy: rsync + docker compose up --build
:: ================================================================
:: Requires Git for Windows (https://git-scm.com/download/win)
::   Provides: bash, ssh
::   rsync is NOT bundled — install separately:
::     winget install --id=Ookii.Rsync
::   OR run from WSL which has rsync natively:
::     wsl bash ./deploy.sh %1
:: ================================================================
setlocal

if "%~1"=="" (
    echo Usage: deploy.bat ROOT@DROPLET_IP
    exit /b 1
)

bash "%~dp0deploy.sh" %*
