@echo off
:: ================================================================
:: server_setup.bat — One-time fresh-droplet Docker setup
:: ================================================================
:: Installs Docker CE on a remote Ubuntu 22.04 server.
:: Does NOT touch other Docker projects already on the server.
:: Requires Git for Windows with bash in PATH, or WSL.
:: Usage: server_setup.bat ROOT@DROPLET_IP
:: ================================================================
setlocal

if "%~1"=="" (
    echo Usage: server_setup.bat ROOT@DROPLET_IP
    exit /b 1
)

bash "%~dp0server_setup.sh" %*
