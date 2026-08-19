@echo off
setlocal
cd /d "%~dp0"
set SCRIPT=%~dp0scripts\dti\bootstrap.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" %*
exit /b %ERRORLEVEL%
