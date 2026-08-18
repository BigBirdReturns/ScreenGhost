@echo off
setlocal
cd /d "%~dp0"
set SCRIPT=%~dp0scripts\dti\preflight.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" %*
exit /b %ERRORLEVEL%
