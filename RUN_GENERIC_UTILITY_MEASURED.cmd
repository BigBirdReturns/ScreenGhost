@echo off
setlocal
set SCRIPT=%~dp0scripts\generic_utility\run-measured.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" %*
exit /b %ERRORLEVEL%
