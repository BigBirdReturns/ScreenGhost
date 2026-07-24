@echo off
setlocal
set SCRIPT=%~dp0scripts\semantic_multibox\run-machine.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" %*
exit /b %ERRORLEVEL%
