@echo off
setlocal
set SCRIPT=%~dp0scripts\semantic_multibox\doctor.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" %*
exit /b %ERRORLEVEL%
