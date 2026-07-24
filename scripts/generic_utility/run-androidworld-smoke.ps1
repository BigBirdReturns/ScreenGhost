[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$AdbPath,
    [string]$Python,
    [int]$ConsolePort = 5554,
    [string]$Out = "log/generic_utility/androidworld",
    [switch]$PerformEmulatorSetup
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $Python) { $Python = (Get-Command python -ErrorAction Stop).Source }
$env:PYTHONPATH = $Repo
$Args = @("-m", "experiments.generic_utility", "androidworld-smoke", "--adb-path", $AdbPath, "--console-port", $ConsolePort, "--out", (Join-Path $Repo $Out))
if ($PerformEmulatorSetup) { $Args += "--perform-emulator-setup" }
& $Python @Args
