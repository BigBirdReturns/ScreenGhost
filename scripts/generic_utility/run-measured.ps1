[CmdletBinding()]
param(
    [string]$Python,
    [string]$Out = "log/generic_utility/measured",
    [string]$LocalModel = "osunlp/UGround-V1-2B",
    [string]$ModelDType = "float16",
    [string]$ModelQuantization,
    [string]$AndroidWorldAdbPath,
    [int]$AndroidWorldConsolePort = 5554,
    [switch]$AndroidWorldSetup,
    [string]$PhysicalDevice,
    [string]$PhysicalSurfaceId = "physical.current",
    [switch]$RequireLocalModel,
    [switch]$RequireAndroidWorld,
    [switch]$RequirePhysical,
    [switch]$RequireBrowser,
    [switch]$RequireFullRepo,
    [switch]$ForceLocalModel,
    [switch]$TrustRemoteCode
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $Python) {
    $candidate = Join-Path $Repo ".venv-generic-utility\Scripts\python.exe"
    if (Test-Path $candidate) { $Python = $candidate }
    elseif ($env:SG_PYTHON) { $Python = $env:SG_PYTHON }
    else { $Python = (Get-Command python -ErrorAction Stop).Source }
}
$env:PYTHONPATH = $Repo
$env:PYTHONDONTWRITEBYTECODE = "1"
$Args = @(
    (Join-Path $Repo "VERIFY_GENERIC_UTILITY_CAMPAIGN.py"),
    "--out", (Join-Path $Repo $Out),
    "--local-model", $LocalModel,
    "--model-dtype", $ModelDType
)
if ($ModelQuantization) { $Args += @("--model-quantization", $ModelQuantization) }
if ($AndroidWorldAdbPath) {
    $Args += @("--androidworld-adb-path", $AndroidWorldAdbPath, "--androidworld-console-port", $AndroidWorldConsolePort)
}
if ($AndroidWorldSetup) { $Args += "--androidworld-setup" }
if ($null -ne $PhysicalDevice -and $PhysicalDevice -ne "") { $Args += @("--physical-device", $PhysicalDevice) }
elseif ($RequirePhysical) { $Args += "--physical-device" }
if ($PhysicalSurfaceId) { $Args += @("--physical-surface-id", $PhysicalSurfaceId) }
if ($RequireLocalModel) { $Args += "--require-local-model" }
if ($RequireAndroidWorld) { $Args += "--require-androidworld" }
if ($RequirePhysical) { $Args += "--require-physical" }
if ($RequireBrowser) { $Args += "--require-browser" }
if ($RequireFullRepo) { $Args += "--require-full-repo" }
if ($ForceLocalModel) { $Args += "--force-local-model" }
if ($TrustRemoteCode) { $Args += "--trust-remote-code" }
& $Python @Args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Measured campaign complete. Receipts: $(Join-Path $Repo $Out)"
