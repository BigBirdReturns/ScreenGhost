[CmdletBinding()]
param(
    [string]$Venv = "artifacts\dti\venv",
    [string]$Profile = "configs\dti\profile.example.json",
    [string]$EvidenceRoot = "artifacts\dti\evidence",
    [switch]$Bootstrap
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Repo

if ($Bootstrap) {
    & (Join-Path $PSScriptRoot "bootstrap.ps1") -Venv $Venv
}

$VenvPython = Join-Path (Join-Path $Repo $Venv) "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "DTI environment is missing. Run BOOTSTRAP_DTI.cmd first or pass -Bootstrap."
}
$ProfilePath = Join-Path $Repo $Profile
if (-not (Test-Path $ProfilePath)) { throw "Profile not found: $ProfilePath" }

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$RunRoot = Join-Path (Join-Path $Repo $EvidenceRoot) $Stamp
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$DoctorPath = Join-Path $RunRoot "doctor.json"
$CapturePath = Join-Path $RunRoot "freeplay-entry.png"
$CaptureReceiptPath = Join-Path $RunRoot "capture.json"
$ManifestPath = Join-Path $RunRoot "preflight-manifest.json"

& $VenvPython -m experiments.dti validate-profile $ProfilePath
if ($LASTEXITCODE -ne 0) { throw "Profile validation failed" }

$DoctorOutput = & $VenvPython -m experiments.dti doctor $ProfilePath
$DoctorExit = $LASTEXITCODE
$DoctorOutput | Set-Content -Encoding utf8 $DoctorPath
$DoctorOutput | Write-Host
if ($DoctorExit -ne 0) {
    Write-Host "DTI doctor refused the live target. Evidence retained at $RunRoot"
    exit $DoctorExit
}

$CaptureOutput = & $VenvPython -m experiments.dti capture $ProfilePath $CapturePath
$CaptureExit = $LASTEXITCODE
$CaptureOutput | Set-Content -Encoding utf8 $CaptureReceiptPath
$CaptureOutput | Write-Host
if ($CaptureExit -ne 0) {
    Write-Host "DTI capture failed. Evidence retained at $RunRoot"
    exit $CaptureExit
}

$Manifest = [ordered]@{
    schema = "screenghost_dti_preflight_manifest_v1"
    created_at = (Get-Date).ToUniversalTime().ToString("o")
    profile = (Resolve-Path $ProfilePath).Path
    doctor = (Resolve-Path $DoctorPath).Path
    capture = (Resolve-Path $CapturePath).Path
    capture_receipt = (Resolve-Path $CaptureReceiptPath).Path
    capture_sha256 = (Get-FileHash -Algorithm SHA256 $CapturePath).Hash.ToLowerInvariant()
    status = "eyes_and_target_custody_passed_round_unqualified"
}
$Manifest | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 $ManifestPath
$Manifest | ConvertTo-Json -Depth 5 | Write-Host
Write-Host "DTI preflight evidence: $RunRoot"
