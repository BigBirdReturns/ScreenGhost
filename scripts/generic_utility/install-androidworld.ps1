[CmdletBinding()]
param(
    [string]$Destination = "tools/android_world",
    [string]$Python,
    [string]$Ref = "3e50888527ef9f29b9157ecd537e408008bb1c85"
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $Python) {
    $candidate = Join-Path $Repo ".venv-generic-utility\Scripts\python.exe"
    if (Test-Path $candidate) { $Python = $candidate }
    else { $Python = (Get-Command python -ErrorAction Stop).Source }
}
$Target = Join-Path $Repo $Destination
if (-not (Test-Path $Target)) {
    & git clone https://github.com/google-research/android_world.git $Target
}
& git -C $Target fetch --tags --prune
& git -C $Target checkout --detach $Ref
& $Python -m pip install -r (Join-Path $Target "requirements.txt")
& $Python -m pip install -e $Target
$Resolved = (& git -C $Target rev-parse HEAD).Trim()
if ($Resolved -ne $Ref) { throw "AndroidWorld checkout mismatch: expected $Ref, got $Resolved" }
Write-Host "AndroidWorld installed at pinned revision $Resolved."
Write-Host "Create the upstream Pixel 6 / API 33 AVD named AndroidWorldAvd and launch it with -grpc 8554."
