[CmdletBinding()]
param(
    [string]$Python,
    [string]$Device,
    [string]$SurfaceId = "physical.current",
    [string]$Out = "log/generic_utility/physical"
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $Python) { $Python = (Get-Command python -ErrorAction Stop).Source }
$env:PYTHONPATH = $Repo
$Args = @("-m", "experiments.generic_utility", "physical-smoke", "--surface-id", $SurfaceId, "--out", (Join-Path $Repo $Out))
if ($Device) { $Args += @("--device", $Device) }
& $Python @Args
