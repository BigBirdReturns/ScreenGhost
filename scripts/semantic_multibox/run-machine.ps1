[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Plan,
    [string]$Python,
    [string]$Out = "log/semantic_multibox/machine-verification",
    [switch]$Apply,
    [switch]$RequireFullRepo
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $Python) {
    $candidate = Join-Path $Repo ".venv-semantic-multibox\Scripts\python.exe"
    if (Test-Path $candidate) { $Python = $candidate }
    elseif ($env:SG_PYTHON) { $Python = $env:SG_PYTHON }
    else { $Python = (Get-Command python -ErrorAction Stop).Source }
}
$PlanPath = (Resolve-Path $Plan).Path
$Args = @(
    (Join-Path $Repo "VERIFY_SEMANTIC_MULTIBOX.py"),
    "--out", (Join-Path $Repo $Out),
    "--machine-plan", $PlanPath,
    "--require-machine"
)
if ($Apply) { $Args += "--apply-machine" }
if ($RequireFullRepo) { $Args += "--require-full-repo" }
$env:PYTHONPATH = $Repo
$env:PYTHONDONTWRITEBYTECODE = "1"
& $Python @Args
exit $LASTEXITCODE
