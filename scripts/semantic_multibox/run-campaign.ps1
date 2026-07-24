[CmdletBinding()]
param(
    [string]$Python,
    [string]$Out = "log/semantic_multibox/verification",
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
$env:PYTHONPATH = $Repo
$env:PYTHONDONTWRITEBYTECODE = "1"
$Args = @((Join-Path $Repo "VERIFY_SEMANTIC_MULTIBOX.py"), "--out", (Join-Path $Repo $Out))
if ($RequireFullRepo) { $Args += "--require-full-repo" }
& $Python @Args
exit $LASTEXITCODE
