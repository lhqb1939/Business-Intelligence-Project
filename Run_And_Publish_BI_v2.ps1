param(
    [switch]$Retrain,
    [string]$Prepared = 'prepared_v2',
    [string]$Results = 'outputs_calendar_v2',
    [string]$Page1 = 'page1_calendar_v2',
    [string]$Start = '2014-07-07',
    [ValidateRange(1,12)][int]$Weeks = 4,
    [ValidateRange(0,2147483647)][int]$RunId = 0,
    [string]$SqlServer = '',
    [string]$Database = 'DW_Promotion_Analytics'
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run from your existing CompanyX LR project with .venv.' }
Push-Location $root
try {
    if ($Retrain) {
        if ($RunId -eq 0) { $RunId = [int][DateTimeOffset]::UtcNow.ToUnixTimeSeconds() }
        $stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') + '_' + [Guid]::NewGuid().ToString('N').Substring(0,6)
        $Results = "runs/$stamp/outputs_calendar_v2"
        $Page1 = "runs/$stamp/page1_calendar_v2"
        & $python 'promotion_lr_calendar_v2.py' train --csv "$Prepared/training.csv" --out $Results --test-weeks 44
        if ($LASTEXITCODE -ne 0) { throw 'Training failed; publication stopped.' }
        & $python 'verify_calendar_v2.py' --prepared $Prepared --results $Results
        if ($LASTEXITCODE -ne 0) { throw 'Verification failed; publication stopped.' }
        & $python 'Expand_Page1_Calendar_v2.py' --prepared $Prepared --model "$Results/model.joblib" --out $Page1 --run-id $RunId --start $Start --weeks $Weeks
        if ($LASTEXITCODE -ne 0) { throw 'Forecast generation failed; publication stopped.' }
    }
    $arguments = @('Publish_BI_v2.py', '--prepared', $Prepared, '--results', $Results, '--page1', $Page1, '--database', $Database)
    if ($SqlServer) { $arguments += @('--sql-server', $SqlServer) }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw 'Publication failed; check the preceding error.' }
} finally { Pop-Location }
