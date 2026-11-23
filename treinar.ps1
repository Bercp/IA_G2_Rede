param([switch]$Retomar, [int]$Epocas = 30, [string]$Execucao = 'runs/reproducao')
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) { throw 'Execute .\instalar.ps1 primeiro.' }
$sourceFiles = @('train.jsonl.gz', 'validation.jsonl.gz', 'test.jsonl.gz', 'games.jsonl.gz', 'report.json')
$missingFiles = @($sourceFiles | Where-Object { -not (Test-Path -LiteralPath (Join-Path 'data/processed/levon_aronian' $_)) })
if ($missingFiles.Count -gt 0) {
    & .\.venv\Scripts\python.exe scripts\build_dataset.py
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao coletar as partidas.' }
}
if (-not (Test-Path -LiteralPath 'data/processed/levon_aronian/verification.json')) {
    & .\.venv\Scripts\python.exe scripts\verify_dataset.py
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao verificar as partidas.' }
}
& .\.venv\Scripts\python.exe -m aronian.data
if ($LASTEXITCODE -ne 0) { throw 'Falha ao preparar os dados.' }
$trainArguments = @('-m', 'aronian.training', '--epochs', $Epocas, '--run', $Execucao)
if ($Retomar) { $trainArguments += '--resume' }
& .\.venv\Scripts\python.exe @trainArguments
if ($LASTEXITCODE -ne 0) { throw 'Falha no treinamento. Verifique a mensagem acima.' }
