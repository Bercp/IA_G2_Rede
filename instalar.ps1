$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Não foi possível criar o ambiente Python.' }
}
& .\.venv\Scripts\python.exe -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'Falha ao instalar PyTorch CPU.' }
& .\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
if ($LASTEXITCODE -ne 0) { throw 'Falha ao instalar dependências.' }
Write-Host 'Ambiente instalado. Execute .\jogar.cmd para jogar ou .\treinar.ps1 para um novo experimento em runs/reproducao.'
