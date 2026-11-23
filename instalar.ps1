param([switch]$Treinamento)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
Set-Location -LiteralPath $PSScriptRoot
$installArguments = @('scripts/install.py')
if ($Treinamento) { $installArguments += '--training' }
python @installArguments
if ($LASTEXITCODE -ne 0) { throw 'Falha na instalação. Verifique o erro acima.' }
