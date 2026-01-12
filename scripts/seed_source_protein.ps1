$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

if (-Not (Test-Path ".venv")) {
  py -3 -m venv .venv
}

. .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

New-Item -ItemType Directory -Force -Path "data\seeds" | Out-Null
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

python -m src.seed_normalize `
  --scope configs/scope.yaml `
  --seeds configs/seeds.yaml `
  --cache data/seeds/uniprot_cache.json `
  --out_tsv data/seeds/seed_normalized.tsv `
  --out_json data/seeds/seed_normalized.json `
  --log logs/step1_seed_normalize.log

if ($LASTEXITCODE -ne 0) { ... exit ... }

Write-Host "[OK] Outputs:"
Write-Host "  data\seeds\seed_normalized.tsv"
Write-Host "  data\seeds\seed_normalized.json"
Write-Host "  logs\step1_seed_normalize.log"


