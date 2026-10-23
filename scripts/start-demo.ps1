<#
.SYNOPSIS
    Start the CiteShield local demo: checks prerequisites, then serves the browser UI and API
    on http://127.0.0.1:<Port> with the local Ollama model (qwen3:4b).

.DESCRIPTION
    Always uses the local Ollama backend: CITESHIELD_LLM__PROVIDER is set to "ollama" for this
    process, so the demo never calls a paid API, whatever .env says. The server binds to
    127.0.0.1 only. Retrieval and NLI models run on the CPU so the 8 GB GPU is left to Ollama
    (the configuration the evaluations used). Nothing is downloaded except the Hugging Face
    models on their first use; the Ollama model must already be pulled.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\start-demo.ps1
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\start-demo.ps1 -Port 8080 -NoBrowser
#>
param(
    [int]$Port = 8000,
    [switch]$NoBrowser,
    [switch]$SkipChecks
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

function Fail([string]$message) {
    Write-Host "ERROR: $message" -ForegroundColor Red
    exit 1
}

if (-not $SkipChecks) {
    Write-Host "Checking prerequisites..." -ForegroundColor Cyan
    try { $uv = & py -m uv --version 2>$null } catch { $uv = $null }
    if (-not $uv) { Fail "uv not found. Install it with:  py -m pip install uv" }
    Write-Host "  $uv"

    if (-not (Test-Path ".venv")) {
        Write-Host "  No virtual environment yet; installing dependencies (first run only)..."
        & py -m uv sync --extra ml
        if ($LASTEXITCODE -ne 0) { Fail "py -m uv sync --extra ml failed" }
    }

    if (-not (Test-Path "data\processed\chunks.jsonl")) { Fail "data\processed\chunks.jsonl is missing (it is part of the repository)." }
    if (-not (Test-Path "data\index\manifest.json")) {
        Fail "No retrieval index. Build it once (embeds the corpus, a few minutes on GPU):  py -m uv run citeshield-index"
    }
    Write-Host "  corpus and index present"

    try {
        $tags = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 5
    } catch {
        Fail "Ollama is not reachable at http://127.0.0.1:11434. Start the Ollama app first."
    }
    if (-not ($tags.models | Where-Object { $_.name -eq "qwen3:4b" })) {
        Fail "The Ollama model qwen3:4b is not pulled. Run:  ollama pull qwen3:4b"
    }
    Write-Host "  Ollama reachable, qwen3:4b present"

    $freeMb = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024)
    Write-Host "  free RAM: $freeMb MB"
    if ($freeMb -lt 8000) {
        Write-Host "  WARNING: under 8 GB free. The models need about 8 GB; close other apps or expect swapping." -ForegroundColor Yellow
    }
}

# Local backend only, and the CPU/GPU split the evaluations used.
$env:CITESHIELD_LLM__PROVIDER = "ollama"
$env:CITESHIELD_LLM__OLLAMA_BASE_URL = "http://127.0.0.1:11434"
$env:CITESHIELD_LLM__OLLAMA_MODEL = "qwen3:4b"
$env:CITESHIELD_LLM__OLLAMA_NUM_CTX = "24576"
$env:CITESHIELD_LLM__OLLAMA_NUM_PREDICT = "8192"
$env:CITESHIELD_RETRIEVER__EMBED_DEVICE = "cpu"
$env:CITESHIELD_RETRIEVER__RERANK_DEVICE = "cpu"
$env:CITESHIELD_RETRIEVER__EMBED_DTYPE = "bfloat16"
$env:CITESHIELD_RETRIEVER__RERANK_BATCH_SIZE = "8"
$env:CITESHIELD_VERIFIER__BATCH_SIZE = "4"
$env:MLFLOW_DISABLE_AGENT_HINT = "1"

$url = "http://127.0.0.1:$Port"
Write-Host ""
Write-Host "Starting CiteShield on $url  (Ctrl+C to stop)" -ForegroundColor Green
Write-Host "The page opens right away; the models take a minute or two to load (status box, top right)."
if (-not $NoBrowser) { Start-Process $url }
& py -m uv run uvicorn citeshield.api.main:app --host 127.0.0.1 --port $Port
