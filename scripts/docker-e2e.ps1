<#
.SYNOPSIS
    Real-model end-to-end test of the Docker deployment: build the image, start it against
    the host's Ollama (qwen3:4b), ask one real question over POST /ask (SSE) and check the
    structured result.

.DESCRIPTION
    Opt-in and local only; CI runs the lighter scripts/docker_smoke.py instead. Needs Docker
    Desktop running, Ollama on http://127.0.0.1:11434 with qwen3:4b pulled, and the built
    retrieval index in data\index. The first run downloads about 7.7 GB of Hugging Face models
    into the Docker volume "citeshield-models" (kept for later runs and shared with
    compose.yaml); loading takes several minutes on CPU.

    The index is copied to a temporary folder for the container, so the test never writes
    to data\index and does not collide with a running native demo. Ollama keeps listening
    on 127.0.0.1: Docker Desktop routes host.docker.internal to the host's loopback.

    Result categories (exit codes):
      0  transport and schema checks passed. The answer-quality line says separately whether
         the model answered and cited GDPR Art. 33(1); with -StrictAnswer a miss exits 4.
      1  prerequisite missing (nothing was started)
      2  infrastructure or transport failure (build, start, readiness, HTTP, SSE, error event)
      3  the final event is not a valid AgentAnswer payload (schema failure)
      4  -StrictAnswer only: transport and schema passed, but the model did not answer the
         question with a citation of GDPR Art. 33(1) (model-quality failure)

    Removes only what it created: its container and the temporary index copy. The image
    and the model volume are kept.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\docker-e2e.ps1
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\docker-e2e.ps1 -SkipBuild -StrictAnswer
#>
param(
    [string]$Image = "citeshield:local",
    [switch]$SkipBuild,
    [string]$Question = "Within how many hours must a controller notify a personal data breach to the supervisory authority under the GDPR?",
    [int]$ReadyTimeoutSec = 2400,
    [int]$AnswerTimeoutSec = 1200,
    [switch]$StrictAnswer,
    [switch]$KeepContainer
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$OllamaHostUrl = "http://127.0.0.1:11434"
$OllamaModel = "qwen3:4b"
$ExpectedChunk = "32016R0679:art33:p1@"   # GDPR Art. 33(1), any version
$Container = "citeshield-e2e-" + [guid]::NewGuid().ToString("N").Substring(0, 8)
$TempIndex = Join-Path ([IO.Path]::GetTempPath()) $Container
$started = $false

function Info([string]$m) { Write-Host $m -ForegroundColor Cyan }
function Ok([string]$m) { Write-Host "  ok   $m" }
function Finish([int]$code, [string]$m) {
    $color = if ($code -eq 0) { "Green" } else { "Red" }
    Write-Host ""
    Write-Host $m -ForegroundColor $color
    exit $code
}
# Native commands whose stderr is expected (engine down, container gone): with
# $ErrorActionPreference = "Stop", Windows PowerShell 5.1 turns redirected stderr into a
# terminating error, so these run with "Continue" and stderr discarded.
function Invoke-DockerQuiet {
    $ErrorActionPreference = "Continue"
    & docker @args 2>$null
}
function Show-Logs {
    if ($script:started) {
        Write-Host "--- container logs (last 150 lines) ---" -ForegroundColor Yellow
        $ErrorActionPreference = "Continue"
        & docker logs --tail 150 $Container 2>&1 | ForEach-Object { Write-Host $_ }
    }
}

# ------------------------------------------------------------------ prerequisites (exit 1)
Info "Checking prerequisites..."
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Finish 1 "PREREQUISITE: the docker CLI is not on PATH. Install and start Docker Desktop."
}
$serverVersion = Invoke-DockerQuiet version --format "{{.Server.Version}}"
if ($LASTEXITCODE -ne 0 -or -not $serverVersion) {
    Finish 1 "PREREQUISITE: the Docker engine is not reachable. Start Docker Desktop and wait until it says 'Engine running'."
}
Ok "Docker engine $serverVersion"

try { $tags = Invoke-RestMethod -Uri "$OllamaHostUrl/api/tags" -TimeoutSec 5 }
catch { Finish 1 "PREREQUISITE: Ollama is not reachable at $OllamaHostUrl. Start the Ollama app." }
if (-not ($tags.models | Where-Object { $_.name -eq $OllamaModel })) {
    Finish 1 "PREREQUISITE: the Ollama model $OllamaModel is not pulled. Run:  ollama pull $OllamaModel"
}
Ok "Ollama reachable on the host, $OllamaModel present"

if (-not (Test-Path "data\processed\chunks.jsonl")) {
    Finish 1 "PREREQUISITE: data\processed\chunks.jsonl is missing (it is part of the repository)."
}
if (-not (Select-String -Path "data\processed\chunks.jsonl" -SimpleMatch "`"chunk_id`":`"$ExpectedChunk" -Quiet)) {
    Finish 1 "PREREQUISITE: the corpus has no GDPR Art. 33(1) chunk ($ExpectedChunk...); the test question needs it."
}
if (-not (Test-Path "data\index\manifest.json")) {
    Finish 1 "PREREQUISITE: no retrieval index. Build it once:  py -m uv run citeshield-index"
}
Ok "corpus covers GDPR Art. 33(1); index present"

# ------------------------------------------------------------------ build, start, wait (exit 2)
try {
    if (-not $SkipBuild) {
        Info "Building $Image ..."
        & docker build -t $Image .
        if ($LASTEXITCODE -ne 0) { Finish 2 "INFRASTRUCTURE: docker build failed." }
    }

    Copy-Item -Recurse "data\index" $TempIndex
    Info "Starting $Container (index copy: $TempIndex)"
    $runArgs = @(
        "run", "--detach", "--name", $Container, "--label", "citeshield.test=e2e",
        "--publish", "127.0.0.1::8000",
        "--volume", "${TempIndex}:/app/data/index",
        "--volume", "citeshield-models:/models",
        "--env", "CITESHIELD_LLM__PROVIDER=ollama",
        "--env", "CITESHIELD_LLM__OLLAMA_BASE_URL=http://host.docker.internal:11434",
        "--env", "CITESHIELD_LLM__OLLAMA_MODEL=$OllamaModel",
        "--env", "CITESHIELD_LLM__OLLAMA_NUM_CTX=24576",
        "--env", "CITESHIELD_LLM__OLLAMA_NUM_PREDICT=8192",
        "--env", "CITESHIELD_RETRIEVER__EMBED_DEVICE=cpu",
        "--env", "CITESHIELD_RETRIEVER__RERANK_DEVICE=cpu",
        "--env", "CITESHIELD_RETRIEVER__EMBED_DTYPE=bfloat16",
        "--env", "CITESHIELD_RETRIEVER__RERANK_BATCH_SIZE=8",
        "--env", "CITESHIELD_VERIFIER__BATCH_SIZE=4",
        $Image
    )
    & docker @runArgs | Out-Null
    if ($LASTEXITCODE -ne 0) { Finish 2 "INFRASTRUCTURE: docker run failed." }
    $started = $true
    $hostPort = ((& docker port $Container 8000/tcp) | Select-Object -First 1).Split(":")[-1]
    $base = "http://127.0.0.1:$hostPort"
    Ok "container started, API at $base"

    Info "Waiting for readiness (first run downloads the models; timeout ${ReadyTimeoutSec}s)..."
    $deadline = (Get-Date).AddSeconds($ReadyTimeoutSec)
    $health = $null; $lastReport = Get-Date
    while ((Get-Date) -lt $deadline) {
        try { $health = Invoke-RestMethod -Uri "$base/health" -TimeoutSec 10 } catch { $health = $null }
        if ($health -and $health.status -eq "ok") { break }
        if ($health -and $health.status -eq "error") {
            Show-Logs
            Finish 2 "INFRASTRUCTURE: the agent failed to load: $($health.error)"
        }
        $running = Invoke-DockerQuiet inspect --format "{{.State.Running}}" $Container
        if ($running -ne "true") { Show-Logs; Finish 2 "INFRASTRUCTURE: the container exited during startup." }
        if (((Get-Date) - $lastReport).TotalSeconds -ge 60) {
            $state = if ($health) { $health.status } else { "not answering yet" }
            Write-Host "  ... $state"
            $lastReport = Get-Date
        }
        Start-Sleep -Seconds 5
    }
    if (-not $health -or $health.status -ne "ok") {
        Show-Logs; Finish 2 "INFRASTRUCTURE: not ready within ${ReadyTimeoutSec}s."
    }
    try { $ready = Invoke-WebRequest -UseBasicParsing -Uri "$base/ready" -TimeoutSec 10 } catch { $ready = $null }
    if (-not $ready -or $ready.StatusCode -ne 200) { Show-Logs; Finish 2 "INFRASTRUCTURE: /health is ok but /ready is not 200." }
    Ok "ready: /health status ok, /ready 200"
    if (-not $health.llm.reachable -or -not $health.llm.model_present) {
        Show-Logs
        Finish 2 "INFRASTRUCTURE: the container cannot use Ollama at $($health.llm.base_url) (reachable=$($health.llm.reachable), model_present=$($health.llm.model_present))."
    }
    Ok "the container reaches Ollama at $($health.llm.base_url); $($health.llm.model) present"

    # -------------------------------------------------------------- one real question (SSE)
    Info "POST /ask: $Question"
    Add-Type -AssemblyName System.Net.Http
    $client = New-Object System.Net.Http.HttpClient
    $client.Timeout = [TimeSpan]::FromSeconds($AnswerTimeoutSec)
    $json = @{ question = $Question; mode = "answer" } | ConvertTo-Json -Compress
    $req = New-Object System.Net.Http.HttpRequestMessage([System.Net.Http.HttpMethod]::Post, "$base/ask")
    $req.Content = New-Object System.Net.Http.StringContent($json, [Text.Encoding]::UTF8, "application/json")
    $t0 = Get-Date
    try {
        $resp = $client.SendAsync($req, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
    } catch {
        Show-Logs; Finish 2 "TRANSPORT: POST /ask failed: $($_.Exception.Message)"
    }
    if (-not $resp.IsSuccessStatusCode) {
        $body = $resp.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        Show-Logs; Finish 2 "TRANSPORT: POST /ask answered $([int]$resp.StatusCode): $body"
    }
    if ($resp.Content.Headers.ContentType.MediaType -ne "text/event-stream") {
        Finish 2 "TRANSPORT: POST /ask did not stream SSE (Content-Type $($resp.Content.Headers.ContentType))."
    }
    $reader = New-Object System.IO.StreamReader($resp.Content.ReadAsStreamAsync().GetAwaiter().GetResult(), [Text.Encoding]::UTF8)
    $final = $null; $errorEvent = $null; $evName = $null; $data = $null; $seen = @{}
    while ($null -ne ($line = $reader.ReadLine())) {
        if ($line.StartsWith("event: ")) { $evName = $line.Substring(7) }
        elseif ($line.StartsWith("data: ")) { $data = $line.Substring(6) }
        elseif ($line -eq "" -and $evName) {
            $seen[$evName] = 1 + [int]$seen[$evName]
            $elapsed = [int]((Get-Date) - $t0).TotalSeconds
            try { $payload = $data | ConvertFrom-Json }
            catch { Finish 2 "TRANSPORT: SSE '$evName' data is not JSON: $data" }
            switch ($evName) {
                "status" { Write-Host ("  [{0,4}s] stage {1}" -f $elapsed, $payload.stage) }
                "guard" { Write-Host ("  [{0,4}s] guard {1}" -f $elapsed, $payload.label) }
                "tool_call" { Write-Host ("  [{0,4}s] tool {1}" -f $elapsed, $payload.name) }
                "draft_rejected" { Write-Host ("  [{0,4}s] draft rejected by validation" -f $elapsed) }
                "final" { $final = $payload }
                "error" { $errorEvent = $payload }
            }
            $evName = $null; $data = $null
        }
    }
    $reader.Dispose(); $client.Dispose()
    $seconds = [int]((Get-Date) - $t0).TotalSeconds
    if ($errorEvent) { Show-Logs; Finish 2 "TRANSPORT: the run ended with an error event after ${seconds}s: $($errorEvent.message)" }
    if (-not $final) { Show-Logs; Finish 2 "TRANSPORT: the stream ended after ${seconds}s without a final event." }
    Ok "SSE stream completed in ${seconds}s with a final event (events: $(($seen.Keys | Sort-Object) -join ', '))"

    # -------------------------------------------------------------- schema checks (exit 3)
    $problems = New-Object System.Collections.Generic.List[string]
    $answer = $final.answer
    if (-not $answer) { $problems.Add("final.answer is missing") }
    else {
        if (@("answered", "partial", "refused") -notcontains $answer.status) { $problems.Add("answer.status '$($answer.status)' is not answered/partial/refused") }
        if ($null -eq $answer.summary) { $problems.Add("answer.summary is missing") }
        $claims = @($answer.claims); $checklist = @($answer.checklist | Where-Object { $_ })
        $cited = @(); foreach ($c in $claims + $checklist) { if ($c) { $cited += @($c.citations) } }
        if ($answer.status -eq "refused") {
            if (-not $answer.refusal_reason) { $problems.Add("a refused answer has no refusal_reason") }
            if ($claims.Count -gt 0 -and $claims[0]) { $problems.Add("a refused answer contains claims") }
        } else {
            if ($cited.Count -eq 0) { $problems.Add("an $($answer.status) answer has no cited claim or checklist item") }
            foreach ($c in $claims + $checklist) {
                if ($c -and @($c.citations).Count -eq 0) { $problems.Add("a claim or checklist item has no citation") }
            }
            foreach ($cit in $cited) {
                if (-not $cit.chunk_id) { $problems.Add("a citation has no chunk_id") }
                if (-not $cit.quote -or $cit.quote.Length -lt 8 -or $cit.quote.Length -gt 400) { $problems.Add("citation $($cit.chunk_id): quote length outside 8-400 characters") }
                if (-not ($final.sources.PSObject.Properties.Name -contains $cit.chunk_id)) { $problems.Add("citation $($cit.chunk_id) is missing from final.sources") }
            }
            if ($null -eq $final.verification) { $problems.Add("final.verification is missing for an $($answer.status) answer") }
        }
    }
    if ($problems.Count -gt 0) {
        $problems | ForEach-Object { Write-Host "  FAIL $_" -ForegroundColor Red }
        Write-Host ($final | ConvertTo-Json -Depth 8 -Compress)
        Finish 3 "SCHEMA: the final event is not a valid AgentAnswer payload ($($problems.Count) problem(s))."
    }
    Ok "final payload valid: status '$($answer.status)', $($cited.Count) citation(s), every cited chunk in sources"

    # -------------------------------------------------------------- answer quality (reported)
    $citesArt33 = @($cited | Where-Object { $_.chunk_id.StartsWith($ExpectedChunk) }).Count -gt 0
    $text = ($answer.summary + " " + (($claims | ForEach-Object { $_.text }) -join " "))
    Write-Host ""
    Write-Host "Outcome: status=$($answer.status) refusal_reason=$($answer.refusal_reason) cites_GDPR_33(1)=$citesArt33 mentions_72_hours=$($text -match '72')"
    Write-Host "Summary: $($answer.summary)"
    foreach ($cit in $cited) { Write-Host "  cites $($cit.chunk_id): `"$($cit.quote)`"" }
    if ($answer.status -ne "refused" -and $citesArt33) {
        Finish 0 "PASS: transport and schema OK; answer quality OK ($($answer.status), cites GDPR Art. 33(1))."
    }
    $miss = if ($answer.status -eq "refused") { "the model refused ($($answer.refusal_reason))" } else { "the $($answer.status) answer does not cite GDPR Art. 33(1)" }
    if ($StrictAnswer) { Finish 4 "MODEL QUALITY: transport and schema OK, but $miss." }
    Finish 0 "PASS (transport and schema only): $miss. This is NOT evidence that the model answered correctly; rerun with -StrictAnswer to fail on it."
}
finally {
    if ($started -and -not $KeepContainer) { Invoke-DockerQuiet rm --force $Container | Out-Null }
    if ($started -and $KeepContainer) { Write-Host "Container kept: $Container  (remove: docker rm -f $Container)" }
    if (Test-Path $TempIndex) { Remove-Item -Recurse -Force $TempIndex -ErrorAction SilentlyContinue }
}
