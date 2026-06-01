param(
    [Parameter(Mandatory = $true)]
    [string]$ArkApiKey
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$envPath = Join-Path $projectRoot ".env"

$lines = @()
if (Test-Path -LiteralPath $envPath) {
    $lines = Get-Content -LiteralPath $envPath | Where-Object {
        $_ -notmatch "^\s*(ARK_API_KEY|TA_TRANSCRIPTION_PROVIDER|TA_LLM_PROVIDER|TA_DOUBAO_TRANSCRIBE_MODEL|TA_DOUBAO_ANSWER_MODEL|TA_AUDIO_STREAMING|TA_AUDIO_SILENCE_SECONDS|TA_AUDIO_MAX_SEGMENT_SECONDS)\s*="
    }
}

$lines += "ARK_API_KEY=$ArkApiKey"
$lines += "TA_TRANSCRIPTION_PROVIDER=doubao"
$lines += "TA_LLM_PROVIDER=doubao"
$lines += "TA_DOUBAO_TRANSCRIBE_MODEL=doubao-seed-2-0-lite-260428"
$lines += "TA_DOUBAO_ANSWER_MODEL=doubao-seed-2-0-lite-260428"
$lines += "TA_AUDIO_STREAMING=1"
$lines += "TA_AUDIO_SILENCE_SECONDS=0.9"
$lines += "TA_AUDIO_MAX_SEGMENT_SECONDS=12"

$lines | Set-Content -LiteralPath $envPath -Encoding UTF8
Write-Host "Saved ARK_API_KEY to $envPath"
