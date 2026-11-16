param(
    [string]$BindAddress = '127.0.0.1',
    [ValidateRange(1, 65535)]
    [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Project Python environment was not found.'
}

# Load project-local settings in the child process, including when this terminal
# already contains model credentials from an earlier configuration.
$bootstrap = @'
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

from dotenv import load_dotenv
import uvicorn

project_root = Path(sys.argv[1]).resolve()
environment_file = project_root / '.env'
if not environment_file.is_file():
    raise SystemExit('Project .env was not found.')
load_dotenv(environment_file, override=True)
model_name = os.getenv('OPENAI_MODEL') or 'qwen3.8-27b'
provider_host = urlsplit(os.getenv('OPENAI_BASE_URL') or '').hostname or 'default'
print(f'Chat model: {model_name}; provider: {provider_host}', flush=True)
uvicorn.run('backend.main:app', host=sys.argv[2], port=int(sys.argv[3]))
'@

Push-Location -LiteralPath $projectRoot
try {
    & $python -c $bootstrap $projectRoot $BindAddress $Port
    $backendExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $backendExitCode
