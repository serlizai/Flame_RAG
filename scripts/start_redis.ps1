param()

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$redisServer = Join-Path $projectRoot '.cache\redis\Redis-8.2.10-Windows-x64-cygwin\redis-server.exe'
$redisConfig = Join-Path $projectRoot '.cache\redis\redis.conf'
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $redisServer) -or -not (Test-Path -LiteralPath $redisConfig)) {
    throw 'Local Redis is not installed. See docs/Windows_Setup.md.'
}

# Do not start another server if this port is already occupied.
$listener = [System.Net.Sockets.TcpClient]::new()
try {
    $listener.Connect('127.0.0.1', 6379)
    Write-Output 'Port 6379 is already in use; checking the configured Redis.'
} catch {
    $process = Start-Process -FilePath $redisServer -ArgumentList '.cache/redis/redis.conf' -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $projectRoot 'logs/redis-startup.log') -RedirectStandardError (Join-Path $projectRoot 'logs/redis-startup-error.log') -PassThru
    Write-Output "Starting local Redis (PID $($process.Id))."
    Start-Sleep -Seconds 2
} finally {
    $listener.Dispose()
}

$healthCheck = @'
import sys
from pathlib import Path
from urllib.parse import urlsplit
from dotenv import dotenv_values
import redis

url = dotenv_values(Path(sys.argv[1]) / '.env').get('REDIS_URL')
target = urlsplit(url)
assert target.hostname in ('127.0.0.1', 'localhost') and (target.port or 6379) == 6379, 'This script manages only local Redis on port 6379.'
client = redis.Redis.from_url(url, socket_connect_timeout=3, socket_timeout=3)
assert client.ping()
print('Redis PONG; version:', client.info('server')['redis_version'])
'@
& $python -c $healthCheck $projectRoot
if ($LASTEXITCODE -ne 0) { throw 'Redis health check failed. Check logs/redis.log.' }
