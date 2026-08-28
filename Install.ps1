[CmdletBinding()]
param(
    [switch]$InstallPython,
    [switch]$SkipModelDownload
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSCommandPath
$RuntimeRoot = Join-Path $ProjectRoot 'runtime'
$RuntimePython = Join-Path $RuntimeRoot 'Scripts\python.exe'
$RequiredPython = '3.11'

function Invoke-Checked {
    param([string]$Description, [scriptblock]$Command)

    Write-Host "`n== $Description ==" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description 失敗，結束碼：$LASTEXITCODE"
    }
}

function Find-Python311 {
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        & $launcher.Source "-$RequiredPython" -c 'import sys; print(sys.executable)' 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            return [PSCustomObject]@{ Executable = $launcher.Source; Arguments = @("-$RequiredPython") }
        }
    }
    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($python) {
        & $python.Source -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)'
        if ($LASTEXITCODE -eq 0) {
            return [PSCustomObject]@{ Executable = $python.Source; Arguments = @() }
        }
    }
    return $null
}

if ($env:OS -ne 'Windows_NT') {
    throw '此安裝器僅支援 Windows。'
}

$nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if (-not $nvidiaSmi) {
    throw '找不到 nvidia-smi.exe。此版本需要相容的 NVIDIA GPU 與驅動程式。'
}
Invoke-Checked '檢查 NVIDIA GPU 與驅動程式' {
    & $nvidiaSmi.Source '--query-gpu=name,driver_version' '--format=csv,noheader'
}

$PythonCommand = Find-Python311
if (-not $PythonCommand -and $InstallPython) {
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw '找不到 Python 3.11，且無法使用 winget 自動安裝。請先安裝 Python 3.11.16。'
    }
    Invoke-Checked '安裝 Python 3.11.16' {
        & $winget.Source install --id Python.Python.3.11 --version 3.11.16 --exact --scope user --accept-package-agreements --accept-source-agreements
    }
    $PythonCommand = Find-Python311
}
if (-not $PythonCommand) {
    throw '找不到 Python 3.11。請安裝 Python 3.11.16 後重跑，或加上 -InstallPython 讓安裝器透過 winget 安裝。'
}

if (-not (Test-Path -LiteralPath $RuntimePython)) {
    Invoke-Checked '建立專案專用 Python 環境' {
        & $PythonCommand.Executable @($PythonCommand.Arguments) -m venv $RuntimeRoot
    }
}

Invoke-Checked '安裝鎖定版 Python 套件與 CUDA PyTorch' {
    & $RuntimePython -m pip install --requirement (Join-Path $ProjectRoot 'requirements.lock') --extra-index-url https://download.pytorch.org/whl/cu128
}

if (-not $SkipModelDownload) {
    Invoke-Checked '下載並驗證鎖定模型版本' {
        & $RuntimePython (Join-Path $ProjectRoot 'download_models.py')
    }
}

Invoke-Checked '驗證可執行環境' {
    & $RuntimePython (Join-Path $ProjectRoot 'verify_installation.py') --require-cuda
}

Write-Host "`n安裝完成。請雙擊「啟動會議逐字稿工具.cmd」。" -ForegroundColor Green
