@echo off
setlocal
set "ROOT=%~dp0"
set "HF_HOME=%ROOT%\models\huggingface"
set "PATH=%ROOT%\runtime\Library\bin;%ROOT%\env\Library\bin;%PATH%"
set "PYTHONW=%ROOT%\runtime\Scripts\pythonw.exe"
if not exist "%PYTHONW%" set "PYTHONW=%ROOT%\env\Scripts\pythonw.exe"
if not exist "%PYTHONW%" set "PYTHONW=%ROOT%\env\pythonw.exe"
if not exist "%PYTHONW%" (
    echo 找不到已安裝的執行環境。
    echo 請先在此資料夾執行：powershell -ExecutionPolicy Bypass -File .\Install.ps1
    pause
    exit /b 1
)
start "" /b "%PYTHONW%" "%ROOT%\meeting_transcriber.py"
