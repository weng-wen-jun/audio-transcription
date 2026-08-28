# 本機會議逐字稿工具

Windows 上使用 NVIDIA CUDA GPU 的本機逐字稿工具。它以 Fun-ASR-Nano-2512 做語音辨識，並以 FSMN-VAD 與 CAM++ 做語音分段及講者辨識。

本專案可直接放在任意資料夾；不依賴固定的本機磁碟路徑。

## 系統需求

- Windows 10 或 Windows 11。
- 相容的 NVIDIA GPU 與已安裝驅動程式；此版本不提供 CPU 執行模式。
- 網路連線，供首次安裝 Python 套件與約 1.9 GB 的 ASR 模型。
- Python 3.11.16。若電腦有 winget，安裝器可在加上 `-InstallPython` 時安裝它。

## 安裝與啟動

在專案資料夾開啟 PowerShell，執行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\Install.ps1 -InstallPython
```

已安裝 Python 3.11.16 時，可省略 `-InstallPython`。安裝器會：

1. 檢查 NVIDIA GPU 與驅動程式。
2. 建立專案內的 `runtime` Python 環境。
3. 安裝 [requirements.lock](requirements.lock) 鎖定的套件版本與 CUDA 12.8 PyTorch。
4. 從上游 Hugging Face 下載鎖定的模型 commit，並驗證權重 SHA-256。
5. 驗證 CUDA、FFmpeg、套件與所有必要模型。

成功後雙擊 [啟動會議逐字稿工具.cmd](啟動會議逐字稿工具.cmd)。啟動器會依自己的所在位置尋找環境，因此 clone 或解壓縮到其他磁碟也能使用。

若只想驗證已安裝的模型而不下載，請執行：

```powershell
.\runtime\Scripts\python.exe .\download_models.py --verify-only
.\runtime\Scripts\python.exe .\verify_installation.py --require-cuda
```

完整 GPU 冒煙測試（會使用上游提供的範例音檔）可執行：

```powershell
.\runtime\Scripts\python.exe .\verify_funasr.py
```

## 模型與可重現性

模型權重、Python 環境、輸出逐字稿與錄音檔都被 `.gitignore` 排除，不會推進一般 GitHub repository。模型會由 [model_manifest.json](model_manifest.json) 從原始 Hugging Face repository 下載；其中保存了不可變 commit 與必要權重檔的 SHA-256。

| 元件 | 來源 | 鎖定 revision | 已驗證權重 |
| --- | --- | --- | --- |
| Fun-ASR-Nano-2512 | `FunAudioLLM/Fun-ASR-Nano-2512` | `272c57b82523ada6fd87095e955f8e29100979ab` | `model.pt` |
| FSMN-VAD | `funasr/fsmn-vad` | `df20e6b30c653645fa4ff125cacfcabd1020a669` | `model.pt` |
| CAM++ | `funasr/campplus` | `e4b6ede7ce16997aff4ae69fbca1f0175e2afede` | `campplus_cn_common.bin` |

三個模型的本機模型卡都標示 `apache-2.0`；發布前仍應再次確認上游模型卡與依賴套件授權條款。請不要將權重直接放進一般 Git repository。

## 日誌與輸出

逐字稿與 JSON 仍會寫入使用者指定的輸出資料夾。處理紀錄不再自動產生；需要時請在程式中按「儲存處理紀錄」，自行選擇檔名與位置。

## 發布前驗收

在尚未安裝 Python、沒有既有 Hugging Face 快取的乾淨 Windows 使用者環境中，重新跑一次安裝流程與完整 GPU 冒煙測試，才可宣稱可交付給其他使用者。本次修改已完成本機腳本與鎖定資料的靜態驗證，但尚未進行該乾淨電腦實測。
