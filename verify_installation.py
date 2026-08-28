"""Verify that a fresh installation is ready before the GUI is launched."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from typing import Any

from download_models import load_manifest, verify_model
from meeting_transcriber import bundled_ffmpeg_executable

EXPECTED_PACKAGES = {
    "funasr": "1.4.4",
    "huggingface-hub": "1.29.0",
    "modelscope": "1.39.1",
    "torch": "2.11.0+cu128",
    "torchaudio": "2.11.0+cu128",
    "transformers": "5.16.1",
}


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in EXPECTED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description="驗證本機會議逐字稿工具安裝狀態。")
    parser.add_argument("--require-cuda", action="store_true", help="將 CUDA 不可用視為驗證失敗。")
    args = parser.parse_args()

    import torch

    versions = package_versions()
    version_errors = [
        f"{package} 應為 {expected}，目前為 {versions[package] or '未安裝'}"
        for package, expected in EXPECTED_PACKAGES.items()
        if versions[package] != expected
    ]
    manifest = load_manifest()
    model_errors = [
        error
        for name, model in manifest["models"].items()
        for error in verify_model(str(name), dict(model))
    ]
    cuda_available = torch.cuda.is_available()
    errors = version_errors + model_errors
    if not bundled_ffmpeg_executable():
        errors.append("找不到 FFmpeg。請重新執行 Install.ps1。")
    if args.require_cuda and not cuda_available:
        errors.append("找不到可用的 NVIDIA CUDA GPU 或相容驅動程式。")

    report: dict[str, Any] = {
        "ready": not errors,
        "python": sys.version.split()[0],
        "packages": versions,
        "cuda_available": cuda_available,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "ffmpeg": bundled_ffmpeg_executable(),
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
