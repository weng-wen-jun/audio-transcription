"""Download the exact model revisions required by the transcription tool.

Weights are intentionally excluded from Git.  This program downloads them from
their upstream Hugging Face repositories and verifies the release manifest
before the GUI is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "model_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict[str, Any]:
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"無法讀取模型清單：{MANIFEST_PATH}\n{exc}") from exc
    if manifest.get("format") != 1 or not isinstance(manifest.get("models"), dict):
        raise RuntimeError("模型清單格式不正確。")
    return manifest


def verify_model(name: str, model: dict[str, Any]) -> list[str]:
    destination = ROOT / str(model["local_dir"])
    errors: list[str] = []
    for relative_path, expected_hash in dict(model["artifacts"]).items():
        artifact = destination / str(relative_path)
        if not artifact.is_file():
            errors.append(f"{name}: 缺少 {relative_path}")
        elif sha256(artifact).lower() != str(expected_hash).lower():
            errors.append(f"{name}: {relative_path} 的 SHA-256 不符合鎖定值")
    return errors


def download_model(name: str, model: dict[str, Any]) -> None:
    from huggingface_hub import snapshot_download

    destination = ROOT / str(model["local_dir"])
    print(f"下載 {name}：{model['repository']} @ {model['revision']}")
    snapshot_download(
        repo_id=str(model["repository"]),
        revision=str(model["revision"]),
        local_dir=destination,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="下載或驗證本工具鎖定的模型版本。")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="不連網下載，只檢查本機模型檔與 SHA-256。",
    )
    args = parser.parse_args()
    manifest = load_manifest()
    models = manifest["models"]

    if not args.verify_only:
        for name, model in models.items():
            download_model(str(name), dict(model))

    errors = [error for name, model in models.items() for error in verify_model(str(name), dict(model))]
    if errors:
        raise RuntimeError("模型驗證失敗：\n- " + "\n- ".join(errors))
    print("模型下載與 SHA-256 驗證完成。")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"失敗：{exc}", file=sys.stderr)
        raise SystemExit(1)
