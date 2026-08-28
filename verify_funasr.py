"""Run a local GPU smoke test for the installed FunASR meeting-transcription stack."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch
from funasr import AutoModel

from meeting_transcriber import (
    CAMPLUS_MODEL_PATH,
    CAMPLUS_MODEL_REPOSITORY,
    CAMPLUS_MODEL_REVISION,
    MODEL_PATH,
    APP_VERSION,
    VAD_MODEL_PATH,
    VAD_MODEL_REPOSITORY,
    VAD_MODEL_REVISION,
    bundled_ffmpeg_executable,
    installed_model_or_repository,
)

ROOT = Path(__file__).resolve().parent
RESULT_PATH = ROOT / "funasr-verification.json"


def main() -> None:
    os.environ.setdefault("HF_HOME", str(ROOT / "models" / "huggingface"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    model = AutoModel(
        model=str(MODEL_PATH),
        trust_remote_code=True,
        remote_code="./model.py",
        vad_model=installed_model_or_repository(VAD_MODEL_PATH, VAD_MODEL_REPOSITORY),
        vad_model_revision=VAD_MODEL_REVISION,
        vad_kwargs={"max_single_segment_time": 30000},
        spk_model=installed_model_or_repository(CAMPLUS_MODEL_PATH, CAMPLUS_MODEL_REPOSITORY),
        spk_model_revision=CAMPLUS_MODEL_REVISION,
        device="cuda:0",
        hub="hf",
        disable_update=True,
    )
    sample = MODEL_PATH / "example" / "zh.mp3"
    record = model.generate(
        input=[str(sample)], cache={}, batch_size=1, language="中文"
    )[0]
    text = record.get("text", "")
    sentences = record.get("sentence_info", [])
    speakers = sorted(
        {str(sentence["spk"]) for sentence in sentences if sentence.get("spk") is not None}
    )
    RESULT_PATH.write_text(
        json.dumps(
            {
                "status": "passed",
                "application_version": f"V{APP_VERSION}",
                "torch": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(0),
                "ffmpeg_available": bundled_ffmpeg_executable() is not None,
                "sample": sample.name,
                "transcript_characters": len(text),
                "diarized_sentences": len(sentences),
                "speaker_labels": speakers,
                "transcript_preview": text[:120],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
