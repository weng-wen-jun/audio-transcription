"""A small Windows GUI for local meeting transcription.

FunASR performs transcription and optional speaker diarization, then the tool
finishes after writing the local transcript files.
"""

from __future__ import annotations

import ctypes
import json
import gc
import hashlib
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
ENV = ROOT / "env"
MODEL_PATH = ROOT / "models" / "fun-asr-nano-2512"
VAD_MODEL_PATH = ROOT / "models" / "fsmn-vad"
CAMPLUS_MODEL_PATH = ROOT / "models" / "campplus"
HF_HOME = ROOT / "models" / "huggingface"
OUTPUT_ROOT = ROOT / "output"
VAD_MODEL_REPOSITORY = "funasr/fsmn-vad"
VAD_MODEL_REVISION = "df20e6b30c653645fa4ff125cacfcabd1020a669"
CAMPLUS_MODEL_REPOSITORY = "funasr/campplus"
CAMPLUS_MODEL_REVISION = "e4b6ede7ce16997aff4ae69fbca1f0175e2afede"
ASR_BLOCK_MILLISECONDS = 5 * 60 * 1000
MIN_ASR_MILLISECONDS = 3 * 1000
TARGET_ASR_MILLISECONDS = 10 * 1000
MAX_ASR_MILLISECONDS = 30 * 1000
ASR_MERGE_GAP_MILLISECONDS = 800
ASR_PAD_BEFORE_MILLISECONDS = 200
ASR_PAD_AFTER_MILLISECONDS = 300
MAX_ASR_HOTWORDS = 50
SPEAKER_BATCH_SIZE = 48
SINGLE_INSTANCE_MUTEX = r"Local\MeetingTranscriberTool-6F99E559-4D26-4D32-AF80-2A7B31506378"
_single_instance_handle: Any | None = None


def configure_environment() -> None:
    """Make the project-local cache and legacy FFmpeg location available."""
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    ffmpeg_dir = ENV / "Library" / "bin"
    os.environ["PATH"] = f"{ffmpeg_dir}{os.pathsep}{os.environ.get('PATH', '')}"


configure_environment()


def installed_model_or_repository(local_path: Path, repository: str) -> str:
    """Use the installer-managed local model, with a cache-compatible fallback."""
    return str(local_path) if local_path.is_dir() else repository


def bundled_ffmpeg_executable() -> str | None:
    """Find an FFmpeg executable from PATH or the pinned Python dependency."""
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, OSError):
        return None


def acquire_single_instance() -> bool:
    """Return false when another GUI instance is already open for this user."""
    if os.name != "nt":
        return True
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel32.CloseHandle.restype = ctypes.c_bool

    ctypes.set_last_error(0)
    handle = kernel32.CreateMutexW(None, False, SINGLE_INSTANCE_MUTEX)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
        kernel32.CloseHandle(handle)
        return False

    global _single_instance_handle
    _single_instance_handle = handle
    return True


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def timestamp_label(milliseconds: Any) -> str:
    try:
        total_seconds = int(float(milliseconds) / 1000)
    except (TypeError, ValueError):
        return "--:--:--"
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def duration_label(seconds: float | None) -> str:
    """Render a duration for the UI and saved process log."""
    if seconds is None:
        return "未知"
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours} 小時 {minutes} 分 {remaining_seconds} 秒"
    return f"{minutes} 分 {remaining_seconds} 秒"


def processing_summary_lines(summary: dict[str, Any]) -> list[str]:
    """Create a readable, persistent per-run timing summary."""
    lines = ["========== 本次處理耗時 =========="]
    lines.append(f"開始時間：{summary['started_at']:%Y-%m-%d %H:%M:%S}")
    lines.append(f"完成時間：{summary['finished_at']:%Y-%m-%d %H:%M:%S}")
    lines.append(f"音檔：{summary['audio_name']}")
    lines.append(f"音檔長度：{duration_label(summary.get('audio_duration_seconds'))}")
    lines.append(
        f"ASR：{summary['asr_segments']} 個片段／{summary['asr_work_blocks']} 個工作區塊"
    )
    if summary.get("speakers"):
        lines.append(f"CAM++ 講者特徵：{summary.get('speaker_windows', 0):,} 個視窗")
    else:
        lines.append("CAM++ 講者辨識：未啟用")
    lines.append("模組耗時：")
    for entry in summary["modules"]:
        note = f"（{entry['note']}）" if entry.get("note") else ""
        lines.append(f"- {entry['name']}：{duration_label(entry['seconds'])}{note}")
    total_seconds = summary["total_seconds"]
    lines.append(f"總處理耗時：{duration_label(total_seconds)}")
    audio_duration = summary.get("audio_duration_seconds")
    if audio_duration and total_seconds > 0:
        rtf = total_seconds / audio_duration
        speed = audio_duration / total_seconds
        lines.append(f"處理效率：RTF {rtf:.3f}（約 {speed:.1f} 倍即時）")
    lines.append("================================")
    return lines


def render_transcript(record: dict[str, Any]) -> str:
    sentences = record.get("sentence_info") or []
    if not sentences:
        return str(record.get("text", "")).strip()

    lines: list[str] = []
    for sentence in sentences:
        speaker = sentence.get("spk")
        label = f"Speaker {speaker}" if speaker is not None else "未辨識講者"
        start = timestamp_label(sentence.get("start"))
        end = timestamp_label(sentence.get("end"))
        text = str(sentence.get("text") or sentence.get("sentence") or "").strip()
        if text:
            lines.append(f"[{start}–{end}] {label}：{text}")
    return "\n".join(lines) or str(record.get("text", "")).strip()


def audio_duration_seconds(audio: Path) -> float | None:
    """Read duration without decoding the whole recording into memory."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            process = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(audio),
                ],
                check=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            duration = float(process.stdout.strip())
            return duration if duration > 0 else None
        except (OSError, subprocess.SubprocessError, ValueError):
            pass

    ffmpeg = bundled_ffmpeg_executable()
    if not ffmpeg:
        return None
    try:
        process = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", str(audio)],
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError:
        return None
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", process.stderr)
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def audio_fingerprint(audio: Path) -> str:
    """Give every source recording a stable, short local work-directory key."""
    metadata = audio.stat()
    raw = f"{audio.resolve()}\0{metadata.st_size}\0{metadata.st_mtime_ns}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def atomic_write_json(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def normalize_vad_segments(value: Any) -> list[list[int]]:
    """Validate the VAD millisecond ranges returned by FunASR."""
    normalized: list[list[int]] = []
    for segment in value or []:
        if not isinstance(segment, (list, tuple)) or len(segment) < 2:
            continue
        try:
            start, end = int(segment[0]), int(segment[1])
        except (TypeError, ValueError):
            continue
        if end > start >= 0:
            normalized.append([start, end])
    return normalized


def group_vad_segments(vad_segments: list[list[int]]) -> list[list[list[int]]]:
    """Make approximately five-minute work blocks, always cutting at VAD silence."""
    groups: list[list[list[int]]] = []
    current: list[list[int]] = []
    group_start = 0
    for segment in vad_segments:
        if current and segment[1] - group_start > ASR_BLOCK_MILLISECONDS:
            groups.append(current)
            current = []
        if not current:
            group_start = segment[0]
        current.append(segment)
    if current:
        groups.append(current)
    return groups


def asr_language_hint(language: str) -> str | None:
    """Translate the GUI choice into the natural-language Nano prompt hint."""
    return {
        "zh": "中文",
        "en": "英文",
        "ja": "日文",
        "ko": "韓文",
        "yue": "粵語",
        "auto": None,
    }.get(language, None)


def asr_hotwords(terminology: str) -> list[str]:
    """Keep only a small, de-duplicated list of user-supplied ASR hotwords."""
    words: list[str] = []
    seen: set[str] = set()
    for line in terminology.splitlines():
        word = line.strip()
        key = word.casefold()
        if not word or key in seen:
            continue
        words.append(word)
        seen.add(key)
        if len(words) == MAX_ASR_HOTWORDS:
            break
    return words


def merge_vad_segments_for_asr(vad_segments: list[list[int]]) -> list[dict[str, Any]]:
    """Merge tiny nearby VAD ranges into 3–30 second ASR inputs with padding."""
    merged: list[dict[str, Any]] = []
    current: list[list[int]] = []

    def flush() -> None:
        if not current:
            return
        start = current[0][0]
        end = current[-1][1]
        merged.append(
            {
                "start": start,
                "end": end,
                "input_start": max(0, start - ASR_PAD_BEFORE_MILLISECONDS),
                "input_end": end + ASR_PAD_AFTER_MILLISECONDS,
                "vad_segments": [list(segment) for segment in current],
            }
        )

    for segment in vad_segments:
        if not current:
            current.append(segment)
            continue

        current_start = current[0][0]
        current_end = current[-1][1]
        gap = segment[0] - current_end
        current_duration = current_end - current_start
        proposed_duration = segment[1] - current_start
        can_merge = gap <= ASR_MERGE_GAP_MILLISECONDS and proposed_duration <= MAX_ASR_MILLISECONDS
        should_merge = current_duration < MIN_ASR_MILLISECONDS or proposed_duration <= TARGET_ASR_MILLISECONDS
        if can_merge and should_merge:
            current.append(segment)
            continue

        flush()
        current = [segment]

    flush()
    return merged


def group_asr_segments(asr_segments: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Put merged ASR inputs into roughly five-minute, silence-boundary work blocks."""
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    group_start = 0
    for segment in asr_segments:
        if current and int(segment["end"]) - group_start > ASR_BLOCK_MILLISECONDS:
            groups.append(current)
            current = []
        if not current:
            group_start = int(segment["start"])
        current.append(segment)
    if current:
        groups.append(current)
    return groups


def batch_vad_segments(vad_segments: list[list[int]]) -> list[list[list[int]]]:
    """Bound padded ASR batches without ever splitting a VAD speech segment."""
    batches: list[list[list[int]]] = []
    current: list[list[int]] = []
    longest = 0
    for segment in vad_segments:
        length = segment[1] - segment[0]
        projected = max(longest, length) * (len(current) + 1)
        if current and projected > ASR_BLOCK_MILLISECONDS:
            batches.append(current)
            current = []
            longest = 0
        current.append(segment)
        longest = max(longest, length)
    if current:
        batches.append(current)
    return batches


def batch_asr_segments(asr_segments: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Bound GPU padding while retaining each merged ASR segment intact."""
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    longest = 0
    for segment in asr_segments:
        length = int(segment["input_end"]) - int(segment["input_start"])
        projected = max(longest, length) * (len(current) + 1)
        if current and projected > ASR_BLOCK_MILLISECONDS:
            batches.append(current)
            current = []
            longest = 0
        current.append(segment)
        longest = max(longest, length)
    if current:
        batches.append(current)
    return batches


def speaker_chunk_count(sample_count: int, sample_rate: int = 16000) -> int:
    """Match CAM++ ``sv_chunk`` without allocating every padded audio window."""
    chunk_length = int(1.5 * sample_rate)
    chunk_shift = int(0.75 * sample_rate)
    count = 0
    previous_end = 0
    for start in range(0, sample_count, chunk_shift):
        end = min(start + chunk_length, sample_count)
        if end <= previous_end:
            break
        previous_end = end
        count += 1
    return count


def progress_percent(completed: int, total: int) -> float:
    """Return a clamped completion percentage for a count-based task."""
    if total <= 0:
        return 0.0
    return min(100.0, max(0.0, completed * 100.0 / total))


def build_record(asr_segments: list[dict[str, Any]], asr_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Use merged ASR ranges as stable timestamps when CTC timing is unavailable."""
    if len(asr_segments) != len(asr_results):
        raise RuntimeError("語音辨識結果與 ASR 片段數量不一致，無法安全合併時間軸。")
    sentences: list[dict[str, Any]] = []
    texts: list[str] = []
    for segment, result in zip(asr_segments, asr_results):
        text = str(result.get("text", "")).strip()
        if not text:
            continue
        sentences.append({"start": segment["start"], "end": segment["end"], "text": text})
        texts.append(text)
    return {"text": " ".join(texts), "sentence_info": sentences}


def configure_torch_compatibility() -> None:
    """Avoid a Torch 2.11 duplicate-cache registration during Transformers import."""
    from torch.compiler._cache import CacheArtifactFactory

    if getattr(CacheArtifactFactory, "_meeting_tool_compatibility", False):
        return
    original_register = CacheArtifactFactory.register

    def idempotent_register(cls: type[Any], artifact_cls: type[Any]) -> type[Any]:
        if artifact_cls.type() in cls._artifact_types:
            return artifact_cls
        return original_register(artifact_cls)

    CacheArtifactFactory.register = classmethod(idempotent_register)
    CacheArtifactFactory._meeting_tool_compatibility = True


def infer_with_recovery(
    model: Any,
    inputs: list[Any],
    component: Any,
    kwargs: dict[str, Any],
    batch_size: int,
) -> list[dict[str, Any]]:
    """Retry items singly if a batched FunASR call loses any results."""
    if not inputs:
        return []

    def run(batch: list[Any]) -> list[dict[str, Any]]:
        result = model.inference(
            batch,
            model=component,
            kwargs=dict(kwargs),
            cache={},
            batch_size=max(1, min(batch_size, len(batch))),
            disable_pbar=True,
        )
        return list(result)

    results = run(inputs)
    if len(results) == len(inputs):
        return results

    recovered: list[dict[str, Any]] = []
    for item in inputs:
        single = run([item])
        if len(single) != 1:
            raise RuntimeError("Fun-ASR 無法從單一語音片段取回結果。")
        recovered.append(single[0])
    return recovered


def merge_adjacent_sentences(sentences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make VAD-level text easier to read while preserving the original timeline."""
    merged: list[dict[str, Any]] = []
    for sentence in sentences:
        if not merged:
            merged.append(dict(sentence))
            continue
        previous = merged[-1]
        same_speaker = previous.get("spk") == sentence.get("spk")
        gap = int(sentence["start"]) - int(previous["end"])
        if same_speaker and gap <= 800:
            previous["end"] = sentence["end"]
            previous["text"] = f"{previous['text']} {sentence['text']}".strip()
        else:
            merged.append(dict(sentence))
    return merged


def check_installation() -> dict[str, Any]:
    import tkinter  # Imported only for the explicit diagnostics mode.
    import torch

    return {
        "model_present": MODEL_PATH.is_dir(),
        "vad_model_present": VAD_MODEL_PATH.is_dir(),
        "campplus_model_present": CAMPLUS_MODEL_PATH.is_dir(),
        "ffmpeg_present": bundled_ffmpeg_executable() is not None,
        "tk_version": tkinter.TkVersion,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


class MeetingTranscriberApp:
    def __init__(self) -> None:
        import tkinter as tk
        from tkinter import filedialog, messagebox, scrolledtext, ttk

        self.tk = tk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.ttk = ttk
        self.scrolledtext = scrolledtext
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.run_log_lines: list[str] = []
        self.root = tk.Tk()
        self.root.title("本機會議逐字稿工具")
        self.root.minsize(780, 630)

        self.audio_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(OUTPUT_ROOT))
        self.language = tk.StringVar(value="zh")
        self.enable_speakers = tk.BooleanVar(value=True)
        self.progress_value = tk.DoubleVar(value=0)
        self.progress_stage = tk.StringVar(value="準備就緒")
        self.confirmed_progress = 0.0
        self.busy_stage: str | None = None
        self.busy_started_at: float | None = None
        self.busy_refresh_active = False
        self.is_running = False

        outer = ttk.Frame(self.root, padding=16)
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(6, weight=1)

        ttk.Label(outer, text="本機會議逐字稿", font=("Microsoft JhengHei UI", 16, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 12)
        )
        ttk.Label(outer, text="錄音檔").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(outer, textvariable=self.audio_path).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(outer, text="選取音檔", command=self.pick_audio).grid(row=1, column=2, sticky="ew")

        ttk.Label(outer, text="輸出資料夾").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(outer, textvariable=self.output_dir).grid(row=2, column=1, sticky="ew", padx=8)
        ttk.Button(outer, text="選取資料夾", command=self.pick_output).grid(row=2, column=2, sticky="ew")

        ttk.Label(outer, text="辨識語言").grid(row=3, column=0, sticky="w", pady=4)
        language_box = ttk.Combobox(
            outer,
            state="readonly",
            textvariable=self.language,
            values=("zh", "auto", "en", "ja", "ko", "yue"),
            width=12,
        )
        language_box.grid(row=3, column=1, sticky="w", padx=8)

        ttk.Checkbutton(outer, text="辨識講者（Speaker 0、Speaker 1…）", variable=self.enable_speakers).grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(8, 2)
        )
        ttk.Label(
            outer,
            text="專有名詞（每行一個；最多 50 個，提供 ASR 熱詞；可留白）",
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(10, 2))
        self.terms = scrolledtext.ScrolledText(outer, height=5, wrap=tk.WORD)
        self.terms.grid(row=6, column=0, columnspan=3, sticky="nsew")

        actions = ttk.Frame(outer)
        actions.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(12, 8))
        actions.columnconfigure(3, weight=1)
        self.start_button = ttk.Button(actions, text="開始產生", command=self.start)
        self.start_button.grid(row=0, column=0, sticky="w")
        ttk.Button(actions, text="開啟輸出資料夾", command=self.open_output).grid(row=0, column=1, padx=8)
        self.save_log_button = ttk.Button(
            actions,
            text="儲存處理紀錄",
            command=self.save_processing_log,
            state="disabled",
        )
        self.save_log_button.grid(row=0, column=2, padx=(0, 8))
        self.status = tk.StringVar(value="請選取錄音檔。")
        ttk.Label(actions, textvariable=self.status).grid(row=0, column=3, sticky="e")
        ttk.Label(actions, textvariable=self.progress_stage).grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(8, 2)
        )
        self.progressbar = ttk.Progressbar(
            actions,
            maximum=100,
            mode="determinate",
            variable=self.progress_value,
        )
        self.progressbar.grid(row=2, column=0, columnspan=4, sticky="ew")

        ttk.Label(outer, text="處理紀錄").grid(row=8, column=0, columnspan=3, sticky="w")
        self.log = scrolledtext.ScrolledText(outer, height=10, state="disabled", wrap=tk.WORD)
        self.log.grid(row=9, column=0, columnspan=3, sticky="nsew")
        outer.rowconfigure(9, weight=1)

        self.root.after(150, self.read_events)

    def pick_audio(self) -> None:
        selected = self.filedialog.askopenfilename(
            title="選取會議錄音",
            filetypes=[("音訊檔", "*.m4a *.mp3 *.wav *.flac *.aac *.ogg"), ("所有檔案", "*.*")],
        )
        if selected:
            self.audio_path.set(selected)

    def pick_output(self) -> None:
        selected = self.filedialog.askdirectory(title="選取輸出資料夾", initialdir=self.output_dir.get())
        if selected:
            self.output_dir.set(selected)

    def open_output(self) -> None:
        output = Path(self.output_dir.get()).expanduser()
        output.mkdir(parents=True, exist_ok=True)
        os.startfile(output)  # type: ignore[attr-defined]

    def append_log(self, text: str) -> None:
        line = f"{datetime.now():%H:%M:%S}  {text}"
        self.run_log_lines.append(line)
        self.log.configure(state="normal")
        self.log.insert("end", f"{line}\n")
        self.log.see("end")
        self.log.configure(state="disabled")
        self.save_log_button.configure(state="normal")

    def clear_log(self) -> None:
        self.run_log_lines.clear()
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.save_log_button.configure(state="disabled")

    def write_processing_log(self, path: Path) -> None:
        path.write_text("\n".join(self.run_log_lines) + "\n", encoding="utf-8")

    def save_processing_log(self) -> None:
        """Let the user choose whether and where to export the current log."""
        if not self.run_log_lines:
            self.messagebox.showinfo("尚無處理紀錄", "目前沒有可儲存的處理紀錄。")
            return
        output = Path(self.output_dir.get()).expanduser()
        initialdir = str(output) if output.is_dir() else str(ROOT)
        audio_name = Path(self.audio_path.get()).stem or "會議逐字稿"
        selected = self.filedialog.asksaveasfilename(
            title="儲存處理紀錄",
            initialdir=initialdir,
            initialfile=f"{audio_name}_{datetime.now():%Y%m%d-%H%M%S}_處理紀錄.txt",
            defaultextension=".txt",
            filetypes=[("文字檔", "*.txt"), ("所有檔案", "*.*")],
        )
        if not selected:
            return
        try:
            self.write_processing_log(Path(selected))
        except OSError as exc:
            self.messagebox.showerror("無法儲存處理紀錄", str(exc))
            return
        self.append_log(f"處理紀錄已手動儲存：{selected}")

    def update_progress(self, value: float, stage: str, busy: bool = False) -> None:
        """Show confirmed pipeline checkpoints and pulse for unknown-duration work."""
        self.progress_stage.set(stage)
        self.progressbar.stop()
        if busy:
            self.busy_stage = stage
            self.busy_started_at = time.monotonic()
            self.progressbar.configure(mode="indeterminate")
            self.progressbar.start(18)
            if not self.busy_refresh_active:
                self.busy_refresh_active = True
                self.root.after(1000, self.refresh_busy_progress)
            return
        self.busy_stage = None
        self.busy_started_at = None
        self.progressbar.configure(mode="determinate")
        self.confirmed_progress = value
        self.progress_value.set(value)

    def refresh_busy_progress(self) -> None:
        """Keep unknown-duration stages visibly alive without inventing a percentage."""
        if not self.busy_stage or self.busy_started_at is None:
            self.busy_refresh_active = False
            return
        elapsed = int(time.monotonic() - self.busy_started_at)
        minutes, seconds = divmod(elapsed, 60)
        self.progress_stage.set(f"{self.busy_stage}（已處理 {minutes:02}:{seconds:02}）")
        self.root.after(1000, self.refresh_busy_progress)

    def start(self) -> None:
        if self.is_running:
            return
        audio = Path(self.audio_path.get()).expanduser()
        output = Path(self.output_dir.get()).expanduser()
        if not audio.is_file():
            self.messagebox.showerror("找不到音檔", "請選取存在的錄音檔。")
            return
        if not MODEL_PATH.is_dir():
            self.messagebox.showerror("找不到模型", f"找不到 Fun-ASR-Nano 模型：\n{MODEL_PATH}")
            return
        self.is_running = True
        self.clear_log()
        self.start_button.configure(state="disabled")
        self.status.set("正在處理…")
        self.update_progress(0, "準備開始…")
        self.append_log(f"開始處理：{audio.name}")
        terminology = self.terms.get("1.0", "end").strip()
        worker = threading.Thread(
            target=self.run_pipeline,
            args=(audio, output, self.language.get(), self.enable_speakers.get(), terminology),
            daemon=True,
        )
        worker.start()

    def run_pipeline(
        self,
        audio: Path,
        output: Path,
        language: str,
        speakers: bool,
        terminology: str,
    ) -> None:
        work_dir: Path | None = None
        try:
            pipeline_started_at = time.monotonic()
            processing_started_at = datetime.now()
            timing_entries: list[dict[str, Any]] = []
            speaker_windows_total = 0

            def record_timing(name: str, started_at: float, note: str = "") -> None:
                timing_entries.append(
                    {"name": name, "seconds": time.monotonic() - started_at, "note": note}
                )

            setup_started_at = time.monotonic()
            output.mkdir(parents=True, exist_ok=True)
            audio_duration = audio_duration_seconds(audio)
            record_timing("準備與讀取音檔資訊", setup_started_at)
            self.events.put(("progress", (5, "正在載入語音辨識模型…", True)))
            self.events.put(("log", "載入 Fun-ASR-Nano 與音檔…"))
            import torch

            configure_torch_compatibility()
            from funasr import AutoModel
            from funasr.models.campplus.cluster_backend import ClusterBackend
            from funasr.models.campplus.utils import distribute_spk, postprocess, sv_chunk
            from funasr.utils.load_utils import load_audio_text_image_video

            if not torch.cuda.is_available():
                raise RuntimeError("找不到可用的 NVIDIA CUDA GPU。")
            asr_model_load_started_at = time.monotonic()
            model_options: dict[str, Any] = {
                "model": str(MODEL_PATH),
                "trust_remote_code": True,
                "remote_code": "./model.py",
                "vad_model": installed_model_or_repository(VAD_MODEL_PATH, VAD_MODEL_REPOSITORY),
                "vad_model_revision": VAD_MODEL_REVISION,
                "vad_kwargs": {"max_single_segment_time": 30000},
                "device": "cuda:0",
                "hub": "hf",
                "disable_update": True,
            }
            model = AutoModel(**model_options)
            # The GUI supplies its own progress updates instead of FunASR's
            # terminal-only progress bar.
            model.kwargs["disable_pbar"] = True
            model.vad_kwargs["disable_pbar"] = True
            record_timing("Fun-ASR-Nano 模型載入", asr_model_load_started_at)

            fingerprint = audio_fingerprint(audio)
            work_dir = output / f"{audio.stem}_{language}_逐字稿工作檔_{fingerprint}"
            work_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = work_dir / "manifest.json"
            manifest: dict[str, Any] = {}
            if manifest_path.is_file():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    manifest = {}

            vad_segments = normalize_vad_segments(manifest.get("vad_segments"))
            manifest_matches = (
                manifest.get("source_fingerprint") == fingerprint
                and manifest.get("asr_block_milliseconds") == ASR_BLOCK_MILLISECONDS
            )
            if manifest_matches and vad_segments:
                timing_entries.append(
                    {"name": "VAD 人聲偵測", "seconds": 0.0, "note": "沿用已保存的工作檔"}
                )
                self.events.put(("log", f"沿用已保存的 VAD 時間軸：{len(vad_segments)} 個語音片段。"))
            else:
                vad_started_at = time.monotonic()
                self.events.put(("progress", (12, "正在分析整份錄音的停頓位置…", True)))
                vad_result = model.inference(
                    [str(audio)],
                    model=model.vad_model,
                    kwargs=dict(model.vad_kwargs),
                    cache={},
                    batch_size=1,
                    disable_pbar=True,
                )
                vad_segments = normalize_vad_segments(vad_result[0].get("value"))
                if not vad_segments:
                    raise RuntimeError("VAD 沒有偵測到人聲片段。請確認錄音內容與音量。")
                manifest = {
                    "format": 3,
                    "source_audio": str(audio),
                    "source_fingerprint": fingerprint,
                    "asr_block_milliseconds": ASR_BLOCK_MILLISECONDS,
                    "vad_segments": vad_segments,
                }
                atomic_write_json(manifest_path, manifest)
                record_timing("VAD 人聲偵測", vad_started_at)
                self.events.put(("log", f"已找到 {len(vad_segments)} 個原始 VAD 片段；將合併短片段後再辨識。"))

            asr_prepare_started_at = time.monotonic()
            asr_segments = merge_vad_segments_for_asr(vad_segments)
            asr_work_blocks = group_asr_segments(asr_segments)
            speaker_work_blocks = group_vad_segments(vad_segments)
            if not asr_work_blocks:
                raise RuntimeError("無法建立語音辨識工作區塊。")
            asr_language = asr_language_hint(language)
            hotwords = asr_hotwords(terminology)
            manifest.update(
                {
                    "format": 3,
                    "asr_segment_settings": {
                        "min_milliseconds": MIN_ASR_MILLISECONDS,
                        "target_milliseconds": TARGET_ASR_MILLISECONDS,
                        "max_milliseconds": MAX_ASR_MILLISECONDS,
                        "merge_gap_milliseconds": ASR_MERGE_GAP_MILLISECONDS,
                        "pad_before_milliseconds": ASR_PAD_BEFORE_MILLISECONDS,
                        "pad_after_milliseconds": ASR_PAD_AFTER_MILLISECONDS,
                    },
                    "asr_language_hint": asr_language,
                    "asr_hotwords": hotwords,
                }
            )
            atomic_write_json(manifest_path, manifest)
            self.events.put(("progress", (
                18,
                f"已將 {len(vad_segments)} 個 VAD 片段合併為 {len(asr_segments)} 個 ASR 片段",
                False,
            )))
            self.events.put(("progress", (
                20,
                f"ASR 工作區塊：0/{len(asr_work_blocks)}",
                False,
            )))
            self.events.put(("log", f"ASR 共 {len(asr_work_blocks)} 個工作區塊，進度條將依完成數量前進。"))
            if asr_language:
                self.events.put(("log", f"Fun-ASR 語言已鎖定為{asr_language}；已啟用文字正規化。"))
            else:
                self.events.put(("log", "Fun-ASR 使用自動語言判斷。"))
            if hotwords:
                self.events.put(("log", f"已套用 {len(hotwords)} 個 ASR 熱詞。"))

            waveform = load_audio_text_image_video(str(audio), fs=16000)
            if not hasattr(waveform, "detach"):
                waveform = torch.as_tensor(waveform)
            waveform = waveform.detach().cpu()
            sample_count = len(waveform)
            record_timing("ASR 分段與音檔解碼", asr_prepare_started_at)

            def audio_between(start_milliseconds: int, end_milliseconds: int) -> Any:
                start = max(0, int(start_milliseconds * 16))
                end = min(sample_count, int(end_milliseconds * 16))
                return waveform[start:end]

            def asr_audio_for(segment: dict[str, Any]) -> Any:
                return audio_between(int(segment["input_start"]), int(segment["input_end"]))

            def speaker_audio_for(segment: list[int]) -> Any:
                return audio_between(segment[0], segment[1]).numpy()

            asr_kwargs = dict(model.kwargs)
            asr_kwargs["itn"] = True
            if asr_language:
                asr_kwargs["language"] = asr_language
            if hotwords:
                asr_kwargs["hotwords"] = hotwords

            asr_started_at = time.monotonic()
            reused_asr_blocks = 0
            all_asr_results: list[dict[str, Any]] = []
            for block_index, block_segments in enumerate(asr_work_blocks):
                checkpoint_path = work_dir / f"asr-block-{block_index + 1:03d}.json"
                block_results: list[dict[str, Any]] | None = None
                if checkpoint_path.is_file():
                    try:
                        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                        candidate = checkpoint.get("asr_results")
                        if checkpoint.get("asr_segments") == block_segments and isinstance(candidate, list):
                            block_results = candidate
                    except (OSError, json.JSONDecodeError):
                        block_results = None

                if block_results is None or len(block_results) != len(block_segments):
                    self.events.put(("progress", (
                        20 + 45 * block_index / len(asr_work_blocks),
                        f"ASR 工作區塊：{block_index}/{len(asr_work_blocks)}（正在辨識第 {block_index + 1} 個）",
                        False,
                    )))
                    self.events.put(("log", f"正在辨識第 {block_index + 1}/{len(asr_work_blocks)} 個工作區塊…"))
                    block_results = []
                    for batch_segments in batch_asr_segments(block_segments):
                        batch_audio = [asr_audio_for(segment) for segment in batch_segments]
                        block_results.extend(
                            json_safe(
                                infer_with_recovery(
                                    model,
                                    batch_audio,
                                    model.model,
                                    asr_kwargs,
                                    len(batch_audio),
                                )
                            )
                        )
                    if len(block_results) != len(block_segments):
                        raise RuntimeError(f"第 {block_index + 1} 個工作區塊無法完整取得辨識結果。")
                    atomic_write_json(
                        checkpoint_path,
                        {"asr_segments": block_segments, "asr_results": block_results},
                    )
                else:
                    reused_asr_blocks += 1
                    self.events.put(("log", f"沿用第 {block_index + 1}/{len(asr_work_blocks)} 個已完成的 ASR 工作區塊。"))

                all_asr_results.extend(block_results)
                self.events.put(("progress", (
                    20 + 45 * (block_index + 1) / len(asr_work_blocks),
                    f"ASR 工作區塊：{block_index + 1}/{len(asr_work_blocks)} 已完成",
                    False,
                )))

            asr_note = f"{len(asr_work_blocks)} 個工作區塊"
            if reused_asr_blocks:
                asr_note += f"，其中 {reused_asr_blocks} 個沿用工作檔"
            record_timing("Fun-ASR 語音辨識", asr_started_at, asr_note)

            record = build_record(asr_segments, all_asr_results)
            transcript = render_transcript(record)
            if not transcript:
                raise RuntimeError(
                    f"已完成 {len(asr_work_blocks)} 個 ASR 工作區塊，但沒有取得逐字稿。"
                    "請確認錄音內容確實有人聲，或改選『auto』再試一次。"
                )

            # Do not retain ASR/VAD while the separate speaker model is running.
            release_asr_started_at = time.monotonic()
            del model
            gc.collect()
            torch.cuda.empty_cache()
            record_timing("釋放 ASR GPU 記憶體", release_asr_started_at)

            if speakers:
                # The 8 GB GPU handled ASR.  Release it completely before
                # loading CAM++; neither model remains resident with the other.
                self.events.put(("progress", (67, "ASR 已完成，正在釋放 GPU 記憶體…", True)))
                self.events.put(("log", "ASR 已完成；正在釋放 Fun-ASR-Nano 與 VAD 的 GPU 記憶體。"))

                speaker_model_load_started_at = time.monotonic()
                self.events.put(("progress", (68, "正在載入 GPU CAM++ 講者模型…", True)))
                speaker_model = AutoModel(
                    model=installed_model_or_repository(CAMPLUS_MODEL_PATH, CAMPLUS_MODEL_REPOSITORY),
                    model_revision=CAMPLUS_MODEL_REVISION,
                    device="cuda:0",
                    hub="hf",
                    disable_update=True,
                )
                speaker_model.kwargs["disable_pbar"] = True
                cluster_backend = ClusterBackend().to("cpu")

                total_speaker_windows = sum(
                    speaker_chunk_count(len(audio_between(segment[0], segment[1]))) for segment in vad_segments
                )
                if total_speaker_windows <= 0:
                    raise RuntimeError("沒有足夠的 CAM++ 語音片段可供講者辨識。")
                speaker_windows_total = total_speaker_windows
                record_timing("CAM++ 模型載入與視窗計算", speaker_model_load_started_at)
                self.events.put(("log", (
                    f"CAM++ 已載入 GPU；共 {total_speaker_windows} 個 embedding 視窗，"
                    f"每批最多 {SPEAKER_BATCH_SIZE} 個。"
                )))
                self.events.put(("progress", (
                    progress_percent(0, total_speaker_windows),
                    f"CAM++ 講者特徵：0/{total_speaker_windows}",
                    False,
                )))
                speaker_windows: list[list[float]] = []
                embedding_batches: list[Any] = []
                completed_speaker_windows = 0
                speaker_embedding_started_at = time.monotonic()
                for block_index, block_segments in enumerate(speaker_work_blocks):
                    voiced_segments = [
                        [segment[0] / 1000, segment[1] / 1000, speaker_audio_for(segment)]
                        for segment in block_segments
                    ]
                    speaker_chunks = sv_chunk(voiced_segments)
                    if not speaker_chunks:
                        continue
                    speaker_windows.extend([[chunk[0], chunk[1]] for chunk in speaker_chunks])
                    block_embeddings: list[Any] = []
                    for start in range(0, len(speaker_chunks), SPEAKER_BATCH_SIZE):
                        chunk_batch = speaker_chunks[start : start + SPEAKER_BATCH_SIZE]
                        batch_end = completed_speaker_windows + len(chunk_batch)
                        self.events.put(("progress", (
                            progress_percent(completed_speaker_windows, total_speaker_windows),
                            f"CAM++ 講者特徵：{completed_speaker_windows}/{total_speaker_windows}"
                            f"（正在處理第 {completed_speaker_windows + 1}–{batch_end} 個）",
                            False,
                        )))
                        batch_started = time.monotonic()
                        speaker_results = infer_with_recovery(
                            speaker_model,
                            [chunk[2] for chunk in chunk_batch],
                            speaker_model.model,
                            speaker_model.kwargs,
                            SPEAKER_BATCH_SIZE,
                        )
                        embeddings = [item.get("spk_embedding") for item in speaker_results]
                        if len(embeddings) != len(chunk_batch) or any(item is None for item in embeddings):
                            raise RuntimeError("CAM++ 沒有完整回傳講者特徵。")
                        block_embeddings.append(torch.cat(embeddings, dim=0).detach().cpu())
                        completed_speaker_windows += len(chunk_batch)
                        elapsed = time.monotonic() - batch_started
                        self.events.put(("progress", (
                            progress_percent(completed_speaker_windows, total_speaker_windows),
                            f"CAM++ 講者特徵：{completed_speaker_windows}/{total_speaker_windows}"
                            f" 已完成（本批 {elapsed:.1f} 秒）",
                            False,
                        )))
                        self.events.put(("log", (
                            f"CAM++ embedding {completed_speaker_windows}/{total_speaker_windows}："
                            f"本批 {elapsed:.1f} 秒。"
                        )))
                    embedding_batches.append(torch.cat(block_embeddings, dim=0))

                record_timing(
                    "CAM++ 講者特徵建立",
                    speaker_embedding_started_at,
                    f"{completed_speaker_windows:,} 個 embedding 視窗",
                )

                if not speaker_windows or not embedding_batches:
                    raise RuntimeError("沒有足夠的 CAM++ 講者特徵可供全場分群。")
                speaker_cluster_started_at = time.monotonic()
                self.events.put(("progress", (82, "CAM++ embedding 已完成，正在做全場講者分群…", True)))
                speaker_embeddings = torch.cat(embedding_batches, dim=0)
                if len(speaker_windows) != speaker_embeddings.shape[0]:
                    raise RuntimeError("CAM++ 特徵與時間片段數量不一致，無法安全標記講者。")
                speaker_labels = cluster_backend(speaker_embeddings, oracle_num=None)
                speaker_timeline = postprocess(
                    speaker_windows,
                    [],
                    speaker_labels,
                    speaker_embeddings.numpy(),
                )
                distribute_spk(record["sentence_info"], speaker_timeline)
                record_timing("全場講者分群與回填", speaker_cluster_started_at)
                del speaker_model, cluster_backend
                gc.collect()
                torch.cuda.empty_cache()
            else:
                timing_entries.append({"name": "講者辨識", "seconds": 0.0, "note": "未啟用"})

            save_started_at = time.monotonic()
            record["sentence_info"] = merge_adjacent_sentences(record["sentence_info"])
            transcript = render_transcript(record)

            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            base = f"{audio.stem}_{stamp}"
            transcript_json = output / f"{base}_逐字稿.json"
            transcript_text = output / f"{base}_逐字稿.txt"
            self.events.put(("progress", (86, "逐字稿已完成，正在儲存檔案…")))
            transcript_json.write_text(
                json.dumps(
                    {
                        "source_audio": str(audio),
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                        "model": "Fun-ASR-Nano-2512",
                        "speaker_diarization": speakers,
                        "speaker_diarization_scope": "global" if speakers else "disabled",
                        "vad_segments": len(vad_segments),
                        "asr_segments": len(asr_segments),
                        "asr_work_blocks": len(asr_work_blocks),
                        "asr_language_hint": asr_language,
                        "asr_hotwords": hotwords,
                        "asr_segment_settings": manifest["asr_segment_settings"],
                        "work_directory": str(work_dir),
                        "result": record,
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            transcript_text.write_text(transcript, encoding="utf-8")
            record_timing("整理並儲存逐字稿", save_started_at)
            self.events.put(("progress", (90, "逐字稿已儲存")))
            self.events.put(("log", f"逐字稿已完成：{transcript_text.name}"))

            processing_finished_at = datetime.now()
            summary = {
                "started_at": processing_started_at,
                "finished_at": processing_finished_at,
                "audio_name": audio.name,
                "audio_duration_seconds": audio_duration,
                "asr_segments": len(asr_segments),
                "asr_work_blocks": len(asr_work_blocks),
                "speakers": speakers,
                "speaker_windows": speaker_windows_total,
                "modules": timing_entries,
                "total_seconds": time.monotonic() - pipeline_started_at,
            }
            self.events.put(("done", (transcript_json, transcript_text, work_dir, summary)))
        except Exception as exc:
            detail = str(exc)
            if work_dir:
                detail += f"\n已保留已完成的工作區塊：\n{work_dir}"
            self.events.put(("error", detail))

    def read_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                try:
                    if event == "progress":
                        value, stage, busy = payload
                        self.update_progress(float(value), str(stage), bool(busy))
                    elif event == "log":
                        self.append_log(str(payload))
                    elif event == "done":
                        json_path, text_path, work_dir, summary = payload
                        self.is_running = False
                        self.start_button.configure(state="normal")
                        self.status.set("完成")
                        self.update_progress(100, "處理完成")
                        self.append_log("全部完成。")
                        for line in processing_summary_lines(summary):
                            self.append_log(line)
                        result_lines = [
                            f"逐字稿：\n{text_path}",
                            f"詳細 JSON：\n{json_path}",
                            f"工作檔（可斷點續跑）：\n{work_dir}",
                        ]
                        self.messagebox.showinfo("處理完成", "\n\n".join(result_lines))
                    elif event == "error":
                        self.is_running = False
                        self.start_button.configure(state="normal")
                        self.status.set("失敗")
                        self.update_progress(self.confirmed_progress, "處理失敗")
                        self.append_log(f"失敗：{payload}")
                        self.messagebox.showerror("處理失敗", str(payload))
                except Exception as exc:
                    self.append_log(f"介面更新警告（{event}）：{exc}")
        except queue.Empty:
            pass
        finally:
            self.root.after(150, self.read_events)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    if "--check" in sys.argv:
        print(json.dumps(check_installation(), ensure_ascii=False, indent=2))
        return
    if not acquire_single_instance():
        ctypes.windll.user32.MessageBoxW(
            None,
            "本機會議逐字稿工具已經開啟。請使用原本的視窗。",
            "本機會議逐字稿工具",
            0x30,
        )
        return
    MeetingTranscriberApp().run()


if __name__ == "__main__":
    main()
