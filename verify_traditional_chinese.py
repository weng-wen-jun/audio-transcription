"""Verify Traditional Chinese transcript export without loading ASR models."""

from __future__ import annotations

from meeting_transcriber import render_transcript, traditional_chinese_output_record


def main() -> None:
    original = {
        "text": "对这个会议的审查意见，会影响软件和视频质量。",
        "sentence_info": [
            {"start": 0, "end": 1200, "spk": 2, "text": "对这个会议的审查意见。"},
            {"start": 1300, "end": 2500, "spk": 2, "text": "软件和视频质量要确认。"},
        ],
    }
    converted = traditional_chinese_output_record(original)

    assert original["text"] == "对这个会议的审查意见，会影响软件和视频质量。"
    assert converted["text"] == "對這個會議的審查意見，會影響軟體和影片質量。"
    assert converted["sentence_info"][0]["text"] == "對這個會議的審查意見。"
    assert converted["sentence_info"][1]["text"] == "軟體和影片質量要確認。"
    assert render_transcript(converted) == (
        "[00:00:00–00:00:01] Speaker 2：對這個會議的審查意見。\n"
        "[00:00:01–00:00:02] Speaker 2：軟體和影片質量要確認。"
    )

    fallback = traditional_chinese_output_record({"text": "这是一份会议记录。"})
    assert render_transcript(fallback) == "這是一份會議記錄。"
    print("Traditional Chinese transcript export verification passed.")


if __name__ == "__main__":
    main()
