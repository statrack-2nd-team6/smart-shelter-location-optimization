---
name: transcribe
description: Convert audio and video files to Korean transcripts using Whisper. Use when asked to transcribe, extract speech/text from media files, convert audio/video to text, or generate transcripts. Supports files in ./sources/ directory with output to ./output/. Works with .mp4, .avi, .mov, .mkv, .webm videos and .mp3, .wav, .m4a, .aac, .ogg, .flac audio files.
---

# Transcribe

Convert audio and video files to Korean text transcripts using OpenAI Whisper.

## Quick Start

Run the transcription script with one or more media files:

```bash
.venv/bin/python .claude/skills/transcribe/scripts/transcribe.py sources/audio.m4a
```

The script:
1. Extracts audio from video files (skips for audio files)
2. Transcribes using Whisper with Korean language setting
3. Generates filename from transcript content + timestamp
4. Saves to `./output/` directory

## Workflow

### Single File Transcription

```bash
.venv/bin/python .claude/skills/transcribe/scripts/transcribe.py sources/[FILE]
```

Example:
- User: "sources/audio.m4a를 전사해줘"
- Action: Run script with default `base` model

### Multiple Files

```bash
.venv/bin/python .claude/skills/transcribe/scripts/transcribe.py sources/file1.m4a sources/file2.mp4
```

Example:
- User: "sources/ 폴더의 모든 m4a 파일을 전사해줘"
- Action: Find all .m4a files, pass to script

### Custom Model

Use `--model` flag to specify Whisper model (tiny, base, small, medium, large):

```bash
.venv/bin/python .claude/skills/transcribe/scripts/transcribe.py sources/video.mp4 --model tiny
```

**Model selection:**
- `tiny`: Fastest, less accurate (good for testing, short files)
- `base`: Default, balanced speed/accuracy
- `small`: Better accuracy, slower
- `medium`/`large`: Best accuracy, very slow

Example:
- User: "tiny 모델로 sources/video.mp4를 전사해줘"
- Action: Run with `--model tiny`

### File Duration Considerations

Check file duration before transcription to set user expectations:

```bash
ffmpeg -i sources/[FILE] 2>&1 | grep Duration
```

**Rough processing time estimates (base model):**
- 2-5 min audio: ~30 seconds
- 10-20 min audio: ~1-2 minutes
- 1+ hour audio: ~5-10 minutes

For long files (>30 min), inform user of expected wait time and suggest using `tiny` model for faster results.

## Output Format

Generated files are saved to `./output/` with naming pattern:

```
[first 30 chars of transcript]_YYYYMMDD_HHMMSS.txt
```

Example: `네 엉덕션을 미야 사는 신의 순놈을_20260108_183045.txt`

## Supported File Types

**Video:** .mp4, .avi, .mov, .mkv, .flv, .wmv, .webm
**Audio:** .mp3, .wav, .m4a, .aac, .ogg, .flac, .wma

## Scripts

### transcribe.py

Main script for transcription workflow. Handles:
- Video to audio extraction (ffmpeg)
- Whisper transcription
- Output file naming and saving

Usage:
```bash
.venv/bin/python .claude/skills/transcribe/scripts/transcribe.py <files...> [--model MODEL] [--output-dir DIR]
```

### extract_media_to_text.py

Lower-level utility for single file transcription. Use `transcribe.py` instead for standard workflows.
