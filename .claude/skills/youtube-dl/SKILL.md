---
name: youtube-dl
description: Download YouTube videos and audio using yt-dlp. Use when asked to download YouTube videos, save YouTube content, extract audio from YouTube, or download playlists. Supports saving to ./sources/ or ./output/ directories. Works with single videos, playlists, and audio-only extraction.
---

# YouTube Download

Download YouTube videos and audio content using yt-dlp.

## Quick Start

Download a YouTube video to the sources directory:

```bash
.venv/bin/python .claude/skills/youtube-dl/scripts/download_youtube.py "[YOUTUBE_URL]"
```

The script:
1. Downloads the video in best available quality
2. Saves to `./sources/` directory (default)
3. Uses the video title as filename
4. Shows download progress in real-time

## Workflow

### Single Video Download

```bash
.venv/bin/python .claude/skills/youtube-dl/scripts/download_youtube.py "https://www.youtube.com/watch?v=..."
```

Example:
- User: "https://www.youtube.com/watch?v=uzB2_LC6fNs 를 다운로드해줘"
- Action: Run script with default settings (video, sources/)

### Audio-Only Download

Use `--audio-only` flag to extract only audio in m4a format:

```bash
.venv/bin/python .claude/skills/youtube-dl/scripts/download_youtube.py "https://www.youtube.com/watch?v=..." --audio-only
```

Example:
- User: "이 유튜브 영상에서 오디오만 추출해줘"
- Action: Run with `--audio-only` flag

**Benefits of audio-only:**
- Smaller file size
- Faster download
- Ready for transcription with the transcribe skill

### Custom Output Directory

Use `--output-dir` to specify save location:

```bash
.venv/bin/python .claude/skills/youtube-dl/scripts/download_youtube.py "https://www.youtube.com/watch?v=..." --output-dir output
```

Example:
- User: "output 폴더에 저장해줘"
- Action: Run with `--output-dir output`

**Common directories:**
- `./sources/` - For media to be processed later (default)
- `./output/` - For final output files

### Playlist Download

Use `--playlist` flag to download entire playlists:

```bash
.venv/bin/python .claude/skills/youtube-dl/scripts/download_youtube.py "https://www.youtube.com/playlist?list=..." --playlist
```

Example:
- User: "이 재생목록 전체를 다운로드해줘"
- Action: Run with `--playlist` flag

Files are numbered: `1_title.webm`, `2_title.webm`, etc.

### Combined Options

Download playlist audio-only to output directory:

```bash
.venv/bin/python .claude/skills/youtube-dl/scripts/download_youtube.py "https://www.youtube.com/playlist?list=..." --audio-only --playlist --output-dir output
```

## Output Format

**Video downloads:**
- Format: Best available quality (usually webm or mp4)
- Filename: `[Video Title].[ext]`
- Example: `자막뉴스 우리가 소심했나 李 후회.webm`

**Audio downloads:**
- Format: m4a (compatible with most players and transcription tools)
- Filename: `[Video Title].m4a`
- Example: `자막뉴스 우리가 소심했나 李 후회.m4a`

**Playlist downloads:**
- Filename: `[Index]_[Video Title].[ext]`
- Example: `1_First Video.webm`, `2_Second Video.webm`

## Integration with Transcribe Skill

After downloading audio, use the transcribe skill:

```bash
# 1. Download audio
.venv/bin/python .claude/skills/youtube-dl/scripts/download_youtube.py "[URL]" --audio-only --output-dir sources

# 2. Transcribe
.venv/bin/python .claude/skills/transcribe/scripts/transcribe.py sources/[filename].m4a
```

Example workflow:
- User: "이 유튜브 영상을 전사해줘: https://www.youtube.com/watch?v=..."
- Actions:
  1. Download audio: `--audio-only --output-dir sources`
  2. Transcribe: Use transcribe skill on downloaded file

## Common Issues

**JavaScript runtime warning:**
- Warning about missing JavaScript runtime is normal
- Downloads work correctly despite the warning
- To suppress: Install deno runtime (optional)

**SABR streaming warnings:**
- Some format warnings are expected
- Best available format is still downloaded
- No action needed

**File already exists:**
- yt-dlp will skip if file exists with same name
- Delete or rename existing file to re-download

## Script Reference

### download_youtube.py

Main download script with full control over download options.

Arguments:
- `url` (required): YouTube URL
- `--output-dir`: Save directory (default: ./sources)
- `--audio-only`: Extract audio only (m4a)
- `--playlist`: Download entire playlist

Usage:
```bash
.venv/bin/python .claude/skills/youtube-dl/scripts/download_youtube.py <url> [options]
```
