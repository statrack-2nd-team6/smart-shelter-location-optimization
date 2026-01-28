#!/usr/bin/env python3
"""
YouTube 영상에서 description과 subtitle 추출 모듈
- yt-dlp를 사용하여 description 추출
- 자막 추출 (공식 자막 → 자동 생성 자막 → Whisper 전사)
"""

import re
from pathlib import Path
import yt_dlp


def get_video_info(youtube_video_id: str) -> dict | None:
    """
    YouTube 영상 정보 (description 포함) 추출

    Args:
        youtube_video_id: YouTube 비디오 ID

    Returns:
        dict: {"title": ..., "description": ..., "duration": ...} 또는 None
    """
    url = f"https://www.youtube.com/watch?v={youtube_video_id}"

    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
        'extract_flat': False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            return {
                'title': info.get('title', ''),
                'description': info.get('description', ''),
                'duration': info.get('duration', 0),
                'channel': info.get('channel', ''),
            }

    except Exception as e:
        print(f"  영상 정보 추출 실패 ({youtube_video_id}): {str(e)[:100]}")
        return None


def download_subtitle(youtube_video_id: str, output_dir: Path) -> Path | None:
    """
    YouTube 자막 다운로드 (공식 자막 우선, 없으면 자동 생성)

    Args:
        youtube_video_id: YouTube 비디오 ID
        output_dir: 자막 저장 디렉토리

    Returns:
        Path: 다운로드된 자막 파일 경로 또는 None
    """
    url = f"https://www.youtube.com/watch?v={youtube_video_id}"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # 이미 다운로드된 자막 확인
    existing = list(output_dir.glob(f"{youtube_video_id}.ko.*"))
    if existing:
        return existing[0]

    ydl_opts = {
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['ko'],
        'subtitlesformat': 'srt',
        'skip_download': True,
        'outtmpl': str(output_dir / youtube_video_id),
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # 다운로드된 자막 파일 찾기
        sub_files = list(output_dir.glob(f"{youtube_video_id}.ko.*"))
        if sub_files:
            return sub_files[0]

        return None

    except Exception as e:
        print(f"  자막 다운로드 실패 ({youtube_video_id}): {str(e)[:50]}")
        return None


def extract_text_from_subtitle(subtitle_path: Path) -> str:
    """
    SRT/VTT 자막 파일에서 텍스트만 추출

    Args:
        subtitle_path: 자막 파일 경로

    Returns:
        str: 추출된 텍스트
    """
    content = Path(subtitle_path).read_text(encoding='utf-8')

    lines = content.split('\n')
    text_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.isdigit():
            continue
        if '-->' in line:
            continue
        if line.startswith(('WEBVTT', 'Kind:', 'Language:')):
            continue
        # HTML 태그 제거
        line = re.sub(r'<[^>]+>', '', line)
        if line:
            text_lines.append(line)

    # 중복 제거 (순서 유지)
    seen = set()
    unique_lines = []
    for line in text_lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)

    return ' '.join(unique_lines)


def get_subtitle_text(youtube_video_id: str, output_dir: Path) -> tuple[str | None, str]:
    """
    YouTube 영상의 자막 텍스트 추출

    Args:
        youtube_video_id: YouTube 비디오 ID
        output_dir: 자막 저장 디렉토리

    Returns:
        tuple: (자막 텍스트 또는 None, 소스 타입 "subtitle" 또는 "none")
    """
    subtitle_path = download_subtitle(youtube_video_id, output_dir)

    if subtitle_path:
        text = extract_text_from_subtitle(subtitle_path)
        return text, "subtitle"

    return None, "none"


def test_extraction(youtube_video_id: str = "_UiZ-Uuw1ro"):
    """추출 테스트"""
    print(f"YouTube 영상 정보 추출 테스트: {youtube_video_id}")
    print("=" * 60)

    # Description 추출
    info = get_video_info(youtube_video_id)
    if info:
        print(f"제목: {info['title']}")
        print(f"채널: {info['channel']}")
        print(f"길이: {info['duration']}초")
        print(f"\nDescription (처음 300자):")
        print("-" * 40)
        desc_preview = info['description'][:300] + "..." if len(info['description']) > 300 else info['description']
        print(desc_preview)
    else:
        print("영상 정보 추출 실패")

    print("\n" + "=" * 60)

    # 자막 추출
    sources_dir = Path(__file__).parent / "sources"
    subtitle_text, source_type = get_subtitle_text(youtube_video_id, sources_dir)

    if subtitle_text:
        print(f"자막 추출 성공 (source: {source_type})")
        print(f"\nSubtitle (처음 300자):")
        print("-" * 40)
        sub_preview = subtitle_text[:300] + "..." if len(subtitle_text) > 300 else subtitle_text
        print(sub_preview)
    else:
        print("자막 없음 - Whisper 전사 필요")


if __name__ == "__main__":
    test_extraction()
