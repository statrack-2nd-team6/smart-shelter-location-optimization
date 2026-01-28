#!/usr/bin/env python3
"""
YouTube 영상 텍스트 추출 스크립트 (3단계 폴백)

YouTube 영상에서 텍스트를 추출합니다.
우선순위에 따라 가장 효율적인 방법을 자동 선택합니다.

추출 우선순위:
    1. 공식 자막 (수동 업로드) - 가장 정확, 가장 빠름
    2. 자동 생성 자막 (YouTube AI) - 빠름, 정확도 보통
    3. Whisper 전사 (오디오 다운로드 필요) - 느림, 정확도 모델에 따라 다름

사용법:
    python extract_youtube_subtitle.py VIDEO_ID
    python extract_youtube_subtitle.py VIDEO_ID1 VIDEO_ID2 --model base
    python extract_youtube_subtitle.py https://www.youtube.com/watch?v=VIDEO_ID

출력:
    ./output/{youtube_video_id}.txt - 추출된 텍스트
    ./output/youtube_data.csv - CSV 파일 (youtube_video_id, description, subtitle)
    ./sources/{youtube_video_id}.ko.srt - 자막 파일 (자막 사용 시)
    ./sources/audio_{youtube_video_id}.m4a - 오디오 파일 (Whisper 사용 시)

필요 패키지:
    pip install yt-dlp openai-whisper
"""

import sys
import re
import csv
import argparse
from pathlib import Path

# ============================================================
# 패키지 확인
# ============================================================

try:
    import yt_dlp
except ImportError:
    print("오류: yt-dlp가 설치되어 있지 않습니다.")
    print("설치: pip install yt-dlp")
    sys.exit(1)

try:
    import whisper
except ImportError:
    print("오류: openai-whisper가 설치되어 있지 않습니다.")
    print("설치: pip install openai-whisper")
    sys.exit(1)


# ============================================================
# 설정
# ============================================================

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent

# 자막/오디오 저장 디렉토리
SOURCES_DIR = PROJECT_ROOT / "sources"

# 텍스트 출력 디렉토리
OUTPUT_DIR = PROJECT_ROOT / "output"

# 자막 언어 우선순위
SUBTITLE_LANGS = ['ko', 'en']


# ============================================================
# 유틸리티 함수
# ============================================================

def extract_youtube_video_id(url_or_id: str) -> str:
    """
    URL 또는 youtube_video_id에서 youtube_video_id만 추출

    예시:
        "dQw4w9WgXcQ" → "dQw4w9WgXcQ"
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ" → "dQw4w9WgXcQ"
        "https://youtu.be/dQw4w9WgXcQ" → "dQw4w9WgXcQ"
    """
    if len(url_or_id) == 11 and not url_or_id.startswith('http'):
        return url_or_id

    patterns = [
        r'(?:v=|/)([0-9A-Za-z_-]{11}).*',
        r'(?:youtu\.be/)([0-9A-Za-z_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)

    return url_or_id


def get_video_description(youtube_video_id: str) -> str | None:
    """
    YouTube 영상의 description을 추출

    Args:
        youtube_video_id: YouTube 비디오 ID

    Returns:
        str: 영상 설명, 실패 시 None
    """
    url = f"https://www.youtube.com/watch?v={youtube_video_id}"

    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            description = info.get('description', '')
            return description if description else None
    except Exception as e:
        print(f"    [실패] Description 추출 오류: {str(e)[:60]}")
        return None


def extract_text_from_subtitle(subtitle_path: Path) -> str:
    """
    SRT/VTT 자막 파일에서 순수 텍스트만 추출

    제거 항목: 시퀀스 번호, 타임스탬프, HTML 태그, 중복 라인
    """
    content = subtitle_path.read_text(encoding='utf-8')
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


# ============================================================
# 1단계: 공식 자막 / 자동생성 자막 다운로드
# ============================================================

def download_subtitle(youtube_video_id: str, sources_dir: Path) -> Path | None:
    """
    YouTube 자막 다운로드

    동작:
        1. 공식 자막(수동 업로드) 확인 → 있으면 다운로드
        2. 없으면 자동 생성 자막 다운로드 시도
        3. 둘 다 없으면 None 반환

    Returns:
        Path: 자막 파일 경로, 없으면 None
    """
    url = f"https://www.youtube.com/watch?v={youtube_video_id}"

    # 이미 다운로드된 자막 확인
    for lang in SUBTITLE_LANGS:
        existing = list(sources_dir.glob(f"{youtube_video_id}.{lang}.*"))
        if existing:
            print(f"    [캐시] 기존 자막: {existing[0].name}")
            return existing[0]

    ydl_opts = {
        'writesubtitles': True,        # 공식 자막
        'writeautomaticsub': True,     # 자동 생성 자막
        'subtitleslangs': SUBTITLE_LANGS,
        'subtitlesformat': 'srt',
        'skip_download': True,
        'outtmpl': str(sources_dir / youtube_video_id),
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        for lang in SUBTITLE_LANGS:
            sub_files = list(sources_dir.glob(f"{youtube_video_id}.{lang}.*"))
            if sub_files:
                print(f"    [다운로드] 자막: {sub_files[0].name}")
                return sub_files[0]

        return None

    except Exception as e:
        print(f"    [실패] 자막 다운로드 오류: {str(e)[:60]}")
        return None


# ============================================================
# 2단계: 오디오 다운로드
# ============================================================

def download_audio(youtube_video_id: str, sources_dir: Path) -> Path | None:
    """
    YouTube에서 오디오만 다운로드 (Whisper 전사용)

    Returns:
        Path: 오디오 파일 경로, 실패 시 None
    """
    url = f"https://www.youtube.com/watch?v={youtube_video_id}"
    output_template = str(sources_dir / f"audio_{youtube_video_id}.%(ext)s")

    # 이미 다운로드된 오디오 확인
    audio_extensions = ['.m4a', '.mp3', '.webm', '.opus', '.ogg']
    for ext in audio_extensions:
        existing = sources_dir / f"audio_{youtube_video_id}{ext}"
        if existing.exists():
            print(f"    [캐시] 기존 오디오: {existing.name}")
            return existing

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        for ext in audio_extensions:
            audio_file = sources_dir / f"audio_{youtube_video_id}{ext}"
            if audio_file.exists():
                print(f"    [다운로드] 오디오: {audio_file.name}")
                return audio_file

        return None

    except Exception as e:
        print(f"    [실패] 오디오 다운로드 오류: {str(e)[:60]}")
        return None


# ============================================================
# 3단계: Whisper 전사
# ============================================================

def transcribe_with_whisper(audio_path: Path, model) -> str | None:
    """
    Whisper를 사용하여 오디오를 텍스트로 변환

    Args:
        audio_path: 오디오 파일 경로
        model: 로드된 Whisper 모델

    Returns:
        str: 전사된 텍스트, 실패 시 None
    """
    try:
        print(f"    [Whisper] 전사 중... (시간이 걸릴 수 있습니다)")
        result = model.transcribe(str(audio_path), language='ko', fp16=False)
        return result['text'].strip()
    except Exception as e:
        print(f"    [실패] Whisper 전사 오류: {str(e)[:60]}")
        return None


# ============================================================
# 통합 처리 함수
# ============================================================

def process_video(youtube_video_id: str, sources_dir: Path, output_dir: Path, whisper_model) -> dict:
    """
    단일 비디오 처리 (3단계 폴백)

    처리 순서:
        1. Description 추출
        2. 캐시된 텍스트 확인
        3. 자막 다운로드 시도 (공식 → 자동생성)
        4. 자막 없으면 오디오 다운로드 후 Whisper 전사

    Returns:
        dict: {youtube_video_id, description, subtitle, status, source_type, text_file}
    """
    result = {
        'youtube_video_id': youtube_video_id,
        'description': None,
        'subtitle': None,
        'status': 'failed',
        'source_type': None,  # 'subtitle', 'whisper', 'cached'
        'text_file': None,
    }

    text_path = output_dir / f"{youtube_video_id}.txt"

    # 0. Description 추출 (자막 추출 전에 먼저 실행)
    print(f"  [0단계] Description 추출 중...")
    description = get_video_description(youtube_video_id)
    if description:
        result['description'] = description
        print(f"  [성공] Description 추출 완료 ({len(description)}자)")
    else:
        result['description'] = ''
        print(f"  [경고] Description 없음")

    # 1. 캐시된 텍스트 확인
    if text_path.exists():
        print(f"  [캐시] 기존 텍스트 사용")
        with open(text_path, 'r', encoding='utf-8') as f:
            result['subtitle'] = f.read()
        result['status'] = 'success'
        result['source_type'] = 'cached'
        result['text_file'] = text_path
        return result

    text = None

    # 2. 자막 시도 (공식 자막 → 자동생성 자막)
    print(f"  [1단계] 자막 확인 중...")
    subtitle_path = download_subtitle(youtube_video_id, sources_dir)

    if subtitle_path:
        text = extract_text_from_subtitle(subtitle_path)
        result['source_type'] = 'subtitle'
        print(f"  [성공] 자막에서 텍스트 추출 완료")

    # 3. 자막 없으면 Whisper 전사
    if not text:
        print(f"  [2단계] 자막 없음 → Whisper 전사 시도...")
        audio_path = download_audio(youtube_video_id, sources_dir)

        if audio_path:
            text = transcribe_with_whisper(audio_path, whisper_model)
            if text:
                result['source_type'] = 'whisper'
                print(f"  [성공] Whisper 전사 완료")

    # 4. 텍스트 저장
    if text:
        result['subtitle'] = text
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text)
        result['status'] = 'success'
        result['text_file'] = text_path
        print(f"  [저장] {text_path.name}")
    else:
        result['subtitle'] = ''
        print(f"  [실패] 텍스트 추출 불가")

    return result


# ============================================================
# CSV 저장 함수
# ============================================================

def save_results_to_csv(results: list, output_dir: Path) -> Path:
    """
    결과를 CSV 파일로 저장

    Args:
        results: 처리 결과 리스트
        output_dir: 출력 디렉토리

    Returns:
        Path: 저장된 CSV 파일 경로
    """
    csv_path = output_dir / "youtube_data.csv"

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        # 헤더 작성
        writer.writerow(['youtube_video_id', 'description', 'subtitle'])

        # 데이터 작성
        for r in results:
            writer.writerow([
                r['youtube_video_id'],
                r['description'] or '',
                r['subtitle'] or ''
            ])

    return csv_path


# ============================================================
# 메인 함수
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='YouTube 영상에서 텍스트 추출 (자막 우선, 없으면 Whisper)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python extract_youtube_subtitle.py dQw4w9WgXcQ
  python extract_youtube_subtitle.py VIDEO_ID1 VIDEO_ID2 --model base
  python extract_youtube_subtitle.py "https://www.youtube.com/watch?v=VIDEO_ID"

추출 우선순위:
  1. 공식 자막 (가장 정확, 가장 빠름)
  2. 자동 생성 자막 (빠름)
  3. Whisper 전사 (느림, 오디오 다운로드 필요)
        """
    )

    parser.add_argument(
        'videos',
        nargs='+',
        help='YouTube 비디오 ID 또는 URL'
    )

    parser.add_argument(
        '--model', '-m',
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper 모델 (기본: base, 자막 없을 때만 사용)'
    )

    parser.add_argument(
        '--sources-dir',
        type=Path,
        default=SOURCES_DIR,
        help=f'자막/오디오 저장 디렉토리 (기본: {SOURCES_DIR})'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=OUTPUT_DIR,
        help=f'텍스트 저장 디렉토리 (기본: {OUTPUT_DIR})'
    )

    parser.add_argument(
        '--csv-output',
        type=Path,
        default=None,
        help='CSV 출력 파일 경로 (기본: {output_dir}/youtube_data.csv)'
    )

    args = parser.parse_args()

    # 디렉토리 생성
    args.sources_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("YouTube 텍스트 추출기 (자막 우선 → Whisper 폴백)")
    print("=" * 60)
    print(f"Whisper 모델: {args.model} (자막 없을 때만 사용)")
    print(f"자막/오디오 저장: {args.sources_dir}")
    print(f"텍스트 저장: {args.output_dir}")
    print(f"처리할 영상: {len(args.videos)}개")
    print("=" * 60)

    # Whisper 모델 로드 (자막 없는 영상 대비)
    print(f"\nWhisper {args.model} 모델 로딩 중...")
    whisper_model = whisper.load_model(args.model)
    print("모델 로드 완료!")

    # 각 비디오 처리
    results = []
    for i, video in enumerate(args.videos, 1):
        youtube_video_id = extract_youtube_video_id(video)
        print(f"\n[{i}/{len(args.videos)}] {youtube_video_id}")

        result = process_video(youtube_video_id, args.sources_dir, args.output_dir, whisper_model)
        results.append(result)

    # CSV 저장
    csv_path = save_results_to_csv(results, args.output_dir)
    print(f"\n[CSV 저장] {csv_path}")

    # 결과 요약
    success = sum(1 for r in results if r['status'] == 'success')
    subtitle_count = sum(1 for r in results if r['source_type'] == 'subtitle')
    whisper_count = sum(1 for r in results if r['source_type'] == 'whisper')
    cached_count = sum(1 for r in results if r['source_type'] == 'cached')
    failed = len(results) - success

    print("\n" + "=" * 60)
    print("처리 완료!")
    print(f"  - 성공: {success}/{len(results)}")
    print(f"    - 자막 추출: {subtitle_count}개")
    print(f"    - Whisper 전사: {whisper_count}개")
    print(f"    - 캐시 사용: {cached_count}개")
    print(f"  - 실패: {failed}/{len(results)}")
    print(f"  - CSV 저장: {csv_path}")
    print("=" * 60)

    if failed > 0:
        print("\n실패한 영상:")
        for r in results:
            if r['status'] == 'failed':
                print(f"  - {r['youtube_video_id']}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
