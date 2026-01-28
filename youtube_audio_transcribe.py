#!/usr/bin/env python3
"""
YouTube 트렌딩 영상 오디오 다운로드 및 전사 스크립트

trending_history.csv에서 rank 1-50까지의 video_id를 읽어
오디오를 다운로드하고 Whisper를 사용하여 한글 전사문을 생성합니다.

사용법:
    .venv/bin/python youtube_audio_transcribe.py           # 전체 50개 처리
    .venv/bin/python youtube_audio_transcribe.py --limit 5 # 처음 5개만 처리
    .venv/bin/python youtube_audio_transcribe.py --test    # 테스트 (처음 2개만)

필요 패키지:
    pip install yt-dlp openai-whisper pandas
"""

import os
import sys
import csv
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# 선택적 import (설치 확인)
try:
    import pandas as pd
except ImportError:
    print("pandas가 설치되어 있지 않습니다. pip install pandas")
    sys.exit(1)

try:
    import whisper
except ImportError:
    print("openai-whisper가 설치되어 있지 않습니다. pip install openai-whisper")
    sys.exit(1)

try:
    import yt_dlp
except ImportError:
    print("yt-dlp가 설치되어 있지 않습니다. pip install yt-dlp")
    sys.exit(1)


# 설정
CSV_PATH = "/mnt/c/Users/Administrator/Desktop/trending_history.csv"
SOURCES_DIR = Path(__file__).parent / "sources"
OUTPUT_DIR = Path(__file__).parent / "output"
MAX_RANK = 50  # 상위 50개만 처리


def setup_directories():
    """sources와 output 디렉토리 생성"""
    SOURCES_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Sources 디렉토리: {SOURCES_DIR}")
    print(f"Output 디렉토리: {OUTPUT_DIR}")


def load_video_ids_from_csv(csv_path: str, max_rank: int = 50) -> list[dict]:
    """
    CSV 파일에서 상위 max_rank개의 video_id와 제목을 추출
    (처음 rank 1~50까지의 데이터만 사용 - 첫 번째 세트)

    Args:
        csv_path: trending_history.csv 파일 경로
        max_rank: 가져올 최대 순위 (기본값: 50)

    Returns:
        list[dict]: video_id와 title을 포함하는 딕셔너리 리스트
    """
    print(f"\nCSV 파일 로드 중: {csv_path}")

    if not Path(csv_path).exists():
        print(f"오류: CSV 파일을 찾을 수 없습니다 - {csv_path}")
        return []

    df = pd.read_csv(csv_path)

    # rank 열이 있는지 확인
    if 'rank' not in df.columns or 'video_id' not in df.columns:
        print("오류: CSV 파일에 'rank' 또는 'video_id' 열이 없습니다.")
        return []

    # 처음 max_rank개 행만 사용 (rank 1부터 50까지가 처음에 있음)
    df_top = df.head(max_rank)

    videos = []
    for _, row in df_top.iterrows():
        video_id = row['video_id']
        # video_id가 유효한지 확인 (날짜 형식이 아닌지)
        if pd.isna(video_id) or len(str(video_id)) < 5 or '-' in str(video_id)[:4]:
            continue
        videos.append({
            'video_id': str(video_id),
            'title': str(row.get('title', 'Unknown')),
            'rank': int(row['rank'])
        })

    print(f"총 {len(videos)}개의 비디오 ID를 로드했습니다.")
    return videos


def download_subtitles(video_id: str, output_dir: Path) -> Path | None:
    """
    yt-dlp를 사용하여 YouTube 자막 다운로드 (공식 자막 우선, 없으면 자동 생성)

    Args:
        video_id: YouTube 비디오 ID
        output_dir: 출력 디렉토리

    Returns:
        Path: 다운로드된 자막 파일 경로 또는 None (자막 없음)
    """
    url = f"https://www.youtube.com/watch?v={video_id}"

    # 이미 다운로드된 자막이 있는지 확인
    existing_subs = list(output_dir.glob(f"{video_id}.ko.*"))
    if existing_subs:
        print(f"  자막 이미 존재: {existing_subs[0].name}")
        return existing_subs[0]

    ydl_opts = {
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['ko'],
        'subtitlesformat': 'srt',
        'skip_download': True,
        'outtmpl': str(output_dir / f"{video_id}"),
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # 다운로드된 자막 파일 찾기
        sub_files = list(output_dir.glob(f"{video_id}.ko.*"))
        if sub_files:
            print(f"  자막 다운로드 완료: {sub_files[0].name}")
            return sub_files[0]

        return None

    except Exception as e:
        print(f"  자막 다운로드 실패: {str(e)[:50]}")
        return None


def extract_text_from_subtitle(subtitle_path: Path) -> str:
    """
    SRT/VTT 자막 파일에서 텍스트만 추출

    Args:
        subtitle_path: 자막 파일 경로

    Returns:
        str: 추출된 텍스트
    """
    import re

    content = subtitle_path.read_text(encoding='utf-8')

    # SRT 형식에서 타임스탬프와 번호 제거
    # 패턴: 숫자만 있는 줄, 타임스탬프 줄
    lines = content.split('\n')
    text_lines = []

    for line in lines:
        line = line.strip()
        # 빈 줄, 숫자만 있는 줄, 타임스탬프 줄 건너뛰기
        if not line:
            continue
        if line.isdigit():
            continue
        if '-->' in line:
            continue
        # VTT 헤더 건너뛰기
        if line.startswith('WEBVTT') or line.startswith('Kind:') or line.startswith('Language:'):
            continue
        # HTML 태그 제거
        line = re.sub(r'<[^>]+>', '', line)
        if line:
            text_lines.append(line)

    # 중복 제거하면서 순서 유지
    seen = set()
    unique_lines = []
    for line in text_lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)

    return ' '.join(unique_lines)


def download_audio(video_id: str, output_dir: Path) -> Path | None:
    """
    yt-dlp를 사용하여 YouTube 비디오에서 오디오만 다운로드

    Args:
        video_id: YouTube 비디오 ID
        output_dir: 출력 디렉토리

    Returns:
        Path: 다운로드된 오디오 파일 경로 또는 None
    """
    output_template = str(output_dir / f"audio_{video_id}.%(ext)s")
    url = f"https://www.youtube.com/watch?v={video_id}"

    # 이미 다운로드된 파일이 있는지 확인
    existing_files = list(output_dir.glob(f"audio_{video_id}.*"))
    audio_extensions = ['.m4a', '.mp3', '.webm', '.opus', '.ogg']
    for f in existing_files:
        if f.suffix.lower() in audio_extensions:
            print(f"  오디오 이미 존재: {f.name}")
            return f

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

        # 다운로드된 파일 찾기
        downloaded_files = list(output_dir.glob(f"audio_{video_id}.*"))
        for f in downloaded_files:
            if f.suffix.lower() in audio_extensions:
                print(f"  오디오 다운로드 완료: {f.name}")
                return f

        print(f"  오디오 다운로드 실패: 파일을 찾을 수 없음")
        return None

    except Exception as e:
        print(f"  오디오 다운로드 오류: {str(e)[:100]}")
        return None


def transcribe_audio(audio_path: Path, output_dir: Path, model_name: str = 'base') -> str | None:
    """
    Whisper를 사용하여 오디오를 한글 전사문으로 변환

    Args:
        audio_path: 오디오 파일 경로
        output_dir: 전사문 저장 디렉토리
        model_name: Whisper 모델 (tiny, base, small, medium, large)

    Returns:
        str: 전사문 텍스트 또는 None
    """
    # 이미 전사된 파일이 있는지 확인
    transcript_path = output_dir / f"{audio_path.stem}.txt"
    if transcript_path.exists():
        print(f"  이미 전사됨: {transcript_path.name}")
        return transcript_path.read_text(encoding='utf-8')

    try:
        # Whisper 모델 로드 (전역 캐싱 권장)
        model = whisper.load_model(model_name)

        # 음성 인식 실행
        result = model.transcribe(str(audio_path), language='ko', fp16=False)
        transcript_text = result['text'].strip()

        # 전사문 저장
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript_text)

        print(f"  전사 완료: {transcript_path.name}")
        return transcript_text

    except Exception as e:
        print(f"  전사 오류: {str(e)[:100]}")
        return None


def save_results_to_csv(results: list[dict], output_path: Path):
    """결과를 CSV 파일로 저장"""
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n결과 저장 완료: {output_path}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='YouTube 트렌딩 영상 오디오 다운로드 및 전사'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=50,
        help='처리할 비디오 개수 (기본값: 50)'
    )
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='테스트 모드 (처음 2개만 처리)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper 모델 (기본값: base)'
    )
    args = parser.parse_args()

    # 테스트 모드면 2개만 처리
    limit = 2 if args.test else args.limit

    print("=" * 70)
    print("YouTube 트렌딩 영상 오디오 다운로드 및 전사")
    print("=" * 70)

    # 디렉토리 설정
    setup_directories()

    # CSV에서 video_id 로드
    videos = load_video_ids_from_csv(CSV_PATH, MAX_RANK)

    # limit 적용
    videos = videos[:limit]
    if not videos:
        print("처리할 비디오가 없습니다.")
        return

    # Whisper 모델 미리 로드 (재사용을 위해)
    print(f"\nWhisper {args.model} 모델 로딩 중...")
    model = whisper.load_model(args.model)
    print("모델 로드 완료!")

    # 결과 저장용
    results = []

    # 각 비디오 처리
    print(f"\n{'=' * 70}")
    print(f"총 {len(videos)}개 비디오 처리 시작")
    print(f"{'=' * 70}")

    for i, video in enumerate(videos, 1):
        video_id = video['video_id']
        title = video['title'][:40] + "..." if len(video['title']) > 40 else video['title']

        print(f"\n[{i}/{len(videos)}] Rank {video['rank']}: {title}")
        print(f"  Video ID: {video_id}")

        transcript = None
        transcript_path = OUTPUT_DIR / f"{video_id}.txt"
        source_type = None  # 'subtitle' or 'whisper'
        audio_path = None

        # 이미 전사된 파일 확인
        if transcript_path.exists():
            print(f"  이미 전사됨: {transcript_path.name}")
            transcript = transcript_path.read_text(encoding='utf-8')
            source_type = 'cached'
        else:
            # 1. 먼저 자막 시도
            subtitle_path = download_subtitles(video_id, SOURCES_DIR)
            if subtitle_path:
                # 자막에서 텍스트 추출
                transcript = extract_text_from_subtitle(subtitle_path)
                source_type = 'subtitle'
                # 전사문 저장
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    f.write(transcript)
                print(f"  자막 추출 완료: {transcript_path.name}")
            else:
                # 2. 자막 없으면 오디오 다운로드 후 Whisper 전사
                print("  자막 없음, Whisper 전사 시도...")
                audio_path = download_audio(video_id, SOURCES_DIR)
                if audio_path:
                    try:
                        result = model.transcribe(str(audio_path), language='ko', fp16=False)
                        transcript = result['text'].strip()
                        source_type = 'whisper'
                        # 전사문 저장
                        with open(transcript_path, 'w', encoding='utf-8') as f:
                            f.write(transcript)
                        print(f"  Whisper 전사 완료: {transcript_path.name}")
                    except Exception as e:
                        print(f"  전사 오류: {str(e)[:100]}")
                        transcript = None

        results.append({
            'rank': video['rank'],
            'video_id': video_id,
            'title': video['title'],
            'status': 'success' if transcript else 'failed',
            'source_type': source_type,  # 'subtitle', 'whisper', or 'cached'
            'audio_file': str(audio_path.name) if audio_path else None,
            'transcript_file': str(transcript_path.name) if transcript else None,
            'transcript_preview': transcript[:200] + "..." if transcript and len(transcript) > 200 else transcript
        })

    # 결과 저장
    result_csv_path = OUTPUT_DIR / f"youtube_transcripts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    save_results_to_csv(results, result_csv_path)

    # 요약 출력
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"\n{'=' * 70}")
    print(f"처리 완료!")
    print(f"  - 성공: {success_count}/{len(videos)}")
    print(f"  - 실패: {len(videos) - success_count}/{len(videos)}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
