#!/usr/bin/env python3
"""
YouTube 영상 description과 subtitle 추출 후 Supabase에 저장

사용법:
    .venv/bin/python extract_and_save.py              # 전체 처리
    .venv/bin/python extract_and_save.py --limit 5    # 5개만 처리
    .venv/bin/python extract_and_save.py --test       # 테스트 (2개만)
    .venv/bin/python extract_and_save.py --whisper    # 자막 없으면 Whisper 사용
"""

import argparse
from pathlib import Path

from supabase_client import get_videos, save_video_description, get_supabase_client
from youtube_extractor import get_video_info, get_subtitle_text

# Whisper는 선택적 import (--whisper 옵션 사용 시에만 필요)
whisper_model = None

SOURCES_DIR = Path(__file__).parent / "sources"


def load_whisper_model(model_name: str = "base"):
    """Whisper 모델 로드 (lazy loading)"""
    global whisper_model
    if whisper_model is None:
        import whisper
        print(f"Whisper {model_name} 모델 로딩 중...")
        whisper_model = whisper.load_model(model_name)
        print("모델 로드 완료!")
    return whisper_model


def download_audio(youtube_video_id: str, output_dir: Path) -> Path | None:
    """yt-dlp로 오디오 다운로드"""
    import yt_dlp

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # 이미 다운로드된 파일 확인
    audio_extensions = ['.m4a', '.mp3', '.webm', '.opus', '.ogg']
    for ext in audio_extensions:
        existing = output_dir / f"audio_{youtube_video_id}{ext}"
        if existing.exists():
            return existing

    url = f"https://www.youtube.com/watch?v={youtube_video_id}"
    output_template = str(output_dir / f"audio_{youtube_video_id}.%(ext)s")

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
        for ext in audio_extensions:
            downloaded = output_dir / f"audio_{youtube_video_id}{ext}"
            if downloaded.exists():
                return downloaded

        return None

    except Exception as e:
        print(f"    오디오 다운로드 실패: {str(e)[:50]}")
        return None


def transcribe_with_whisper(youtube_video_id: str, model_name: str = "base") -> str | None:
    """Whisper로 음성 전사"""
    model = load_whisper_model(model_name)

    # 오디오 다운로드
    audio_path = download_audio(youtube_video_id, SOURCES_DIR)
    if not audio_path:
        return None

    try:
        print(f"    Whisper 전사 중...")
        result = model.transcribe(str(audio_path), language='ko', fp16=False)
        return result['text'].strip()
    except Exception as e:
        print(f"    전사 실패: {str(e)[:50]}")
        return None


def get_processed_video_ids() -> set:
    """이미 처리된 video_id 목록 조회"""
    client = get_supabase_client()
    try:
        response = client.table("videos_description").select("video_id").execute()
        return {row['video_id'] for row in response.data}
    except Exception:
        return set()


def process_videos(limit: int = None, use_whisper: bool = False, whisper_model_name: str = "base"):
    """
    메인 처리 함수

    Args:
        limit: 처리할 최대 개수
        use_whisper: 자막 없을 때 Whisper 사용 여부
        whisper_model_name: Whisper 모델 이름
    """
    print("=" * 60)
    print("YouTube Description & Subtitle 추출 → Supabase 저장")
    print("=" * 60)

    # 1. Supabase에서 videos 조회
    print("\n[1] Supabase에서 videos 조회 중...")
    videos = get_videos(limit=limit)
    print(f"    총 {len(videos)}개 영상 조회됨")

    # 2. 이미 처리된 영상 제외
    processed_ids = get_processed_video_ids()
    videos_to_process = [v for v in videos if v['id'] not in processed_ids]
    print(f"    이미 처리됨: {len(processed_ids)}개")
    print(f"    처리 대상: {len(videos_to_process)}개")

    if not videos_to_process:
        print("\n처리할 영상이 없습니다.")
        return

    # 3. 각 영상 처리
    SOURCES_DIR.mkdir(exist_ok=True)

    success_count = 0
    fail_count = 0

    print(f"\n[2] 영상 처리 시작")
    print("-" * 60)

    for i, video in enumerate(videos_to_process, 1):
        video_id = video['id']
        youtube_id = video['youtube_video_id']

        print(f"\n[{i}/{len(videos_to_process)}] id={video_id}, youtube_id={youtube_id}")

        # Description 추출
        info = get_video_info(youtube_id)
        if info:
            description = info['description']
            print(f"    Description: {len(description)}자")
        else:
            description = ""
            print(f"    Description: 추출 실패")

        # Subtitle 추출
        subtitle, source_type = get_subtitle_text(youtube_id, SOURCES_DIR)

        if subtitle:
            print(f"    Subtitle: {len(subtitle)}자 (source: {source_type})")
        elif use_whisper:
            print(f"    Subtitle: 자막 없음, Whisper 전사 시도...")
            subtitle = transcribe_with_whisper(youtube_id, whisper_model_name)
            if subtitle:
                source_type = "whisper"
                print(f"    Subtitle: {len(subtitle)}자 (source: whisper)")
            else:
                subtitle = ""
                print(f"    Subtitle: Whisper 전사 실패")
        else:
            subtitle = ""
            print(f"    Subtitle: 자막 없음 (Whisper 비활성화)")

        # Supabase에 저장
        if save_video_description(video_id, description, subtitle):
            print(f"    저장 완료!")
            success_count += 1
        else:
            print(f"    저장 실패!")
            fail_count += 1

    # 4. 결과 요약
    print("\n" + "=" * 60)
    print("처리 완료!")
    print(f"  - 성공: {success_count}")
    print(f"  - 실패: {fail_count}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='YouTube description/subtitle 추출 후 Supabase 저장'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='처리할 최대 개수'
    )
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='테스트 모드 (2개만 처리)'
    )
    parser.add_argument(
        '--whisper', '-w',
        action='store_true',
        help='자막 없을 때 Whisper 전사 사용'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper 모델 (기본값: base)'
    )

    args = parser.parse_args()

    limit = 2 if args.test else args.limit

    process_videos(
        limit=limit,
        use_whisper=args.whisper,
        whisper_model_name=args.model
    )


if __name__ == "__main__":
    main()
