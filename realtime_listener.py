#!/usr/bin/env python3
"""
Supabase Realtime으로 videos 테이블 변경 감지 및 자동 처리

videos 테이블에 새 레코드가 INSERT되면 자동으로:
1. YouTube description 추출
2. subtitle 추출 (없으면 Whisper 전사)
3. videos_description 테이블에 저장

사용법:
    .venv/bin/python realtime_listener.py              # 기본 실행
    .venv/bin/python realtime_listener.py --whisper    # 자막 없으면 Whisper 사용
    .venv/bin/python realtime_listener.py --whisper --model small
"""

import argparse
import asyncio
import os
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv
from supabase._async.client import create_client as create_async_client

from supabase_client import save_video_description
from youtube_extractor import get_video_info, get_subtitle_text

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_API_KEY")

# Whisper 모델 (lazy loading)
whisper_model = None
SOURCES_DIR = Path(__file__).parent / "sources"


def load_whisper_model(model_name: str = "base"):
    """Whisper 모델 로드"""
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


def process_video(video_id: int, youtube_video_id: str, use_whisper: bool = False, whisper_model_name: str = "base"):
    """
    단일 영상 처리

    Args:
        video_id: videos 테이블의 id
        youtube_video_id: YouTube 비디오 ID
        use_whisper: Whisper 사용 여부
        whisper_model_name: Whisper 모델 이름
    """
    print(f"\n[처리 중] id={video_id}, youtube_id={youtube_video_id}")

    # Description 추출
    info = get_video_info(youtube_video_id)
    if info:
        description = info['description']
        print(f"    Description: {len(description)}자")
    else:
        description = ""
        print(f"    Description: 추출 실패")

    # Subtitle 추출
    SOURCES_DIR.mkdir(exist_ok=True)
    subtitle, source_type = get_subtitle_text(youtube_video_id, SOURCES_DIR)

    if subtitle:
        print(f"    Subtitle: {len(subtitle)}자 (source: {source_type})")
    elif use_whisper:
        print(f"    Subtitle: 자막 없음, Whisper 전사 시도...")
        subtitle = transcribe_with_whisper(youtube_video_id, whisper_model_name)
        if subtitle:
            print(f"    Subtitle: {len(subtitle)}자 (source: whisper)")
        else:
            subtitle = ""
            print(f"    Subtitle: Whisper 전사 실패")
    else:
        subtitle = ""
        print(f"    Subtitle: 자막 없음")

    # 저장
    if save_video_description(video_id, description, subtitle):
        print(f"    저장 완료!")
    else:
        print(f"    저장 실패!")


class RealtimeListener:
    def __init__(self, use_whisper: bool = False, whisper_model_name: str = "base"):
        self.use_whisper = use_whisper
        self.whisper_model_name = whisper_model_name
        self.client = None
        self.channel = None
        self.running = True

    def handle_insert(self, payload):
        """INSERT 이벤트 핸들러"""
        try:
            # payload 구조 확인
            record = payload.get('data', {}).get('record', {})
            if not record:
                record = payload.get('new', {})
            if not record:
                record = payload.get('record', {})

            video_id = record.get('id')
            youtube_video_id = record.get('youtube_video_id')

            if video_id and youtube_video_id:
                print(f"\n{'='*60}")
                print(f"[NEW] videos 테이블에 새 레코드 감지!")
                process_video(
                    video_id,
                    youtube_video_id,
                    self.use_whisper,
                    self.whisper_model_name
                )
                print(f"{'='*60}")
            else:
                print(f"[DEBUG] payload: {payload}")

        except Exception as e:
            print(f"[ERROR] INSERT 처리 중 오류: {e}")

    async def start(self):
        """Realtime 구독 시작"""
        print("=" * 60)
        print("Supabase Realtime Listener 시작")
        print("=" * 60)
        print(f"Whisper 사용: {self.use_whisper}")
        if self.use_whisper:
            print(f"Whisper 모델: {self.whisper_model_name}")
        print("-" * 60)
        print("videos 테이블 INSERT 이벤트 대기 중...")
        print("종료하려면 Ctrl+C를 누르세요.")
        print("-" * 60)

        # Async 클라이언트 생성
        self.client = await create_async_client(SUPABASE_URL, SUPABASE_KEY)

        # Realtime 채널 생성 및 구독
        self.channel = self.client.channel('videos-changes')

        self.channel.on_postgres_changes(
            event='INSERT',
            schema='public',
            table='videos',
            callback=self.handle_insert
        )

        await self.channel.subscribe()
        print("[OK] Realtime 구독 완료!")

        # 메인 루프 유지
        while self.running:
            await asyncio.sleep(1)

    def stop(self):
        """리스너 중지"""
        self.running = False
        print("\n리스너 중지됨.")


def main():
    parser = argparse.ArgumentParser(
        description='Supabase Realtime으로 videos 테이블 변경 감지 및 자동 처리'
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

    listener = RealtimeListener(
        use_whisper=args.whisper,
        whisper_model_name=args.model
    )

    # 비동기 실행
    try:
        asyncio.run(listener.start())
    except KeyboardInterrupt:
        listener.stop()
        print("종료됨.")


if __name__ == "__main__":
    main()
