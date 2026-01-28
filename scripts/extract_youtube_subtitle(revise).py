#!/usr/bin/env python3
"""
YouTube 영상 description과 subtitle 추출 후 Supabase에 저장

[기능]
1. Supabase videos 테이블에서 youtube_video_id 조회
2. YouTube description 추출 (yt-dlp)
3. subtitle 추출 (공식 자막 → 자동 생성 자막 → Whisper 전사)
4. videos_description 테이블에 저장

[사용법]
    .venv/bin/python "scripts/extract_youtube_subtitle(revise).py"              # 전체 처리 (기본: Whisper 사용)
    .venv/bin/python "scripts/extract_youtube_subtitle(revise).py" --limit 5    # 5개만 처리
    .venv/bin/python "scripts/extract_youtube_subtitle(revise).py" --test       # 테스트 (2개만)
    .venv/bin/python "scripts/extract_youtube_subtitle(revise).py" --no-whisper # Whisper 사용 안 함

[필요 패키지]
    pip install supabase python-dotenv yt-dlp openai-whisper
"""

import argparse
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client, Client
import yt_dlp

# 환경변수 로드
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_API_KEY")

# Whisper 모델 (lazy loading)
whisper_model = None

# 디렉토리 설정
SOURCES_DIR = Path(__file__).parent.parent / "sources"


# =============================================================================
# 1단계: Supabase 클라이언트
# =============================================================================

def get_supabase_client() -> Client:
    """Supabase 클라이언트 생성"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL 또는 SUPABASE_API_KEY가 설정되지 않았습니다.")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def get_videos(limit: int = None, start_id: int = None, end_id: int = None) -> list[dict]:
    """
    videos 테이블에서 id, youtube_video_id 조회

    Args:
        limit: 최대 개수
        start_id: 시작 ID (이상)
        end_id: 종료 ID (이하)

    Returns:
        list[dict]: [{"id": 1, "youtube_video_id": "abc123"}, ...]
    """
    client = get_supabase_client()
    query = client.table("videos").select("id, youtube_video_id")

    if start_id:
        query = query.gte("id", start_id)
    if end_id:
        query = query.lte("id", end_id)
    if limit:
        query = query.limit(limit)

    response = query.execute()
    return response.data


def get_processed_ids(include_empty_subtitle: bool = False) -> set:
    """
    이미 처리된 id 목록 조회

    Args:
        include_empty_subtitle: True면 subtitle이 비어있는 영상도 재처리 대상에서 제외
                               False면 subtitle이 비어있는 영상은 재처리 대상에 포함
    """
    client = get_supabase_client()
    try:
        response = client.table("videos_description").select("id, subtitle").execute()
        if include_empty_subtitle:
            # 모든 처리된 영상 제외
            return {row['id'] for row in response.data}
        else:
            # subtitle이 있는 영상만 제외 (비어있으면 재처리 대상)
            return {row['id'] for row in response.data if row.get('subtitle')}
    except Exception:
        return set()


def get_processed_youtube_ids() -> dict:
    """
    이미 처리된 youtube_video_id와 그 결과(description, subtitle) 조회

    Returns:
        dict: {youtube_video_id: {"description": ..., "subtitle": ...}, ...}
    """
    client = get_supabase_client()
    try:
        # videos 테이블과 videos_description 테이블 조인하여 조회
        # videos_description에서 subtitle이 있는 것만 조회
        response = client.table("videos_description").select(
            "id, description, subtitle"
        ).neq("subtitle", "").execute()

        if not response.data:
            return {}

        # id -> youtube_video_id 매핑을 위해 videos 테이블 조회
        ids = [row['id'] for row in response.data]
        videos_response = client.table("videos").select("id, youtube_video_id").in_("id", ids).execute()

        id_to_youtube = {v['id']: v['youtube_video_id'] for v in videos_response.data}

        # youtube_video_id -> 결과 매핑
        result = {}
        for row in response.data:
            youtube_id = id_to_youtube.get(row['id'])
            if youtube_id and youtube_id not in result:
                result[youtube_id] = {
                    "description": row.get('description', ''),
                    "subtitle": row.get('subtitle', '')
                }

        return result
    except Exception as e:
        print(f"처리된 youtube_id 조회 실패: {e}")
        return {}


def save_video_description(id: int, description: str, subtitle: str) -> bool:
    """
    videos_description 테이블에 결과 저장 (insert 또는 update)

    Args:
        id: videos 테이블의 id (PK)
        description: YouTube 영상 설명
        subtitle: 자막 텍스트

    Returns:
        bool: 성공 여부
    """
    client = get_supabase_client()

    try:
        # 기존 레코드 확인 (id 기준)
        existing = client.table("videos_description").select("id").eq("id", id).execute()

        if existing.data:
            # 기존 레코드 있으면 update
            client.table("videos_description").update({
                "description": description,
                "subtitle": subtitle
            }).eq("id", id).execute()
        else:
            # 없으면 insert
            client.table("videos_description").insert({
                "id": id,
                "description": description,
                "subtitle": subtitle
            }).execute()

        return True
    except Exception as e:
        print(f"저장 실패 (id={id}): {e}")
        return False


# =============================================================================
# 2단계: YouTube 추출 (description, subtitle)
# =============================================================================

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


# =============================================================================
# 3단계: Whisper 전사 (선택)
# =============================================================================

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


def cleanup_source_files(youtube_video_id: str, output_dir: Path) -> None:
    """
    Supabase 저장 완료 후 sources 폴더의 임시 파일 삭제

    Args:
        youtube_video_id: YouTube 비디오 ID
        output_dir: 파일이 저장된 디렉토리
    """
    output_dir = Path(output_dir)
    deleted_files = []

    # 자막 파일 삭제 ({youtube_video_id}.ko.*)
    subtitle_files = list(output_dir.glob(f"{youtube_video_id}.ko.*"))
    for f in subtitle_files:
        try:
            f.unlink()
            deleted_files.append(f.name)
        except Exception as e:
            print(f"    파일 삭제 실패 ({f.name}): {e}")

    # 오디오 파일 삭제 (audio_{youtube_video_id}.*)
    audio_files = list(output_dir.glob(f"audio_{youtube_video_id}.*"))
    for f in audio_files:
        try:
            f.unlink()
            deleted_files.append(f.name)
        except Exception as e:
            print(f"    파일 삭제 실패 ({f.name}): {e}")

    if deleted_files:
        print(f"    임시 파일 삭제: {', '.join(deleted_files)}")


# =============================================================================
# 메인 처리 함수
# =============================================================================

def process_videos(limit: int = None, start_id: int = None, end_id: int = None, use_whisper: bool = False, whisper_model_name: str = "base"):
    """
    메인 처리 함수

    Args:
        limit: 처리할 최대 개수
        start_id: 시작 ID (이상)
        end_id: 종료 ID (이하)
        use_whisper: 자막 없을 때 Whisper 사용 여부
        whisper_model_name: Whisper 모델 이름
    """
    print("=" * 60)
    print("YouTube Description & Subtitle 추출 → Supabase 저장")
    print("=" * 60)

    # 1. Supabase에서 videos 조회
    print("\n[1] Supabase에서 videos 조회 중...")
    videos = get_videos(limit=limit, start_id=start_id, end_id=end_id)
    print(f"    총 {len(videos)}개 영상 조회됨")

    # 2. 이미 처리된 영상 제외 (subtitle 비어있으면 재처리)
    processed_ids = get_processed_ids(include_empty_subtitle=False)
    videos_to_process = [v for v in videos if v['id'] not in processed_ids]
    print(f"    이미 처리됨 (subtitle 있음): {len(processed_ids)}개")
    print(f"    처리 대상 (신규 또는 subtitle 없음): {len(videos_to_process)}개")

    # 3. 이미 처리된 youtube_video_id 결과 조회 (중복 영상 재사용)
    processed_youtube_ids = get_processed_youtube_ids()
    print(f"    처리된 youtube_video_id: {len(processed_youtube_ids)}개 (중복 영상 재사용 가능)")

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

        # 중복 youtube_video_id 체크 - 이미 처리된 영상이면 결과 복사
        if youtube_id in processed_youtube_ids:
            cached = processed_youtube_ids[youtube_id]
            description = cached['description']
            subtitle = cached['subtitle']
            print(f"    [중복 영상] 기존 결과 복사")
            print(f"    Description: {len(description)}자 (복사)")
            print(f"    Subtitle: {len(subtitle)}자 (복사)")
        else:
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

            # 새로 처리한 영상은 캐시에 추가 (같은 배치 내 중복 처리 방지)
            if subtitle:
                processed_youtube_ids[youtube_id] = {
                    "description": description,
                    "subtitle": subtitle
                }

        # Supabase에 저장
        if save_video_description(video_id, description, subtitle):  # video_id는 videos.id
            print(f"    저장 완료!")
            success_count += 1
            # 저장 성공 후 임시 파일 삭제
            cleanup_source_files(youtube_id, SOURCES_DIR)
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
        '--start-id',
        type=int,
        default=None,
        help='시작 ID (이상)'
    )
    parser.add_argument(
        '--end-id',
        type=int,
        default=None,
        help='종료 ID (이하)'
    )
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='테스트 모드 (2개만 처리)'
    )
    parser.add_argument(
        '--no-whisper',
        action='store_true',
        help='자막 없을 때 Whisper 전사 사용 안 함 (기본: Whisper 사용)'
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
        start_id=args.start_id,
        end_id=args.end_id,
        use_whisper=not args.no_whisper,  # 기본값: Whisper 사용
        whisper_model_name=args.model
    )


if __name__ == "__main__":
    main()
