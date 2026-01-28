#!/usr/bin/env python3
"""
Supabase 클라이언트 모듈
- videos 테이블에서 id, youtube_video_id 조회 (읽기 전용)
- videos_description 테이블에 결과 저장
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_API_KEY")


def get_supabase_client() -> Client:
    """Supabase 클라이언트 생성"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL 또는 SUPABASE_API_KEY가 설정되지 않았습니다.")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def get_videos(limit: int = None) -> list[dict]:
    """
    videos 테이블에서 id, youtube_video_id 조회

    Returns:
        list[dict]: [{"id": 1, "youtube_video_id": "abc123"}, ...]
    """
    client = get_supabase_client()
    query = client.table("videos").select("id, youtube_video_id")

    if limit:
        query = query.limit(limit)

    response = query.execute()
    return response.data


def save_video_description(video_id: int, description: str, subtitle: str) -> bool:
    """
    videos_description 테이블에 결과 저장

    Args:
        video_id: videos 테이블의 id (FK)
        description: YouTube 영상 설명
        subtitle: 자막 텍스트

    Returns:
        bool: 성공 여부
    """
    client = get_supabase_client()

    data = {
        "video_id": video_id,
        "description": description,
        "subtitle": subtitle
    }

    try:
        response = client.table("videos_description").upsert(data).execute()
        return True
    except Exception as e:
        print(f"저장 실패 (video_id={video_id}): {e}")
        return False


def test_connection():
    """Supabase 연결 테스트"""
    print("Supabase 연결 테스트 중...")
    print(f"URL: {SUPABASE_URL}")

    try:
        client = get_supabase_client()

        # videos 테이블에서 샘플 조회
        response = client.table("videos").select("id, youtube_video_id").limit(3).execute()

        print("연결 성공!")
        print(f"videos 테이블 샘플:")
        for row in response.data:
            print(f"  id={row['id']}, youtube_video_id={row['youtube_video_id']}")

        return True

    except Exception as e:
        print(f"연결 실패: {e}")
        return False


if __name__ == "__main__":
    test_connection()
