#!/usr/bin/env python3
"""
YouTube 영상/오디오 다운로드 스크립트
yt-dlp를 사용하여 YouTube 콘텐츠 다운로드
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def download_youtube(url, output_dir='./sources', audio_only=False, playlist=False):
    """
    YouTube 영상 또는 오디오 다운로드

    Args:
        url: YouTube URL
        output_dir: 저장 디렉토리
        audio_only: True이면 오디오만 다운로드
        playlist: True이면 재생목록 다운로드

    Returns:
        bool: 성공 여부
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("YouTube 다운로드 시작")
    print("=" * 60)
    print(f"URL: {url}")
    print(f"저장 위치: {output_dir}")
    print(f"모드: {'오디오만' if audio_only else '비디오+오디오'}")
    if playlist:
        print("재생목록 다운로드")
    print("=" * 60)

    # yt-dlp 명령어 구성
    cmd = ['.venv/bin/yt-dlp']

    if audio_only:
        # 오디오만 추출
        cmd.extend(['-x', '--audio-format', 'm4a'])
        output_template = str(output_dir / '%(title)s.%(ext)s')
    else:
        # 비디오 다운로드 (최고 품질)
        output_template = str(output_dir / '%(title)s.%(ext)s')

    # 출력 템플릿 설정
    cmd.extend(['-o', output_template])

    # 재생목록 옵션
    if playlist:
        # 재생목록 인덱스를 파일명에 포함
        output_template_playlist = str(output_dir / '%(playlist_index)s_%(title)s.%(ext)s')
        cmd[-1] = output_template_playlist
    else:
        # 재생목록이 아닌 경우 단일 영상만 다운로드
        cmd.append('--no-playlist')

    # URL 추가
    cmd.append(url)

    print(f"\n실행 명령어: {' '.join(cmd)}\n")

    try:
        # yt-dlp 실행
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=False  # 실시간으로 출력 표시
        )

        print("\n" + "=" * 60)
        print("✓ 다운로드 완료!")
        print("=" * 60)

        # 다운로드된 파일 목록 표시
        print(f"\n다운로드된 파일:")
        for file in sorted(output_dir.iterdir()):
            if file.is_file():
                size = file.stat().st_size / (1024 * 1024)  # MB
                print(f"  - {file.name} ({size:.1f} MB)")

        return True

    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 60)
        print("✗ 다운로드 실패")
        print("=" * 60)
        print(f"오류: {e}")
        return False
    except Exception as e:
        print(f"\n✗ 예기치 않은 오류: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='YouTube 영상/오디오를 다운로드합니다.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 비디오 다운로드
  python download_youtube.py "https://www.youtube.com/watch?v=..."

  # 오디오만 다운로드
  python download_youtube.py "https://www.youtube.com/watch?v=..." --audio-only

  # output 폴더에 저장
  python download_youtube.py "https://www.youtube.com/watch?v=..." --output-dir output

  # 재생목록 다운로드
  python download_youtube.py "https://www.youtube.com/playlist?list=..." --playlist
        """
    )

    parser.add_argument(
        'url',
        help='YouTube URL (영상 또는 재생목록)'
    )

    parser.add_argument(
        '--output-dir',
        default='./sources',
        help='저장 디렉토리 (기본: ./sources)'
    )

    parser.add_argument(
        '--audio-only',
        action='store_true',
        help='오디오만 다운로드 (m4a 형식)'
    )

    parser.add_argument(
        '--playlist',
        action='store_true',
        help='재생목록 전체 다운로드'
    )

    args = parser.parse_args()

    # 다운로드 실행
    success = download_youtube(
        args.url,
        args.output_dir,
        args.audio_only,
        args.playlist
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
