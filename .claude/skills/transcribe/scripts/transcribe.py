#!/usr/bin/env python3
"""
미디어 파일 전사 메인 스크립트
- ./sources/ 폴더의 오디오/비디오 파일을 전사
- ./output/ 폴더에 전사문 저장
- 전사문 내용 기반 파일명 + 타임스탬프
"""

import os
import sys
import argparse
import whisper
import subprocess
from pathlib import Path
from datetime import datetime
import re


def is_video_file(file_path):
    """비디오 파일 여부 확인"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    return Path(file_path).suffix.lower() in video_extensions


def is_audio_file(file_path):
    """오디오 파일 여부 확인"""
    audio_extensions = ['.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.wma']
    return Path(file_path).suffix.lower() in audio_extensions


def extract_audio_from_video(video_path, output_audio_path):
    """ffmpeg를 사용하여 비디오에서 오디오 추출"""
    print(f"  [비디오→오디오] 추출 중...")

    try:
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'libmp3lame',
            '-q:a', '2', '-y', output_audio_path
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"  ✓ 오디오 추출 완료")
        return True
    except Exception as e:
        print(f"  ✗ 오디오 추출 실패: {str(e)}")
        return False


def generate_filename_from_text(text, timestamp):
    """전사문 내용으로 파일명 생성"""
    # 전사문의 첫 50자를 추출하고 파일명에 사용 가능한 문자로 변환
    clean_text = text[:50].strip()
    # 파일명에 사용할 수 없는 문자 제거
    clean_text = re.sub(r'[<>:"/\\|?*\n\r]', ' ', clean_text)
    # 여러 공백을 하나로
    clean_text = re.sub(r'\s+', ' ', clean_text)
    # 파일명 길이 제한 (30자)
    clean_text = clean_text[:30].strip()

    # 타임스탬프 추가
    filename = f"{clean_text}_{timestamp}.txt"

    return filename


def transcribe_media(media_path, model_name='base', output_dir='./output'):
    """미디어 파일을 전사하고 결과 저장"""
    media_path = Path(media_path)
    output_dir = Path(output_dir)

    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"전사 시작: {media_path.name}")
    print(f"{'='*60}")

    # 파일 존재 확인
    if not media_path.exists():
        print(f"✗ 파일을 찾을 수 없습니다: {media_path}")
        return None

    # 파일 타입 확인
    if not (is_video_file(media_path) or is_audio_file(media_path)):
        print(f"✗ 지원하지 않는 파일 형식: {media_path.suffix}")
        return None

    temp_audio_path = None

    # 비디오인 경우 오디오 추출
    if is_video_file(media_path):
        temp_audio_path = media_path.parent / f".temp_{media_path.stem}.mp3"
        if not extract_audio_from_video(str(media_path), str(temp_audio_path)):
            return None
        audio_path = temp_audio_path
    else:
        audio_path = media_path

    try:
        # Whisper 모델 로드
        print(f"  [Whisper] {model_name} 모델 로딩 중...")
        model = whisper.load_model(model_name)

        # 전사 실행
        print(f"  [Whisper] 전사 중... (파일 크기에 따라 시간이 걸릴 수 있습니다)")
        result = model.transcribe(str(audio_path), language='ko', fp16=False)
        transcript_text = result['text'].strip()

        # 타임스탬프 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 파일명 생성
        output_filename = generate_filename_from_text(transcript_text, timestamp)
        output_path = output_dir / output_filename

        # 전사문 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcript_text)

        print(f"  ✓ 전사 완료!")
        print(f"  ✓ 저장 위치: {output_path}")
        print(f"\n[전사문 미리보기]")
        print("-" * 60)
        preview = transcript_text[:200] + "..." if len(transcript_text) > 200 else transcript_text
        print(preview)
        print("-" * 60)

        return output_path

    except Exception as e:
        print(f"  ✗ 전사 실패: {str(e)}")
        return None

    finally:
        # 임시 오디오 파일 삭제
        if temp_audio_path and temp_audio_path.exists():
            temp_audio_path.unlink()
            print(f"  ✓ 임시 파일 삭제")


def main():
    parser = argparse.ArgumentParser(
        description='미디어 파일을 한글 전사문으로 변환합니다.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python transcribe.py sources/audio.m4a
  python transcribe.py sources/video.mp4 --model tiny
  python transcribe.py sources/audio1.m4a sources/audio2.m4a --model base
        """
    )

    parser.add_argument(
        'files',
        nargs='+',
        help='전사할 미디어 파일 경로 (1개 이상)'
    )

    parser.add_argument(
        '--model',
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper 모델 (기본: base)'
    )

    parser.add_argument(
        '--output-dir',
        default='./output',
        help='출력 디렉토리 (기본: ./output)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("미디어 전사 도구")
    print(f"모델: {args.model}")
    print(f"출력 디렉토리: {args.output_dir}")
    print(f"파일 개수: {len(args.files)}")
    print("=" * 60)

    # 각 파일 처리
    success_count = 0
    for file_path in args.files:
        result = transcribe_media(file_path, args.model, args.output_dir)
        if result:
            success_count += 1

    # 최종 결과
    print("\n" + "=" * 60)
    print(f"전사 완료: {success_count}/{len(args.files)} 파일")
    print("=" * 60)

    sys.exit(0 if success_count == len(args.files) else 1)


if __name__ == "__main__":
    main()
