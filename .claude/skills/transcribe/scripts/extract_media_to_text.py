#!/usr/bin/env python3
"""
미디어 파일(비디오/오디오)에서 한글 전사문을 추출하는 스크립트
- ffmpeg를 사용하여 비디오에서 오디오 추출
- Whisper Python API를 사용하여 한글 전사문 생성
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import whisper


def is_video_file(file_path):
    """비디오 파일 여부 확인"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    return Path(file_path).suffix.lower() in video_extensions


def is_audio_file(file_path):
    """오디오 파일 여부 확인"""
    audio_extensions = ['.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.wma']
    return Path(file_path).suffix.lower() in audio_extensions


def extract_audio_from_video(video_path, output_audio_path):
    """
    ffmpeg를 사용하여 비디오에서 오디오 추출

    Args:
        video_path: 입력 비디오 파일 경로
        output_audio_path: 출력 오디오 파일 경로

    Returns:
        bool: 성공 여부
    """
    print(f"비디오에서 오디오 추출 중: {video_path}")

    try:
        # ffmpeg 명령어: 비디오에서 오디오만 추출
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # 비디오 스트림 제외
            '-acodec', 'libmp3lame',  # MP3 코덱 사용
            '-q:a', '2',  # 오디오 품질 (0-9, 낮을수록 고품질)
            '-y',  # 덮어쓰기 허용
            output_audio_path
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )

        print(f"오디오 추출 완료: {output_audio_path}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"오디오 추출 실패: {e.stderr.decode('utf-8')}")
        return False
    except Exception as e:
        print(f"오디오 추출 중 오류 발생: {str(e)}")
        return False


def transcribe_audio_with_whisper(audio_path, model_name='tiny'):
    """
    Whisper Python API를 사용하여 오디오를 한글 전사문으로 변환

    Args:
        audio_path: 입력 오디오 파일 경로
        model_name: Whisper 모델 이름 (tiny, base, small, medium, large 중 선택)

    Returns:
        str: 전사문 텍스트 또는 None (실패 시)
    """
    print(f"Whisper를 사용하여 전사 중: {audio_path}")

    try:
        # Whisper 모델 로드
        print(f"Whisper {model_name} 모델 로딩 중...")
        model = whisper.load_model(model_name)

        # 음성 인식 실행
        # language='ko': 한국어 지정
        # fp16=False: CPU에서도 실행 가능하도록 설정
        print("Whisper 실행 중... (시간이 걸릴 수 있습니다)")
        result = model.transcribe(audio_path, language='ko', fp16=False)

        transcript_text = result['text'].strip()

        # 전사문을 텍스트 파일로 저장
        audio_filename = Path(audio_path).stem
        transcript_path = Path(audio_path).parent / f"{audio_filename}.txt"

        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript_text)

        print("전사 완료!")
        print(f"전사문이 저장되었습니다: {transcript_path}")

        return transcript_text

    except Exception as e:
        print(f"전사 중 오류 발생: {str(e)}")
        return None


def extract_media_to_text(media_path, keep_temp_audio=False):
    """
    미디어 파일(비디오/오디오)에서 한글 전사문 추출

    Args:
        media_path: 입력 미디어 파일 경로
        keep_temp_audio: 임시 오디오 파일 보관 여부 (비디오인 경우)

    Returns:
        str: 전사문 텍스트 또는 None (실패 시)
    """
    media_path = Path(media_path)

    # 파일 존재 여부 확인
    if not media_path.exists():
        print(f"오류: 파일을 찾을 수 없습니다 - {media_path}")
        return None

    # 파일 타입 확인
    if not (is_video_file(media_path) or is_audio_file(media_path)):
        print(f"오류: 지원하지 않는 파일 형식입니다 - {media_path.suffix}")
        print("지원 형식: 비디오(.mp4, .avi, .mov 등), 오디오(.mp3, .wav, .m4a 등)")
        return None

    temp_audio_path = None

    # 비디오 파일인 경우 오디오 추출
    if is_video_file(media_path):
        print("\n[단계 1/2] 비디오에서 오디오 추출")
        temp_audio_path = media_path.parent / f"{media_path.stem}_extracted.mp3"

        if not extract_audio_from_video(str(media_path), str(temp_audio_path)):
            return None

        audio_path = temp_audio_path
    else:
        print("\n[단계 1/2] 오디오 파일 확인 완료")
        audio_path = media_path

    # Whisper로 전사
    print("\n[단계 2/2] Whisper를 사용한 전사 시작")
    transcript = transcribe_audio_with_whisper(str(audio_path))

    # 임시 오디오 파일 삭제 (비디오에서 추출한 경우)
    if temp_audio_path and temp_audio_path.exists() and not keep_temp_audio:
        temp_audio_path.unlink()
        print(f"\n임시 오디오 파일 삭제: {temp_audio_path}")

    return transcript


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='미디어 파일에서 한글 전사문을 추출합니다.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python extract_media_to_text.py video.mp4
  python extract_media_to_text.py audio.m4a
  python extract_media_to_text.py video.mp4 --keep-audio
        """
    )

    parser.add_argument(
        'media_path',
        help='비디오 또는 오디오 파일 경로'
    )

    parser.add_argument(
        '--keep-audio',
        action='store_true',
        help='비디오에서 추출한 오디오 파일을 보관합니다'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("미디어 파일 → 한글 전사문 추출 도구")
    print("=" * 60)

    # 전사 실행
    transcript = extract_media_to_text(args.media_path, args.keep_audio)

    if transcript:
        print("\n" + "=" * 60)
        print("전사 완료!")
        print("=" * 60)
        print("\n[전사문 미리보기]")
        print("-" * 60)
        # 처음 500자만 출력
        preview = transcript[:500] + "..." if len(transcript) > 500 else transcript
        print(preview)
        print("-" * 60)
    else:
        print("\n전사 실패")
        sys.exit(1)


if __name__ == "__main__":
    main()
