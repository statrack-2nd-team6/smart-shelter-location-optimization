# ## 세번째 영상만 작동함
#pip install pandas google-api-python-client yt-dlp requests

# import pandas as pd
# from googleapiclient.discovery import build
# import yt_dlp
# import requests
# import time
# import re

# # 1. API 설정
# API_KEY = 'YOUTUBE_API_KEY'
# youtube = build('youtube', 'v3', developerKey=API_KEY)

# def get_actual_transcript(v_id):
#     """자막 URL을 찾아 실제 텍스트 본문을 반환하는 함수"""
#     url = f"https://www.youtube.com/watch?v={v_id}"
    
#     ydl_opts = {
#         'skip_download': True,
#         'writesubtitles': True,
#         'writeautomaticsub': True,
#         'subtitleslangs': ['ko'], # 한국어 우선
#         'quiet': True,
#     }
    
#     try:
#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             info = ydl.extract_info(url, download=False)
#             subtitles = info.get('requested_subtitles')
            
#             # 한국어 자막 주소가 있는지 확인
#             if subtitles and 'ko' in subtitles:
#                 sub_url = subtitles['ko']['url']
                
#                 # 자막 파일(VTT/JSON 형식 등) 다운로드
#                 response = requests.get(sub_url)
#                 if response.status_code == 200:
#                     # XML/VTT 태그 및 타임스탬프 제거 (텍스트만 추출)
#                     clean_text = re.sub(r'<[^>]*>', '', response.text) # 태그 제거
#                     clean_text = re.sub(r'\d{2}:\d{2}:\d{2}.\d{3} --> \d{2}:\d{2}:\d{2}.\d{3}', '', clean_text) # 시간대 제거
#                     # 줄바꿈 및 중복 공백 정리
#                     lines = [line.strip() for line in clean_text.split('\n') if line.strip() and not line.strip().isdigit()]
#                     return " ".join(dict.fromkeys(lines)) # 중복 문장 제거 후 합침
#             return "자막 데이터 없음"
#     except Exception as e:
#         return f"추출 에러: {str(e)}"

# def main():
#     target_ids = ['IIHQ1Z2yu80', 'BnqvaC65vvM', 'or7Xc8tZg-g'] # 분석할 ID 리스트
#     all_data = []

#     # 기본 정보 수집 (제목, 설명 등)
#     request = youtube.videos().list(part="snippet,statistics", id=",".join(target_ids))
#     response = request.execute()

#     for item in response.get('items', []):
#         v_id = item['id']
#         snippet = item['snippet']
        
#         print(f"데이터 추출 중: {snippet['title'][:20]}...")
#         transcript = get_actual_transcript(v_id) # 자막 본문 추출 함수 호출
        
#         all_data.append({
#             'video_id': v_id,
#             'title': snippet['title'],
#             'description': snippet['description'].replace('\n', ' '),
#             'view_count': item['statistics'].get('viewCount', 0),
#             'transcript': transcript
#         })
#         time.sleep(1)

#     # CSV 저장
#     df = pd.DataFrame(all_data)
#     df.to_csv('youtube_final_transcript.csv', index=False, encoding='utf-8-sig')
#     print("\n[완료] 'youtube_final_transcript.csv' 파일을 확인하세요!")

# if __name__ == "__main__":
#     main()

import pandas as pd
from googleapiclient.discovery import build
import yt_dlp
import whisper
import os
import time

# 1. API 설정
API_KEY = 'AIzaSyDrgy04Jjh-ks11VKWfyj_bD5-Xz0m0lDg'
youtube = build('youtube', 'v3', developerKey=API_KEY)

# 2. Whisper 모델 로드
print("Whisper 모델 로딩 중 (잠시만 기다려주세요)...")
model = whisper.load_model("base")

def get_transcript_no_ffmpeg(v_id):
    """FFmpeg 없이 원본 오디오를 바로 Whisper로 전달하는 함수"""
    url = f"https://www.youtube.com/watch?v={v_id}"
    
    # FFmpeg 변환 과정을 완전히 빼버린 설정
    ydl_opts = {
        'format': 'm4a/bestaudio/best', # 가장 가벼운 원본 오디오 포맷
        'outtmpl': f'audio_{v_id}.%(ext)s',
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        # 1. 오디오 다운로드 (변환 없이 원본 그대로)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # 실제 다운로드된 파일명 확인 (확장자가 m4a 등일 수 있음)
            filename = ydl.prepare_filename(info)
        
        # 2. Whisper로 음성 인식 실행
        # Whisper는 FFmpeg이 없어도 오디오 라이브러리가 깔려있으면 일부 포맷을 읽을 수 있습니다.
        print(f"[{v_id}] 음성 분석 중...")
        result = model.transcribe(filename, fp16=False, language='ko')
        
        # 3. 파일 삭제
        if os.path.exists(filename):
            os.remove(filename)
            
        return result['text'].strip()
    
    except Exception as e:
        return f"추출 실패: {str(e)}"

def main():
    target_ids = ['IIHQ1Z2yu80', 'BnqvaC65vvM', 'or7Xc8tZg-g'] 
    all_data = []

    request = youtube.videos().list(part="snippet,statistics", id=",".join(target_ids))
    response = request.execute()

    for item in response.get('items', []):
        v_id = item['id']
        snippet = item['snippet']
        print(f"\n작업 시작: {snippet['title'][:20]}...")
        
        transcript = get_transcript_no_ffmpeg(v_id)
        
        all_data.append({
            'video_id': v_id,
            'title': snippet['title'],
            'transcript': transcript
        })

    df = pd.DataFrame(all_data)
    df.to_csv('youtube_final_result.csv', index=False, encoding='utf-8-sig')
    print("\n[성공] youtube_final_result.csv 파일이 생성되었습니다!")

if __name__ == "__main__":
    main()