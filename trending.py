import os
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()

def get_trending_videos(api_key, region, max_results):
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    # 인기 급상승 동영상 목록 호출
    request = youtube.videos().list(
        part="snippet,statistics,contentDetails",
        chart="mostPopular",      # 핵심 파라미터: 인기 급상승 차트
        regionCode=region,        # 국가 코드 (한국은 KR)
        maxResults=max_results
    )
    response = request.execute()

    trending_data = []

    for item in response['items']:
        thumbnails = item['snippet']['thumbnails']
        # 보통 'maxres' (최고화질) 또는 'high' (고화질)를 사용합니다.
        thumbnail_url = thumbnails.get('maxres', thumbnails.get('high')).get('url')
        
        trending_data.append({
            'rank': len(trending_data) + 1,
            'title': item['snippet']['title'],
            'channel': item['snippet']['channelTitle'],
            'view_count': item['statistics'].get('viewCount', 0),
            'like_count': item['statistics'].get('likeCount', 0),
            'comment_count': item['statistics'].get('commentCount', 0),
            'published_at': item['snippet']['publishedAt'],
            'video_url': f"https://www.youtube.com/watch?v={item['id']}",
            'thumbnail': thumbnail_url,
            'video_id': item['id']
        })

    return pd.DataFrame(trending_data)

# 실행하기
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
df_trending = get_trending_videos(YOUTUBE_API_KEY,'KR',50)

file_path = 'trending_history.csv'
kst = timezone(timedelta(hours=9))
now_kst = datetime.now(kst) 
df_trending['created_at'] = now_kst.strftime('%Y-%m-%d %H:%M:%S')# 언제 수집했는지 기록

if not os.path.exists(file_path):
    # 파일이 없으면 새로 생성 (헤더 포함)
    df_trending.to_csv(file_path, index=False, encoding='utf-8-sig')
else:
    # 파일이 있으면 기존 데이터 밑에 추가 (헤더 제외)
    df_trending.to_csv(file_path, mode='a', header=False, index=False, encoding='utf-8-sig')

