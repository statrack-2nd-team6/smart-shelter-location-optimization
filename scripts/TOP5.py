import os
import googleapiclient.discovery
import pandas as pd
import isodate
from dotenv import load_dotenv

load_dotenv()

# API 설정
DEVELOPER_KEY = os.getenv("YOUTUBE_API_KEY")
channel_id = "UCQNE2JmbasNYbjGAcuBiRRg"

youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=DEVELOPER_KEY)

def get_length_vs_engagement(channel_id):
    video_data = []
    ch_response = youtube.channels().list(part="contentDetails", id=channel_id).execute()
    uploads_id = ch_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
    res = youtube.playlistItems().list(part="contentDetails", playlistId=uploads_id, maxResults=50).execute()
    v_ids = [item['contentDetails']['videoId'] for item in res['items']]
    stats_res = youtube.videos().list(part="contentDetails,statistics,snippet", id=",".join(v_ids)).execute()

    for v in stats_res['items']:
        duration_sec = isodate.parse_duration(v['contentDetails']['duration']).total_seconds()
        views = int(v['statistics'].get('viewCount', 0))
        likes = int(v['statistics'].get('likeCount', 0))
        comments = int(v['statistics'].get('commentCount', 0))
        engagement_rate = ((likes + comments) / views * 100) if views > 0 else 0
        video_data.append({
            'Title': v['snippet']['title'],
            'Duration_Min': duration_sec / 60,
            'Views': views,
            'Engagement_Rate': engagement_rate
        })
    return pd.DataFrame(video_data)

df = get_length_vs_engagement(channel_id)

# 인게이지먼트 비율(ER)이 가장 높은 상위 5개 영상 확인
top_engagement = df.sort_values(by='Engagement_Rate', ascending=False).head(5)
print("--- 인게이지먼트 비율 상위 영상 ---")
print(top_engagement[['Title', 'Duration_Min', 'Engagement_Rate']])

# 영상 길이가 60분 이상인 영상들만 따로 보기
long_videos = df[df['Duration_Min'] >= 60]
print("\n--- 60분 이상 장편 영상 통계 ---")
print(long_videos[['Title', 'Duration_Min', 'Engagement_Rate']])