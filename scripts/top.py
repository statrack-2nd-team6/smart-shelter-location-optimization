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
            'Engagement_Rate': engagement_rate,
            'Video_ID': v['id']
        })
    return pd.DataFrame(video_data)

df = get_length_vs_engagement(channel_id)
top_video = df.sort_values(by='Engagement_Rate', ascending=False).iloc[0]

# 결과 출력
print("=== 인게이지먼트가 가장 높은 영상 정보 ===")
print(f"제목: {top_video['Title']}")
print(f"영상 길이: {top_video['Duration_Min']:.2f}분")
print(f"인게이지먼트 비율: {top_video['Engagement_Rate']:.2f}%")
print(f"조회수: {top_video['Views']}회")

# 4. 댓글 분석에 바로 사용할 수 있도록 ID 추출
# 기존 수집 단계에서 'Video_ID'를 df에 저장해두었다면 아래와 같이 가져옵니다.
# (앞선 코드에 Video_ID 저장이 누락되었다면 아래 '보완된 수집 코드'를 참고하세요)
best_video_id = top_video['Video_ID'] 
print(f"영상 ID (v= 뒷부분): {best_video_id}")