import os
import googleapiclient.discovery
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import isodate # 영상 길이 파싱을 위해 필요 (pip install isodate)
from dotenv import load_dotenv

load_dotenv()

# 1. API 및 채널 설정
DEVELOPER_KEY = os.getenv("YOUTUBE_API_KEY")
channel_id = "UCQNE2JmbasNYbjGAcuBiRRg" # 조코딩 채널 ID

youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=DEVELOPER_KEY)

def get_length_vs_engagement(channel_id):
    video_data = []
    
    # 채널의 업로드 목록 ID 가져오기
    ch_response = youtube.channels().list(part="contentDetails", id=channel_id).execute()
    uploads_id = ch_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

    # 최근 영상 50개 가져오기
    res = youtube.playlistItems().list(part="contentDetails", playlistId=uploads_id, maxResults=50).execute()
    v_ids = [item['contentDetails']['videoId'] for item in res['items']]

    # 영상 상세 정보 가져오기 (길이, 조회수, 좋아요, 댓글)
    stats_res = youtube.videos().list(part="contentDetails,statistics,snippet", id=",".join(v_ids)).execute()

    for v in stats_res['items']:
        # ISO 8601 기간 형식을 초 단위로 변환 (예: PT10M30S -> 630)
        duration_sec = isodate.parse_duration(v['contentDetails']['duration']).total_seconds()
        
        # 통계 데이터 추출 (댓글 기능이 꺼져있을 수 있으므로 get 사용)
        views = int(v['statistics'].get('viewCount', 0))
        likes = int(v['statistics'].get('likeCount', 0))
        comments = int(v['statistics'].get('commentCount', 0))
        
        # 인게이지먼트 비율 계산 (조회수 대비 좋아요+댓글)
        engagement_rate = ((likes + comments) / views * 100) if views > 0 else 0
        
        video_data.append({
            'Title': v['snippet']['title'],
            'Duration_Min': duration_sec / 60, # 분석하기 쉽게 분 단위로 변환
            'Views': views,
            'Engagement_Rate': engagement_rate
        })
        
    return pd.DataFrame(video_data)

# 2. 데이터 수집
df = get_length_vs_engagement(channel_id)

# 3. 시각화 (산점도 및 회귀선)
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='Duration_Min', y='Engagement_Rate', scatter_kws={'alpha':0.5})
plt.title('Correlation: Video Length vs. Engagement Rate')
plt.xlabel('Video Duration (Minutes)')
plt.ylabel('Engagement Rate (%)')
plt.grid(True)
plt.show()

# 상관계수 출력
correlation = df['Duration_Min'].corr(df['Engagement_Rate'])
print(f"영상 길이와 호응도 간의 상관계수: {correlation:.2f}")