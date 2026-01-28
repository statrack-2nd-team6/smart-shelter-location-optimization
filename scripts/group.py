import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import googleapiclient.discovery
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

# 1. 영상 길이에 따른 그룹화
def categorize_duration(min):
    if min < 1: return 'Shorts'
    elif min < 60: return 'General'
    else: return 'Long-form (60m+)'

df['Group'] = df['Duration_Min'].apply(categorize_duration)

# 2. 그룹별 평균 계산
group_stats = df.groupby('Group').agg({
    'Views': 'mean',
    'Engagement_Rate': 'mean',
    'Title': 'count'
}).rename(columns={'Title': 'Video_Count'}).reset_index()

print("--- 그룹별 상세 통계 ---")
print(group_stats)

# 3. 시각화: 조회수 vs 인게이지먼트 비율 비교
fig, ax1 = plt.subplots(figsize=(10, 6))

# 바 차트: 평균 조회수
sns.barplot(data=group_stats, x='Group', y='Views', alpha=0.6, ax=ax1, palette='Blues')
ax1.set_ylabel('Average Views')

# 꺾은선 차트: 인게이지먼트 비율 (이중 축 사용)
ax2 = ax1.twinx()
sns.lineplot(data=group_stats, x='Group', y='Engagement_Rate', marker='o', color='red', ax=ax2, linewidth=3)
ax2.set_ylabel('Average Engagement Rate (%)')

plt.title('Content Strategy: Views vs. Engagement by Video Length')
plt.show()