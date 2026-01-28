import os
import googleapiclient.discovery
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# 1. API 설정
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = os.getenv("YOUTUBE_API_KEY")

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)

# 2. 채널의 최신 영상 데이터 가져오기 (조코딩 채널 ID 예시)
# 채널 ID는 조코딩 채널 홈 주소 끝부분이나 '정보' 탭에서 확인 가능합니다.
channel_id = "UCQNE2JmbasNYbjGAcuBiRRg" 

def get_upload_time_analysis(channel_id):
    video_data = []
    
    # 채널의 업로드 목록 ID(uploads playlist ID) 가져오기
    ch_request = youtube.channels().list(part="contentDetails", id=channel_id)
    ch_response = ch_request.execute()
    uploads_id = ch_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

    # 최근 영상 50개 가져오기
    request = youtube.playlistItems().list(
        part="snippet,contentDetails",
        playlistId=uploads_id,
        maxResults=50
    )
    response = request.execute()

    for item in response['items']:
        video_id = item['contentDetails']['videoId']
        published_at = item['snippet']['publishedAt'] # '2023-10-27T09:00:00Z' 형식
        
        # 영상 상세 통계(조회수) 가져오기
        v_request = youtube.videos().list(part="statistics", id=video_id)
        v_response = v_request.execute()
        
        if v_response['items']:
            view_count = int(v_response['items'][0]['statistics']['viewCount'])
            
            # 시간 데이터 파싱 (UTC 기준을 로컬 시간으로 변환하려면 처리가 필요하지만, 여기서는 기본 파싱)
            dt = datetime.strptime(published_at, '%Y-%m-%dT%H:%M:%SZ')
            
            video_data.append({
                'published_at': dt,
                'day_of_week': dt.strftime('%A'), # 요일 이름
                'hour': dt.hour,                  # 시간(0-23)
                'views': view_count
            })
            
    return pd.DataFrame(video_data)

# 3. 데이터 수집 및 전처리
df = get_upload_time_analysis(channel_id)

# 요일 순서 정렬
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=days_order, ordered=True)

# 4. 시각화 1: 요일별 평균 조회수
plt.figure(figsize=(10, 5))
sns.barplot(data=df, x='day_of_week', y='views', estimator='mean', palette='viridis')
plt.title('Average Views by Upload Day')
plt.xticks(rotation=45)
plt.show()

# 5. 시각화 2: 시간대별 평균 조회수
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='hour', y='views', marker='o')
plt.title('Average Views by Upload Hour (UTC)')
plt.xticks(range(0, 24))
plt.grid(True)
plt.show()