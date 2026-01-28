import os
import googleapiclient.discovery
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

# 1. API 설정
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = os.getenv("YOUTUBE_API_KEY")

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)

# 2. 재생목록 내의 모든 영상 정보 가져오기 함수
def get_playlist_stats(playlist_id):
    video_data = []
    
    # 재생목록 아이템 호출
    request = youtube.playlistItems().list(
        part="snippet,contentDetails",
        playlistId=playlist_id,
        maxResults=50
    )
    response = request.execute()

    for item in response['items']:
        video_id = item['contentDetails']['videoId']
        title = item['snippet']['title']
        
        # 각 영상의 상세 통계(조회수) 호출
        video_request = youtube.videos().list(
            part="statistics",
            id=video_id
        )
        video_response = video_request.execute()
        
        view_count = int(video_response['items'][0]['statistics']['viewCount'])
        video_data.append({'Title': title, 'Views': view_count})
    
    return pd.DataFrame(video_data)

# 3. 데이터 실행 및 유지율 계산
# 조코딩의 '수익형 앱 만들기' 재생목록 ID 예시
playlist_id = "PLU9-uwewPMe2AX9o9hFgv-nRvOcBdzvP5" 
df = get_playlist_stats(playlist_id)

# 첫 번째 영상 조회수 기준 유지율(%) 계산
first_view = df.iloc[0]['Views']
df['Retention_Rate'] = (df['Views'] / first_view) * 100

print(df[['Title', 'Views', 'Retention_Rate']])

# 4. 시각화
plt.figure(figsize=(12, 6))
plt.plot(df.index + 1, df['Retention_Rate'], marker='o', color='b', linestyle='-')
plt.title('Course Retention Rate Analysis')
plt.xlabel('Video Sequence (Episode)')
plt.ylabel('Retention Rate (%)')
plt.grid(True)
plt.show()
