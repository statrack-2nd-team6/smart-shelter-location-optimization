import googleapiclient.discovery
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# 1. API 설정
DEVELOPER_KEY = "YOUTUBE_API_KEY"
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=DEVELOPER_KEY)

# 2. 특정 영상의 댓글 수집 함수
def get_video_comments(video_id):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100  # 필요에 따라 조절
    )
    
    try:
        response = request.execute()
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment.lower())
    except Exception as e:
        print(f"댓글을 가져올 수 없습니다: {e}")
        
    return comments

# 3. 재방문 및 학습 의지 관련 키워드 설정
target_keywords = ['저장', '다시', '완강', '복습', '나중에', '반복', '최고', '도움', '감사', '성지순례']

# 4. 분석 실행 (그래프에서 확인한 가장 우상단의 140분 영상 ID 예시)
# 예: video_id = "v_ids[0]" (앞선 코드에서 추출한 리스트 활용 가능)
video_id = "YOUR_VIDEO_ID" 
comments_list = get_video_comments(video_id)

# 5. 키워드 빈도 계산
keyword_counts = Counter()
for comment in comments_list:
    for word in target_keywords:
        if word in comment:
            keyword_counts[word] += 1

# 6. 결과 시각화
df_keywords = pd.DataFrame(keyword_counts.items(), columns=['Keyword', 'Count']).sort_values(by='Count', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(df_keywords['Keyword'], df_keywords['Count'], color='orange')
plt.title(f'Revisit & Learning Intent Keywords in Video: {video_id}')
plt.ylabel('Frequency')
plt.show()

print(df_keywords)