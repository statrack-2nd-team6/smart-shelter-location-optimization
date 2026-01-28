import os
import googleapiclient.discovery
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

# API 설정
DEVELOPER_KEY = os.getenv("YOUTUBE_API_KEY")

youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=DEVELOPER_KEY)

def get_video_comments(video_id):
    comments = []
    # 한 번에 최대 100개의 댓글을 가져옵니다.
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100 
    )
    
    try:
        response = request.execute()
        for item in response['items']:
            # 댓글 텍스트 추출 및 소문자 변환
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment.lower())
    except Exception as e:
        print(f"오류 발생: {e}")
        
    return comments

# [입력 항목 2] 분석하고 싶은 영상의 ID를 여기에 넣으세요
# 예: https://www.youtube.com/watch?v=q8H_v_y_H8k 이라면 "q8H_v_y_H8k" 입력
video_id = "r6jn4Aj7PjU" 

# [입력 항목 3] 찾고 싶은 재방문 관련 키워드를 리스트에 넣으세요
target_keywords = ['저장', '다시', '완강', '복습', '나중에', '반복', '최고', '도움', '감사', '성지순례']

# 분석 실행
comments_list = get_video_comments(video_id)

# 키워드 빈도 계산
keyword_counts = Counter()
for comment in comments_list:
    for word in target_keywords:
        if word in comment:
            keyword_counts[word] += 1

# 결과 시각화
df_keywords = pd.DataFrame(keyword_counts.items(), columns=['Keyword', 'Count']).sort_values(by='Count', ascending=False)

if not df_keywords.empty:
    plt.figure(figsize=(10, 6))
    plt.bar(df_keywords['Keyword'], df_keywords['Count'], color='orange')
    plt.title(f'Engagement Keywords Analysis (Video ID: {video_id})')
    plt.xlabel('Revisit Keywords')
    plt.ylabel('Frequency')
    plt.show()
    print(df_keywords)
else:
    print("검색된 키워드가 없습니다. 키워드 목록을 수정하거나 다른 영상으로 시도해보세요.")