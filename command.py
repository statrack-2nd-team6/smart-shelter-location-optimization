import os
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

if not YOUTUBE_API_KEY:
    raise ValueError("YOUTUBE_API_KEY가 설정되지 않았습니다.")

youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# '파이썬 데이터 분석' 키워드로 검색 결과 5개 가져오기
request = youtube.search().list(
    part="snippet",
    q="파이썬 데이터 분석",
    maxResults=5,
    type="video"
)
response = request.execute()

for item in response["items"]:
    print(f"제목: {item['snippet']['title']}")
    print(f"동영상 ID: {item['id']['videoId']}\n")