import requests
from PIL import Image
from io import BytesIO

# 구글에서 발급받은 API 키 입력
GOOGLE_API_KEY = 'AIzaSyBRV6HVSYMwD40mmaEFikiFTMZmCFk6UQw'

def get_google_satellite_image(lat, lon):
    """
    구글 정적 지도 API를 이용해 위성 사진 가져오기
    """
    url = "https://maps.googleapis.com/maps/api/staticmap"
    
    params = {
        'center': f'{lat},{lon}', # 구글은 위도(lat), 경도(lon) 순서입니다!
        'zoom': 18,               # 확대 레벨 (1~21)
        'size': '640x640',        # 이미지 크기 (최대 640x640)
        'maptype': 'satellite',   # 위성 사진 모드
        'key': GOOGLE_API_KEY
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        print(f"❌ 에러 발생: {response.status_code}")
        print(f"메시지: {response.text}")
        return None

# 테스트: 홍대입구역
print("🧪 구글 맵 테스트 시작: 홍대입구역")
img = get_google_satellite_image(lat=37.556641, lon=126.923466)

if img:
    img.save('google_hongdae.png')
    print("✅ 성공! google_hongdae.png 저장됨")
else:
    print("❌ 실패!")