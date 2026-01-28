#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Maps Static API를 이용한 버스 정류장 위성 이미지 다운로드
"""

import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import time
import os

# ============================================
# 🔑 여기에 Google API Key를 입력하세요!
# ============================================
GOOGLE_API_KEY = 'AIzaSyBRV6HVSYMwD40mmaEFikiFTMZmCFk6UQw'  # ⬅️ 여기 수정!

# ============================================
# 📁 설정
# ============================================
INPUT_CSV = 'dataset_engineered.csv'  # 입력 파일
OUTPUT_DIR = 'bus_stop_images'        # 출력 폴더

# ============================================
# 🛠️ 함수 정의
# ============================================

def get_google_satellite_image(lat, lon, api_key=GOOGLE_API_KEY):
    """
    Google Maps Static API로 위성 이미지 다운로드
    
    Parameters:
    -----------
    lat : float
        위도 (Latitude)
    lon : float
        경도 (Longitude)
    api_key : str
        Google API Key
        
    Returns:
    --------
    PIL.Image or None
        성공 시 이미지 객체, 실패 시 None
    """
    url = "https://maps.googleapis.com/maps/api/staticmap"
    
    params = {
        'center': f'{lat},{lon}',      # 중심 좌표 (위도,경도)
        'zoom': 18,                     # 확대 수준 (1~20, 18이 적당)
        'size': '400x400',              # 이미지 크기 (최대 640x640)
        'maptype': 'satellite',         # satellite(위성), roadmap(일반)
        'key': api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            # 정상 응답
            img = Image.open(BytesIO(response.content))
            return img
        elif response.status_code == 403:
            print(f"    ❌ API Key 오류 또는 할당량 초과")
            return None
        else:
            print(f"    ⚠️  HTTP {response.status_code}: {response.text[:100]}")
            return None
            
    except Exception as e:
        print(f"    ⚠️  예외 발생: {e}")
        return None


def download_with_retry(lat, lon, api_key, max_retries=3):
    """
    재시도 기능이 포함된 다운로드
    
    Parameters:
    -----------
    lat : float
        위도
    lon : float
        경도
    api_key : str
        API Key
    max_retries : int
        최대 재시도 횟수
        
    Returns:
    --------
    PIL.Image or None
    """
    for attempt in range(max_retries):
        img = get_google_satellite_image(lat, lon, api_key)
        
        if img:
            return img
        
        if attempt < max_retries - 1:
            print(f"    🔄 재시도 {attempt + 1}/{max_retries}...")
            time.sleep(1)
    
    return None


# ============================================
# 🚀 메인 실행
# ============================================

def main():
    print("=" * 80)
    print("🗺️  Google Maps 위성 이미지 다운로드 시작")
    print("=" * 80)
    
    # API Key 확인
    if GOOGLE_API_KEY == 'YOUR_GOOGLE_API_KEY_HERE':
        print("\n❌ 에러: API Key를 입력하지 않았습니다!")
        print("\n📝 해결 방법:")
        print("   1. Google Cloud Console에서 API Key 발급")
        print("   2. 이 파일 상단의 GOOGLE_API_KEY 변수에 입력")
        print("   3. 다시 실행")
        return
    
    # 폴더 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n📁 출력 폴더: {OUTPUT_DIR}")
    
    # 데이터 로드
    print(f"📥 데이터 로드 중: {INPUT_CSV}")
    
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"\n❌ 에러: '{INPUT_CSV}' 파일을 찾을 수 없습니다!")
        print(f"\n📝 해결 방법:")
        print(f"   1. 파일이 현재 폴더에 있는지 확인")
        print(f"   2. 파일명이 정확한지 확인")
        return
    
    print(f"✅ {len(df):,}개 정류장 로드 완료")
    
    # 필요한 컬럼 확인
    required_cols = ['lat', 'lon', 'name']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"\n❌ 에러: 필요한 컬럼이 없습니다: {missing_cols}")
        print(f"\n현재 컬럼: {list(df.columns)}")
        return
    
    # 통계 변수
    success_count = 0
    fail_count = 0
    skip_count = 0
    start_time = time.time()
    
    print(f"\n🚀 다운로드 시작...\n")
    
    # 이미지 다운로드
    for idx, row in df.iterrows():
        lat = row['lat']
        lon = row['lon']
        stop_name = row['name']
        
        # 파일명 (특수문자 제거)
        safe_name = "".join(c for c in stop_name if c.isalnum() or c in (' ', '-', '_'))
        filename = f"{OUTPUT_DIR}/{idx:05d}_{safe_name[:30]}.png"
        
        # 이미 존재하면 스킵
        if os.path.exists(filename):
            skip_count += 1
            if (idx + 1) % 100 == 0:
                print(f"[{idx+1:5d}/{len(df)}] ⏭️  스킵 (이미 존재)")
            continue
        
        # 다운로드
        img = download_with_retry(lat, lon, GOOGLE_API_KEY)
        
        if img:
            img.save(filename)
            success_count += 1
            
            # 진행률 표시 (10개마다)
            if (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                eta = (len(df) - idx - 1) / rate
                
                print(f"[{idx+1:5d}/{len(df)}] ✅ {stop_name[:20]:20s} "
                      f"| {rate:.1f}개/초 | ETA: {eta/60:.0f}분")
        else:
            fail_count += 1
            print(f"[{idx+1:5d}/{len(df)}] ❌ {stop_name[:20]:20s}")
        
        # API 제한 고려 (초당 50건까지 가능하지만 여유있게)
        time.sleep(0.1)
    
    # 완료 통계
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("🎉 다운로드 완료!")
    print("=" * 80)
    print(f"\n📊 통계:")
    print(f"   ✅ 성공: {success_count:,}개")
    print(f"   ❌ 실패: {fail_count:,}개")
    print(f"   ⏭️  스킵: {skip_count:,}개")
    print(f"   📦 총합: {len(df):,}개")
    print(f"\n⏱️  소요 시간: {total_time/60:.1f}분")
    
    if success_count > 0:
        print(f"   평균 속도: {success_count/total_time:.1f}개/초")
    
    if fail_count > 0:
        print(f"\n⚠️  {fail_count}개 실패")
        print(f"   → 다시 실행하면 실패한 것만 재시도됩니다.")
    
    # 무료 한도 경고
    total_downloaded = success_count + skip_count
    free_limit = 28500  # 월 무료 한도
    
    if total_downloaded > free_limit:
        excess = total_downloaded - free_limit
        cost = (excess / 1000) * 7  # $7 per 1000 requests
        print(f"\n💰 비용 예상:")
        print(f"   무료 한도 초과: {excess:,}개")
        print(f"   예상 비용: ${cost:.2f} (약 {cost * 1300:.0f}원)")


if __name__ == '__main__':
    main()
