# 🏠 스마트 쉼터 최적 입지 선정 시스템

서울시 버스정류장의 **대기오염도**, **이용객 수**, **배차간격**을 분석하여 스마트 쉼터 설치 우선순위를 예측하는 AI 시스템

## 📊 프로젝트 개요

### 핵심 개념: 오염 노출 위험도
```
우선순위 점수 = CAI × 40% + 승객수 × 40% + 대기시간 × 20%
```

- **대기오염도 (CAI)**: 얼마나 공기가 나쁜가?
- **이용객 수**: 얼마나 많은 사람이 노출되는가?
- **대기시간**: 얼마나 오래 노출되는가?

### 데이터셋
- **총 정류장**: 10,694개 (서울시 전체)
- **데이터 기간**: 2024.12 ~ 2025.11
- **Feature 수**: 35개 (파생 변수 포함)
- **결측치**: 0%

### 모델 성능
| 모델 | Test R² | Test RMSE | Test MAE |
|------|---------|-----------|----------|
| **Linear Regression** | **1.000000** | 0.000000 | 0.000000 |
| Ridge Regression | 0.999997 | 0.000111 | 0.000062 |
| Random Forest | 0.996083 | 0.004265 | 0.001813 |
| Gradient Boosting | 0.996792 | 0.003859 | 0.001940 |

**Best Model**: Linear Regression (R² = 1.0)
- 완벽한 예측: Target이 Features의 선형 결합이기 때문

---

## 🚀 빠른 시작

### 1. 필요한 라이브러리 설치
```bash
pip install streamlit folium streamlit-folium plotly scikit-learn pandas numpy
```

### 2. 모델 학습 (이미 완료됨)
```bash
python3 train_ml_models.py
```

생성되는 파일:
- `best_model.pkl`: 최고 성능 모델
- `scaler.pkl`: Feature 정규화 스케일러
- `model_metadata.json`: 모델 메타데이터
- `seoul_data.json`: 서울시 정류장 데이터
- `model_*.pkl`: 각 모델별 파일

### 3. 웹 데모 실행
```bash
streamlit run streamlit_app.py
```

브라우저에서 `http://localhost:8501` 접속

---

## 📂 파일 구조

```
smart_shelter/
├── README.md                          # 이 파일
├── EDA_Report_Final.md                # 완전한 EDA 보고서
├── dataset_engineered.csv             # Feature Engineering 완료 데이터
├── smart_shelter_dataset_final.csv    # 원본 통합 데이터
│
├── train_ml_models.py                 # ML 모델 학습 스크립트
├── streamlit_app.py                   # Streamlit 웹 애플리케이션
│
├── best_model.pkl                     # 최고 성능 모델 (Linear Regression)
├── scaler.pkl                         # StandardScaler
├── model_metadata.json                # 모델 메타데이터
├── seoul_data.json                    # 서울시 정류장 데이터 (10,694개)
│
└── models/                            # 개별 모델 파일
    ├── model_linear_regression.pkl
    ├── model_ridge_regression.pkl
    ├── model_random_forest.pkl
    └── model_gradient_boosting.pkl
```

---

## 🎯 웹 데모 기능

### Tab 1: 타 지역 데이터 입력 & 예측

**목적**: 다른 시/도에서 자체 데이터로 우선순위 예측

**입력 데이터**:
1. **대기측정소 정보**
   - 위치 (위도, 경도)
   - PM2.5, PM10, CAI 농도

2. **버스정류장 정보**
   - 위치 (위도, 경도)
   - 승차 인원
   - 배차 간격

**처리 과정**:
```
1. 주소 → 위경도 변환 (Geocoding)
2. IDW (Inverse Distance Weighting)로 정류장별 대기오염도 계산
3. Feature Engineering
4. 학습된 ML 모델로 우선순위 예측
5. 결과 시각화
```

**IDW (Inverse Distance Weighting)**:
- 거리 기반 보간법
- 가까운 측정소의 영향이 더 크게 반영
- 공식: `weight = 1 / distance^2`

### Tab 2: 서울시 지도 시각화

**기능**:
- 서울시 10,694개 정류장 지도 표시
- 우선순위 기준 색상 코딩
  - 🔴 빨강: 최우선 (상위 30%)
  - 🟠 주황: 우선 (30~60%)
  - 🔵 파랑: 일반 (60~80%)
  - 🟢 초록: 저우선 (하위 20%)
- 클릭 시 상세 정보 팝업
- Top N 필터링

### Tab 3: 대시보드 & 분석

**제공 정보**:
1. **전체 통계**
   - 총 정류장 수
   - 평균 우선순위
   - 최우선 설치 대상 개수
   - 평균 CAI

2. **자치구별 분석**
   - 평균 우선순위 (Top 10)
   - 정류장 수 분포

3. **분포 분석**
   - 우선순위 점수 히스토그램
   - CAI 분포

4. **상관관계**
   - CAI vs 승객수 산점도
   - 우선순위 색상 매핑

---

## 🔬 Feature Engineering

### 원본 Features (10개)
- `cai`: 통합대기환경지수
- `pm25`: 초미세먼지
- `pm10`: 미세먼지
- `o3`: 오존
- `no2`: 이산화질소
- `ridership`: 승차 인원
- `dispatch_interval`: 배차 간격
- `lat`, `lon`: 위치
- `district`: 자치구

### 파생 Features (7개)
1. **로그 변환**
   - `ridership_log = log(ridership + 1)`
   - Long-tail 분포 완화

2. **상호작용 변수**
   - `cai_ridership = cai × ridership`
   - `pollution_exposure_v2 = cai × dispatch_half`
   - `total_exposure = cai × ridership × dispatch_half`

3. **이진 변수**
   - `is_high_traffic`: ridership > Q3 (177,197명)
   - `is_high_pollution`: cai > Q3 (59.80)
   - `is_long_wait`: dispatch_half > median (6.0분)

### Feature Importance (Random Forest)
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | pm25 | 76.68% |
| 2 | total_exposure | 11.58% |
| 3 | pollution_exposure_v2 | 6.54% |
| 4 | cai | 2.65% |
| 5 | cai_ridership | 0.75% |

**인사이트**: PM2.5가 압도적으로 중요 (76.7%)

---

## 📊 데이터 분석 결과

### 주요 통계

#### 승차 인원
- 평균: 150,537명
- 중앙값: 58,940명
- 최대: 3,976,503명 (홍대입구역)

#### 배차 간격
- 평균: 12.72분
- 평균 대기시간: 6.36분

#### CAI
- 평균: 58.77
- 범위: 51.30 ~ 64.80
- 전체 '보통' 등급

### 상관관계 분석
| 변수1 | 변수2 | 상관계수 |
|-------|-------|----------|
| CAI | PM2.5 | 0.9907 ⭐ |
| 우선순위 | CAI | 0.8859 ⭐ |
| 우선순위 | 승객수 | 0.3713 |
| 우선순위 | 대기시간 | 0.2588 |
| CAI | 승객수 | 0.0169 ✅ |

**핵심**: CAI와 승객수는 독립적 → 다중공선성 없음 ✅

### 우선순위 TOP 5
| 순위 | 정류장명 | 자치구 | 점수 | CAI | 승객수 |
|------|----------|--------|------|-----|--------|
| 1 | 홍대입구역 | 마포구 | 0.7158 | 59.3 | 3,976,503 |
| 2 | 구로디지털단지역(중) | 관악구 | 0.7035 | 61.8 | 3,304,771 |
| 3 | 신도림역 | 구로구 | 0.6753 | 62.8 | 2,637,133 |
| 4 | 고속터미널 | 서초구 | 0.6661 | 59.7 | 3,569,629 |
| 5 | 구로디지털단지역 | 구로구 | 0.6641 | 61.8 | 2,896,908 |

---

## 🔧 기술 스택

### 데이터 처리
- **pandas**: 데이터 조작
- **numpy**: 수치 계산

### 머신러닝
- **scikit-learn**: ML 모델
  - LinearRegression
  - Ridge
  - RandomForestRegressor
  - GradientBoostingRegressor
- **scipy**: IDW 보간

### 시각화
- **plotly**: 인터랙티브 차트
- **folium**: 지도 시각화

### 웹 프레임워크
- **Streamlit**: 웹 애플리케이션

---

## 📈 모델 성능 해석

### Q: 왜 Linear Regression이 R² = 1.0인가?

**A**: Target 변수가 Features의 **선형 결합**이기 때문

```python
priority_v4 = cai_normalized × 0.4 + 
              ridership_normalized × 0.4 + 
              dispatch_half_normalized × 0.2
```

모든 변수가 이미 Features에 포함되어 있으므로, Linear Regression이 완벽하게 역산 가능!

### 실전 활용
- **회귀 모델**: 연속적인 우선순위 점수 (0~1)
- **분류 모델**: 4단계 등급 (최우선/우선/일반/저우선)
- **해석력**: Linear 모델 → 투명한 의사결정

---

## 🎓 프로젝트 하이라이트

### 1. 실용적 문제 해결
- ✅ 실제 스마트 쉼터 입지 선정에 활용 가능
- ✅ 대기오염 × 이용객 × 대기시간 통합 분석
- ✅ 타 지역 확장 가능 (IDW 활용)

### 2. 데이터 품질
- ✅ 결측치 0%
- ✅ 매칭률 100%
- ✅ 10,694개 샘플 (충분한 데이터)

### 3. Feature Engineering
- ✅ 로그 변환 (Long-tail 분포)
- ✅ 상호작용 변수 (오염 노출 지수)
- ✅ 이진 변수 (Q3 기준)

### 4. 모델 다양성
- ✅ ML: Linear, Ridge, Random Forest, Gradient Boosting
- ✅ 성능 비교 및 해석
- ✅ Best Model 자동 선택

### 5. 웹 데모
- ✅ 3개 탭 구성
- ✅ IDW 보간 실시간 계산
- ✅ 인터랙티브 지도 시각화
- ✅ 대시보드 분석

---

## 🚦 사용 예시

### 예시 1: 부산시 데이터 입력

**대기측정소 (3개)**:
1. 부산진구측정소: (35.16, 129.05), PM2.5=20, CAI=60
2. 해운대측정소: (35.16, 129.16), PM2.5=18, CAI=58
3. 사하구측정소: (35.10, 128.97), PM2.5=22, CAI=62

**버스정류장 (5개)**:
1. 서면역: (35.15, 129.06), 승객=200만, 배차=10분
2. 해운대역: (35.16, 129.16), 승객=150만, 배차=12분
3. ...

**→ IDW로 각 정류장 대기오염도 계산 후 우선순위 예측**

### 예시 2: 서울시 지도에서 Top 100 확인

1. Tab 2 선택
2. 슬라이더로 "100" 설정
3. 지도에서 빨간 마커 클릭
4. 우선순위 정보 확인

---

## 📝 향후 개선 사항

### 1. 모델 확장
- [ ] 딥러닝 모델 (MLP, TabNet)
- [ ] SHAP 분석 추가
- [ ] 시계열 예측 (시간대별 우선순위)

### 2. 기능 추가
- [ ] Geocoding API 연동 (주소 → 위경도)
- [ ] 실시간 대기오염 데이터 연동
- [ ] PDF 보고서 자동 생성
- [ ] 엑셀 다운로드

### 3. 성능 최적화
- [ ] 대용량 데이터 처리 (10만+ 정류장)
- [ ] 캐싱 강화
- [ ] 지도 렌더링 최적화

---

## 👥 기여자

- **주현**: SeSAC 데이터 분석 프로그램

---

## 📜 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.

---

## 📧 문의

프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.

---

**마지막 업데이트**: 2025-01-23  
**버전**: 1.0.0
