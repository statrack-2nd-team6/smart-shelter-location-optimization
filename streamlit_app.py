"""
스마트 쉼터 최적 입지 선정 - Streamlit 웹 데모
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from joblib import load  
from scipy.interpolate import Rbf
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Page config
st.set_page_config(
    page_title="스마트 쉼터 최적 입지 선정",
    page_icon="🏠",
    layout="wide"
)

# Load models and data
@st.cache_resource
def load_models():
    model = load(BASE_DIR / "best_model.joblib")
    scaler = load(BASE_DIR / "scaler.joblib")

    with open(BASE_DIR / "model_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return model, scaler, metadata


@st.cache_data
def load_seoul_data():
    with open(BASE_DIR / "seoul_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


try:
    model, scaler, metadata = load_models()
    seoul_df = load_seoul_data()
    feature_cols = metadata["feature_cols"]
except FileNotFoundError as e:
    st.error(f"❌ 파일을 찾을 수 없습니다: {e}")
    st.write("BASE_DIR:", BASE_DIR)
    st.write("BASE_DIR 파일 목록:", [p.name for p in BASE_DIR.iterdir()])
    st.stop()
except Exception as e:
    # ✅ (추가) 로딩/역직렬화 오류를 명확히 보여줌
    st.error(f"❌ 모델 로딩 중 오류: {type(e).__name__}: {e}")
    st.write("BASE_DIR:", BASE_DIR)
    st.stop()

# Title
st.title("🏠 스마트 쉼터 최적 입지 선정 시스템")
st.markdown("**서울시 버스정류장 대기오염 및 이용객 기반 우선순위 분석**")
st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs([
    "📍 타 지역 데이터 입력 & 예측",
    "🗺️ 서울시 지도 시각화",
    "📊 대시보드 & 분석"
])

# =============================================================================
# TAB 1: 타 지역 데이터 입력 & 예측
# =============================================================================
with tab1:
    st.header("📍 타 지역 스마트 쉼터 우선순위 예측")
    st.markdown("""
    다른 시/도의 데이터를 입력하여 해당 지역의 버스정류장 우선순위를 예측합니다.
    **IDW (Inverse Distance Weighting)**를 사용하여 대기측정소 데이터로부터 정류장 대기오염도를 추정합니다.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1️⃣ 대기측정소 데이터 입력")

        n_stations = st.number_input(
            "대기측정소 개수",
            min_value=1,
            max_value=20,
            value=3,
            help="지역 내 대기측정소 개수를 입력하세요"
        )

        stations_data = []
        for i in range(n_stations):
            with st.expander(f"🏭 측정소 {i+1}", expanded=(i == 0)):
                st_name = st.text_input(
                    f"측정소 이름",
                    value=f"측정소{i+1}",
                    key=f"st_name_{i}"
                )
                st_lat = st.number_input(
                    "위도",
                    value=37.5 + i * 0.1,
                    format="%.6f",
                    key=f"st_lat_{i}"
                )
                st_lon = st.number_input(
                    "경도",
                    value=127.0 + i * 0.1,
                    format="%.6f",
                    key=f"st_lon_{i}"
                )
                st_pm25 = st.number_input(
                    "PM2.5 (㎍/㎥)",
                    value=18.0,
                    min_value=0.0,
                    key=f"st_pm25_{i}"
                )
                st_pm10 = st.number_input(
                    "PM10 (㎍/㎥)",
                    value=32.0,
                    min_value=0.0,
                    key=f"st_pm10_{i}"
                )
                st_cai = st.number_input(
                    "CAI",
                    value=58.0,
                    min_value=0.0,
                    key=f"st_cai_{i}"
                )

                stations_data.append({
                    "name": st_name,
                    "lat": st_lat,
                    "lon": st_lon,
                    "pm25": st_pm25,
                    "pm10": st_pm10,
                    "cai": st_cai
                })

    with col2:
        st.subheader("2️⃣ 버스정류장 데이터 입력")

        n_stops = st.number_input(
            "버스정류장 개수",
            min_value=1,
            max_value=50,
            value=5,
            help="예측할 버스정류장 개수를 입력하세요"
        )

        stops_data = []
        for i in range(n_stops):
            with st.expander(f"🚏 정류장 {i+1}", expanded=(i == 0)):
                stop_name = st.text_input(
                    "정류장 이름",
                    value=f"정류장{i+1}",
                    key=f"stop_name_{i}"
                )
                stop_lat = st.number_input(
                    "위도",
                    value=37.52 + i * 0.05,
                    format="%.6f",
                    key=f"stop_lat_{i}"
                )
                stop_lon = st.number_input(
                    "경도",
                    value=127.02 + i * 0.05,
                    format="%.6f",
                    key=f"stop_lon_{i}"
                )
                stop_ridership = st.number_input(
                    "승차 인원 (명)",
                    value=100000,
                    min_value=0,
                    key=f"stop_ridership_{i}"
                )
                stop_dispatch = st.number_input(
                    "배차 간격 (분)",
                    value=12.0,
                    min_value=1.0,
                    key=f"stop_dispatch_{i}"
                )

                stops_data.append({
                    "name": stop_name,
                    "lat": stop_lat,
                    "lon": stop_lon,
                    "ridership": stop_ridership,
                    "dispatch_interval": stop_dispatch
                })

    st.markdown("---")
    if st.button("🔮 우선순위 예측하기", type="primary", use_container_width=True):
        st.subheader("📊 예측 결과")

        def idw_interpolation(stations, stops, power=2):
            """IDW (Inverse Distance Weighting)"""
            results = []

            for stop in stops:
                stop_lat, stop_lon = stop["lat"], stop["lon"]

                distances = []
                for station in stations:
                    dlat = np.radians(station["lat"] - stop_lat)
                    dlon = np.radians(station["lon"] - stop_lon)
                    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(stop_lat)) * \
                        np.cos(np.radians(station["lat"])) * np.sin(dlon / 2) ** 2
                    c = 2 * np.arcsin(np.sqrt(a))
                    distance = 6371 * c  # km
                    distances.append(distance)

                distances = np.array(distances)

                if np.min(distances) < 0.001:
                    idx = np.argmin(distances)
                    weights = np.zeros(len(distances))
                    weights[idx] = 1.0
                else:
                    weights = 1 / (distances ** power)
                    weights = weights / np.sum(weights)

                pm25 = sum(w * s["pm25"] for w, s in zip(weights, stations))
                pm10 = sum(w * s["pm10"] for w, s in zip(weights, stations))
                cai = sum(w * s["cai"] for w, s in zip(weights, stations))

                results.append({
                    **stop,
                    "pm25": pm25,
                    "pm10": pm10,
                    "cai": cai
                })

            return results

        stops_with_pollution = idw_interpolation(stations_data, stops_data)

        predictions = []
        for stop in stops_with_pollution:
            ridership = stop["ridership"]
            ridership_log = np.log(ridership + 1)
            dispatch_half = stop["dispatch_interval"] / 2
            cai = stop["cai"]
            pm25 = stop["pm25"]
            pm10 = stop["pm10"]

            o3 = 0.033
            no2 = 0.018

            cai_ridership = cai * ridership / 1000000
            pollution_exposure_v2 = cai * dispatch_half
            total_exposure = cai * ridership * dispatch_half / 1000000

            is_high_traffic = 1 if ridership > 177197 else 0
            is_high_pollution = 1 if cai > 59.80 else 0
            is_long_wait = 1 if dispatch_half > 6.0 else 0

            features = [
                cai, pm25, pm10, o3, no2,
                ridership, ridership_log,
                stop["dispatch_interval"], dispatch_half,
                stop["lat"], stop["lon"],
                cai_ridership, pollution_exposure_v2, total_exposure,
                is_high_traffic, is_high_pollution, is_long_wait
            ]

            features_scaled = scaler.transform([features])
            priority = model.predict(features_scaled)[0]

            predictions.append({
                "정류장명": stop["name"],
                "위도": stop["lat"],
                "경도": stop["lon"],
                "승차인원": f"{ridership:,}명",
                "배차간격": f'{stop["dispatch_interval"]:.1f}분',
                "CAI": f"{cai:.2f}",
                "PM2.5": f"{pm25:.2f}㎍/㎥",
                "우선순위점수": f"{priority:.4f}",
                "등급": "최우선" if priority >= 0.6 else "우선" if priority >= 0.4 else "일반" if priority >= 0.2 else "저우선",
                "_priority": priority
            })

        predictions.sort(key=lambda x: x["_priority"], reverse=True)
        for p in predictions:
            del p["_priority"]

        df_pred = pd.DataFrame(predictions)
        df_pred.insert(0, "순위", range(1, len(df_pred) + 1))

        st.dataframe(df_pred, use_container_width=True, hide_index=True)

        st.subheader("📈 우선순위 분포")
        fig = px.bar(
            df_pred,
            x="정류장명",
            y=df_pred["우선순위점수"].apply(lambda x: float(x)),
            color="등급",
            color_discrete_map={
                "최우선": "#FF4136",
                "우선": "#FF851B",
                "일반": "#FFDC00",
                "저우선": "#2ECC40"
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 2: 서울시 지도 시각화
# =============================================================================
with tab2:
    st.header("🗺️ 서울시 버스정류장 우선순위 지도")

    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("⚙️ 설정")

        top_n = st.slider(
            "표시할 정류장 수",
            min_value=10,
            max_value=500,
            value=100,
            step=10
        )

        show_all = st.checkbox("전체 정류장 표시 (느릴 수 있음)", value=False)

    with col1:
        seoul_center = [37.5665, 126.9780]
        m = folium.Map(
            location=seoul_center,
            zoom_start=11,
            tiles="OpenStreetMap"
        )

        df_sorted = seoul_df.sort_values("priority", ascending=False)

        df_to_show = df_sorted if show_all else df_sorted.head(top_n)

        max_priority = df_to_show["priority"].max()
        min_priority = df_to_show["priority"].min()

        for _, row in df_to_show.iterrows():
            normalized = (row["priority"] - min_priority) / (max_priority - min_priority)

            if normalized > 0.7:
                color = "red"
                icon = "exclamation-sign"
            elif normalized > 0.4:
                color = "orange"
                icon = "warning-sign"
            elif normalized > 0.2:
                color = "lightblue"
                icon = "info-sign"
            else:
                color = "green"
                icon = "ok-sign"

            folium.Marker(
                location=[row["lat"], row["lon"]],
                popup=folium.Popup(f"""
                    <b>{row["name"]}</b><br>
                    자치구: {row["district"]}<br>
                    CAI: {row["cai"]:.2f}<br>
                    승객: {row["ridership"]:,}명<br>
                    <b>우선순위: {row["priority"]:.4f}</b>
                """, max_width=300),
                icon=folium.Icon(color=color, icon=icon)
            ).add_to(m)

        folium_static(m, width=None, height=600)
        st.info(f"📍 표시된 정류장: {len(df_to_show):,}개 / 전체 {len(seoul_df):,}개")

# =============================================================================
# TAB 3: 대시보드 & 분석
# =============================================================================
with tab3:
    st.header("📊 서울시 버스정류장 대시보드")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("총 정류장 수", f"{len(seoul_df):,}개")

    with col2:
        st.metric("평균 우선순위", f'{seoul_df["priority"].mean():.4f}')

    with col3:
        high_priority = len(seoul_df[seoul_df["priority"] >= 0.5])
        st.metric("최우선 설치 대상", f"{high_priority}개")

    with col4:
        avg_cai = seoul_df["cai"].mean()
        st.metric("평균 CAI", f"{avg_cai:.2f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📍 자치구별 평균 우선순위")
        district_avg = seoul_df.groupby("district")["priority"].mean().sort_values(ascending=False).head(10)

        fig = px.bar(
            x=district_avg.values,
            y=district_avg.index,
            orientation="h",
            labels={"x": "평균 우선순위", "y": "자치구"},
            color=district_avg.values,
            color_continuous_scale="Reds"
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🚌 자치구별 정류장 수")
        district_count = seoul_df["district"].value_counts().head(10)

        fig = px.bar(
            x=district_count.values,
            y=district_count.index,
            orientation="h",
            labels={"x": "정류장 수", "y": "자치구"},
            color=district_count.values,
            color_continuous_scale="Blues"
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 우선순위 점수 분포")
        fig = px.histogram(
            seoul_df,
            x="priority",
            nbins=50,
            labels={"priority": "우선순위 점수"},
            color_discrete_sequence=["#FF4136"]
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🌫️ CAI 분포")
        fig = px.histogram(
            seoul_df,
            x="cai",
            nbins=50,
            labels={"cai": "CAI"},
            color_discrete_sequence=["#0074D9"]
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("🔗 CAI vs 승객수 산점도")
    fig = px.scatter(
        seoul_df.sample(min(1000, len(seoul_df))),
        x="cai",
        y="ridership",
        color="priority",
        size="priority",
        hover_data=["name", "district"],
        labels={"cai": "CAI", "ridership": "승차 인원", "priority": "우선순위"},
        color_continuous_scale="Reds"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🏠 <b>스마트 쉼터 최적 입지 선정 시스템</b></p>
    <p>서울시 버스정류장 10,694개 분석 | ML 모델: Linear Regression (R² = 1.0)</p>
    <p>SeSAC 데이터 분석 프로젝트 2025</p>
</div>
""", unsafe_allow_html=True)