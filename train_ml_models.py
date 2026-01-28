#!/usr/bin/env python3
"""
스마트 쉼터 우선순위 예측 - ML 모델 학습
- 배포(Streamlit Cloud) 안정성을 위해 pickle 대신 joblib 사용
- 실행 위치(cwd)와 무관하게 동작하도록 경로를 파일 기준으로 고정
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset_engineered.csv"

SCALER_PATH = BASE_DIR / "scaler.joblib"
BEST_MODEL_PATH = BASE_DIR / "best_model.joblib"
METADATA_PATH = BASE_DIR / "model_metadata.json"
SEOUL_DATA_PATH = BASE_DIR / "seoul_data.json"


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    print("=" * 80)
    print("🤖 ML 모델 학습 시작")
    print("=" * 80)
    print()

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("📥 데이터 로딩...")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"dataset_engineered.csv를 찾을 수 없습니다.\n"
            f"- 기대 경로: {DATASET_PATH}\n"
            f"- 현재 파일 기준 폴더(BASE_DIR): {BASE_DIR}"
        )

    with open(DATASET_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        data = list(reader)

    print(f"✅ 데이터 로드 완료: {len(data):,}개 샘플")
    print()

    # -------------------------------------------------------------------------
    # Prepare features and target
    # -------------------------------------------------------------------------
    print("🔧 Feature 및 Target 준비...")

    feature_cols = [
        "cai", "pm25", "pm10", "o3", "no2",
        "ridership", "ridership_log",
        "dispatch_interval", "dispatch_half",
        "lat", "lon",
        "cai_ridership", "pollution_exposure_v2", "total_exposure",
        "is_high_traffic", "is_high_pollution", "is_long_wait",
    ]

    X: list[list[float]] = []
    y: list[float] = []
    seoul_rows = []  # For web app (seoul_data.json)

    missing_cols = set()
    for row in data:
        for col in feature_cols + ["priority_v4", "stop_id", "name", "district", "lat", "lon", "cai", "ridership"]:
            if col not in row:
                missing_cols.add(col)

    if missing_cols:
        raise KeyError(f"CSV에 필요한 컬럼이 없습니다: {sorted(missing_cols)}")

    for row in data:
        features = [float(row[col]) for col in feature_cols]
        target = float(row["priority_v4"])

        X.append(features)
        y.append(target)

        # Save metadata for web app (서울 정류장 리스트)
        seoul_rows.append({
            "stop_id": row["stop_id"],
            "name": row["name"],
            "district": row["district"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "cai": float(row["cai"]),
            "ridership": int(float(row["ridership"])),
            "priority": target,
        })

    X_np = np.array(X, dtype=np.float64)
    y_np = np.array(y, dtype=np.float64)

    print(f"✅ Feature shape: {X_np.shape}")
    print(f"✅ Target shape: {y_np.shape}")
    print(f"✅ Feature 목록 ({len(feature_cols)}개):")
    for i, col in enumerate(feature_cols, 1):
        print(f"   {i:2d}. {col}")
    print()

    # -------------------------------------------------------------------------
    # Train-test split
    # -------------------------------------------------------------------------
    print("✂️  Train-Test Split (80:20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=0.2, random_state=42
    )
    print(f"✅ Train: {len(X_train):,}개")
    print(f"✅ Test: {len(X_test):,}개")
    print()

    # -------------------------------------------------------------------------
    # Feature scaling (IMPORTANT: keep consistent with web app)
    # -------------------------------------------------------------------------
    print("📊 Feature Scaling (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✅ Scaling 완료")
    print()

    # -------------------------------------------------------------------------
    # Train models
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("🎯 모델 학습 시작")
    print("=" * 80)
    print()

    # NOTE:
    # Streamlit app에서 scaler.transform(features)를 한 뒤 model.predict()를 호출하고 있으므로,
    # 모든 모델을 "scaled 입력" 기준으로 학습/예측하도록 통일.
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
        ),
    }

    results: dict[str, dict[str, float]] = {}

    for name, model in models.items():
        print(f"🔄 Training {name}...")

        model.fit(X_train_scaled, y_train)

        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)

        train_rmse = _rmse(y_train, y_pred_train)
        train_mae = float(mean_absolute_error(y_train, y_pred_train))
        train_r2 = float(r2_score(y_train, y_pred_train))

        test_rmse = _rmse(y_test, y_pred_test)
        test_mae = float(mean_absolute_error(y_test, y_pred_test))
        test_r2 = float(r2_score(y_test, y_pred_test))

        results[name] = {
            "train_rmse": train_rmse,
            "train_mae": train_mae,
            "train_r2": train_r2,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "test_r2": test_r2,
        }

        print(f"   Train - RMSE: {train_rmse:.6f}, MAE: {train_mae:.6f}, R²: {train_r2:.6f}")
        print(f"   Test  - RMSE: {test_rmse:.6f}, MAE: {test_mae:.6f}, R²: {test_r2:.6f}")
        print()

    # -------------------------------------------------------------------------
    # Results summary
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("📊 모델 성능 비교")
    print("=" * 80)
    print()
    print(f'{"Model":<25s} {"Test RMSE":<12s} {"Test MAE":<12s} {"Test R²":<12s}')
    print("-" * 80)
    for name, metrics in results.items():
        print(
            f'{name:<25s} '
            f'{metrics["test_rmse"]:<12.6f} '
            f'{metrics["test_mae"]:<12.6f} '
            f'{metrics["test_r2"]:<12.6f}'
        )
    print()

    best_model_name = max(results, key=lambda x: results[x]["test_r2"])
    best_model = models[best_model_name]

    print(f"🏆 Best Model: {best_model_name}")
    print(f'   Test R² Score: {results[best_model_name]["test_r2"]:.6f}')
    print()

    # -------------------------------------------------------------------------
    # Feature importance (tree models)
    # -------------------------------------------------------------------------
    if best_model_name in ["Random Forest", "Gradient Boosting"]:
        print("=" * 80)
        print("📊 Feature Importance (Top 10)")
        print("=" * 80)

        importances = best_model.feature_importances_
        feature_importance = sorted(
            zip(feature_cols, importances),
            key=lambda x: x[1],
            reverse=True,
        )

        for i, (feat, imp) in enumerate(feature_importance[:10], 1):
            print(f"{i:2d}. {feat:<30s}: {imp:.6f}")
        print()

    # -------------------------------------------------------------------------
    # Save artifacts (joblib + json)
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("💾 모델 및 전처리기 저장")
    print("=" * 80)

    # Scaler
    dump(scaler, SCALER_PATH)
    print(f"✅ Scaler 저장: {SCALER_PATH.name}")

    # Best model
    dump(best_model, BEST_MODEL_PATH)
    print(f"✅ Best Model 저장: {BEST_MODEL_PATH.name} ({best_model_name})")

    # All models (optional)
    for name, model in models.items():
        safe_name = name.replace(" ", "_").lower()
        path = BASE_DIR / f"model_{safe_name}.joblib"
        dump(model, path)
        print(f"✅ {name} 저장: {path.name}")

    # Metadata for app
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_cols": feature_cols,
                "best_model_name": best_model_name,
                "results": results,
                "data_info": {
                    "total_samples": int(len(data)),
                    "train_samples": int(len(X_train)),
                    "test_samples": int(len(X_test)),
                    "n_features": int(len(feature_cols)),
                },
                # 중요: 앱에서 스케일링 후 예측하도록 학습도 scaled로 통일했음을 명시
                "note": "All models were trained on StandardScaler-transformed features for deployment consistency.",
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"✅ Metadata 저장: {METADATA_PATH.name}")

    # Seoul data for map/dashboard
    with open(SEOUL_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(seoul_rows, f, ensure_ascii=False)
    print(f"✅ Seoul data 저장: {SEOUL_DATA_PATH.name}")

    print()
    print("=" * 80)
    print("✅ ML 모델 학습 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()