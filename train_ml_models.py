#!/usr/bin/env python3
"""
스마트 쉼터 우선순위 예측 - ML 모델 학습
"""

import csv
import json
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

print('='*80)
print('🤖 ML 모델 학습 시작')
print('='*80)
print()

# Load data
print('📥 데이터 로딩...')
with open('dataset_engineered.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    data = list(reader)

print(f'✅ 데이터 로드 완료: {len(data):,}개 샘플')
print()

# Prepare features and target
print('🔧 Feature 및 Target 준비...')

# Select features
feature_cols = [
    'cai', 'pm25', 'pm10', 'o3', 'no2',
    'ridership', 'ridership_log',
    'dispatch_interval', 'dispatch_half',
    'lat', 'lon',
    'cai_ridership', 'pollution_exposure_v2', 'total_exposure',
    'is_high_traffic', 'is_high_pollution', 'is_long_wait'
]

X = []
y = []
metadata = []  # For later use in web app

for row in data:
    features = [float(row[col]) for col in feature_cols]
    target = float(row['priority_v4'])
    
    X.append(features)
    y.append(target)
    
    # Save metadata for web app
    metadata.append({
        'stop_id': row['stop_id'],
        'name': row['name'],
        'district': row['district'],
        'lat': float(row['lat']),
        'lon': float(row['lon']),
        'cai': float(row['cai']),
        'ridership': int(row['ridership']),
        'priority': target
    })

X = np.array(X)
y = np.array(y)

print(f'✅ Feature shape: {X.shape}')
print(f'✅ Target shape: {y.shape}')
print(f'✅ Feature 목록 ({len(feature_cols)}개):')
for i, col in enumerate(feature_cols, 1):
    print(f'   {i:2d}. {col}')
print()

# Train-test split
print('✂️  Train-Test Split (80:20)...')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f'✅ Train: {len(X_train):,}개')
print(f'✅ Test: {len(X_test):,}개')
print()

# Feature scaling
print('📊 Feature Scaling (StandardScaler)...')
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print('✅ Scaling 완료')
print()

# Train models
print('='*80)
print('🎯 모델 학습 시작')
print('='*80)
print()

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
}

results = {}

for name, model in models.items():
    print(f'🔄 Training {name}...')
    
    # Train
    if name in ['Linear Regression', 'Ridge Regression']:
        model.fit(X_train_scaled, y_train)
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
    
    # Evaluate
    train_mse = mean_squared_error(y_train, y_pred_train)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_r2 = r2_score(y_train, y_pred_train)
    
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    results[name] = {
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2
    }
    
    print(f'   Train - RMSE: {train_rmse:.6f}, MAE: {train_mae:.6f}, R²: {train_r2:.6f}')
    print(f'   Test  - RMSE: {test_rmse:.6f}, MAE: {test_mae:.6f}, R²: {test_r2:.6f}')
    print()

# Results summary
print('='*80)
print('📊 모델 성능 비교')
print('='*80)
print()
print(f'{"Model":<25s} {"Test RMSE":<12s} {"Test MAE":<12s} {"Test R²":<12s}')
print('-'*80)
for name, metrics in results.items():
    print(f'{name:<25s} {metrics["test_rmse"]:<12.6f} {metrics["test_mae"]:<12.6f} {metrics["test_r2"]:<12.6f}')
print()

# Find best model
best_model_name = max(results, key=lambda x: results[x]['test_r2'])
best_model = models[best_model_name]

print(f'🏆 Best Model: {best_model_name}')
print(f'   Test R² Score: {results[best_model_name]["test_r2"]:.6f}')
print()

# Feature importance (for tree-based models)
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    print('='*80)
    print('📊 Feature Importance (Top 10)')
    print('='*80)
    
    importances = best_model.feature_importances_
    feature_importance = sorted(
        zip(feature_cols, importances),
        key=lambda x: x[1],
        reverse=True
    )
    
    for i, (feat, imp) in enumerate(feature_importance[:10], 1):
        print(f'{i:2d}. {feat:<30s}: {imp:.6f}')
    print()

# Save models
print('='*80)
print('💾 모델 및 전처리기 저장')
print('='*80)

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print('✅ Scaler 저장: scaler.pkl')

# Save best model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f'✅ Best Model 저장: best_model.pkl ({best_model_name})')

# Save all models
for name, model in models.items():
    safe_name = name.replace(' ', '_').lower()
    with open(f'model_{safe_name}.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f'✅ {name} 저장: model_{safe_name}.pkl')

# Save metadata
with open('model_metadata.json', 'w', encoding='utf-8') as f:
    json.dump({
        'feature_cols': feature_cols,
        'best_model_name': best_model_name,
        'results': results,
        'data_info': {
            'total_samples': len(data),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': len(feature_cols)
        }
    }, f, indent=2, ensure_ascii=False)
print('✅ Metadata 저장: model_metadata.json')

# Save sample predictions for web app
with open('seoul_data.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False)
print('✅ Seoul data 저장: seoul_data.json')

print()
print('='*80)
print('✅ ML 모델 학습 완료!')
print('='*80) 