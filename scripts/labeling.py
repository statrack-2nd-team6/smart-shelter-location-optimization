import os
import pandas as pd
import re

def main():
    # 1. 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(base_dir, "..", "bus_stop_images")
    
    if not os.path.exists(target_dir):
        print("❌ 오류: 이미지 폴더를 찾을 수 없습니다.")
        return

    all_files = os.listdir(target_dir)
    # '스마트'가 포함된 CSV 파일 찾기
    pos_files = [f for f in all_files if '스마트' in f and f.endswith('.csv')]
    # 이미지 파일 목록 정렬 (일관성 유지)
    img_files = sorted([f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if not pos_files:
        print("❌ 오류: 리스트 CSV 파일을 찾을 수 없습니다.")
        return

    # 2. Positive 기준 리스트 로드 (인코딩 안전하게 처리)
    pos_list_path = os.path.join(target_dir, pos_files[0])
    df_pos = None
    for enc in ['utf-8-sig', 'cp949', 'utf-8', 'euc-kr']:
        try:
            df_pos = pd.read_csv(pos_list_path, encoding=enc)
            print(f"✅ 리스트 로드 성공 (인코딩: {enc})")
            break
        except:
            continue
    
    if df_pos is None:
        print("❌ 오류: 리스트 파일을 읽을 수 없습니다.")
        return

    # 리스트에서 한글만 추출하여 중복 없는 세트 생성
    pos_names_raw = df_pos.iloc[:, 0].dropna().astype(str).tolist()
    clean_pos_set = set(re.sub(r'[^가-힣]', '', name) for name in pos_names_raw if re.sub(r'[^가-힣]', '', name))
    
    print(f"✅ 기준 명단: {len(clean_pos_set)}개 (중복 제거 완료)")

    # 3. 매칭 루프 (중복 매칭 방지)
    image_data = []
    used_pos_names = set() # 이미 1(Positive)로 할당된 정류장 이름 추적

    for file_name in img_files:
        label = 0
        name_part = file_name.split('.')[0]
        # 파일명에서 한글만 추출
        clean_file_name = re.sub(r'[^가-힣]', '', name_part)
        
        # 조건: 리스트에 존재하고, 아직 이 정류장 이름으로 Positive를 할당하지 않았을 때
        if clean_file_name in clean_pos_set and clean_file_name not in used_pos_names:
            label = 1
            used_pos_names.add(clean_file_name) # 사용됨으로 기록
        
        image_data.append({'file_name': file_name, 'label': label})

    # 4. 결과 저장
    df_final = pd.DataFrame(image_data)
    output_path = os.path.join(base_dir, "final_pu_dataset.csv")
    df_final.to_csv(output_path, index=False, encoding='utf-8-sig')

    print("-" * 30)
    print(f"🎉 작업 완료!")
    print(f"📍 최종 Positive (P): {df_final['label'].sum()}개")
    print(f"📍 나머지 Unlabeled (U): {len(df_final) - df_final['label'].sum()}개")
    print(f"⚠️ 매칭되지 않은 리스트 항목: {len(clean_pos_set) - len(used_pos_names)}개")
    print(f"📍 저장 위치: {output_path}")
    print("-" * 30)

if __name__ == "__main__":
    main()