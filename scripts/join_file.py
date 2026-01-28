"""
3개의 테이블을 조인하고 정류소별 평균 배차간격을 계산하는 스크립트

입력 파일:
- ROUTE_ID 01(final).csv: 노선 기본 정보 (공항버스 제외)
- ROUTE_ID 02.xlsx: 노선별 정류소 정보
- NODE_ID 01.csv: 정류소별 승객 정보

조인 조건:
1. ROUTE_ID 01(final) + ROUTE_ID 02: ROUTE_ID 기준
2. 위 결과 + NODE_ID 01: NODE_ID = 표준버스정류장ID 기준

출력 파일:
- join_revise.csv: 3개 테이블 조인 결과
- join_final.csv: 정류소별 요약 (노선목록, avg_caralc, 평균배차간격 포함)
"""

import pandas as pd
import os

# 파일 경로 설정
DESKTOP_PATH = "/mnt/c/Users/Administrator/Desktop"


def load_data():
    """데이터 파일 로드"""
    # ROUTE_ID 01(final).csv는 cp949 인코딩
    route1 = pd.read_csv(
        os.path.join(DESKTOP_PATH, "ROUTE_ID 01(final).csv"),
        encoding='cp949'
    )
    route2 = pd.read_excel(os.path.join(DESKTOP_PATH, "ROUTE_ID 02.xlsx"))
    node = pd.read_csv(os.path.join(DESKTOP_PATH, "NODE_ID 01.csv"))

    print(f"ROUTE_ID 01(final) 로드 완료: {route1.shape[0]}행")
    print(f"ROUTE_ID 02 로드 완료: {route2.shape[0]}행")
    print(f"NODE_ID 01 로드 완료: {node.shape[0]}행")

    return route1, route2, node


def join_tables(route1, route2, node):
    """3개 테이블 조인"""
    # 1단계: ROUTE_ID 01(final) + ROUTE_ID 02 조인 (ROUTE_ID 기준)
    merged_routes = pd.merge(
        route2,
        route1,
        on="ROUTE_ID",
        how="left"
    )
    print(f"\n1단계 조인 완료 (ROUTE_ID 기준): {merged_routes.shape[0]}행")

    # 2단계: 위 결과 + NODE_ID 조인 (NODE_ID = 표준버스정류장ID)
    final_merged = pd.merge(
        merged_routes,
        node,
        left_on="NODE_ID",
        right_on="표준버스정류장ID",
        how="left"
    )
    print(f"2단계 조인 완료 (NODE_ID = 표준버스정류장ID): {final_merged.shape[0]}행")

    return final_merged


def save_result(df, filename, path=DESKTOP_PATH):
    """결과 저장"""
    output_file = os.path.join(path, filename)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {output_file}")
    print(f"총 {df.shape[0]}행, {df.shape[1]}열")


def create_station_summary(df):
    """정류소별 버스 노선 정리 (모든 컬럼 유지)"""
    # 노선목록, 버스수 집계
    agg_dict = {
        '노선목록': ('노선명', lambda x: ', '.join(sorted(x.dropna().unique().astype(str)))),
        '버스수': ('노선명', 'nunique'),
    }

    # 나머지 컬럼들은 first로 유지
    for col in df.columns:
        if col not in ['NODE_ID', '정류소명', '자치구', '노선명']:
            agg_dict[col] = (col, 'first')

    station_summary = df.groupby(['NODE_ID', '정류소명', '자치구']).agg(**agg_dict).reset_index()

    print(f"\n정류소별 정리 완료: {station_summary.shape[0]}개 정류소")

    return station_summary


def calc_avg_caralc(df, revise_df):
    """정류소별 평균 배차간격 계산"""
    # ROUTE_NM별 CARALC 매핑 (중복 제거)
    route_caralc = revise_df.drop_duplicates(subset=['ROUTE_NM'])[['ROUTE_NM', 'CARALC']]
    route_caralc['ROUTE_NM'] = route_caralc['ROUTE_NM'].astype(str)
    caralc_dict = dict(zip(route_caralc['ROUTE_NM'], route_caralc['CARALC']))

    print(f"ROUTE_NM-CARALC 매핑: {len(caralc_dict)}개 노선")

    # 각 정류소별로 노선목록에서 버스들의 CARALC 평균 계산
    def calculate(row):
        routes = [r.strip() for r in str(row['노선목록']).split(',')]
        total_caralc = 0
        count = 0
        for route in routes:
            if route in caralc_dict:
                total_caralc += caralc_dict[route]
                count += 1
        if count > 0:
            return total_caralc / count
        return None

    df['avg_caralc'] = df.apply(calculate, axis=1)

    # 평균배차간격: avg_caralc / 2
    df['평균배차간격'] = df['avg_caralc'] / 2

    print("avg_caralc, 평균배차간격 계산 완료")

    return df


def main():
    print("=" * 50)
    print("테이블 조인 시작")
    print("=" * 50)

    # 데이터 로드
    route1, route2, node = load_data()

    # 테이블 조인
    join_revise = join_tables(route1, route2, node)

    # join_revise.csv 저장
    print("\n" + "-" * 50)
    save_result(join_revise, "join_revise.csv")

    # 정류소별 요약 생성
    print("\n" + "=" * 50)
    print("정류소별 요약 생성")
    print("=" * 50)

    station_summary = create_station_summary(join_revise)

    # avg_caralc, 평균배차간격 계산
    print("\n" + "-" * 50)
    station_summary = calc_avg_caralc(station_summary, join_revise)

    # join_final.csv 저장
    print("\n" + "-" * 50)
    save_result(station_summary, "join_final.csv")

    # 결과 미리보기
    print("\n" + "=" * 50)
    print("결과 미리보기")
    print("=" * 50)
    print("\n컬럼:", list(station_summary.columns))
    print("\n샘플 데이터:")
    print(station_summary[['NODE_ID', '정류소명', '노선목록', '버스수', 'avg_caralc', '평균배차간격']].head())


if __name__ == "__main__":
    main()
