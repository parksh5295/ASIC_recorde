"""
매핑 일관성 검증 스크립트

ISV 파일의 chunk 단위 매핑이 Main_Association_Rule_ex_Batch.py의 전체 데이터 매핑과
동일한 결과를 생성하는지 검증합니다.
"""
import pandas as pd
import argparse
import os
import sys

# Add project root to path
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    if '..' not in sys.path:
        sys.path.insert(0, '..')

from utils.time_transfer import time_scalar_transfer
from Modules.Heterogeneous_module import choose_heterogeneous_method
from Heterogeneous_Method.separate_group_mapping import map_intervals_to_groups
from utils.class_row import get_label_columns_to_exclude
from Dataset_Choose_Rule.association_data_choose import get_clustered_data_path


def verify_mapping_consistency(file_type, file_number, chunk_size=500, n_splits=40):
    """
    ISV 방식과 Main_Association_Rule_ex_Batch 방식의 매핑 결과를 비교합니다.
    
    Args:
        file_type: 데이터셋 타입
        file_number: 파일 번호
        chunk_size: ISV에서 사용하는 chunk 크기
        n_splits: interval split 개수
    """
    print(f"\n{'='*70}")
    print(f"매핑 일관성 검증: {file_type} (file {file_number})")
    print(f"{'='*70}\n")
    
    # 1. 데이터 로드
    file_path, _ = get_clustered_data_path(file_type, file_number)
    print(f"[1/4] 데이터 로드: {file_path}")
    
    # 전체 데이터 로드 (Main_Association_Rule_ex_Batch 방식)
    full_data = pd.read_csv(file_path, low_memory=False)
    print(f"  전체 데이터: {len(full_data)} rows, {len(full_data.columns)} columns")
    
    # 첫 번째 chunk 로드 (ISV 방식)
    first_chunk = pd.read_csv(file_path, nrows=chunk_size, low_memory=False)
    print(f"  첫 번째 chunk: {len(first_chunk)} rows")
    
    # 2. 전처리
    print(f"\n[2/4] 전처리 (time_scalar_transfer)")
    full_data_processed = time_scalar_transfer(full_data.copy(), file_type)
    first_chunk_processed = time_scalar_transfer(first_chunk.copy(), file_type)
    
    label_cols_to_exclude = get_label_columns_to_exclude(file_type)
    full_features = full_data_processed.drop(columns=label_cols_to_exclude, errors='ignore')
    first_chunk_features = first_chunk_processed.drop(columns=label_cols_to_exclude, errors='ignore')
    
    print(f"  전체 데이터 features: {len(full_features.columns)} columns")
    print(f"  첫 번째 chunk features: {len(first_chunk_features.columns)} columns")
    
    # 3. Main_Association_Rule_ex_Batch 방식: 전체 데이터로 매핑 생성
    print(f"\n[3/4] Main_Association_Rule_ex_Batch 방식: 전체 데이터로 매핑 생성")
    full_embedded, _, full_category_mapping, full_data_list = choose_heterogeneous_method(
        full_features,
        file_type, 'Interval_inverse', 'N', n_splits_override=n_splits
    )
    full_mapped, _ = map_intervals_to_groups(full_embedded, full_category_mapping, full_data_list, 'N')
    print(f"  전체 데이터 매핑 완료: {len(full_mapped)} rows, {len(full_mapped.columns)} columns")
    
    # 4. ISV 방식: 첫 번째 chunk로 매핑 생성 후, 전체 데이터를 매핑
    print(f"\n[4/4] ISV 방식: 첫 번째 chunk로 매핑 생성 후, 전체 데이터를 매핑")
    
    # 첫 번째 chunk로 category_mapping 생성 (ISV에서 하는 것처럼)
    chunk_embedded, _, chunk_category_mapping, chunk_data_list = choose_heterogeneous_method(
        first_chunk_features,
        file_type, 'Interval_inverse', 'N', n_splits_override=n_splits
    )
    print(f"  첫 번째 chunk로 category_mapping 생성 완료")
    
    # 전체 데이터를 chunk_category_mapping으로 매핑 (ISV에서 각 chunk에 대해 하는 것처럼)
    # 하지만 전체 데이터를 한 번에 처리
    full_embedded_for_chunk_mapping, _, _, full_data_list_for_chunk = choose_heterogeneous_method(
        full_features,
        file_type, 'Interval_inverse', 'N', n_splits_override=n_splits
    )
    chunk_mapped_full, _ = map_intervals_to_groups(
        full_embedded_for_chunk_mapping, 
        chunk_category_mapping,  # 첫 번째 chunk에서 생성한 mapping 사용
        full_data_list_for_chunk, 
        'N'
    )
    print(f"  전체 데이터를 chunk_category_mapping으로 매핑 완료: {len(chunk_mapped_full)} rows")
    
    # 5. 비교
    print(f"\n{'='*70}")
    print("매핑 결과 비교")
    print(f"{'='*70}\n")
    
    # 공통 컬럼 찾기
    common_cols = set(full_mapped.columns) & set(chunk_mapped_full.columns)
    common_cols = common_cols - {'label', 'cluster', 'adjusted_cluster'}  # label 컬럼 제외
    
    print(f"비교 대상 컬럼 수: {len(common_cols)}")
    if len(common_cols) == 0:
        print("  ⚠️  경고: 비교할 컬럼이 없습니다!")
        return
    
    # 각 컬럼별로 비교
    mismatches = []
    matches = []
    
    for col in sorted(common_cols)[:20]:  # 처음 20개 컬럼만 비교 (너무 많으면 시간 오래 걸림)
        full_values = full_mapped[col].astype(str)
        chunk_values = chunk_mapped_full[col].astype(str)
        
        # 길이가 다르면 비교 불가
        if len(full_values) != len(chunk_values):
            mismatches.append((col, f"길이 불일치: {len(full_values)} vs {len(chunk_values)}"))
            continue
        
        # 값 비교
        diff_mask = full_values != chunk_values
        diff_count = diff_mask.sum()
        diff_rate = diff_count / len(full_values) * 100
        
        if diff_count > 0:
            mismatches.append((col, f"{diff_count}/{len(full_values)} ({diff_rate:.2f}%) 불일치"))
            # 샘플 출력
            diff_indices = full_values[diff_mask].index[:5]
            for idx in diff_indices:
                print(f"    샘플 불일치 [{idx}]: 전체={full_values.loc[idx]}, chunk={chunk_values.loc[idx]}")
        else:
            matches.append(col)
    
    print(f"\n일치하는 컬럼: {len(matches)}/{len(common_cols)}")
    if matches:
        print(f"  예시: {matches[:5]}")
    
    print(f"\n불일치하는 컬럼: {len(mismatches)}/{len(common_cols)}")
    if mismatches:
        print("  불일치 상세:")
        for col, msg in mismatches[:10]:
            print(f"    - {col}: {msg}")
    
    # 6. category_mapping 비교
    print(f"\n{'='*70}")
    print("category_mapping 비교")
    print(f"{'='*70}\n")
    
    if 'interval' in full_category_mapping and 'interval' in chunk_category_mapping:
        full_interval = full_category_mapping['interval']
        chunk_interval = chunk_category_mapping['interval']
        
        print(f"전체 데이터 interval 컬럼 수: {len(full_interval.columns)}")
        print(f"첫 번째 chunk interval 컬럼 수: {len(chunk_interval.columns)}")
        
        common_interval_cols = set(full_interval.columns) & set(chunk_interval.columns)
        print(f"공통 interval 컬럼 수: {len(common_interval_cols)}")
        
        if len(common_interval_cols) > 0:
            # 샘플 컬럼 비교
            sample_col = list(common_interval_cols)[0]
            full_rules = set(full_interval[sample_col].dropna())
            chunk_rules = set(chunk_interval[sample_col].dropna())
            
            print(f"\n샘플 컬럼 '{sample_col}' 비교:")
            print(f"  전체 데이터 규칙 수: {len(full_rules)}")
            print(f"  첫 번째 chunk 규칙 수: {len(chunk_rules)}")
            print(f"  공통 규칙 수: {len(full_rules & chunk_rules)}")
            print(f"  전체 데이터만 있는 규칙: {len(full_rules - chunk_rules)}")
            print(f"  첫 번째 chunk만 있는 규칙: {len(chunk_rules - full_rules)}")
            
            if full_rules != chunk_rules:
                print(f"\n  ⚠️  경고: interval 규칙이 다릅니다!")
                if len(full_rules - chunk_rules) > 0:
                    print(f"    전체 데이터만 있는 규칙 예시: {list(full_rules - chunk_rules)[:3]}")
                if len(chunk_rules - full_rules) > 0:
                    print(f"    첫 번째 chunk만 있는 규칙 예시: {list(chunk_rules - full_rules)[:3]}")
            else:
                print(f"  ✓ interval 규칙이 일치합니다!")
    
    # 7. 결론
    print(f"\n{'='*70}")
    print("결론")
    print(f"{'='*70}\n")
    
    if len(mismatches) == 0:
        print("✓ 매핑이 완전히 일치합니다!")
    else:
        print(f"⚠️  매핑에 차이가 있습니다: {len(mismatches)}/{len(common_cols)} 컬럼 불일치")
        print("\n원인 분석:")
        print("  1. 각 chunk에 대해 choose_heterogeneous_method를 호출하면,")
        print("     그 chunk의 데이터 분포에 따라 다른 interval이 생성될 수 있습니다.")
        print("  2. 전체 데이터의 interval과 chunk의 interval이 다르면,")
        print("     map_intervals_to_groups가 다른 그룹 번호를 할당할 수 있습니다.")
        print("\n해결 방법:")
        print("  - 전체 데이터셋을 스캔해서 생성한 category_mapping을 사용하거나")
        print("  - existing_mapping 파라미터를 활용하여 일관된 interval을 보장해야 합니다.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='매핑 일관성 검증')
    parser.add_argument('--file_type', type=str, default='CICIDS2017', help='데이터셋 타입')
    parser.add_argument('--file_number', type=int, default=1, help='파일 번호')
    parser.add_argument('--chunk_size', type=int, default=500, help='ISV chunk 크기')
    parser.add_argument('--n_splits', type=int, default=40, help='interval split 개수')
    
    args = parser.parse_args()
    verify_mapping_consistency(args.file_type, args.file_number, args.chunk_size, args.n_splits)
