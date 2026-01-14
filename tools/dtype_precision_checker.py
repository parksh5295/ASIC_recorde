#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 dtype 정밀도 영향도 점검 스크립트.

사용법 예시:
    python tools/dtype_precision_checker.py --csv ../Dataset_ex/sample.csv
    python tools/dtype_precision_checker.py --csv ../Dataset_ex/sample.csv --target-cols col1 col2 col3 --runs 5
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_numeric_subset(csv_path: str, columns=None, sample_size: int | None = None):
    df = pd.read_csv(csv_path)
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"지정한 컬럼이 존재하지 않습니다: {missing}")
        df = df[columns]
    else:
        df = df.select_dtypes(include=[np.number])
        if df.empty:
            raise ValueError("숫자형 컬럼을 찾을 수 없습니다. --target-cols 로 명시해 주세요.")

    df = df.dropna()
    if df.empty:
        raise ValueError("모든 행이 NaN 으로 제거되었습니다.")

    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
        logger.info(f"샘플링: {sample_size} 행을 사용합니다.")

    return df.to_numpy()


def compare_precisions(array64: np.ndarray, n_clusters: int):
    # NaN/Inf 제거
    finite_mask = np.isfinite(array64).all(axis=1)
    if not finite_mask.all():
        removed = len(array64) - finite_mask.sum()
        logger.warning(f"비정상 값(NaN/Inf)을 포함한 {removed}개 행을 제거합니다.")
        array64 = array64[finite_mask]
        if len(array64) == 0:
            raise ValueError("정상 값이 남아있지 않습니다. 소스를 확인하세요.")

    scaler = StandardScaler()
    scaled64 = scaler.fit_transform(array64)
    scaled32 = scaled64.astype(np.float32, copy=True)

    model64 = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=4096, n_init="auto")
    model32 = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=4096, n_init="auto")

    model64.fit(scaled64)
    model32.fit(scaled32)

    inertia64 = float(model64.inertia_)
    inertia32 = float(model32.inertia_)
    inertia_gap = abs(inertia64 - inertia32) / max(inertia64, 1e-9)

    centers_gap = np.linalg.norm(model64.cluster_centers_ - model32.cluster_centers_.astype(np.float64)) / (
        np.linalg.norm(model64.cluster_centers_) + 1e-9
    )

    sil64 = silhouette_score(scaled64, model64.labels_)
    sil32 = silhouette_score(scaled64, model32.labels_)  # 같은 데이터에서 비교
    sil_gap = abs(sil64 - sil32)

    return {
        "inertia_float64": inertia64,
        "inertia_float32": inertia32,
        "inertia_relative_gap": inertia_gap,
        "centers_relative_gap": centers_gap,
        "silhouette_float64": sil64,
        "silhouette_float32": sil32,
        "silhouette_abs_gap": sil_gap,
    }


def main():
    parser = argparse.ArgumentParser(description="float64 vs float32 정밀도 비교 도구")
    parser.add_argument("--csv", required=True, help="입력 CSV 경로")
    parser.add_argument("--target-cols", nargs="*", help="분석에 사용할 컬럼 목록 (미지정 시 숫자형 전체)")
    parser.add_argument("--sample-size", type=int, default=20000, help="최대 샘플 크기 (기본 20,000)")
    parser.add_argument("--clusters", type=int, default=10, help="KMeans 클러스터 수")
    parser.add_argument("--runs", type=int, default=3, help="반복 실행 횟수 (기본 3)")
    parser.add_argument("--clip-value", type=float, default=None, help="값의 절대크기를 지정 값으로 클리핑 (예: 1e6)")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {args.csv}")

    logger.info("데이터 로딩 중...")
    data64 = load_numeric_subset(args.csv, args.target_cols, args.sample_size)
    if args.clip_value is not None:
        np.clip(data64, -args.clip_value, args.clip_value, out=data64)
    logger.info(f"데이터 shape: {data64.shape}")

    logger.info("float64 vs float32 비교를 시작합니다...")
    all_metrics = []
    for run_idx in range(1, args.runs + 1):
        logger.info(f"[실행 {run_idx}/{args.runs}]")
        metrics = compare_precisions(data64, args.clusters)
        all_metrics.append(metrics)
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.6f}")

    # 평균/최대 gap 계산
    def avg(key):
        return sum(m[key] for m in all_metrics) / len(all_metrics)

    summary = {
        "inertia_relative_gap_avg": avg("inertia_relative_gap"),
        "centers_relative_gap_avg": avg("centers_relative_gap"),
        "silhouette_abs_gap_avg": avg("silhouette_abs_gap"),
        "inertia_relative_gap_max": max(m["inertia_relative_gap"] for m in all_metrics),
        "centers_relative_gap_max": max(m["centers_relative_gap"] for m in all_metrics),
        "silhouette_abs_gap_max": max(m["silhouette_abs_gap"] for m in all_metrics),
    }

    print("\n요약:")
    print(f" - Inertia 차이(평균 상대): {summary['inertia_relative_gap_avg']:.6e} (최대 {summary['inertia_relative_gap_max']:.6e})")
    print(f" - 중심 좌표 차이(평균 상대): {summary['centers_relative_gap_avg']:.6e} (최대 {summary['centers_relative_gap_max']:.6e})")
    print(f" - 실루엣 점수 차이(평균 절대): {summary['silhouette_abs_gap_avg']:.6e} (최대 {summary['silhouette_abs_gap_max']:.6e})")
    print("위 값들이 충분히 작다면 float32로 다운캐스팅해도 영향이 미미하다고 볼 수 있습니다.")


if __name__ == "__main__":
    main()


