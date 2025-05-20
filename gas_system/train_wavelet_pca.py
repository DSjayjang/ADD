# train_wavelet_pca.py
"""
학습 스크립트: Wavelet-PCA 분류기를 학습하고 pickle 파일로 저장합니다.
WaveletPCAClassifier.fit(X, y) 인터페이스를 사용하도록 수정했습니다.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump
from models.waveletPCA_classifier import WaveletPCAClassifier
from typing import List

# 설정
DATA_DIR = Path("wavelet/data_250418")
CSV_FILES = [
    "000_Et_H_CO_n.csv",
    "002_Et_H_CO_H.csv",
    "008_Et_H_CO_L.csv",
    "028_Et_H_CO_M.csv",
]
N_PER_CLASS   = 100
ERROR_SD      = 30.0  # 노이즈 표준편차 (사용 안 함)
WAVELET       = "db2"
VAR_THRESHOLD = 0.8
OUTPUT_PATH   = Path("models/wavelet_pca.pkl")


def load_and_prepare(data_dir: Path, csv_files: List[str]) -> (np.ndarray, np.ndarray):
    """
    1) CSV 로드
    2) 노이즈 없이 원본 데이터를 하나의 3D 배열로 합쳐 반환
       반환: X3d (T, sensors, classes * N_PER_CLASS), y (length classes * N_PER_CLASS)
    """
    originals = []
    for fname in csv_files:
        df = pd.read_csv(data_dir / fname, header=None)
        originals.append(df.iloc[:, 3:].to_numpy())

    # 샘플 개수
    classes = len(originals)
    T, sensors = originals[0].shape
    total = classes * N_PER_CLASS

    # 각 클래스당 N_PER_CLASS 샘플 랜덤 추출 또는 복제
    X3d = np.empty((T, sensors, total))
    y   = np.empty(total, dtype=int)
    idx = 0
    for k, arr in enumerate(originals):
        # 반복 샘플링: 같은 시계열을 N_PER_CLASS개로 복제
        for i in range(N_PER_CLASS):
            X3d[..., idx] = arr
            y[idx] = k
            idx += 1
    return X3d, y


def main():
    # 1) 데이터 준비
    X3d, y = load_and_prepare(DATA_DIR, CSV_FILES)
    # 2) 3D -> 2D for fit: (samples, seq_len, sensors)
    #    X3d: (T, sensors, total) -> X: (total, T, sensors)
    X = np.transpose(X3d, (2, 0, 1))

    # 3) Wavelet-PCA 분류기 생성 및 학습
    clf = WaveletPCAClassifier(
        wavelet=WAVELET,
        var_threshold=VAR_THRESHOLD
    )
    clf.fit(X, y)

    # 4) 모델 저장
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(clf, OUTPUT_PATH)
    print(f"▶ Wavelet-PCA classifier saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
