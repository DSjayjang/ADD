# models/waveletPCA_preprocessing.py

import numpy as np
import pandas as pd
from typing import List
from pathlib import Path

class preprocessing:
    @staticmethod
    def load_csv(path: Path, file_list: List[str]) -> List[np.ndarray]:
        """
        path: CSV가 들어있는 폴더
        file_list: 읽을 파일 이름 리스트
        """
        originals: List[np.ndarray] = []
        for fname in file_list:
            # 헤더가 첫 줄에 있으므로 header=0으로 읽고,
            # 3번째 인덱스(4번째 컬럼)부터 실수로 변환합니다.
            df = pd.read_csv(path / fname, header=0)
            arr = df.iloc[:, 3:].to_numpy(dtype=np.float64)
            originals.append(arr)
        return originals

    @staticmethod
    def make_noise_data(originals: List[np.ndarray], n: int, err_sd: float) -> np.ndarray:
        """
        originals: list of 2D-arrays (T×F), 모두 float64 여야 함
        n: 클래스당 샘플 수
        err_sd: 노이즈 표준편차
        """
        noisy = []
        for ary in originals:
            ary = ary.astype(np.float64)
            rows, cols = ary.shape
            arr = np.empty((rows, cols, n), dtype=np.float64)
            for i in range(n):
                rng = np.random.default_rng(seed=i+1)
                arr[..., i] = ary + rng.normal(scale=err_sd, size=(rows, cols))
            noisy.append(arr)

        # 4개 클래스를 하나의 3D-array로 결합
        rows, cols, _ = noisy[0].shape
        data_original = np.empty((rows, cols, 4 * n), dtype=np.float64)
        for k, arr in enumerate(noisy):
            data_original[..., k*n:(k+1)*n] = arr
        return data_original
