import numpy as np
import pandas as pd
from .base_classifier import BaseClassifier
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import os

# pandas DataFrame <-> R DataFrame 변환 활성화
pandas2ri.activate()

class RClassifier(BaseClassifier):
    """
    R 기반 분류기 래퍼 (Wavelet-PCA inference-only)

    __init__ args:
      r_script_path: R 스크립트 경로 (wave_pca_classifier.R)
      data_paths: 길이4 CSV 파일 경로 리스트
    """
    def __init__(self, r_script_path: str, data_paths: list):
        super().__init__()
        # 절대 경로로 R 스크립트 로드
        abs_path = os.path.abspath(r_script_path).replace('\\','/')
        dir_path = os.path.dirname(abs_path)
        ro.r(f'setwd("{dir_path}")')
        ro.r(f'source("{abs_path}")')
        # Wavelet-PCA 모델 생성 (data_paths list)
        train_fn = ro.globalenv['WavePCAClassifier']
        r_data_paths = ro.vectors.StrVector(data_paths)
        self.r_model = train_fn(r_data_paths)
        # 예측 함수 참조
        self.predict_fn = ro.globalenv['predict.WavePCAClassifier']
        self.is_fitted = True

    def preprocess_input(self, X: np.ndarray) -> np.ndarray:
        return X

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = []
        for window in X:
            df = pd.DataFrame(window,
                              columns=[f"V{i}" for i in range(window.shape[1])])
            r_df = pandas2ri.py2rpy(df)
            r_out = self.predict_fn(self.r_model, r_df)
            probs.append(np.array(r_out))
        return np.vstack(probs)