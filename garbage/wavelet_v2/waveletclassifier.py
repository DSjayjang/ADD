from __future__ import annotations
import math
import numpy as np
import pandas as pd
import pywt
from sklearn.decomposition import PCA

# wave_pca_class
class wave:

    def __init__(self, data_original: list[np.ndarray], n:int, wavelet : str = "db2", var_threshold:float = 0.80):
        
        # ------------------------------------- [Input] ------------------------------------------
        # data_original : preprocessing 클래스의 make_noise_data 함수를 통해 얻은 학습 데이터
        # n : 클래스 당 표본 수
        # wavelet = "db2"는 R 코드에서 filter.number = 2, family = "DaubExPhase"에 대응한다.
        # var_threshold : PCA에 대한 누적 분산 임계값
        # ----------------------------------------------------------------------------------------

        self.data_original = data_original
        self.T_original, self.n_sensors, self.n_samples = data_original.shape
        self.T = 1 << (self.T_original - 1).bit_length()
        self.pad = self.T - self.T_original
        self.w_per_sensor = self.T_original

        self.n = n
        self.wavelet = wavelet
        self.wavelet_features : np.ndarray = self._extract_wavelet_features(data_original, wavelet)

        # res : 분류기 학습 결과(mean, cov)
        # pca : pca 결과
        self.res, self.pca = self._wave_pca_class(var_threshold)
    

    # 웨이블렛 계수 추출 함수
    def _extract_wavelet_features(self, data: np.ndarray, wavelet : str) -> np.ndarray:

        n_sensors, n_samples = data.shape[1:3]
        wavelet_features = np.zeros((n_samples, n_sensors * self.w_per_sensor))
        for i in range(n_samples):
            for j in range(n_sensors):
                series = data[:, j, i]
                if self.pad:
                    # 뒤를 0으로 채우기
                    series = np.pad(series, (0, self.pad))

                # 웨이블렛 분해 결과 리스트
                coeff_lists = pywt.wavedec(series, wavelet=wavelet, mode="periodization")

                # 모든 레벨의 계수 + 근사 계수
                coeffs = np.concatenate(coeff_lists[1:] + [coeff_lists[0]])

                # 상위 w_per_sensor 개 계수 선택 (절대값 기준)
                # np.argsort = 오름차순으로 정렬되는 인덱스 배열 반환
                # [::-1] = 역순
                # [:w_per_sensor] = 상위 w_per_sensor개 만큼 자르기
                idx = np.argsort(np.abs(coeffs))[::-1][:self.w_per_sensor]
                sel = coeffs[idx]

                # w_per_sensor 단위로 대입
                wavelet_features[i, j * self.w_per_sensor : (j + 1) * self.w_per_sensor] = sel

        return wavelet_features

    # 분류기 학습
    def _wave_pca_class(self, var_threshold: float) -> tuple[dict[str, np.ndarray], PCA]:

        pca = PCA()
        wavelet_features_reduced = pca.fit_transform(self.wavelet_features)
        cum_var = np.cumsum(pca.explained_variance_ratio_)

        # 누적 분산이 처음 var_threshold(기본값) 넘는 인덱스
        n_pcs = int(np.searchsorted(cum_var, var_threshold) + 1)
        print(f"Number of PCs explaining >= {var_threshold*100:.0f}% variance: {n_pcs}")

        # 데이터 축약
        wavelet_features_reduced = wavelet_features_reduced[:, :n_pcs]

        # 분류기 학습 및 저장
        res: dict[str, np.ndarray] = {}

        # np.eye : 주어진 크기만큼 대각행렬을 생성한다.
        # singular problem 문제를 해소하기 위해 작은 regularisation 대각행렬 더할 예정임
        regularisation = np.eye(n_pcs) * 1e-4

        # 분류기 학습(naive bayes)
        for k in range(4):
            # cls : shape(100, 4)
            cls = wavelet_features_reduced[k * self.n : (k + 1) * self.n]
            res[f"mean{k+1}"] = cls.mean(axis=0)
            # np.cov : 공분산 행렬 구하는 함수
            # rowvar = FALSE : 열을 변수로 보고 행을 관측치로 보겠다는 뜻
            res[f"cov{k+1}"] = np.cov(cls, rowvar=False) + regularisation

        return res, pca

    # 4개 가스 샘플 분류 예측
    def prediction(self, sample: np.ndarray):

        # 전처리된 데이터 샘플에 대해 wavelet 분해
        sample_feats = self._extract_wavelet_features(sample, self.wavelet)

        # 데이터 샘플을 학습된 고정 축에 투영
        n_pcs = self.res["mean1"].shape[0]
        x_reduced = self.pca.transform(sample_feats)[:, : n_pcs]

        preds = np.empty(x_reduced.shape[0], dtype=int)
        preds_prob: list[np.array] = []
        # log-likelihoods
        for i, x in enumerate(x_reduced):
            logpdfs: list = []
            for k in range(1, 5):
                mean = self.res[f"mean{k}"]
                cov = self.res[f"cov{k}"]
                diff = x - mean
                log_det = math.log(np.linalg.det(cov))
                mahal = diff @ np.linalg.inv(cov) @ diff
                logpdfs.append(-0.5 * (log_det + mahal))
            
            preds_prob.append(np.exp(logpdfs)/sum(np.exp(logpdfs)))
            preds[i] = int(np.argmax(logpdfs) + 1)

        # 결과 출력
        print(f"Predicted class: {preds}")

        return preds_prob
        




    

    

        



