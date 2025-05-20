import math
import numpy as np
import pywt
from sklearn.decomposition import PCA
from typing import Optional
import joblib

from models.base_classifier import BaseClassifier

def pad_or_truncate(seq: np.ndarray, target_len: int) -> np.ndarray:
    """
    seq: 1D 배열, 원본 시퀀스
    target_len: 학습 시 사용한 시퀀스 길이 T_original
    반환: 길이 target_len로 패드(0)하거나, 넘치면 앞부분을 자른 시퀀스
    """
    L = seq.shape[0]
    if L < target_len:
        # 뒤쪽에 0 패딩
        return np.pad(seq, (0, target_len - L))
    elif L > target_len:
        # 최근 target_len 포인트 우선
        return seq[-target_len:]
    else:
        return seq
    
class WaveletPCAClassifier(BaseClassifier):
    """
    Wavelet → PCA → GaussianNB 분류기.
    fit(X, y)로 학습한 뒤, predict_proba 호출 전 preprocess_input에서
    padding/truncate를 자동 처리합니다.
    """
    def __init__(
        self,
        wavelet: str = "db2",
        var_threshold: float = 0.8,
        regularization: float = 1e-4,
        seq_len: Optional[int] = None
    ):
        super().__init__()
        self.wavelet        = wavelet
        self.var_threshold  = var_threshold
        self.regularization = regularization
        # 학습 시 결정될 원본 시퀀스 길이
        self.seq_len: Optional[int] = seq_len

        # 학습 후 설정되는 속성
        self.pca: Optional[PCA] = None
        self.n_pcs: int = 0
        self.means: list[np.ndarray] = []
        self.cov_invs: list[np.ndarray] = []
        self.logdets: list[float] = []

    def preprocess_input(self, X: np.ndarray) -> np.ndarray:
        """
        X: (n_samples, seq_len_input, n_sensors)
        pad_or_truncate로 길이 맞춘 뒤 웨이블릿 특징 → PCA 투영
        """
        assert self.is_fitted, "먼저 fit()을 호출하세요."

        # seq_len이 클래스에 없거나 None일 때 복구
        if not hasattr(self, "seq_len") or self.seq_len is None:
            n_sensors = X.shape[2]
            self.seq_len = self.pca.n_features_in_ // n_sensors

        n_samples, _, n_sensors = X.shape
        T = self.seq_len
        pad = (1 << (T - 1).bit_length()) - T

        feats = np.zeros((n_samples, T * n_sensors), dtype=float)
        for i in range(n_samples):
            offset = 0
            for j in range(n_sensors):
                seq = X[i, :, j]
                seq = pad_or_truncate(seq, T)
                if pad > 0:
                    seq = np.pad(seq, (0, pad))
                coeffs = pywt.wavedec(seq, wavelet=self.wavelet, mode="periodization")
                flat = np.concatenate(coeffs[1:] + [coeffs[0]])
                idx = np.argsort(np.abs(flat))[::-1][:T]
                feats[i, offset:offset+T] = flat[idx]
                offset += T

        Z = self.pca.transform(feats)[:, :self.n_pcs]
        return Z

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X: (n_samples, seq_len, n_sensors)
        y: (n_samples,) 클래스 레이블 (0부터 시작)
        """
        n_samples, seq_len, n_sensors = X.shape
        # 원본 시퀀스 길이와 센서 수 저장
        self.seq_len   = seq_len
        self.n_sensors = n_sensors

        # 재학습 시 상태 초기화
        self.means    = []
        self.cov_invs = []
        self.logdets  = []

        # 1) feature extraction
        pad = (1 << (seq_len - 1).bit_length()) - seq_len
        feats = np.zeros((n_samples, seq_len * n_sensors), dtype=float)
        for i in range(n_samples):
            idx_start = 0
            for j in range(n_sensors):
                seq = X[i, :, j]
                if pad > 0:
                    seq = np.pad(seq, (0, pad))
                coeffs = pywt.wavedec(seq, wavelet=self.wavelet, mode="periodization")
                flat = np.concatenate(coeffs[1:] + [coeffs[0]])
                select = np.argsort(np.abs(flat))[::-1][:seq_len]
                feats[i, idx_start:idx_start+seq_len] = flat[select]
                idx_start += seq_len

        # 2) PCA 학습 및 주성분 결정
        pca_full = PCA().fit(feats)
        cumvar   = np.cumsum(pca_full.explained_variance_ratio_)
        self.n_pcs = int(np.searchsorted(cumvar, self.var_threshold) + 1)
        self.pca   = PCA(n_components=self.n_pcs).fit(feats)

        # 3) Gaussian NB 파라미터 계산
        Z = self.pca.transform(feats)
        classes = np.unique(y)
        for k in classes:
            block = Z[y == k]
            mu  = block.mean(axis=0)
            cov = np.cov(block, rowvar=False) + np.eye(self.n_pcs) * self.regularization
            inv = np.linalg.inv(cov)
            ld  = math.log(np.linalg.det(cov))
            self.means.append(mu)
            self.cov_invs.append(inv)
            self.logdets.append(ld)

        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = self.preprocess_input(X)
        n_samples = Z.shape[0]
        n_classes = len(self.means)
        probs     = np.zeros((n_samples, n_classes), dtype=float)
        for i in range(n_samples):
            logps = []
            for mu, inv, ld in zip(self.means, self.cov_invs, self.logdets):
                diff = Z[i] - mu
                logp = -0.5 * (diff @ inv @ diff + ld)
                logps.append(logp)
            arr = np.array(logps)
            exp = np.exp(arr - np.max(arr))
            probs[i] = exp / exp.sum()
        return probs

    def save(self, filepath: str):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str):
        return joblib.load(filepath)