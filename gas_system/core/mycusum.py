import numpy as np

class CUSUMDetector:
    def __init__(self, phase1_len=50, threshold=5.0, drift=0.5):
        """
        phase1_len: Phase I(기준) 구간 길이 (샘플 수)
        threshold: decision.interval (σ 단위)
        drift: se.shift (σ 단위)
        """
        self.phase1_len = phase1_len
        self.threshold = threshold
        self.drift = drift
        self.reset()

    def reset(self):
        """내부 상태 초기화"""
        self.mean_ = None
        # self.sigma_ = None
        self.cov_ = None
        self.k = None
        self.h = None
        self.S_pos = 0.0
        self.S_neg = 0.0

        self.S_0 = 0.0

    #####
    # multivariate로 변경해야함 현재는 그냥 univariate 취급하듯이 전체 센서들의 평균이 구해짐;;
    #####
    def fit(self, X: np.ndarray):
        """
        Phase I 데이터로부터 μ₀, σ₀, k, h 계산
        X: 1D array-like (phase1_len 길이)
        """

        # x1 = np.asarray(X, dtype=float) # 필요없음

        if X.ndim != 1 or len(X) < self.phase1_len:
            raise ValueError("fit()에는 길이 phase1_len 이상의 1D 배열을 넣어야 합니다.")

        # 기준 구간 (초기값 세팅)
        p1 = X[:self.phase1_len]

        self.mean_ = np.mean(p1, axis = 0) # axis = 0 추가해서 평균벡터로 만들기
        self.cov_ = np.cov(p1, rowvar = False) # 공분산 행렬
        self.S_0 = np.zeros_like(self.mean_)

        # qcc::cusum 매핑
        # k = se.shift * σ₀ / 2
        self.k = (self.drift * self.sigma_) / 2.0
        # h = decision.interval * σ₀
        self.h = self.threshold * self.sigma_
        # 누적합 초기화 
        # ### 하지만 reset에서 초기화되기 때문에 굳이 필요하진 않음.
        # self.S_pos = 0.0
        # self.S_neg = 0.0

    def update(self, X: np.ndarray) -> bool:
        # """
        # Phase II 한 점씩 호출
        # x: 단일 관측값 (스칼라)
        # Returns:
        #   True  → S_pos > h 이거나 –S_neg > h 이면 (상향/하향 탐지)
        #   False → 아직 탐지 아님
        # """
        # if self.mean_ is None:
        #     raise RuntimeError("먼저 fit()을 호출하여 Phase I를 설정하세요.")
        # xi = float(x)
        # s = xi - self.mean_ - self.k
        # self.S_pos = max(0.0, self.S_pos + s)
        # self.S_neg = min(0.0, self.S_neg + s)
        # # 상향 또는 하향 임계 초과 여부
        # return (self.S_pos > self.h) or ((-self.S_neg) > self.h)
        
        k = 0.5
        
        inv_cov = np.linalg.inv(self.cov_)
        results = []

        C_t = np.zeros_like(self.mean_)
        MC_t_prev = 0
        n_t = 1

        for t in range(self.phase1_len, len(X)):
            x_t = X[t]
            diff = x_t - self.mean_

            # cusum 벡터 업데이트
            if MC_t_prev > 0:
                C_t += diff
                n_t += 1
            else:
                C_t = diff.copy()
                n_t = 1

            # 탐지 통계량
            MC_t = max(0, np.sqrt(C_t @ inv_cov @ C_t.T) - k * n_t)
            MC_t_prev = MC_t

            # 이상 탐지 여부
            is_outlier = MC_t > self.threshold
            results.append({
                'index' : t,
                'MC_t' : MC_t,
                'n_t' : n_t,
                'C_t' : C_t.copy()
            })
        return results

