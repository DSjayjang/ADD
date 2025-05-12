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
        self.sigma_ = None
        self.k = None
        self.h = None
        self.S_pos = 0.0
        self.S_neg = 0.0

    def fit(self, X: np.ndarray):
        """
        Phase I 데이터로부터 μ₀, σ₀, k, h 계산
        X: 1D array-like (phase1_len 길이)
        """
        x1 = np.asarray(X, dtype=float)
        if x1.ndim != 1 or len(x1) < self.phase1_len:
            raise ValueError("fit()에는 길이 phase1_len 이상의 1D 배열을 넣어야 합니다.")
        # 기준 구간
        p1 = x1[:self.phase1_len]
        self.mean_  = p1.mean()
        self.sigma_ = p1.std(ddof=1)
        # qcc::cusum 매핑
        # k = se.shift * σ₀ / 2
        self.k = (self.drift * self.sigma_) / 2.0
        # h = decision.interval * σ₀
        self.h = self.threshold * self.sigma_
        # 누적합 초기화
        self.S_pos = 0.0
        self.S_neg = 0.0

    def update(self, x: np.ndarray) -> bool:
        """
        Phase II 한 점씩 호출
        x: 단일 관측값 (스칼라)
        Returns:
          True  → S_pos > h 이거나 –S_neg > h 이면 (상향/하향 탐지)
          False → 아직 탐지 아님
        """
        if self.mean_ is None:
            raise RuntimeError("먼저 fit()을 호출하여 Phase I를 설정하세요.")
        xi = float(x)
        s = xi - self.mean_ - self.k
        self.S_pos = max(0.0, self.S_pos + s)
        self.S_neg = min(0.0, self.S_neg + s)
        # 상향 또는 하향 임계 초과 여부
        return (self.S_pos > self.h) or ((-self.S_neg) > self.h)
