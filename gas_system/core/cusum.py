import numpy as np

class uniCUSUM:
    def __init__(self, phase1_len=100, threshold=5, delta=0.5):
        self._delta = float(delta)
        self._threshold = float(threshold)
        self._phase1_len = phase1_len

        # 초기값
        self._mu0 = None
        self._sigma = None
        self._s_prev_pos = 0.0
        self._s_prev_neg = 0.0

        self.violation_upper = []
        self.violation_lower = []   

    # Phase 1
    def fit(self, X: np.ndarray):
        phase1_data = X[:self._phase1_len]

        self._mu0 = np.mean(phase1_data)
        self._sigma = np.std(phase1_data, ddof = 1)
    
    # Phase 2
    def update(self, X: np.ndarray):
        phase2_X = X[self._phase1_len:]

        for idx, x in enumerate(phase2_X):
            s_t_pos = max(0.0, self._s_prev_pos + (x - self._mu0 - self._delta/2) / self._sigma)
            s_t_neg = max(0.0, self._s_prev_neg + (self._mu0 - x - self._delta/2) / self._sigma)

            # 벡터 누적합 업데이트
            self._s_prev_pos = s_t_pos
            self._s_prev_neg = s_t_neg

            if self._s_prev_pos > self._threshold:
                start_idx = self._phase1_len + idx
                self.violation_upper.append(start_idx)

            if self._s_prev_neg > self._threshold:
                start_idx = self._phase1_len + idx
                self.violation_lower.append(start_idx)

        return self.violation_lower, self.violation_upper
    
class multiCUSUM:
    def __init__(self, phase1_len=100, threshold=5, k=1.0):
        self._k = float(k)
        self._threshold = float(threshold)
        self._phase1_len = phase1_len

        # 초기값
        self._mu0 = []
        self._cov0 = []
        self.violation = []

    # Phase 1
    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        phase1_data = X[:self._phase1_len, :]

        self._mu0 = np.mean(phase1_data, axis = 0)
        self._cov0 = np.cov(phase1_data, rowvar = False)

    # Phase 2
    def update(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        # 1D 입력(특징 하나짜리) → (n_samples,1) 로 reshape
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # 변화가 있는지 T/F만 출력하면 됨
        self.n_t = 1
        self.C_t = np.zeros(X.shape[1]) # 초기 누적합 벡터
        self._inv_cov0 = np.linalg.inv(self._cov0)

        # 시점 t에서의 누적합 벡터
        C_t = np.sum(X - self._mu0, axis = 0)

        # 탐지 통계량
        stat = C_t.T @ self._inv_cov0 @ C_t
        MC_t = max(0, np.sqrt(stat) - self._k * self.n_t)

        result = MC_t > self._threshold
        print(result)

        return result
