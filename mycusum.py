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
        phase2_X = X[self._phase1_len:, :]

        self.n_t = 1
        self.C_t = np.zeros(phase2_X.shape[1]) # 초기 누적합 벡터
        self._inv_cov0 = np.linalg.inv(self._cov0)

        for t in range(self._phase1_len, phase2_X.shape[0]):
            # 시점 t에서의 누적합 벡터
            window = X[t-self.n_t+1 : t+1]
            C_t = np.sum(window - self._mu0, axis = 0)

            # 탐지 통계량
            stat = C_t @ self._inv_cov0 @ C_t.T
            MC_t = max(0, np.sqrt(stat) - self._k * self.n_t)

            if MC_t > self._threshold:
                start_idx = self._phase1_len + t
                self.violation.append(start_idx)

            # 다음 시점의 n_t 갱신
            if MC_t > 0: self.n_t += 1
            else: self.n_t = 1

        return self.violation
