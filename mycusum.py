import numpy as np

class uniCUSUM:
    def __init__(self, phase1_len=100, h=5, delta=0.5):
        self._delta = float(delta)
        self._h = float(h)
        self._phase1_len = phase1_len

        # 초기값
        self._mu0 = None
        self._sigma = None
        self._s_prev_pos = 0.0
        self._s_prev_neg = 0.0

        self.violation_upper = []
        self.violation_lower = []   

    def fit(self, phase1_X: np.array):
        # Phase 1
        self.phase1_data = phase1_X[:self._phase1_len]
        self._mu0 = np.mean(self.phase1_data)
        self._sigma = np.std(self.phase1_data, ddof = 1)
    
    def update(self, phase2_X):        
        for idx, x in enumerate(phase2_X):
            s_t_pos = max(0.0, self._s_prev_pos + (x - self._mu0 - self._delta/2) / self._sigma)
            s_t_neg = max(0.0, self._s_prev_neg + (self._mu0 - x - self._delta/2) / self._sigma)

            # 벡터 누적합 업데이트
            self._s_prev_pos = s_t_pos
            self._s_prev_neg = s_t_neg

            if self._s_prev_pos > self._h:
                start_idx = self._phase1_len + idx
                self.violation_upper.append(start_idx)
                # self.reset()

            if self._s_prev_neg > self._h:
                start_idx = self._phase1_len + idx
                self.violation_lower.append(start_idx)
                # self.reset()

        return self.violation_lower, self.violation_upper

    # def reset(self):
    #     # 벡터 누적합 리셋
    #     self._s_prev_pos = 0.0
    #     self._s_prev_neg = 0.0
