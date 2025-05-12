from abc import ABC, abstractmethod
import numpy as np

class BaseClassifier(ABC):
    def __init__(self):
        self.is_fitted = False

    @abstractmethod
    def preprocess_input(self, X: np.ndarray) -> np.ndarray:
        """
        X: (n_samples, seq_len, n_features)
        리턴: model.fit/predict_proba 가 기대하는 형태
        """

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xp = self.preprocess_input(X)
        self.model.fit(Xp, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.is_fitted, "먼저 fit()을 호출하세요."
        Xp = self.preprocess_input(X)
        return self.model.predict(Xp)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.is_fitted, "먼저 fit()을 호출하세요."
        Xp = self.preprocess_input(X)
        return self.model.predict_proba(Xp)
