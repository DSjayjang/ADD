import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifierCV
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sktime.datatypes._panel._convert import from_3d_numpy_to_nested
from models.base_classifier import BaseClassifier
import joblib

class MiniRocketClassifier(BaseClassifier, ClassifierMixin):
    def __init__(self):
        super().__init__()
        self.label_encoder = LabelEncoder()
        self.pipe = make_pipeline(
            MiniRocketMultivariate(),
            RidgeClassifierCV()
        )
        self.classes_ = None

    @classmethod
    def load(cls, filepath: str) -> "MiniRocketClassifier":
        """
        저장된 파이프라인(.pkl) 파일에서 MiniRocketClassifier 객체를 불러옵니다.
        """
        return joblib.load(filepath)

    def preprocess_input(self, X) -> pd.DataFrame:
        # … (기존 preprocess_input 코드 그대로) …
        if isinstance(X, pd.DataFrame):
            return X
        if isinstance(X, list):
            arrs = []
            for w in X:
                w = np.asarray(w)
                if w.ndim != 2:
                    raise ValueError("Each list element must be a 2D array of shape (seq_len, n_features)")
                if w.shape[0] < 9:
                    w = np.pad(w, ((0, 9 - w.shape[0]), (0, 0)), mode="edge")
                arrs.append(w)
            X_3d = np.stack(arrs, axis=0)
        elif isinstance(X, np.ndarray):
            if X.ndim == 3:
                X_3d = X
            elif X.ndim == 2:
                X_3d = X[np.newaxis, ...]
            else:
                raise ValueError("NumPy array input must be 2D or 3D")
        else:
            raise ValueError("Input must be a nested DataFrame, list of arrays, or ndarray")
        if X_3d.shape[1] < 9:
            pad_len = 9 - X_3d.shape[1]
            X_3d = np.pad(X_3d, ((0,0),(0,pad_len),(0,0)), mode="edge")
        X_tf = np.transpose(X_3d, (0, 2, 1))
        return from_3d_numpy_to_nested(X_tf)

    def fit(self, X, y):
        # … (기존 fit 코드) …
        X_nested = self.preprocess_input(X)
        y_enc = self.label_encoder.fit_transform(y)
        if len(y_enc) != len(X_nested):
            raise ValueError(f"Inconsistent sample size: X {len(X_nested)}, y {len(y_enc)}")
        self.classes_ = self.label_encoder.classes_
        self.pipe.fit(X_nested, y_enc)
        return self

    def predict(self, X):
        # … (기존 predict 코드) …
        X_nested = X if isinstance(X, pd.DataFrame) else self.preprocess_input(X)
        preds = self.pipe.predict(X_nested)
        return self.label_encoder.inverse_transform(preds)

    def predict_proba(self, X):
        # … (기존 predict_proba 코드) …
        X_nested = X if isinstance(X, pd.DataFrame) else self.preprocess_input(X)
        scores = self.pipe.decision_function(X_nested)
        if scores.ndim == 1:
            probs = np.vstack([1 - scores, scores]).T
        else:
            scores = scores - np.max(scores, axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs
