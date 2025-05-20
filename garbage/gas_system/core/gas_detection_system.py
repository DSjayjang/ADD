import time
import numpy as np
from joblib import Parallel, delayed

# helper for parallel prediction
# clf: either BaseClassifier (has preprocess_input) or sklearn classifier wrapped in SklearnWrapper
# X: batched window array, shape (1, seq_len, n_features)
def _predict_one(clf, X):
    window = X[0]  # shape (seq_len, n_features)
    # BaseClassifier (e.g., MiniRocket) supports list of arrays input
    if hasattr(clf, "preprocess_input"):
        proba = clf.predict_proba([window])[0]
    else:
        # sklearn models expect flat 2D arrays
        flat = window.flatten().reshape(1, -1)
        proba = clf.predict_proba(flat)[0]
    return proba

class GasDetectionSystem:
    def __init__(
        self,
        base_classifiers,
        meta_classifier,
        cusum_fn,
        window_size=None,
        label_mapping=None,
        pretrained: bool = True
    ):
        self.base_classifiers       = base_classifiers
        self.meta_classifier        = meta_classifier
        self.cusum_fn               = cusum_fn
        self.window_size            = window_size
        self.label_mapping          = label_mapping or {}
        self.is_fitted              = pretrained

        # 누적(inference) 시간을 기록할 변수
        self.total_inference_time   = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        for clf in self.base_classifiers:
            clf.fit(X, y)
        probas = [clf.predict_proba(X) for clf in self.base_classifiers]
        X_meta = np.concatenate(probas, axis=1)
        self.meta_classifier.fit(X_meta, y)
        self.is_fitted            = True
        # 학습 후에는 누적 시간 초기화
        self.total_inference_time = 0.0

    def fit_expanding(self, windows_list: list, y: np.ndarray):
        for clf in self.base_classifiers:
            clf.fit(windows_list, y)
        last_window = [windows_list[-1]]
        probas = [clf.predict_proba(last_window)[0] for clf in self.base_classifiers]
        X_meta = np.concatenate(probas).reshape(1, -1)
        self.meta_classifier.fit(X_meta, [y[-1]])
        self.is_fitted            = True
        self.total_inference_time = 0.0

    def run_pipeline(self, window: np.ndarray):
        """
        Single-window inference
        window: (seq_len, n_features)
        """
        assert self.is_fitted, "Call fit() first"
        start = time.time()

        # 배치 차원 추가
        X1 = window[np.newaxis, ...]  # shape (1, seq_len, n_features)
        probas = []
        for clf in self.base_classifiers:
            # 모델별 전처리
            Xp = clf.preprocess_input(X1)
            # predict_proba 결과 (n_samples, n_classes)
            p = clf.predict_proba(Xp)
            probas.append(p[0])  # 첫 번째(유일한) 샘플 확률

        # 메타 입력 생성 및 예측
        meta_in = np.concatenate(probas).reshape(1, -1)
        pred = self.meta_classifier.predict(meta_in)[0]

        inference_time = time.time() - start
        self.total_inference_time += inference_time

        return {
            "meta_pred":             pred,
            "probas":                probas,
            "inference_time":        inference_time,
            "total_inference_time":  self.total_inference_time
        }

