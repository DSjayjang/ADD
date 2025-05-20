import numpy as np
import pandas as pd
import tensorflow as tf
import gpflow
import joblib
import gpsig
from sklearn.base import ClassifierMixin
from .base_classifier import BaseClassifier
from gpsig.preprocessing import tabulate_list_of_sequences

# GPflow autobuild 설정
try:
    gpflow.config.set_default_autobuild(False)
except Exception:
    pass
try:
    tf.reset_default_graph()
except AttributeError:
    pass

class GPSigClassifier(BaseClassifier, ClassifierMixin):
    def __init__(self, ckpt_dir: str, scaler_path: str):
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.scaler_path = scaler_path
        self.scaler = joblib.load(self.scaler_path)
        with tf.Graph().as_default():
            self.model = self._build_model()
            self.sess = tf.Session()
            with self.model.enquire_session(session=self.sess):
                saver = tf.train.Saver()
                ckpt_path = tf.train.latest_checkpoint(self.ckpt_dir)
                if ckpt_path is None:
                    raise FileNotFoundError(f"No checkpoint in {self.ckpt_dir}")
                saver.restore(self.sess, ckpt_path)

    @classmethod
    def load(cls, ckpt_dir: str, scaler_path: str):
        """
        Checkpoint 디렉토리와 스케일러 경로를 받아
        모델 인스턴스를 생성하여 반환합니다.
        """
        return cls(ckpt_dir=ckpt_dir, scaler_path=scaler_path)

    def _build_model(self):
        num_levels, num_inducing, len_ex, n_feat = 4, 200, 297, 9
        input_dim = len_ex * (n_feat + 1)
        ten_levels = num_levels * (num_levels + 1) // 2
        Z = np.zeros((ten_levels, num_inducing, 2, n_feat), np.float64)
        feat = gpsig.inducing_variables.InducingTensors(Z, num_levels, increments=True)
        lengthscales = np.ones(n_feat, np.float64)
        kern = gpsig.kernels.SignatureRBF(
            input_dim, n_feat, num_levels,
            lengthscales=lengthscales
        )
        lik = gpflow.likelihoods.MultiClass(4)
        return gpsig.models.SVGP(
            X=np.zeros((1, input_dim)),
            Y=np.zeros((1,1), np.int64),
            kern=kern,
            feat=feat,
            likelihood=lik,
            num_latent=4,
            minibatch_size=1
        )

    def preprocess_input(self, X: np.ndarray) -> np.ndarray:
        """
        X: (n_samples, seq_len, n_features)
        GP-SIG SVGP 모델이 기대하는 2D 형태로 변환합니다.
        """
        # 1) (n_samples, seq_len, n_features) → 리스트 of (seq_len, n_features)
        seq_list = [x for x in X]
        # 2) GP-SIG 전처리 함수로 2D 탭울레이션
        X_tab = tabulate_list_of_sequences(seq_list)
        # 3) (n_samples, input_dim) 형태로 reshape
        return X_tab.reshape(len(seq_list), -1)

    def predict(self, path: str) -> int:
        X = self._preprocess_csv(path)
        X_tab = gpsig.preprocessing.tabulate_list_of_sequences([X]).reshape(1, -1)
        with self.model.enquire_session(session=self.sess):
            probs, _ = self.model.predict_y(X_tab)
        return int(np.argmax(probs, axis=1)[0])

    def predict_proba(self, path: str) -> np.ndarray:
        X = self._preprocess_csv(path)
        X_tab = gpsig.preprocessing.tabulate_list_of_sequences([X]).reshape(1, -1)
        with self.model.enquire_session(session=self.sess):
            probs, _ = self.model.predict_y(X_tab)
        return probs[0]
