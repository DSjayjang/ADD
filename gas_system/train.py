# train.py
import os
import time
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from core.dataset import load_data_long, create_training_windows
from core.cusum import multiCUSUM
from models.load_base_models import train_base_models, load_base_models


# 1) 설정
DATA_CSV    = "data/Et_H_CO.csv"
MODELS_DIR  = "models"
WINDOW_SIZE = 10
PHASE1_LEN  = 20
CUSUM_TH    = 5.0

def main():
    # 2) 데이터 불러와 슬라이딩 윈도우 생성
    X_raw, y_raw, _ = load_data_long(DATA_CSV)
    X_train, y_train = create_training_windows(X_raw, y_raw, WINDOW_SIZE)

    # 3) 베이스 모델 학습 및 저장
    train_base_models(X_train, y_train, outdir=MODELS_DIR)

    # 4) 학습된 베이스 모델 로드하여 메타 피처 생성
    base_clfs = load_base_models(MODELS_DIR)
    probas = [clf.predict_proba(X_train) for _, clf in base_clfs]
    X_meta  = np.hstack(probas)
    y_enc   = LabelEncoder().fit_transform(y_train)

    # 5) 메타 XGBoost 학습 및 저장
    meta = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
    meta.fit(X_meta, y_enc)
    os.makedirs(MODELS_DIR, exist_ok=True)
    dump(meta, f"{MODELS_DIR}/meta_xgb.pkl")
    meta.save_model(f"{MODELS_DIR}/meta_xgb.model")

    print("✅ Training complete. Models saved in", MODELS_DIR)


if __name__ == "__main__":
    main()
