from core.dataset import create_training_windows, create_expanding_windows
from core.cusum import CUSUMDetector
from core.gas_detection_system import GasDetectionSystem
from joblib import dump
import os
import time
import numpy as np
import pandas as pd

# MiniRocket 래퍼
from models.minirocket_classifier import MiniRocketClassifier
# GPSig 보류
# from models.gpsig_classifier import GPSigClassifier

def load_data_long(file_path: str):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    pivot = df.pivot_table(
        index=["time", "Label"],
        columns="Variable",
        values="Value",
        aggfunc="first"
    ).sort_index()
    X_raw = pivot.to_numpy(dtype=float)
    y_raw = np.array([lab for (_, lab) in pivot.index], dtype=int)
    return X_raw, y_raw, {c: c for c in np.unique(y_raw)}


def main():
    # 1) 데이터 불러오기
    X_raw, y_raw, label_mapping = load_data_long("data/Et_H_CO.csv")

    # 2) 슬라이딩 윈도우 생성
    window_size = 5
    X_train, y_train = create_training_windows(X_raw, y_raw, window_size)

    # 3) 미리 학습된 MiniRocket 모델 세 개 로드
    mr1 = MiniRocketClassifier.load("models/minirocket.pkl")
    mr2 = MiniRocketClassifier.load("models/minirocket.pkl")
    mr3 = MiniRocketClassifier.load("models/minirocket.pkl")
    base_classifiers = [mr1, mr2, mr3]

    # 4) 메타 입력 생성 (mr1, mr2, mr3)
    proba1 = mr1.predict_proba(X_train)
    proba2 = mr2.predict_proba(X_train)
    proba3 = mr3.predict_proba(X_train)
    X_meta = np.hstack([proba1, proba2, proba3])

    # 4.5) 메타용 라벨 인코딩
    from sklearn.preprocessing import LabelEncoder
    le_meta = LabelEncoder()
    y_meta = le_meta.fit_transform(y_train)

    # 5) 메타 모델 학습 및 저장
    from xgboost import XGBClassifier
    meta_clf = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
    meta_clf.fit(X_meta, y_meta)
    dump(meta_clf, "models/meta_xgb.pkl")

    # 6) CUSUM Detector 준비
    cusum = CUSUMDetector(phase1_len=20, threshold=5.0, drift=0.0)
    cusum.fit(X_raw[:20].mean(axis=1))

    # 7) 시스템 생성
    system = GasDetectionSystem(
        base_classifiers=base_classifiers,
        meta_classifier=meta_clf,
        cusum_fn=cusum,
        window_size=window_size,
        label_mapping=label_mapping,
        pretrained=True
    )

    # 8) Expanding window으로 추론
    X_exp, _ = create_expanding_windows(X_raw, y_raw, min_size=1)
    print(f"▶ Running inference on {len(X_exp)} windows")

    has_started = False
    change_time = None
    n_features = X_raw.shape[1]

    for i, window in enumerate(X_exp):
        if window.shape[0] % window_size != 0:
            continue

        # 1) CUSUM 변화점 검사
        raw_change = cusum.update(window.mean(axis=1)[-1])
        is_change = raw_change and not has_started
        if is_change:
            has_started = True
            change_time = time.time()

        # 2) 분류 모델 추론 (변화 후에만)
        if has_started:
            # 전체 파이프라인
            start = time.time()
            res = system.run_pipeline(window)
            total_latency = res["inference_time"]
            elapsed = time.time() - change_time
            pred_str = str(res["meta_pred"])

            # 개별 모델별 latency 측정
            segments = window.reshape(-1, window_size, n_features)
            model_times = {}
            for idx, clf in enumerate(base_classifiers, 1):
                t0 = time.time()
                clf.predict_proba(segments)
                model_times[f"model{idx}"] = time.time() - t0
            # 빠른 순서대로 정렬
            sorted_models = sorted(model_times.items(), key=lambda x: x[1])
            lat_str = ", ".join(f"{name}: {t:.4f}s" for name, t in sorted_models)
        else:
            total_latency = 0.0
            elapsed = 0.0
            pred_str = "—"
            lat_str = ""

        # 3) 출력
        change_str = "🔴 Change" if is_change else ""
        started_str = "True" if has_started else "False"
        print(
            f"Window #{i:4d} | {change_str:>12} | Started: {started_str:5} "
            f"| Pred: {pred_str:<3} | Total Latency: {total_latency:.4f}s "
            f"| Elapsed: {elapsed:.4f}s"
        )
        if lat_str:
            print(f"  ▶ Model latencies (fastest→slowest): {lat_str}")

if __name__ == "__main__":
    main()
