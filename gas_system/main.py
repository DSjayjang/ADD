# main.py
import os
import time
from pathlib import Path

import numpy as np

from core.dataset          import load_data_long
from core.cusum            import multiCUSUM
from core.gas_detection_system import GasDetectionSystem
from models.load_base_models   import load_base_models
from models.load_meta_model    import load_meta_model

def main():
    # 1) 긴 포맷 CSV 로드
    X_raw, y_raw, label_map = load_data_long("data/Et_H_CO.csv")

    # 2) 미리 학습된 Base classifiers & Meta classifier 불러오기
    base_clfs = load_base_models("models")         # [(name, clf), ...]
    meta      = load_meta_model("models/meta_xgb") # .pkl 또는 .model 자동 감지

    # 3) multiCUSUM 초기화 (vector 입력)
    cusum = multiCUSUM(phase1_len=20, threshold=5.0)
    cusum.fit(X_raw[:20])  # 첫 20개 벡터로 Phase-I 학습

    # 4) GasDetectionSystem 생성 (CPU-affinity 프로세스 풀 내장)
    window_size = 5
    system = GasDetectionSystem(
        base_classifiers=base_clfs,
        meta_classifier=meta,
        cusum_fn=cusum,
        window_size=window_size,
        label_mapping=label_map
    )

    # 5) 실시간 스트리밍 추론
    print("▶ 실시간 추론 시작")
    has_started = False
    t0          = None
    start_time  = None

    for t, sample in enumerate(X_raw):
        # 5-1) 변화점 탐지 (vector 그대로)
        if not has_started:
            if cusum.update(sample.reshape(1, -1)):
                has_started = True
                t0          = t
                start_time  = time.time()
                print(f"[Change point detected at t₀ = {t0}]")
            continue

        # 5-2) 변화점 이후, window_size 간격으로만 추론
        if (t - t0 + 1) % window_size != 0:
            continue

        # 5-3) t₀ 부터 t까지의 윈도우로 추론
        window = X_raw[t0 : t + 1]
        res    = system.run_pipeline_imap(window)
        elapsed = time.time() - start_time

        print(
            f"window[{t0}→{t}] len={window.shape[0]} | "
            f"Fastest={res['first_model_name']} "
            f"(pred={res['first_pred']},"
            f"time={res['first_time']:.4f}s) | "
            f"models_latency={res['models_latency']:.4f}s | "
            f"meta_pred={res['meta_pred']} | "
            f"elapsed={elapsed:.4f}s"
        )

if __name__ == "__main__":
    main()
