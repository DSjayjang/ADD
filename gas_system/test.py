import time
from pathlib import Path

import numpy as np

from core.dataset import load_data_long, RollingWindowBuffer
from core.cusum import multiCUSUM
from core.gas_detection_system import GasDetectionSystem
from models.load_base_models import load_base_models
from models.load_meta_model  import load_meta_model

# 1) 설정
DATA_CSV    = "data/Et_H_CO.csv"   # 또는 레이블 없는 테스트 파일
MODELS_DIR  = "models"
WINDOW_SIZE = 10
PHASE1_LEN  = 20
CUSUM_TH    = 5.0

def main():
    # 2) 긴 포맷 CSV 로드 (레이블 없어도 OK)
    X_raw, y_raw, label_map = load_data_long(DATA_CSV)

    # 3) 미리 학습된 베이스/메타 모델 로드
    base_clfs = load_base_models(MODELS_DIR)
    meta      = load_meta_model(f"{MODELS_DIR}/meta_xgb")

    # 4) CUSUM 설정 (벡터 단위)
    cusum = multiCUSUM(phase1_len=PHASE1_LEN, threshold=CUSUM_TH)
    cusum.fit(X_raw[:PHASE1_LEN])

    # 5) rolling buffer 준비
    buffer = RollingWindowBuffer(WINDOW_SIZE, X_raw.shape[1])

    # 6) 가스 검출 시스템 초기화
    system = GasDetectionSystem(
        base_classifiers=base_clfs,
        meta_classifier=meta,
        cusum_fn=cusum,
        window_size=WINDOW_SIZE,
        label_mapping=label_map,
    )

    # 7) 실시간 스트리밍 추론 루프
    print("▶ Streaming inference start")
    has_started = False
    t0 = None
    ts = None

    for t, sample in enumerate(X_raw):
        # 7-1) 변화점 검출 전
        if not has_started:
            if cusum.update(sample.reshape(1, -1)):
                has_started = True
                t0 = t
                ts = time.time()
                print(f"[Change point detected at t₀ = {t0}]")
            continue

        # 7-2) 변화점 이후 WINDOW_SIZE 간격일 때만
        if (t - t0 + 1) % WINDOW_SIZE != 0:
            buffer.add(sample)
            continue

        # 7-3) 버퍼에서 윈도우 꺼내 추론
        window = buffer.add(sample)
        if window is None:
            continue

        res     = system.run_pipeline_imap(window)
        elapsed = time.time() - ts

        print(
            f"t={t} window[{t0}→{t}] len={window.shape[0]} | "
            f"Fastest={res['first_model_name']} "
            f"(pred={res['first_pred']}, time={res['first_time']:.3f}s) | "
            f"models_latency={res['models_latency']:.3f}s | "
            f"meta={res['meta_pred']} | elapsed={elapsed:.3f}s"
        )


if __name__ == "__main__":
    main()
