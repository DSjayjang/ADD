import os
import time
import numpy as np
import multiprocessing as mp
import psutil
from typing import List, Tuple, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed


def _worker_predict(args):
    """
    워커 함수: 지정된 코어에서 predict_proba 실행
    args: (core_id, 모델명, clf, segments)
    """
    core_id, name, clf, segments = args
    # 이 프로세스를 지정된 CPU 코어에 고정
    proc = psutil.Process(os.getpid())
    try:
        proc.cpu_affinity([core_id])
    except AttributeError:
        # cpu_affinity 미지원 시스템용 대체 처리
        try:
            os.sched_setaffinity(0, {core_id})
        except Exception:
            pass
    # 예측 실행 및 소요 시간 측정
    start = time.time()
    proba = clf.predict_proba(segments)[0]
    duration = time.time() - start
    return name, proba, duration


class GasDetectionSystem:
    def __init__(
        self,
        base_classifiers: List[Tuple[str, Any]],
        meta_classifier: Any,
        cusum_fn: Any,
        window_size: int,
        label_mapping: Dict[Any, Any],
        pretrained: bool = True,
    ):
        """
        가스 검출 시스템 초기화

        base_classifiers: [(모델 이름, BaseClassifier 인스턴스), ...]
        meta_classifier: 학습된 메타 분류기
        cusum_fn: 변화점 탐지기
        window_size: 분할 윈도우 크기
        label_mapping: 레이블 매핑 딕셔너리
        pretrained: 베이스 분류기 학습 여부
        """
        self.base_classifiers = base_classifiers
        self.meta = meta_classifier
        self.cusum = cusum_fn
        self.ws = window_size
        self.labels = label_mapping
        self.is_fitted = pretrained
        # 프로세스 풀 생성: 분류기 수만큼 워커 프로세스
        self.pool = mp.Pool(processes=len(self.base_classifiers))
        # 사용 가능한 CPU 코어 수 확인
        self.n_cores = os.cpu_count() or len(self.base_classifiers)
        self.total_time = 0.0

    def run_pipeline_map(self, window: np.ndarray) -> Dict[str, Any]:
        """
        단일 윈도우에 대한 추론 파이프라인 실행:
          1) window를 segments 단위로 분할
          2) 각 베이스 분류기를 별도 프로세스/코어에서 병렬 실행
          3) 가장 빠른 응답과 모든 확률 수집
          4) 메타 분류기로 최종 예측
        """
        # (n_segments, ws, n_features) 모양으로 재구성
        n_feat = window.shape[1]
        segments = window.reshape(-1, self.ws, n_feat)

        # 작업 목록 준비: 각 모델을 코어에 할당
        tasks = []
        for idx, (name, clf) in enumerate(self.base_classifiers):
            core_id = idx % self.n_cores
            tasks.append((core_id, name, clf, segments))

        # 병렬 predict_proba 실행
        results = self.pool.map(_worker_predict, tasks)

        # 가장 빠른 모델 결과 선택
        first_name, first_proba, first_time = min(results, key=lambda x: x[2])
        # 모든 모델의 확률을 순서대로 정렬
        probas_map = {nm: p for nm, p, _ in results}
        ordered = [probas_map[name] for name, _ in self.base_classifiers]

        # 메타 분류기 예측
        meta_input = np.hstack(ordered).reshape(1, -1)
        meta_pred = self.meta.predict(meta_input)[0]

        # 베이스 모델 전체 병렬 소요 시간은 최대값으로 정의
        models_latency = max(d for _, _, d in results)

        return {
            'first_model_name': first_name,                       # 가장 빠른 모델 이름
            'first_pred':       int(np.argmax(first_proba)),      # 첫 모델 예측 클래스
            'first_time':       first_time,                       # 첫 응답 소요 시간
            'models_latency':   models_latency,                   # 병렬 전체 소요 시간
            'meta_pred':        meta_pred                         # 메타 분류기 예측
        }
    
    def run_pipeline_imap(self, window: np.ndarray) -> Dict[str, Any]:
        n_feat   = window.shape[1]
        segments = window.reshape(-1, self.ws, n_feat)
        # (core_id, name, clf, segments) 태스크 준비
        tasks = [
            (idx % self.n_cores, name, clf, segments)
            for idx, (name, clf) in enumerate(self.base_classifiers)
        ]

        first_result = None
        results      = []

        # self.pool은 __init__에서 만들어 두었으니 재사용
        for name, proba, duration in self.pool.imap_unordered(_worker_predict, tasks):
            if first_result is None:
                first_result = (name, proba, duration)
                print(f"[First] 모델: {name}, 예측: {np.argmax(proba)}, 시간: {duration:.4f}s")
            results.append((name, proba, duration))

        # 메타 분류기 입력 준비
        probas_map = {nm: p for nm, p, _ in results}
        ordered    = [probas_map[name] for name, _ in self.base_classifiers]
        meta_input = np.hstack(ordered).reshape(1, -1)
        meta_pred  = self.meta.predict(meta_input)[0]
        models_latency = max(d for _, _, d in results)

        return {
            'first_model_name': first_result[0],
            'first_pred':       int(np.argmax(first_result[1])),
            'first_time':       first_result[2],
            'models_latency':   models_latency,
            'meta_pred':        meta_pred
        }