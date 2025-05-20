from pathlib import Path
import joblib
import xgboost as xgb

class MetaXGBoost:
    def __init__(self, **kwargs):
        # GPU 가속용 기본 옵션 추가
        gpu_kwargs = {
            "tree_method": "gpu_hist",
            "predictor":   "gpu_predictor",
            "gpu_id":      0,           # 사용할 GPU ID
        }
        # 사용자가 전달한 인자와 병합 (사용자 인자가 우선)
        gpu_kwargs.update(kwargs)
        self.model = xgb.XGBClassifier(**gpu_kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

def load_meta_model(model_path: str):
    """
    저장된 메타 분류기(XGBoost)를 로드하여 반환합니다.
    """
    model_path = Path(model_path)
    # joblib로 저장한 경우
    try:
        return joblib.load(model_path.with_suffix('.pkl'))
    except FileNotFoundError:
        # XGBoost native 포맷(.model)인 경우
        clf = xgb.XGBClassifier()
        clf.load_model(str(model_path))
        return clf