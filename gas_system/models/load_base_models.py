from pathlib import Path
from models.minirocket_classifier import MiniRocketClassifier
from models.waveletPCA_classifier import WaveletPCAClassifier

def load_base_models(model_dir: str):
    """
    주어진 모델 디렉토리에서 MiniRocket 3개와 Wavelet-PCA 분류기를 로드해
    [(name, instance), …] 형태로 반환합니다.
    """
    model_dir = Path(model_dir)

    # 1) MiniRocket 모델 불러오기
    mr1 = MiniRocketClassifier.load(model_dir / "minirocket.pkl")
    mr2 = MiniRocketClassifier.load(model_dir / "minirocket.pkl")
    mr3 = MiniRocketClassifier.load(model_dir / "minirocket.pkl")

    # 2) Wavelet-PCA 모델 불러오기
    wavelet_clf = WaveletPCAClassifier.load(model_dir / "wavelet_pca.pkl")

    return [
        ("MR1", mr1),
        ("MR2", mr2),
        ("MR3", mr3),
        ("WaveletPCA", mr3),
    ]