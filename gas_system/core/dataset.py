import numpy as np
from torch.utils.data import Dataset


'''
탐지되는 시점부터 range()를 잡으면 될거같다.
예: 
30초에 탐지가 되면, 데이터는 1초 간격으로 업데이트(append)되지만, 
window size마다 데이터를 보고 결과 출력
 >> X 아닌듯
ㄴㅇㄴㅇ
'''
def create_training_windows(X: np.ndarray, y: np.ndarray, window_size: int, step: int = 1):
    if X.ndim == 2 and X.shape[0] < X.shape[1]:
        X = X.T
    T, F = X.shape
    windows, labels = [], []
    for start in range(0, T - window_size + 1, step):
        end = start + window_size
        win = X[start:end]
        windows.append(win)
        labels.append(y[end - 1])
    X_windows = np.stack(windows)
    y_windows = np.array(labels)
    return X_windows, y_windows

def create_expanding_windows(X: np.ndarray, y: np.ndarray, min_size: int = 1):
    if X.ndim == 2 and X.shape[0] < X.shape[1]:
        X = X.T
    T, F = X.shape
    windows_list = []
    labels = []
    for end in range(min_size, T + 1):
        win = X[:end]
        windows_list.append(win)
        labels.append(y[end - 1])
    y_windows = np.array(labels)
    return windows_list, y_windows

class GasDataset(Dataset):
    def __init__(self, windows_list: list, y_windows: np.ndarray):
        assert len(windows_list) == len(y_windows)
        self.windows = [w.astype(np.float32) for w in windows_list]
        self.y = y_windows.astype(np.int64)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.y[idx]