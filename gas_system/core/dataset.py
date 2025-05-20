import numpy as np
from torch.utils.data import Dataset
import pandas as pd

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
    label_map = {c: c for c in np.unique(y_raw)}
    return X_raw, y_raw, label_map

from typing import Optional

class RollingWindowBuffer:
    """
    고정 크기 순환 버퍼. 새 샘플을 add() 하면,
    윈도우가 꽉 찼을 때 (window_size, n_features) 배열을 리턴,
    그렇지 않으면 None을 리턴합니다.
    """
    def __init__(self, window_size: int, n_features: int):
        self.buf = np.zeros((window_size, n_features), dtype=float)
        self.window_size = window_size
        self.n_features = n_features
        self.idx = 0
        self.count = 0

    def add(self, x: np.ndarray) -> Optional[np.ndarray]:
        """
        x: shape (n_features,) 새 샘플
        반환: 윈도우가 완성되면 shape (window_size, n_features) ndarray,
              아직 불완전하면 None
        """
        self.buf[self.idx] = x
        self.idx = (self.idx + 1) % self.window_size
        self.count += 1

        if self.count < self.window_size:
            return None

        # idx 위치부터 순서대로 뷰로 합쳐서 반환
        if self.idx == 0:
            return self.buf.copy()
        top = self.buf[self.idx:]
        bot = self.buf[:self.idx]
        return np.vstack((top, bot))