import numpy as np
import pandas as pd


class preprocessing:

    @staticmethod
    def load_csv(path, file_list) -> list[np.ndarray]:
        originals: list[np.ndarray] = [] # 타입힌트 list[np.ndarray]
        for fname in file_list:
            data_dir = path / fname

            # pandas data.frame
            df = pd.read_csv(data_dir, header=None)

            # to.numpy()
            originals.append(df.iloc[:, 3:].to_numpy())
        
        return originals
        
    @staticmethod
    def make_noise_data(originals, n, err_sd) -> np.ndarray:
        # N(0, err_sd^2) noise 생성
        noisy = []
        for ary in originals:
            rows, cols = ary.shape
            arr = np.empty((rows, cols, n))
            for i in range(n):
                rng = np.random.default_rng(seed= i+1)
                arr[..., i] = ary + rng.normal(scale=err_sd, size=(rows, cols))
            noisy.append(arr)

        """
        noisy: list[np.ndarray] = [shape(2970, 8, 100), shape(2970, 8, 100), 
                                shape(2970, 8, 100), shape(2970, 8, 100)]
        (noisy는 총 4개의 다차원 array를 담고 있는 list임)
        해당 형식을 shape(2970, 8, 400)으로 하나의 array 형태로 재편집해야 함
        data_original : shape (T, sensors, 4*n)
        """

        # 행, 열(센서의 수), n(샘플 수)
        rows, cols, n = noisy[0].shape
    
        # data_original
        data_original = np.empty((rows, cols, 4 * n))
        for k, ary in enumerate(noisy):
            data_original[..., k * n : (k + 1) * n] = ary
        
        return data_original
    
