# 타입 힌트 처리용 모듈
from __future__ import annotations

# 파일 경로 찾아주는 함수 Path
from pathlib import Path

# 그냥 os
import os

# math.log, math.exp 등의 함수 사용하기 위한 모듈
import math

# 데이터 전처리 및 분석용 모듈
import numpy as np
import pandas as pd

# wavelet
import pywt

# PCA
from sklearn.decomposition import PCA

# waveletclassifier.py 및 preprocessing.py 파일의 클래스 가져오기
from waveletclassifier import wave
from preprocessing import preprocessing


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# 분석할 csv 파일 지정

# base_dir : main.py 파일이 저장된 폴더
base_dir = Path("data_250418").resolve().parent

# data_dir : 예제 데이터가 저장된 폴더
data_dir = base_dir / "data_250418"

# csv 파일명
CSV_FILES = [
    "000_Et_H_CO_n.csv",
    "002_Et_H_CO_H.csv",
    "008_Et_H_CO_L.csv",
    "028_Et_H_CO_M.csv",
]

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# 메인 함수
def main() -> None:

    # sample, err_sd 설정
    n = 100
    err_sd = 30.0

    # csv 파일 불러오기
    raw_data: list[np.ndarray] = preprocessing.load_csv(path=data_dir, file_list=CSV_FILES)

    # 노이즈를 추가한 데이터
    data_original: np.ndarray = preprocessing.make_noise_data(raw_data, n, err_sd)

    # 분류기의 객체 인스턴스 생성
    wavelet = wave(data_original, n, wavelet= "db2", var_threshold= 0.8)
   
    # 분류기 사용 예시 : 가스 데이터에 대한 예측
    example_data = data_original[:, :, [95, 106, 309, 275, 65, 355]] # 1, 2, 4, 3, 1, 4번 가스의 데이터 샘플
    pred = wavelet.prediction(example_data)
    print(pred) # 예측값 출력
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------



# 메인 함수 실행
if __name__ == "__main__":
    main()