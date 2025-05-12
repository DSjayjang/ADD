"""
# rpy2 설치
# 명령 프롬프트
(add) C:\Users\user>conda install -c conda-forge rpy2 -y

## 명령 프롬프트에서 r —version 실행, 설치되어 있는 R 버전이 일치해야 함
(add) C:\Users\user>r --version

# 버전이 다르다면 아래 명령어를 통해 확인
(add) C:\Users\user>where r

# 오류 발생 시 아래를 설치
pip install --upgrade rpy2 ipykernel
conda install -c conda-forge typing_extensions
"""

# 환경변수 설정
## 설치된 R 경로 설정
import os
os.environ['R_HOME'] = r'C:\Programming\R\R-4.4.2'

# rpy2
import rpy2
from rpy2.robjects import r
print('rpy2 version:', rpy2.__version__)

# 경로 설정
r_path = os.path.abspath(r'C:\Programming\Github\ADD')

# cusum 실행 테스트
r.source(r_path + '\Cusum\cusum_fitting_share.R')
print('R 객체 리스트:\n', r.ls())

start_point = r['start.point']
time_labels = r['time.labels']
detect_time = time_labels[start_point[0]]

# 탐지시간 출력
# type: str
print('start_point:', start_point[0])
print('detect_time:', detect_time)

# wavelet pca 실행 테스트
r.source(r_path + '\Wavelet\ADD_function.R')
print('R 객체 리스트:\n', r.ls())

r['res']