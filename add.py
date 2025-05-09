# %%
# rpy2 및 환경변수 설정
import os
os.environ['R_HOME'] = r'C:\Programming\R\R-4.4.2'

import rpy2
print('rpy2 version:', rpy2.__version__)

# %%
from rpy2.robjects import r

# cusum load
r_path = os.path.abspath(r'C:\Programming\Github\ADD\Cusum')
r.source(r_path + '\cusum_fitting_share.R')

print('R 객체 리스트:\n', r.ls())

# %%
start_point = r['start.point']
time_labels = r['time.labels']

detect_time = time_labels[start_point[0]]

print('start_point:', start_point[0])
print('detect_time:', detect_time)
# %%
r("print(start.point)")
# %%
