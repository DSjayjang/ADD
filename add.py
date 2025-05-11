# %%
import pandas as pd
import numpy as np
import os
import rpy2
from rpy2.robjects import r

print('rpy2 version:', rpy2.__version__)

# 환경변수 설정
# PCRL PC
os.environ['R_HOME'] = r'C:\Programming\R\R-4.4.2'
r_path = os.path.abspath(r'C:\Programming\Github\ADD')

# # My labtop
# os.environ['R_HOME'] = r'C:\Programming\R\R-4.3.3'
# r_path = os.path.abspath(r'C:\Programming\Git\Github\ADD')

# %%
# cusum load
r.source(r_path + '\Cusum\cusum_fitting_share.R')
print('R 객체 리스트:\n', r.ls())

start_point = r['start.point']
time_labels = r['time.labels']
detect_time = time_labels[start_point[0]]

print('start_point:', start_point[0])
print('detect_time:', detect_time)

# %%
# wavelet pca load
r.source(r_path + '\Wavelet\ADD_function.R')
print('R 객체 리스트:\n', r.ls())

# %%
r['res_mean1']

