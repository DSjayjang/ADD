import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mycusum import uniCUSUM
from mycusum import multiCUSUM

SEED = 2025
np.random.seed(SEED)

os.chdir(r'C:\Users\user\Desktop\add') # pcrl
# os.chdir(r'C:\Users\linde\Desktop\add') # laptop

######################################################################################
# 데이터 불러오기 : gas
######################################################################################
gas_data = pd.read_csv('GAS_v2.csv', index_col = 0)
gas_data1 = gas_data + np.random.normal(0, 30, size = (236, 80)) # noise 추가 : 241015

gas_random = gas_data1.iloc[:, 0] # 우선 univariate에 대해서만
time_labels = gas_data1.index

# Univariate CUSUM
threshold = 10
phase1_len = 100 # 초기 데이터 길이 (현재는 100행 -> 몇 초 동안 볼 것인지 t로 변경해야할 필요)

cusum = uniCUSUM(phase1_len, threshold)
cusum.fit(gas_random)
violoation_lower, violoation_upper = cusum.update(gas_random)

start_point = min(violoation_lower + violoation_upper)
print(start_point)
print(time_labels[start_point])


######################################################################################
# 데이터 불러오기 : CO2
######################################################################################
co2_data = pd.read_csv('CO2.csv', index_col = 0, ).drop(columns = ['Unnamed: 81'])
co2_data = co2_data.iloc[1:, :]
co2_data = co2_data.astype(float)

co2_data1 = co2_data + np.random.normal(0, 10, size = (136, 80)) # noise 추가 : 241015

co2_random = co2_data1.iloc[:, 0] # 우선 univariate에 대해서만
time_labels = co2_data1.index

# Univariate CUSUM
threshold = 10
phase1_len = 38 # 초기 데이터 길이 (현재는 38행 -> 몇 초 동안 볼 것인지 t로 변경해야할 필요)

cusum = uniCUSUM(phase1_len, threshold)
cusum.fit(co2_random)
violoation_lower, violoation_upper = cusum.update(co2_random)

start_point = min(violoation_lower + violoation_upper)
print(start_point)
print(time_labels[start_point])



######################################################################################
# 다변량
######################################################################################

######################################################################################
# 데이터 불러오기 : gas
######################################################################################
gas_data = pd.read_csv('GAS_v2.csv', index_col = 0)
gas_data1 = gas_data + np.random.normal(0, 30, size = (236, 80)) # noise 추가 : 241015

random_col = np.random.randint(0, gas_data1.shape[1], size = 4)
gas_random = gas_data1.iloc[:, random_col]

time_labels = gas_data1.index

# Multivariate CUSUM
threshold = 10
phase1_len = 100 # 초기 데이터 길이 (현재는 100행 -> 몇 초 동안 볼 것인지 t로 변경해야할 필요)

cusum = multiCUSUM(phase1_len, threshold)
cusum.fit(gas_random)

violoation = cusum.update(gas_random)

start_point = min(violoation)
print(start_point)
print(time_labels[start_point])