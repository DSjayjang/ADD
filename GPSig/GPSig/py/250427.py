# noise 90으로 고정, 인스턴스 30개
#%%
# imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
sys.path.append('..')

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import gpflow as gp
import gpsig

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

#%% define
def add_noise_df(df, noise=0.05):
    numeric_cols = [col for col in df.columns 
                    if pd.api.types.is_numeric_dtype(df[col]) 
                    and col != 'Time']
    
    df_noised = df.copy()
    
    for col in numeric_cols:
        col_sd = df[col].std()
        noise_values = np.random.normal(
            loc=0, 
            scale=90,
            #scale=col_sd * noise, 
            size=len(df)
        )
        df_noised[col] = df[col] + noise_values
    return df_noised

def generate_instances(data, repetitions=10, noise_=0.03):
    all_data = data.copy()
    
    for _ in range(repetitions):
        noisy_data = add_noise_df(data, noise=noise_)
        all_data = pd.concat([all_data, noisy_data], axis=0, ignore_index=True)
    
    return all_data

#%%
NUM_REPETITIONS = 30
NOISE_LEVEL = 90

path = "B:/PCR_lab/202505/data/"

df1 = pd.read_csv(path + "000_Et_H_CO_n.csv")
df2 = pd.read_csv(path + "002_Et_H_CO_H.csv")
df3 = pd.read_csv(path + "008_Et_H_CO_L.csv")
df4 = pd.read_csv(path + "028_Et_H_CO_M.csv")

dfs = [df1, df2, df3, df4]
#%%
dfs_processed = []

for df in dfs:
    df.drop(labels=["Time(s)", "Temperature(oC)", "Relative_Humidity(%)"], axis=1, inplace=True)
    df = df.groupby(df.index // 10).mean().reset_index(drop=True)
    df_noised_train = generate_instances(df, repetitions=NUM_REPETITIONS, noise_=NOISE_LEVEL)
    dfs_processed.append(df_noised_train.iloc[297:,:])
#%%
df_train = pd.concat(dfs_processed, axis=0)

dfs_test = []
for df in dfs:
    df = df.groupby(df.index // 10).mean().reset_index(drop=True)
    df_noised_train = generate_instances(df, repetitions=NUM_REPETITIONS, noise_=NOISE_LEVEL)
    dfs_test.append(df_noised_train.iloc[297:,:])
df_test = pd.concat(dfs_test, axis=0)
#%%
X_train = pd.DataFrame(columns=['dim_0', 'dim_1', 'dim_2', 'dim_3', 'dim_4', 'dim_5', 'dim_6', 'dim_7'])
X_test = pd.DataFrame(columns=['dim_0', 'dim_1', 'dim_2', 'dim_3', 'dim_4', 'dim_5', 'dim_6', 'dim_7'])

for i in range(120):
    start_idx = i * 297
    end_idx = (i + 1) * 297
    
    dim_0_series = pd.Series(df_train.iloc[start_idx:end_idx, 0].values)
    dim_1_series = pd.Series(df_train.iloc[start_idx:end_idx, 1].values)
    dim_2_series = pd.Series(df_train.iloc[start_idx:end_idx, 2].values)
    dim_3_series = pd.Series(df_train.iloc[start_idx:end_idx, 3].values)
    dim_4_series = pd.Series(df_train.iloc[start_idx:end_idx, 4].values)
    dim_5_series = pd.Series(df_train.iloc[start_idx:end_idx, 5].values)
    dim_6_series = pd.Series(df_train.iloc[start_idx:end_idx, 6].values)
    dim_7_series = pd.Series(df_train.iloc[start_idx:end_idx, 7].values)
    
    X_train = X_train.append({
        'dim_0': dim_0_series,
        'dim_1': dim_1_series,
        'dim_2': dim_2_series,
        'dim_3': dim_3_series,
        'dim_4': dim_4_series,
        'dim_5': dim_5_series,
        'dim_6': dim_6_series,
        'dim_7': dim_7_series
    }, ignore_index=True)

for i in range(120):
    start_idx = i * 297
    end_idx = (i + 1) * 297
    
    dim_0_series = pd.Series(df_test.iloc[start_idx:end_idx, 0].values)
    dim_1_series = pd.Series(df_test.iloc[start_idx:end_idx, 1].values)
    dim_2_series = pd.Series(df_test.iloc[start_idx:end_idx, 2].values)
    dim_3_series = pd.Series(df_test.iloc[start_idx:end_idx, 3].values)
    dim_4_series = pd.Series(df_test.iloc[start_idx:end_idx, 4].values)
    dim_5_series = pd.Series(df_test.iloc[start_idx:end_idx, 5].values)
    dim_6_series = pd.Series(df_test.iloc[start_idx:end_idx, 6].values)
    dim_7_series = pd.Series(df_test.iloc[start_idx:end_idx, 7].values)
    
    X_test = X_test.append({
        'dim_0': dim_0_series,
        'dim_1': dim_1_series,
        'dim_2': dim_2_series,
        'dim_3': dim_3_series,
        'dim_4': dim_4_series,
        'dim_5': dim_5_series,
        'dim_6': dim_6_series,
        'dim_7': dim_7_series
    }, ignore_index=True)

#%%
y_train = np.array([i for i in range(1, 5) for _ in range(30)])
y_test = y_train
y_test
#%%
labels_dict = {c : i for i, c in enumerate(np.unique(y_train))}
y_train = np.asarray([labels_dict[c] for c in y_train])
y_test = np.asarray([labels_dict[c] for c in y_test])


#
X_train = [np.stack(x, axis=1) for x in X_train.values]
X_test = [np.stack(x, axis=1) for x in X_test.values]
X_train = gpsig.preprocessing.add_time_to_list(X_train)
X_test = gpsig.preprocessing.add_time_to_list(X_test)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, stratify=y_train)

scaler = StandardScaler()
scaler.fit(np.concatenate(X_train, axis=0))

X_train = [scaler.transform(x) for x in X_train]
X_val = [scaler.transform(x) for x in X_val]
X_test = [scaler.transform(x) for x in X_test]


X_train = gpsig.preprocessing.tabulate_list_of_sequences(X_train)
X_val = gpsig.preprocessing.tabulate_list_of_sequences(X_val)
X_test = gpsig.preprocessing.tabulate_list_of_sequences(X_test)

num_train, len_examples, num_features = X_train.shape
num_val = X_val.shape[0]
num_test = X_test.shape[0]
num_classes = np.unique(y_train).size

num_levels = 4
num_inducing = 200
num_lags = 0

Z_init = gpsig.utils.suggest_initial_inducing_tensors(X_train, num_levels, num_inducing, labels=y_train, increments=True, num_lags=num_lags)
l_init = gpsig.utils.suggest_initial_lengthscales(X_train, num_samples=1000)

input_dim = len_examples * num_features
X_train = X_train.reshape([-1, input_dim])
X_val = X_val.reshape([-1, input_dim]) if X_val is not None else None
X_test = X_test.reshape([-1, input_dim])

feat = gpsig.inducing_variables.InducingTensors(Z_init, num_levels=num_levels, increments=True)

k = gpsig.kernels.SignatureRBF(input_dim, num_features, num_levels, lengthscales=l_init)

if num_classes == 2:
    lik = gp.likelihoods.Bernoulli()
    num_latent = 1
else:
    lik = gp.likelihoods.MultiClass(num_classes)
    num_latent = num_classes

m = gpsig.models.SVGP(X_train, y_train[:, None], kern=k, feat=feat, likelihood=lik, num_latent=num_latent, minibatch_size=50)

acc = lambda m, X, y: accuracy_score(y, np.argmax(m.predict_y(X)[0], axis=1))
nlpp = lambda m, X, y: -np.mean(m.predict_density(X, y[:, None]))

val_acc = lambda m: acc(m, X_val, y_val)
val_nlpp = lambda m: nlpp(m, X_val, y_val)

test_acc = lambda m: acc(m, X_test, y_test)
test_nlpp = lambda m: nlpp(m, X_test, y_test)

opt = gpsig.training.NadamOptimizer

m.kern.set_trainable(False)
hist = gpsig.training.optimize(m, opt(1e-3), max_iter=300, print_freq=10, save_freq=100, val_scorer=[val_acc, val_nlpp])

m.kern.set_trainable(True)
hist = gpsig.training.optimize(m, opt(1e-2), max_iter=100, print_freq=10, save_freq=100, history=hist,
                               val_scorer=[val_acc, val_nlpp], save_best_params=True, lower_is_better=True, patience=5000)
m.assign(hist['best']['params'])

X_train, y_train = np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0)
num_train = X_train.shape[0]
m.X, m.Y = X_train, y_train
m.num_data = num_train

m.kern.set_trainable(False)
hist = gpsig.training.optimize(m, opt(1e-3), max_iter=300, print_freq=10, save_freq=100, history=hist)

print('Test nlpp.: {:.03f}'.format(test_nlpp(m)))
print('Test acc.: {:.03f}'.format(test_acc(m)))
print('Test classification report:')
print(classification_report(y_test, np.argmax(m.predict_y(X_test)[0], axis=1)))

m.as_pandas_table()

print('The variances of each level:')
plt.bar(range(num_levels+1), m.kern.sigma.value * m.kern.variances.value)

print('The lengthsacles of each coordinate:')
plt.bar(range(num_features), m.kern.lengthscales.value)

time = [y['time'] for x, y in hist.items() if str(x).isnumeric()]
elbo = [y['elbo'] for x, y in hist.items() if str(x).isnumeric()]
val_acc = [y['val'][0] for x, y in hist.items() if str(x).isnumeric() and 'val' in y]
val_nlpp = [y['val'][1] for x, y in hist.items() if str(x).isnumeric() and 'val' in y]

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

axes[0].plot(time, elbo)
axes[0].set_title('ELBO over time')
axes[1].plot(time[:len(val_nlpp)], val_acc, 'tab:orange')
axes[1].set_title('Validation acc. over time')
axes[2].plot(time[:len(val_nlpp)], val_nlpp, 'tab:red')
axes[2].set_title('Validation nlpp. over time')

