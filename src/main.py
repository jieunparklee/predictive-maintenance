import argparse
import matplotlib.pyplot as plt
import mplcursors
import numpy as np
import os
import pandas as pd
import tensorflow

from keras.layers import Dense, LSTM
from keras.models import Sequential
from numpy.random import seed
from scipy import stats
from sklearn import preprocessing
from utils import * 

parser = argparse.ArgumentParser(description='Motor maintenance prediction LSTM model')

parser.add_argument('--dataset', type=int, default=2)
parser.add_argument('--new-testset', type=int, default=0)
parser.add_argument('--train-ratio', type=float, default=0.2)
parser.add_argument('--past', type=int, default=5)
parser.add_argument('--future', type=int, default=3)
parser.add_argument('--batch', type=int, default=20)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--hidden-units', type=int, default=10)
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--save-new', type=int, default=0)

args = parser.parse_args()

output_dir = 'results/'
if not os.path.isdir(output_dir):
    os.makedirs(outpute_dir)

merged_data=pd.read_csv('dataset/IMS/merged_dataset_'+str(args.dataset)+'.csv',index_col=0, header=0)
merged_data.index = pd.to_datetime(merged_data.index, format='%Y-%m-%d %H:%M:%S')

scaler = preprocessing.MinMaxScaler()

scaled_df = scaler.fit_transform(merged_data)
scaled_df = pd.DataFrame(scaled_df, columns=['Bearing 1','Bearing 2','Bearing 3','Bearing 4'], index=merged_data.index)

split = args.train_ratio
dataset_train = scaled_df[:str(scaled_df.iloc[int(len(scaled_df)*split)].name)] # normal operating conditions
dataset_test = scaled_df[str(scaled_df.iloc[int(len(scaled_df)*split)].name):]

X_train = dataset_train
X_test = dataset_test

seed(10)
tensorflow.random.set_seed(10)
act_func = 'tanh'
num_features = len(X_train.columns)

# design network
model = Sequential()
model.add(LSTM(args.hidden_units, activation=act_func)) 
model.add(Dense(len(merged_data.columns)))
model.compile(loss='mae', optimizer='adam')

BATCH_SIZE = args.batch
NUM_EPOCHS = args.epochs

foresee = args.future 
seeing_past = args.past

y_model_train = X_train[seeing_past:].values 
X_model_train = np.array([np.array(X_train[i:i+seeing_past]) for i in range(len(X_train)-seeing_past)]) 
y_train = np.array([np.array(X_train[i:i+foresee]) for i in range(seeing_past, len(X_train)-foresee+1)]) 
X_train = X_train[:-foresee].values.reshape(X_train[:-foresee].shape[0], 1, num_features)

y_test = np.array([np.array(X_test[i:i+foresee]) for i in range(seeing_past, len(X_test)-foresee+1)]) 
X_test = X_test[:-foresee].values.reshape(X_test[:-foresee].shape[0], 1, num_features)

train_index = dataset_train.index[seeing_past-1:-foresee]
test_index = dataset_test.index[seeing_past-1:-foresee]
total_index = train_index.append(test_index)

history=model.fit(X_model_train, y_model_train,
                  batch_size=BATCH_SIZE, 
                  epochs=NUM_EPOCHS,
                  validation_split=0.05,
                  verbose = 1, 
                  shuffle=False)

y_hat_train = forecast(X_train, model, seeing_past, foresee)
scored_train = y_train-y_hat_train

mu = np.mean(scored_train.reshape(scored_train.shape[0], scored_train.shape[1]*scored_train.shape[2]), axis=0)
sigma = np.cov(scored_train.reshape(scored_train.shape[0], scored_train.shape[1]*scored_train.shape[2]), rowvar=0)

y_hat_test = forecast(X_test, model, seeing_past, foresee)
scored_test = y_test-y_hat_test

scored = np.abs(np.append(scored_train, scored_test, axis=0))
scored = pd.DataFrame(scored.reshape(scored.shape[0], scored.shape[1]*scored.shape[2]))

likelihood = scored.apply(lambda row: compute_likelihood(row, mu, sigma), axis=1)
likelihood.index = total_index

if args.save : 
	likelihood.to_csv('results/likelihood'+str(args.dataset)+'_p'+str(args.past)+'_f'+str(args.future)+'.csv')

if args.new_testset :
	merged_data2=pd.read_csv('dataset/IMS/merged_dataset_'+str(args.new_testset)+'.csv',index_col=0, header=0)
	merged_data2.index = pd.to_datetime(merged_data2.index, format='%Y-%m-%d %H:%M:%S')
	scaled_df2 = scaler.fit_transform(merged_data2)
	scaled_df2 = pd.DataFrame(scaled_df2, columns=['Bearing 1','Bearing 2','Bearing 3','Bearing 4'], index=merged_data2.index)
	X_test2 = scaled_df2

	y_test2 = np.array([np.array(X_test2[i:i+foresee]) for i in range(seeing_past, len(X_test2)-foresee+1)]) 
	X_test2 = X_test2[:-foresee].values.reshape(X_test2[:-foresee].shape[0], 1, num_features)

	total_index2 = scaled_df2.index[seeing_past-1:-foresee]

	y_hat_test2 = forecast(X_test2, model, seeing_past, foresee)
	scored_test2 = y_test2-y_hat_test2

	scored2 = np.abs(scored_test2)
	scored2 = pd.DataFrame(scored2.reshape(scored2.shape[0], scored2.shape[1]*scored2.shape[2]))

	likelihood2 = scored2.apply(lambda row: compute_likelihood(row, mu, sigma), axis=1)
	likelihood2.index = total_index2

	if args.save_new : 
		likelihood2.to_csv('results/likelihood_add'+str(args.new_testset)+'_p'+str(args.past)+'_f'+str(args.future)+'.csv')


