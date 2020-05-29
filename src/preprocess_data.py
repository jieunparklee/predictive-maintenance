import argparse
import numpy as np
import os
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=int, default=2, help='options=[2, 3]')

args = parser.parse_args()

data_dir = 'dataset/IMS/'
if args.dataset == 2 :
	data_dir+='2nd_test'
else :
	data_dir+='4th_test/txt'

merged_data = pd.DataFrame()

for filename in os.listdir(data_dir): # getting the average value of each bearing per file 
    dataset=pd.read_csv(os.path.join(data_dir, filename), sep='\t')
    dataset_mean_abs = np.array(dataset.abs().mean())
    dataset_mean_abs = pd.DataFrame(dataset_mean_abs.reshape(1,4))
    dataset_mean_abs.index = [filename]
    merged_data = merged_data.append(dataset_mean_abs)

merged_data.columns = ['Bearing 1','Bearing 2','Bearing 3','Bearing 4']

merged_data.index = pd.to_datetime(merged_data.index, format='%Y.%m.%d.%H.%M.%S')
merged_data = merged_data.sort_index()

merged_data.to_csv('dataset/IMS/merged_dataset_'+str(args.dataset)+'.csv')