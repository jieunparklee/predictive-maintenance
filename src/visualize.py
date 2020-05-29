import argparse
import matplotlib.pyplot as plt
import mplcursors
import numpy as np
import pandas as pd
from datetime import timedelta, datetime

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=int, default=3)
parser.add_argument('--new-testset', type=int, default=0)
parser.add_argument('--train-ratio', type=float, default=0.2)
parser.add_argument('--threshold', type=float, default=-0.5)
parser.add_argument('--past', type=int, default=5)
parser.add_argument('--future', type=int, default=3)
parser.add_argument('--drop-metric', type=str, default='-')

args = parser.parse_args()

if args.new_testset :
	likelihood=pd.read_csv('results/likelihood_add'+str(args.new_testset)+'_p'+str(args.past)+'_f'+str(args.future)+'.csv',index_col=0, names=['likelihood'], header=None)
	if args.new_testset == 2 : 
		fault = datetime.strptime('2004.02.19.06.22.39', '%Y.%m.%d.%H.%M.%S')
	else :
		fault = datetime.strptime('2004.04.18.02.42.55', '%Y.%m.%d.%H.%M.%S')
else :
	likelihood=pd.read_csv('results/likelihood'+str(args.dataset)+'_p'+str(args.past)+'_f'+str(args.future)+'.csv',index_col=0, names=['likelihood'], header=None)
	if args.dataset == 2 : 
		fault = datetime.strptime('2004.02.19.06.22.39', '%Y.%m.%d.%H.%M.%S')
	else : 
		fault = datetime.strptime('2004.04.18.02.42.55', '%Y.%m.%d.%H.%M.%S')

likelihood.index = pd.to_datetime(likelihood.index, format='%Y-%m-%d %H:%M:%S')
likelihood = likelihood['likelihood'] # series 

if args.drop_metric == '%' :
	change = pd.Series((np.array(likelihood[1:]) - np.array(likelihood[:-1])) / np.abs(np.array(likelihood[:-1])))
elif args.drop_metric == '-' :
	change = pd.Series((np.array(likelihood[1:]) - np.array(likelihood[:-1])))

change.index = likelihood[1:].index


plt.rcParams["figure.figsize"] = (17,3)

if args.new_testset : 
	plt.plot(likelihood, 'g')
else : 
	plt.plot(likelihood[:int(len(likelihood)*args.train_ratio)], 'g--')
	plt.plot(likelihood[int(len(likelihood)*args.train_ratio):], 'g')

suspicious = change[change < args.threshold]
print("[Suspicious behavior]")
print(suspicious)

print("\n[Motor Fault]")
print(str(fault)+'\n')

consecutive = 0
prev_s_i = suspicious.index[0]
color = 'gold' 
marker = '^'
plt.plot(prev_s_i, likelihood[prev_s_i], marker=marker, color=color, markersize=10)
print('Warning ' + str(fault - prev_s_i) + ' prior motor fault')	

imminent = None

for s_i in suspicious.index[1:] :
	if s_i-prev_s_i <= timedelta(minutes=11) :
		consecutive = consecutive + 1
	else :
		consecutive = 0

	prev_s_i = s_i

	if (consecutive >= 2) :
		imminent = s_i 
		break

	print('Warning ' + str(fault - s_i) + ' prior motor fault')	
	plt.plot(s_i, likelihood[s_i], marker=marker, color=color, markersize=10)

if imminent :
	plt.plot(likelihood[imminent: ], 'r')
	print('\nPredicted Imminent Failure ' + str(fault - imminent) + ' prior motor fault\n')
else : 
	print('\nFailed to predict Imminent Failure\n')

mplcursors.cursor()
plt.show() 


