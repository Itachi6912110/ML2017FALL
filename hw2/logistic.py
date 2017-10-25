#hw2 training logistic regression model for classification
import pandas as pd
import numpy as np
from pandas import DataFrame, read_csv
import sys

#################################################
#                  parameters                   #
#################################################

train_x  = sys.argv[1]  #'/home/louiefu/Desktop/ML_HW2/X_train'
train_y  = sys.argv[2]  #'/home/louiefu/Desktop/ML_HW2/Y_train'
test_x   = sys.argv[3]  #'/home/louiefu/Desktop/ML_HW2/X_test'
out_file = sys.argv[4]  #'/home/louiefu/Desktop/ML_HW2/Y_log_1M_norm_0.01_1.csv' 
normalization = True
total_iterations = 1000
lamda = 0
lr = 1
lr_b = 0
lr_w = np.zeros((106,1))
b = 0
w = np.ones((106,1))

#################################################
#                training state                 #
#################################################

#read in files
#read in X all -> All_datas :(total lines in X_train-1, 106) array
All_datas = np.genfromtxt(train_x, delimiter=',')
All_datas = np.delete(All_datas,0,0)

#read in Y_exact all -> Y_exact : (total lines in Y_train-1, 106) array
Y_exact = np.reshape(np.genfromtxt(train_y, delimiter=','),(-1,1))
Y_exact = np.delete(Y_exact,0,0)

#normalization
if normalization :
	#mean/var: 1*106
	mean = np.reshape(np.sum(All_datas,axis=0)/All_datas.shape[0],(1,-1))
	var  = np.sqrt(np.reshape(np.sum(All_datas**2,axis=0)/All_datas.shape[0],(1,-1)) - mean**2)
	Mean_all = np.repeat(mean,All_datas.shape[0],axis=0)
	Var_all  = np.repeat(var,All_datas.shape[0],axis=0)
	All_datas = (All_datas - Mean_all) / Var_all
	All_datas = np.nan_to_num(All_datas)

#picking datas
#wait to do
Feat_datas = All_datas

#gradient descent
print("start training...")
for ith in range(total_iterations):
	#initialize all grads
	b_grads  = np.zeros((1,1))
	w_grads = np.zeros((106,1))

	#calculate all diff
	z = np.dot(Feat_datas,w) + b
	Y_predict = 1 / (1 + np.exp(-z))
	diff = Y_exact - Y_predict

	#calculate gradients
	b_grads = b_grads - np.reshape(np.matrix(diff).sum()*1.0,(1,1))
	w_grads = w_grads - np.dot(np.transpose(Feat_datas),diff) + 2.0*lamda*w
	
	#update customarily lr
	lr_b = lr_b + b_grads ** 2
	lr_w = lr_w + w_grads ** 2

	#update parameters
	b = b - lr / np.sqrt(lr_b) * b_grads[0,0]
	w = w - lr / np.sqrt(lr_w) * w_grads

	print("iteration: %d" %ith, end = "\r")

print("end of training...")

#################################################
#                  w & b output                 #
#################################################
print("start writing b,w...")
#f = open('b.csv','w')
#f.write(str(b))
#f.close()
b = np.resize(b,(1,1))
np.savetxt('b.csv', b , delimiter=",")
np.savetxt('w.csv', w , delimiter=",")

#################################################
#                  test result                  #
#################################################

#read in test file
#read in X_test all -> Test_datas :(total lines in X_test-1, 106) array
Test_datas = np.genfromtxt(test_x, delimiter=',')
Test_datas = np.delete(Test_datas,0,0)

#normalization
if normalization :
	#mean/var: 1*106
	mean_t = np.reshape(np.sum(Test_datas,axis=0)/Test_datas.shape[0],(1,-1))
	var_t  = np.sqrt(np.reshape(np.sum(Test_datas**2,axis=0)/Test_datas.shape[0],(1,-1)) - mean_t**2)
	Mean_t_all = np.repeat(mean_t,Test_datas.shape[0],axis=0)
	Var_t_all  = np.repeat(var_t,Test_datas.shape[0],axis=0)
	Test_datas = (Test_datas - Mean_t_all) / Var_t_all 
	Test_datas = np.nan_to_num(Test_datas)

#calculate prob. of being class 1
z = np.dot(Test_datas,w) + b
Y_estimated = 1 / (1 + np.exp(-z))

#making result
result_r, result_c = Y_estimated.shape
result = np.zeros(result_r)

for i in range(result_r):
	if Y_estimated.item((i,0)) >= 0.5:
		result[i] = 1
	else:
		result[i] = 0

#write to result_logistic.csv
print("start writing result_logistic.csv")
id_num = []
for i in range(result_r):
	id_num.append(i+1)
id_array = np.array(id_num)
out =  np.column_stack((id_array, result))
np.savetxt( out_file , out, delimiter=',', fmt="%i", header = 'id,label',comments='') 

print("finishing writing "+out_file+" !")
