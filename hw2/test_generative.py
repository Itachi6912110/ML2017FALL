import pandas as pd
import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
from pandas import DataFrame, read_csv
import sys

#################################################
#                  parameters                   #
#################################################

train_x  = sys.argv[1] # '/home/louiefu/Desktop/ML_HW2/X_train'
train_y  = sys.argv[2] #'/home/louiefu/Desktop/ML_HW2/Y_train'
test_x   = sys.argv[3] #'/home/louiefu/Desktop/ML_HW2/X_test'
out_file = sys.argv[4] #'/home/louiefu/Desktop/ML_HW2/Y_generative.csv'

#################################################
#               function to use                 #
#################################################
def Gauss_prob( m, inv_var, det_var, x):
	D , col = m.shape
	#e_exp = np.reshape(np.diag(np.dot(np.dot(np.transpose(x-m),inv(var)),x-m)),(-1,1))
	#print(np.exp(-0.5*e_exp) / (((2*np.pi)**(D/2)) * (det(var)**0.5)))
	return np.exp(-0.5*np.dot(np.dot(np.transpose(x-m),inv_var),x-m)) / (((2*np.pi)**(D/2)) * (det_var**0.5)) 

def Post_prob( P0, P1, gaus0 , gaus1):
	return gaus1*P1 / (gaus1*P1 + gaus0*P0)



#################################################
#                  test result                  #
#################################################

#read in test file
#read in X_test all -> Test_datas :(total lines in X_test-1, 106) array
#goal: make Test_datas(106,total datas)
Test_datas = np.genfromtxt(test_x, delimiter=',')
Test_datas = np.delete(Test_datas,0,0) 
Test_datas = np.transpose(Test_datas)
test_feat_count, test_data_count = Test_datas.shape

mean0 = np.reshape(np.genfromtxt('./m0.csv', delimiter=','),(-1,1))
mean1 = np.reshape(np.genfromtxt('./m1.csv', delimiter=','),(-1,1))
sigma = np.genfromtxt('./sigma.csv', delimiter=',')
P0 = np.genfromtxt('./P0.csv', delimiter=',')
P1 = np.genfromtxt('./P1.csv', delimiter=',')
#print(mean0.shape)
#print(mean1.shape)
#print(sigma.shape)
#input()

#making result
result = np.zeros(test_data_count)
inv_sig = inv(sigma)
det_sig = det(sigma)
#Mean0_test = np.repeat(mean0,test_data_count,axis=1)
#Mean1_test = np.repeat(mean1,test_data_count,axis=1)
#PP = Post_prob(P0,P1,Gauss_prob(Mean0_test,sigma,Test_datas),Gauss_prob(Mean1_test,sigma,Test_datas))

for col_num in range(test_data_count):
	x = np.reshape(Test_datas[:,col_num],(-1,1))
	PP = Post_prob(P0,P1,Gauss_prob(mean0,inv_sig,det_sig,x),Gauss_prob(mean1,inv_sig,det_sig,x))
	if PP >= 0.8:
		result[col_num] = 1
	else:
		result[col_num] = 0
	print("iteration: %d" %col_num, end = "\r")

#write to result_logistic.csv
#print("start writing result_logistic.csv")
id_num = []
for i in range(test_data_count):
	id_num.append(i+1)
id_array = np.array(id_num)
out =  np.column_stack((id_array, result))
np.savetxt( out_file , out, delimiter=',', fmt='%i', header = 'id,label',comments='') 

print("finishing writing "+out_file+" !")