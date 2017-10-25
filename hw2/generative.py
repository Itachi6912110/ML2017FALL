#hw2 training probabilistic generative model for classification
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
#                training state                 #
#################################################

#read in files
#read in X all -> All_datas :(total lines in X_train-1, 106) array
#print("start read in train files...")
All_datas = np.genfromtxt(train_x, delimiter=',')
All_datas = np.delete(All_datas,0,0)
data_count, feat_count = All_datas.shape

#read in Y_exact all -> Y_exact : (total lines in Y_train-1, 106) array
Y_exact = np.reshape(np.genfromtxt(train_y, delimiter=','),(-1,1))
Y_exact = np.delete(Y_exact,0,0)

#classify to 2 groups, class0 , class1
#class0/1 : (106, class_data_count)
#print("start classifying training datas...")
class0 = np.array([])
class1 = np.array([])
for i in range(data_count):
	if Y_exact[i][0] == 1:
		class1 = np.hstack((class1,All_datas[i,:]))
	else:
		class0 = np.hstack((class0,All_datas[i,:]))

class0 = np.transpose(np.reshape(class0,(-1,feat_count)))
class1 = np.transpose(np.reshape(class1,(-1,feat_count)))
feat_count0 , data_count0 = class0.shape
feat_count1 , data_count1 = class1.shape
P0 = data_count0 / (data_count0 + data_count1)
P1 = data_count1 / (data_count0 + data_count1)

#picking datas
#wait to do

#print("start calculating mean and var...")
#start generating mean0/1, sigma0/1, sigma
#mean = [sum of class 0/1 Xs]/element num of class 0/1
#mean: goal to make (106,1)
mean0 = np.reshape(np.sum(class0,axis=1)/data_count0,(-1,1))
mean1 = np.reshape(np.sum(class1,axis=1)/data_count1,(-1,1))

#sigma0/1 = [sum of (Xi-mean0/1)(Xi-mean0/1)-T] / count0/1
#sigma: goal to make (106,106)
Mean0 = np.repeat(mean0,data_count0,axis=1)
Mean1 = np.repeat(mean1,data_count1,axis=1)
sigma0 = np.dot(class0-Mean0,np.transpose(class0-Mean0)) / data_count0
sigma1 = np.dot(class1-Mean1,np.transpose(class1-Mean1)) / data_count1
sigma = P0 * sigma0 + P1 * sigma1

#################################################
#              m & sigma output                 #
#################################################
#print("start writing m0,m1,sigma...")
np.savetxt('m0.csv', mean0 , delimiter=",")
np.savetxt('m1.csv', mean1 , delimiter=",")
np.savetxt('sigma.csv', sigma , delimiter=",")

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
