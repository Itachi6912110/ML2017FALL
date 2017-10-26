#used to run test for known model
import numpy as np
import sys

#################################################
#                  parameters                   #
#################################################

train_x  = sys.argv[1]  #'/home/louiefu/Desktop/ML_HW2/X_train'
train_y  = sys.argv[2]  #'/home/louiefu/Desktop/ML_HW2/Y_train'
test_x   = sys.argv[3]  #'/home/louiefu/Desktop/ML_HW2/X_test'
out_file = sys.argv[4]  #'/home/louiefu/Desktop/ML_HW2/Y_log_1M_norm_0.01_1.csv' 
b_file   = './b.csv'
w_file   = './w.csv'
normalization = True

#################################################
#                  test result                  #
#################################################

#read in test file
#read in X_test all -> Test_datas :(total lines in X_test-1, 106) array
Test_datas = np.genfromtxt(test_x, delimiter=',')
Test_datas = np.delete(Test_datas,0,0)
w = np.reshape(np.genfromtxt(w_file, delimiter=','),(-1,1))
b = np.genfromtxt(b_file, delimiter=',')
#print(w.shape)
#print(b.shape)
#input()

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
