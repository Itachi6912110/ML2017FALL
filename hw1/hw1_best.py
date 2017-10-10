#hw1_best.py for hw1_best.sh
#using code_ans result_2_9_9hr
import sys
import pandas as pd
import numpy as np
from pandas import DataFrame, read_csv

order = 2
val_set = 0
train_hours = 9
train_feat = [2,7,8,9,10,12,15,16]
train_feat2 = [2,7,8,9,10,12]
test_loc = sys.argv[1]
out_file = sys.argv[2]

###################test##################

#read M_hw1_best.csv
M = np.genfromtxt('M_hw1_best.csv' , delimiter=',')

#read test.csv in
df2 = pd.read_csv(test_loc, encoding = 'big5' , header = None)
del df2[0]
del df2[1]

#change NR to 0
for i in range(2,11):
        df2[i].replace(to_replace=['NR'],value=0,inplace=True)

#convert whole df to float
df2 = df2.convert_objects(convert_numeric=True)

#data combining - generating Test_datas(np.array)
drop_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
Test_datas = df2.head(18).as_matrix(columns=df2.head(18).columns[:])
df2.drop(df2.index[drop_list],inplace=True)
while not df2.empty:
        df2_temp = df2.head(18)
        Temp_datas = df2_temp.as_matrix(columns=df2_temp.columns[:])
        Test_datas = np.concatenate((Test_datas, Temp_datas), axis=1)
        df2.drop(df2.index[drop_list],inplace=True)

#calculate result
T_r, T_c = Test_datas.shape
result = np.zeros(T_c//9)

#print("start calculating test result ...")
for i in range(9,T_c+9,9):
	datas_t_temp = np.array([])
	for feat in train_feat:
		datas_t_temp = np.hstack((datas_t_temp,Test_datas[feat,i-train_hours:i]))
	if order == 2:
		for feat in train_feat2:
			datas_t_temp = np.hstack((datas_t_temp,Test_datas[feat,i-train_hours:i] ** 2))
	datas_t_temp = np.hstack((datas_t_temp,np.array([1])))
	datas_t = np.reshape(datas_t_temp,(1,-1))
	y_estimated = np.dot(datas_t , M)
	result[i//9 -1] = y_estimated
	#print("test result: %d" %i, end = "\r")

#write to result.csv
#print("start writing " + out_file)
id_num = []
for i in range(T_c//9):
	id_num.append("id_%d" %i)
id_array = np.array(id_num)
out =  np.column_stack((id_array, result))
np.savetxt( out_file , out, delimiter=',', fmt="%s", header = 'id,value',comments='')