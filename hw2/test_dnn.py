import numpy as np
import sys

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

train_x  = sys.argv[1]  #'/home/louiefu/Desktop/ML_HW2/X_train'
train_y  = sys.argv[2]  #'/home/louiefu/Desktop/ML_HW2/Y_train'
test_x   = sys.argv[3]  #'/home/louiefu/Desktop/ML_HW2/X_test'
out_file = sys.argv[4]  #'/home/louiefu/Desktop/ML_HW2/cheat/Y_adaboost_5.csv'

#read in X_test all -> Test_datas :(total lines in X_test-1, 106) array
X_test = np.genfromtxt(test_x, delimiter=',')
X_test = np.delete(X_test,0,0)

bdt = joblib.load('dnn_model.pkl')

#prediction
P = bdt.predict(X_test)
result = np.reshape(P,(-1,1))

#write to result_logistic.csv
print("start writing result_dnn.csv")
id_num = []
for i in range(result.shape[0]):
	id_num.append(i+1)
id_array = np.array(id_num)
out =  np.column_stack((id_array, result))
np.savetxt( out_file , out, delimiter=',', fmt="%i", header = 'id,label',comments='') 

print("finishing writing "+out_file+" !")
