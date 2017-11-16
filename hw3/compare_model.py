from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import keras
import sys

test_file  = sys.argv[1]
x_test_file = 'x_test.csv'
id_test_file = 'id_test.csv'
all_test_file = 'all_test.csv'
input_model1 = 'model_sample_4.h5'
input_model2  = 'restruct_mdl7.h5'
input_model3 = 'model5_500.h5'
input_model4 = 'model_sample_3.h5'


out_file = sys.argv[2]

batch_size = 128
total_epochs = 0
PATIENCE = 50

img_row = 48
img_col = 48
input_shape = (img_row, img_col, 1)
num_classes = 7

df = pd.read_csv(test_file,header=None)

df_label = df[0]

df_label.drop( df_label.index[[0]], inplace=True)
df_label = df_label.convert_objects(convert_numeric=True)
Lables = df_label.as_matrix()
np.savetxt('id_test.csv', Lables, delimiter=",")

df = df[1].str.split(' ', 48*48, expand=True)
df.drop( df.index[[0]], inplace=True)
df = df.convert_objects(convert_numeric=True)
All_datas = df.as_matrix(columns=df.columns[:])
np.savetxt('x_test.csv', All_datas, delimiter=",")

x = np.genfromtxt(x_test_file, delimiter=',')
y = np.genfromtxt(id_test_file, delimiter=',')
y = np.reshape(y,(y.shape[0],1))
All_datas = np.hstack((y,x))
#np.savetxt('all_test.csv', All_datas, delimiter=",")

model1 = load_model(input_model1)
model2 = load_model(input_model2)
model3 = load_model(input_model3)
model4 = load_model(input_model4)

#read in test file
print("read in test file ...")
All_test  = All_datas
X  = All_test[:,1:]
X = np.reshape(X,(X.shape[0], img_row, img_col, 1))
ID = np.reshape(All_test[:,0], (-1,1))
ID = ID.astype(int)

X /= 255

#predict
print("start prediction ...")
test_prob1 = model1.predict(X)
test_prob2 = model2.predict(X)
test_prob3 = model3.predict(X)
test_prob4 = model4.predict(X)

test_prob = test_prob1*65+test_prob2*63+test_prob3*62.5+test_prob4*64
results = test_prob.argmax(axis=-1)
results = np.reshape(results,(-1,1))
results = results.astype(int)

#write to result
print("start writing result")
out =  np.column_stack((ID, results))
np.savetxt( out_file , out, delimiter=',', fmt="%s", header = 'id,label',comments='')