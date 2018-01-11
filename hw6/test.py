import numpy as np
from skimage.io import ImageCollection, imsave, imshow, imread
from keras.callbacks import EarlyStopping, ModelCheckpoint 
from keras.layers import Dense, Dropout, Input, Flatten
from keras.models import Model, load_model
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
import csv
import tensorflow as tf
import sys

#################################################
#                 Parameters                    #
#################################################
img_data_file = sys.argv[1]
test_file = sys.argv[2]
model_file = 'model_autodnn15.h5'
out_file = sys.argv[3]
valid_size = 0
PATIENCE = 10
ENCODE_DIM = 400

#################################################
#                  Functions                    #
#################################################
def make_shallow_autoencoder(input_size, encoding_dim):
	I1 = Input(shape=(input_size,))
	E = Dense(units=512, activation='relu')(I1)
	code = Dense(units=encoding_dim, activation='relu')(E)
	D = Dense(units=512, activation='relu')(code)
	out = Dense(units=input_size, activation='sigmoid')(D)
	
	autoencoder = Model(inputs=I1, outputs=out)
	
	encoder = Model(inputs=I1, outputs=code)

	# create a placeholder for an encoded (32-dimensional) input
	encoded_input = Input(shape=(encoding_dim,))
	# retrieve the last layer of the autoencoder model
	decoder_layer = autoencoder.layers[-2]
	# create the decoder model
	decoder = Model(encoded_input, decoder_layer(encoded_input))
	
	return autoencoder, encoder, decoder


#################################################
#                     Data                      #
#################################################
"""
imgs_ori = np.load(img_data_file)
img_size = imgs_ori.shape[1]
imgs_ori = imgs_ori.astype(np.float32) / 255.0
"""
#################################################
#                 Cluster Model                 #
#################################################
"""
print("loading model ... ")
autoencoder, encoder, decoder = make_shallow_autoencoder(img_size, ENCODE_DIM)
autoencoder.load_weights(model_file)

encoded_imgs = encoder.predict(imgs_ori)
print(encoded_imgs.shape)

print("doing k-means ... ")
kmeans = KMeans(n_clusters=2).fit(encoded_imgs)
all_labels = kmeans.labels_
np.save("labels.npy", all_labels)
"""
#################################################
#                     Test                      #
#################################################
test_data = pd.read_csv( test_file ,sep=",",engine="python",dtype='U').values
all_labels = np.load("labels_pca600.npy")
x_test1 = test_data[:,1].astype(int)
x_test2 = test_data[:,2].astype(int)

result = []

print("predicting ... ")

for i in range(x_test1.shape[0]):
	if all_labels[x_test1[i]] == all_labels[x_test2[i]]:
		result.append(1)
	else:
		result.append(0)

#################################################
#                  Write Out                    #
#################################################
print("writing resuls ... ")
output_file=open(out_file,'w')
writer=csv.writer(output_file)

writer.writerow(["ID","Ans"])

for i in range(len(result)):
	writer.writerow([i,result[i]])

output_file.close()