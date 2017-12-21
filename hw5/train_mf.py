from keras.layers import Embedding, Dot, Add, Dense, Flatten, Input, Dropout, Multiply
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, ModelCheckpoint 
from keras.optimizers import SGD
from keras.models import Model
from CFModel import CFModel, DeepModel, MFModel, FeatureModel
import tensorflow as tf
import numpy as np
import pickle
import csv
import sys
import os

#################################################
#                    GPU                        #
#################################################
"""
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))
"""
#################################################
#                 Parameters                    #
#################################################
VECTOR_DIM = 24
BATCH_SIZE = 128
EPOCH      = 20
PATIENCE   = 3
valid_size = 0
do_normalize = True
genre_num = 5
train_file = 'data/train.csv'
model_file = 'model/model_mf_final.h5'
normalize_file = 'data/normalize.npy'
user_token_file = 'model/user_tok.pickle'
movie_token_file = 'model/movie_tok.pickle'
genre_file = 'data/movie_genre.pickle'
#################################################
#                 Functions                     #
#################################################
def read_data(file):
	f = open(file,'r')
	data = np.array([line for line in csv.reader(f)])
	return data[1:,1:]

def normalization(data):
	m = np.mean(data)
	s = np.std(data)
	np.save(normalize_file, np.array([m,s]))
	return (data-m)/s

def genres_vectors(l):
	vecs = np.zeros((len(l)+1,genre_num))
	for i in range(len(l)):
		vecs[i+1][l[i]] = 1
	print(vecs)
	return vecs

def build_CFModel():
	global tok_u
	global tok_m

	model = CFModel(len(tok_u.word_index)+1, len(tok_m.word_index)+1, VECTOR_DIM)
	model.compile(loss="mse", optimizer="Adamax")
	model.summary()
	return model

def build_MFModel():
	global tok_u
	global tok_m

	model = MFModel(len(tok_u.word_index)+1, len(tok_m.word_index)+1, VECTOR_DIM)
	model.compile(loss="mse", optimizer="Adamax")
	model.summary()
	return model

def build_DeepModel():
	global tok_u
	global tok_m

	model = DeepModel(len(tok_u.word_index)+1, len(tok_m.word_index)+1, VECTOR_DIM)
	model.compile(loss="mse", optimizer="Adamax")
	model.summary()
	return model

def build_FeatureModel(_gvec):
	global tok_u
	global tok_m

	model = FeatureModel(len(tok_u.word_index)+1, len(tok_m.word_index)+1, genre_num ,_gvec, VECTOR_DIM)
	model.compile(loss="mse", optimizer="Adamax")
	model.summary()
	return model

#################################################
#                   Data                        #
#################################################
train = read_data(train_file)
np.random.shuffle(train)

users = train[:,0]
movies = train[:,1]
rate = (train[:,2].reshape(train.shape[0]).astype(np.float))

if do_normalize:
	print(rate)
	rate = normalization(rate)
	print(rate)
else:
	print(rate)
	rate /= 5
	print(rate)

with open( user_token_file , 'rb') as handle:
    tok_u = pickle.load(handle)

with open(movie_token_file, 'rb') as handle:
    tok_m = pickle.load(handle)

with open( genre_file , 'rb') as handle:
    genre = pickle.load(handle)

user_list = tok_u.texts_to_sequences(users)
movie_list = tok_m.texts_to_sequences(movies)

user_list = np.array(user_list)
movie_list = np.array(movie_list)

genre = genres_vectors(genre)

X_train_u = user_list[valid_size:] 
X_train_m = movie_list[valid_size:]
Y_train   = rate[valid_size:]  
X_valid_u = user_list[:valid_size] 
X_valid_m = movie_list[:valid_size]
Y_valid   = rate[:valid_size]
#print(X_train_u)

#################################################
#                    Model                      #
#################################################
#model = build_CFModel()
model = build_MFModel()
#model = build_DeepModel()
#model = build_FeatureModel(genre)

#################################################
#                  Train                        #
#################################################

earlystopping = EarlyStopping(monitor='val_loss', patience = PATIENCE, verbose=1, mode='min')

checkpoint = ModelCheckpoint(filepath=model_file, 
                             verbose=1,
                             save_best_only=True,
                             monitor='val_loss',
                             mode='min')

History = model.fit([X_train_u,X_train_m,np.zeros(X_train_u.shape[0])],Y_train,
		batch_size=BATCH_SIZE,epochs=EPOCH,
		validation_data=([X_valid_u,X_valid_m,np.zeros(X_valid_u.shape[0])],Y_valid),
		#validation_split=0.1,
		#shuffle=True,
		callbacks=[checkpoint, earlystopping])
"""
History = model.fit([X_train_u,X_train_m],Y_train,
		batch_size=BATCH_SIZE,epochs=EPOCH,
		validation_data=([X_valid_u,X_valid_m],Y_valid),
		#validation_split=0.1,
		#shuffle=True,
		callbacks=[checkpoint, earlystopping])
"""

if(valid_size == 0):
	model.save(model_file)
	print("save model to: ", model_file)
