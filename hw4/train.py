from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Dropout, GRU, SimpleRNN
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.constraints import unit_norm
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
import tensorflow as tf
from gensim.models import word2vec as w2v
import matplotlib.pyplot as plt
import numpy as np
import pickle
import csv
import sys
import os

#################################################
#               GPU memory limit                #
#################################################
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.25
set_session(tf.Session(config=config))

#################################################
#                  Parameters                   #
#################################################
embed_model_file = 'embed_model_128'
corpus_file = 'corpus.txt'
save_path = 'gensim_pre3.h5'
BATCH_SIZE = 128
EPOCH = 20
PATIENCE = 3

#################################################
#                  Functions                    #
#################################################

def read_train(file):
	f = open(file,'r')
	x = list()
	y = list()
	for line in f.readlines():
		x.append(line[10:-1])
		y.append(line[0])
	return np.array(x),np.array(y).astype(int)

def read_corpus(file):
	f = open(file,'r')
	x = list()
	y = list()
	for line in f.readlines():
		x.append(line[:-1])
	f.close()
	return np.array(x)

def word2idx(model,text):
	vocab = dict([(k,v.index) for k,v in model.wv.vocab.items()]);
	content = np.zeros((len(text),32))
	for i in range(len(text)):
		line = text[i]
		j = 0;
		for word in line.split(' '):
			if j > 31: #padding(limit) sequences to lenght 32
				break
			if word in vocab:
				content[i][j] = vocab[word]
			j = j + 1
		#print(i)
	return content

def build_model(vocab_size,weights):
	model = Sequential()              
	model.add(Embedding(vocab_size,128,input_length=32,weights=[weights],trainable=True))
	#RNN
	model.add(GRU(units=512,activation="relu",dropout=0.3,recurrent_dropout=0.3,return_sequences=True))
	model.add(GRU(units=512,activation="relu",dropout=0.3,recurrent_dropout=0.3,return_sequences=True))
	model.add(GRU(units=256,activation="relu",dropout=0.2,recurrent_dropout=0.2,return_sequences=True))	
	model.add(GRU(units=256,activation="relu",dropout=0.2,recurrent_dropout=0.2,return_sequences=True))	
	model.add(GRU(units=256,activation="relu",dropout=0.2,recurrent_dropout=0.2,return_sequences=True))
	model.add(GRU(units=256,activation="relu",dropout=0.2,recurrent_dropout=0.2))	
	
	#DNN
	model.add(Dense(units=128,activation="relu"))
	model.add(Dense(units=128,activation="relu"))
	model.add(Dense(units=1,activation="sigmoid"))

	model.compile(loss="binary_crossentropy",optimizer="Adamax",metrics=["accuracy"])

	model.summary()
	return model

#################################################
#             Load Word2Vec & Data              #
#################################################
#load gensim word2vec model
#embed_model = w2v.Word2Vec.load("model/embed_model_q4_128")
embed_model = w2v.Word2Vec.load(embed_model_file)
weights = np.array(embed_model.wv.syn0)

#load data
#X_temp, Y = read_train(sys.argv[1])
f = open(sys.argv[1],'r')
y = []
for line in f.readlines():
	y.append(line[0])
Y = np.array(y).astype(int)
f.close()
#data = read_corpus(corpus_file)
f = open(corpus_file,'r')
x = []
for line in f.readlines():
	x.append(line[:-1])
data = np.array(x)
f.close()

X = data[:200000]
semi = data[200000:400000]

#change words to index
X = word2idx(embed_model,X)
semi = word2idx(embed_model, semi)

X_train = X
Y_train = Y
X_semi = semi

#################################################
#                Construct model                #
#################################################

vocab_size = weights.shape[0]

model = build_model(vocab_size,weights)

#################################################
#                  Training                     #
#################################################

earlystopping = EarlyStopping(monitor='val_acc', patience = PATIENCE, verbose=1, mode='max')

checkpoint = ModelCheckpoint(filepath=save_path, verbose=1, save_best_only=True,
                             monitor='val_acc', mode='max')

history = model.fit(X_train,Y_train,batch_size=BATCH_SIZE,epochs=EPOCH,
		validation_split=0.1 ,callbacks=[checkpoint, earlystopping])

#################################################
#              plot training process            #
#################################################
"""
#plot the training process
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
"""
#################################################
#            Semi-Supervised learning           #
#################################################

"""
#Semi supervise learning for q5
model = load_model("gensim_pre3.h5")
cycle_count = 0
labeled = 0
CYCLE = 3
while labeled < len(X_semi) and cycle_count < CYCLE:
	print("******************")
	print("CYCLE %d / %d" %(cycle_count+1, CYCLE))
	print("******************")
	
	semi_data = data[200000:400000]

	#classify for unlabeled data
	print("start prediction ...")
	results = model.predict(X_semi, verbose=1)
	results = np.reshape(results,(-1,1))

	labeled_0 = np.where( results < 0.1 )
	labeled_1 = np.where( results > 0.9 )
	labeled = labeled_0[0].shape[0] + labeled_1[0].shape[0]

	print("x_expand 0 making ...")
	x_expand = [semi_data[i] for i in labeled_0[0]]
	y_expand = np.zeros((labeled_0[0].shape[0],1))
	print("x_expand 1 making ...")
	x_expand = x_expand + [semi_data[i] for i in labeled_1[0]]
	y_expand = np.vstack((y_expand, np.ones((labeled_1[0].shape[0],1))))
	y_expand = np.reshape(y_expand,(-1,))

	x_expand = word2idx(embed_model,x_expand)
	print(type(x_expand))
	print(type(X_train))
	print(x_expand.shape)
	print(X_train.shape)

	Y_train = np.reshape(Y_train,(-1,1))
	y_expand = np.reshape(y_expand,(-1,1))
	x_train_expand = np.vstack((X_train, x_expand))
	y_train_expand = np.vstack((Y_train, y_expand))
	y_train_expand = np.reshape(y_train_expand,(-1,))

	print("start training ...")
	#model construct
	model = build_model(vocab_size,weights)
	earlystopping = EarlyStopping(monitor='val_acc', patience = 3, verbose=1, mode='max')

	save_path = "model/gensim_semi_3.h5"
	checkpoint = ModelCheckpoint(filepath=save_path, 
                             verbose=1,
                             save_best_only=True,
                             monitor='val_acc',
                             mode='max')

	history = model.fit(x_train_expand,y_train_expand,batch_size=128,epochs=20,
		validation_split=0.1 ,callbacks=[checkpoint, earlystopping])

	#plot the training process
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	plt.show()
	cycle_count += 1
"""