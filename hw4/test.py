from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from gensim.models import word2vec as w2v
import numpy as np
import pickle
import h5py
import csv
import sys

#################################################
#               GPU memory limit                #
#################################################
"""
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.25
set_session(tf.Session(config=config))
"""

#################################################
#                  Parameters                   #
#################################################
embed_model_file = 'embed_model_128'
model_file = 'gensim_pre3.h5'
BATCH_SIZE = 128

#################################################
#                  Functions                    #
#################################################
def word2idx(model,text):
	vocab = dict([(k,v.index) for k,v in model.wv.vocab.items()]);
	content = np.zeros((len(text),32))
	for i in range(len(text)):
		line = text[i]
		j = 0;
		for word in line.split(' '):
			if j > 31:
				break
			if word in vocab:
				content[i][j] = vocab[word]
			j = j + 1
	return content

#################################################
#             Load Word2Vec & Data              #
#################################################

embed_model = w2v.Word2Vec.load(embed_model_file)
rnn_model = load_model(model_file)

f = open(sys.argv[1],'r')
test_set = []
i = -1
for line in f.readlines():
	test_set.append(line[len(str(i))+1:-1])
	i = i + 1
test_set.pop(0)
f.close()

X_test = word2idx(embed_model,test_set)

#################################################
#                   Predict                     #
#################################################

predict = rnn_model.predict(X_test,batch_size=BATCH_SIZE)
result = np.round(predict).reshape(X_test.shape[0]).astype(int)

#################################################
#                Output Results                 #
#################################################
ID = [ x for x in range(result.shape[0])]
id_test = np.reshape(np.array(ID),(-1,1))
print("start writing result")
out =  np.column_stack((id_test, result))
np.savetxt( sys.argv[2] , out, delimiter=',', fmt="%s", header = 'id,label',comments='') 

print("finishing writing "+sys.argv[2]+" !")