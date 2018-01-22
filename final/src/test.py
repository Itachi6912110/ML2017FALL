from keras.models import load_model
from gensim.models import word2vec
import numpy as np
import pickle
import h5py
import csv
import sys

from build import build_model

w2v = word2vec.Word2Vec.load("../model/w2v") 
weights = np.array(w2v.wv.syn0)

with open("../preprocess/test_P.pkl", 'rb') as f:
    test_P = pickle.load(f)

with open("../preprocess/test_Q.pkl", 'rb') as f:
	test_Q = pickle.load(f)

def word2index(model,text,length):
	vocab = dict([(k,v.index) for k,v in model.wv.vocab.items()]);
	matrix = np.zeros((len(text),length))
	for i in range(len(text)):
		line = text[i]
		j = 0;
		for word in line.split(' '):
			if j > length-1: #padding(limit) sequences to lenght
				break
			if word in vocab:
				matrix[i][j] = vocab[word]
			j = j + 1
	return matrix

test_P_vec = word2index(w2v,test_P,150)
test_Q_vec = word2index(w2v,test_Q,10)

#########################################################################

model_1 = build_model(weights)
model_1.load_weights("../model/FastQA")

[start_1,end_1] = model_1.predict([test_P_vec,test_Q_vec],batch_size=128)

start = np.argmax(np.array(start_1),axis=1).astype(int)
end = np.argmax(np.array(end_1),axis=1).astype(int)

#####################################################################

answer = []
for i in range(len(test_P)):
	cut = test_P[i].split()
	sentence = ''.join(cut)
	s = start[i]
	e = end[i]
	if s >= len(cut): 
		s = len(cut) - 1
	if s < 0:
		s = 0
	if e >= len(cut): 
		e = len(cut) - 1
	if e < 0:
		e = 0
	sp = sentence.find(cut[s])
	if sp == -1:
		sp = 0
	ep = sentence.find(cut[e])
	if ep == -1:
		ep = len(cut) - 1
	ep = ep + len(cut[e]) - 1
	answer.append([sp,ep])

with open("../preprocess/test_context_len.pkl", 'rb') as f:
    context_len = pickle.load(f)

selector = np.load("../preprocess/test_selector.npy")

for i in range(len(answer)):
	answer[i][0] = answer[i][0] + context_len[i][selector[i]]
	answer[i][1] = answer[i][1] + context_len[i][selector[i]]

answer = [np.arange(ans[0],ans[1]+1) for ans in answer]

#############################################################################

output = open(sys.argv[1], 'w')
writer = csv.writer(output)
writer.writerow(["id","answer"])

with open("../preprocess/test_id.pkl", 'rb') as f:
    id_list = pickle.load(f)

for i in range(len(answer)):
	writer.writerow([id_list[i],str(answer[i])[1:-1]])
