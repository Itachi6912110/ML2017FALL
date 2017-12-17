from keras.models import load_model
from CFModel import CFModel, DeepModel, MFModel
import numpy as np
import pickle
import h5py
import csv
import sys

#################################################
#                 Parameters                    #
#################################################
test_file = sys.argv[3]
normalize_file = 'model/normalize.npy'
user_token_file = 'model/user_tok.pickle'
movie_token_file = 'model/movie_tok.pickle'
do_normalize = int(sys.argv[4])
model_type = sys.argv[5]
out_file = sys.argv[2]
model_file = sys.argv[1]
#genre_num = 5
#genre_file = 'data/movie_genre.pickle'

#################################################
#                 Functions                     #
#################################################
def read_data(file):
	f = open(file,'r')
	data = np.array([line for line in csv.reader(f)])
	return data[1:,1:]

def genres_vectors(l):
	vecs = np.zeros((len(l)+1,genre_num))
	for i in range(len(l)):
		vecs[i+1][l[i]] = 1
	print(vecs)
	return vecs

#################################################
#                   Data                        #
#################################################
test = read_data(test_file)

users = test[:,0]
movies = test[:,1]

with open( user_token_file, 'rb') as handle:
    tok_u = pickle.load(handle)

with open(movie_token_file, 'rb') as handle:
    tok_m = pickle.load(handle)

#with open( genre_file , 'rb') as handle:
#    genre = pickle.load(handle)

#genre = genres_vectors(genre)

user_list = tok_u.texts_to_sequences(users)
movie_list = tok_m.texts_to_sequences(movies)
user_list = np.array(user_list)
movie_list = np.array(movie_list)

#################################################
#                    Model                      #
#################################################
if model_type == '0': 
	model = MFModel(len(tok_u.word_index)+1, len(tok_m.word_index)+1, 24)
elif model_type == '1': 
	model = CFModel(len(tok_u.word_index)+1, len(tok_m.word_index)+1, 24)
elif model_type == '2': 
	model = DeepModel(len(tok_u.word_index)+1, len(tok_m.word_index)+1, 24)
else: 
	model = MFModel(len(tok_u.word_index)+1, len(tok_m.word_index)+1, 24)

#################################################
#                  Predict                      #
#################################################
model.load_weights(model_file)
if model_type == '0':
	predict =  model.predict([user_list,movie_list,np.zeros((user_list.shape[0],1))],verbose=1)
elif model_type == '1' or model_type == '2':
	predict =  model.predict([user_list,movie_list],verbose=1)
else:
	predict =  model.predict([user_list,movie_list,np.zeros((user_list.shape[0],1))],verbose=1)

result = predict

if do_normalize:
	ms = np.load(normalize_file)
	predict = (predict*ms[1])+ms[0]
	result = np.clip(np.squeeze(predict),1,5)
else:
	result = np.clip(np.squeeze(predict*5),1,5)

#################################################
#                  Write Out                    #
#################################################
output_file=open(out_file,'w')
writer=csv.writer(output_file)

writer.writerow(["TestDataId","Rating"])

for i in range(result.size):
	writer.writerow([i+1,result[i]])

output_file.close()