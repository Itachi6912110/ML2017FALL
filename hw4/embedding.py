from gensim.models import word2vec as w2v
import numpy as np

#################################################
#                  Parameters                   #
#################################################
corpus_file = 'corpus.txt'
save_file = 'embed_model_128'

#################################################
#                     Class                     #
#################################################
class GenSentence(object):
	def __init__(self,dir_name):
		self.dir_name = dir_name

	def __iter__(self):
		f = open(self.dir_name)
		for line in f:
			yield line.split()

#################################################
#                  Embedding                    #
#################################################
sentence = GenSentence(corpus_file)
model = w2v.Word2Vec(sentence,max_vocab_size=20000,size=128,iter=10,sorted_vocab=0)

model.save(save_file)