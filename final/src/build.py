import keras
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Embedding, Lambda, Dropout
from keras.layers import RepeatVector, Activation, multiply, concatenate
from keras.layers.recurrent import GRU, LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
import keras.backend as K
import numpy as np

def build_model(weights):
	P_len = 150
	Q_len = 10
	word_vec_len = 300

	P_in = Input(shape=(P_len,))
	Q_in = Input(shape=(Q_len,))

	P = Embedding(weights.shape[0],word_vec_len,weights=[weights],trainable=True)(P_in)
	Q = Embedding(weights.shape[0],word_vec_len,weights=[weights],trainable=True)(Q_in)

	encoder = Bidirectional(LSTM(word_vec_len,dropout=0.2,return_sequences=True))

	P_encode = encoder(P)
	P_encode = TimeDistributed(Dense(word_vec_len,
		weights=np.concatenate((np.eye(word_vec_len),np.eye(word_vec_len)),axis=1),
		use_bias=False,trainable=True))(P_encode)

	Q_encode = encoder(Q)
	Q_encode = TimeDistributed(Dense(word_vec_len,
		weights=np.concatenate((np.eye(word_vec_len),np.eye(word_vec_len)),axis=1),
		use_bias=False,trainable=True))(Q_encode)

	Q_attention = TimeDistributed(Dense(1))(Q_encode)
	Q_attention = Lambda(lambda q: keras.activations.softmax(q, axis=1))(Q_attention)
	Q_attention = Lambda(lambda q: q[0]*q[1])([Q_encode,Q_attention])
	Q_attention = Lambda(lambda q: K.sum(q,axis=1))(Q_attention)
	Q_attention = RepeatVector(P_len)(Q_attention)

	pointer_s = Lambda(lambda arg: concatenate([arg[0],arg[1],arg[2]]))([P_encode,Q_attention,multiply([P_encode,Q_attention])])
	pointer_s = TimeDistributed(Dense(word_vec_len,activation="relu"))(pointer_s)
	pointer_s = TimeDistributed(Dense(1))(pointer_s)
	pointer_s = Flatten()(pointer_s)
	pointer_s = Activation("softmax")(pointer_s)

	pos = Lambda(lambda x: K.tf.cast(K.argmax(x,axis=1),dtype=K.tf.int32))(pointer_s)
	extract_feature = Lambda(lambda arg: K.tf.gather_nd(arg[0], K.tf.stack(
		[K.tf.range(K.tf.shape(arg[1])[0]), K.tf.cast(arg[1], K.tf.int32)], axis=1)))([P_encode, pos])
	extract_feature = RepeatVector(P_len)(extract_feature)

	pointer_e = Lambda(lambda arg: concatenate([arg[0],arg[1],arg[2],multiply([arg[0],arg[1]]),multiply([arg[0],arg[2]])]))([P_encode,Q_attention,extract_feature])
	pointer_e = TimeDistributed(Dense(word_vec_len,activation="relu"))(pointer_e)
	pointer_e = TimeDistributed(Dense(1))(pointer_e)
	pointer_e = Flatten()(pointer_e)
	pointer_e = Activation("softmax")(pointer_e)

	model = Model([P_in,Q_in],[pointer_s,pointer_e])

	model.compile(loss="categorical_crossentropy",optimizer="Adamax")

	model.summary()

	return model