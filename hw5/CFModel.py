# CFModel.py
import numpy as np
from keras.layers import Embedding, Dropout, Dense, Input, Add, Dot, Flatten, Concatenate
from keras.models import Sequential, Model

class CFModel(Model):

    def __init__(self, n_users, m_items, k_factors, **kwargs):
        I1 = Input(shape=(1,))
        E1 = Embedding(n_users, k_factors,input_length=1)(I1)
        F1 = Flatten()(E1)

        I2 = Input(shape=(1,))
        E2 = Embedding(m_items, k_factors,input_length=1)(I2)
        F2 = Flatten()(E2)

        M = Dot(axes=1)([F1,F2])

        super(CFModel, self).__init__(**kwargs,
                                      inputs=[I1,I2], 
                                      outputs=M)
        

    def rate(self, user_id, item_id):
        return self.predict([np.array([user_id]), np.array([item_id])])[0][0]

class DeepModel(Sequential):

    def __init__(self, n_users, m_items, k_factors, p_dropout=0.1, **kwargs):
        I1 = Input(shape=(1,))
        E1 = Embedding(n_users, k_factors,input_length=1)(I1)
        F1 = Flatten()(E1)

        I2 = Input(shape=(1,))
        E2 = Embedding(m_items, k_factors,input_length=1)(I2)
        F2 = Flatten()(E2)

        C1 = Concatenate()([E1,E2])

        Dp1 = Dropout(p_dropout)(C1)
        H1 = Dense(150, activation='relu')(Dp1)
        Dp2 = Dropout(p_dropout)(H1)
        H2 = Dense(50, activation='relu')(Dp2)
        out = Dense(1, activation='linear')(H2)

        super(DeepModel, self).__init__(**kwargs,
                                        inputs=[I1,I2], 
                                        outputs=out)

    def rate(self, user_id, item_id):
        return self.predict([np.array([user_id]), np.array([item_id])])[0][0]

class MFModel(Model):

    def __init__(self, n_users, m_items, k_factors, **kwargs):
        I1 = Input(shape=(1,))
        E1 = Embedding(n_users, k_factors,input_length=1)(I1)
        F1 = Flatten()(E1)

        I2 = Input(shape=(1,))
        E2 = Embedding(m_items, k_factors,input_length=1)(I2)
        F2 = Flatten()(E2)

        M = Dot(axes=1)([F1,F2])

        I3 = Input(shape=(1,))
        B = Dense(units=1,activation=None)(I3)

        out = Add()([M,B])
        super(MFModel, self).__init__(**kwargs, 
                                      inputs=[I1,I2,I3], 
                                      outputs=out)

class FeatureModel(Model):

    def __init__(self, n_users, m_items, f_len, _gvec , k_factors, **kwargs):
        I1 = Input(shape=(1,))
        E1 = Embedding(n_users, k_factors,input_length=1)(I1)
        F1 = Flatten()(E1)

        I2 = Input(shape=(1,))
        E2 = Embedding(m_items, k_factors,input_length=1)(I2)
        F2 = Flatten()(E2)

        M1 = Concatenate()([F1,F2])

        E3 = Embedding(m_items, f_len,input_length=1, weights=[_gvec], trainable=False)(I2)
        F3 = Flatten()(E3)
        Feat = Dense(30, activation='relu')(F3)
        M2 = Concatenate()([M1,Feat])

        I3 = Input(shape=(1,))
        B = Dense(units=1,activation=None)(I3)

        H1 = Dense(500, activation='relu')(M2)
        H1 = Dropout(0.6)(H1)
        H2 = Dense(250, activation='relu')(H1)
        H2 = Dropout(0.6)(H2)
        H3 = Dense(50, activation='relu')(H2)
        H4 = Dense(1)(H3)
        out = Add()([H4, B])
        super(FeatureModel, self).__init__(**kwargs, 
                                      inputs=[I1,I2,I3], 
                                      outputs=out)