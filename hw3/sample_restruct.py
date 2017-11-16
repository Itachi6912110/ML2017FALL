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

train_file = sys.argv[1]
#test_file  = 'test.csv'
x_train_file = 'x_train.csv'
x_test_file = 'x_test.csv'
y_train_file = 'y_train.csv'
id_test_file = 'id_test.csv'
all_train_file = 'all_train.csv'
all_test_file = 'all_test.csv'
model_file  = 'model/model5_500.h5'

batch_size = 128
total_epochs = 0
PATIENCE = 50

img_row = 48
img_col = 48
input_shape = (img_row, img_col, 1)
num_classes = 7

#read in datas

df = pd.read_csv(train_file,header=None)

df_label = df[0]

df_label.drop( df_label.index[[0]], inplace=True)
df_label = df_label.convert_objects(convert_numeric=True)
Lables = df_label.as_matrix()
np.savetxt('y_train.csv', Lables, delimiter=",")

df = df[1].str.split(' ', 48*48, expand=True)
df.drop( df.index[[0]], inplace=True)
df = df.convert_objects(convert_numeric=True)
All_datas = df.as_matrix(columns=df.columns[:])
np.savetxt('x_train.csv', All_datas, delimiter=",")

x = np.genfromtxt(x_train_file, delimiter=',')
y = np.genfromtxt(y_train_file, delimiter=',')
y = np.reshape(y,(y.shape[0],1))
All_datas = np.hstack((y,x))
np.savetxt('all_train.csv', All_datas, delimiter=",")

def my_build_model():
    input_img = Input(shape=(48, 48, 1))
    block1 = Conv2D(64, (3, 3), padding='valid', activation='relu')(input_img)
    block1 = MaxPooling2D(pool_size=(2, 2))(block1)

    block2 = Conv2D(128, (3, 3), activation='relu')(block1)
    block2 = MaxPooling2D(pool_size=(2, 2))(block2)
    block2 = Dropout(0.4)(block2)

    block3 = Conv2D(256, (3, 3), activation='relu')(block2)
    block3 = MaxPooling2D(pool_size=(2, 2))(block3)
    block3 = Dropout(0.4)(block3)

    block3 = Flatten()(block3)


    fc1 = Dense(1024, activation='relu')(block3)
    fc1 = Dropout(0.5)(fc1)

    fc2 = Dense(1024, activation='relu')(fc1)
    fc2 = Dropout(0.5)(fc2)

    fc3 = Dense(512, activation='relu')(fc2)
    fc3 = Dropout(0.5)(fc3)

    predict = Dense(7)(fc3)
    predict = Activation('softmax')(predict)
    model = Model(inputs=input_img, outputs=predict)

    #opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    opt = Adam(lr=1e-3)
    #opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model

model = my_build_model()

#generate validation set and train set
print("generate training data ...")
#All_datas = np.genfromtxt(all_train_file, delimiter=',')
np.random.shuffle(All_datas)
x_test = All_datas[:4000,1:]
x_test = np.reshape(x_test,(x_test.shape[0], img_row, img_col, 1))
y_test = np.reshape(All_datas[:4000, 0],(-1,1))
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = All_datas[4000:,1:]
y_train = np.reshape(All_datas[4000:, 0],(-1,1))
x_train = np.reshape(x_train,(x_train.shape[0], img_row, img_col, 1))
y_train = keras.utils.to_categorical(y_train, num_classes)

x_train /= 255
x_test /= 255

train_datagen = ImageDataGenerator(
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip=True)


best_acc = 0
early_stop_counter = 0
flag_run = True

while flag_run:
    if early_stop_counter < PATIENCE:
        #run = input("want to do fitting? (y/n) ")
        run = 'y'
    else:
        run = 'n'
    
    if run != 'y' and run != 'Y':
        print("total_epochs = %d" %(total_epochs*5), end="\n")
        print("best accuracy = %.5f" %best_acc)
        break

    #fit model
    #epochs = eval(input("number of epochs: "))
    epochs = 100
    #shuffle every epoch
    for e in range(epochs):
        print("####################")
        print("Epoch cycle: %d / %d" %(e+1,epochs))
        print("####################")
        rand_data = np.random.permutation(All_datas[4000:,:])
        x_train = rand_data[:,1:]
        y_train = np.reshape(rand_data[:, 0],(-1,1))
        y_train = keras.utils.to_categorical(y_train, num_classes)

        #rand_flip_data = np.random.permutation(flip_datas)
        #x_train = np.concatenate((x_train, rand_flip_data[:,1:]), axis=0)
        #y_train = np.concatenate((y_train, keras.utils.to_categorical(np.reshape(rand_flip_data[:, 0],(-1,1)), num_classes)), axis=0)
        x_train = np.reshape(x_train,(x_train.shape[0], img_row, img_col, 1))

        for i in range(5):
            model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=batch_size),
                                steps_per_epoch=len(x_train)/batch_size, 
                                epochs=1, 
                                validation_data=(x_test,y_test))

            score = model.evaluate(x_test, y_test, verbose=0)
            #print('Val loss:', score[0])
            #print('Val accuracy:', score[1])
            if score[1] > best_acc:
                best_acc = score[1]
                early_stop_counter = 0
                model.save(model_file)
 
            else:
                early_stop_counter += 1

        if early_stop_counter > PATIENCE:
            print("incounter early stop! best acc: %d" %best_acc)
            print("total epoch cycle: %d" %e)
            total_epochs += e
            break
        #if never early stop
        early_stop_counter = 0

        print("now best_acc = %.5f" %best_acc)

    total_epochs += epochs
    flag_run = False