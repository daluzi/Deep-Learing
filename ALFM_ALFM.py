from keras.layers import Input, Embedding, LSTM, Dense, Lambda, merge
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
import numpy as np
import keras.backend as K
from random import randint

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
description_num = 5
current_path ='./Sports/'
local_path = '/Users/jiayuhan 1/PycharmProjects/CODE/Data Processing/Amazon/Raw_Data/Movies/'


def test_count(filename):
    test_num = 0
    for line in open(filename):
        line = eval(line)
        test_num += len(line[1])
    return test_num
def Construct_X_Train(filename):
    X_train=[]
    for line in open(filename):
        line=eval(line)
        des_list=line[2]
        X_train.append(des_list)
    # return des_list
    X_train=np.array(X_train)
    return X_train

def Construct_V_Train(filename):
    V_train=[]
    for line in open(filename):
        line=eval(line)
        des_list=line[2][0]
        V_train.append(des_list)
    # return des_list
    V_train=np.array(V_train)
    return V_train

def Construct_Y_Train(filename):
    Y_train=[]
    for line in open(filename):
        line=eval(line)
        score_list=line[3]
        Y_train.append(score_list)
    # return score_list
    Y_train=np.array(Y_train)
    return Y_train
#
# set parameters:
# V = 142381
# V = 226928
V = 94514
embedding_dim=50
max_len=50
filters_num =64
kernel_size=3


# description_input=Input(shape=(10,50))
# description_validation=Input(shape=(50,))
input_1=Input(shape=(description_num,50))
input_2=Input(shape=(50,))
embedding = Embedding(input_dim=V,
                 output_dim=embedding_dim,
                 input_length=max_len)
conv1d = Conv1D(filters=filters_num,
                            kernel_size=kernel_size,
                            activation='relu')
maxpool1d = MaxPool1D(48)
dense = Dense(15)

# input_vector=TimeDistributed(embedding)(description_input)
# validation_vector=embedding(description_validation)

input_vector=TimeDistributed(embedding)(input_1)
validation_vector=embedding(input_2)

convolutional_vector=TimeDistributed(conv1d)(input_vector)
validation_conv=conv1d(validation_vector)

maxpooling_vector=TimeDistributed(maxpool1d)(convolutional_vector)
validation_maxpooling=maxpool1d(validation_conv)

middle_output=TimeDistributed(dense)(maxpooling_vector)
middle_validation= dense(validation_maxpooling)

def change_dim_1(X):
    return K.squeeze(X,1)
def change_dim_2(X):
    return K.squeeze(X,2)
def repeat(X):
    return K.repeat_elements(X,description_num,1)
def repeat1(X):
    return K.repeat_elements(X,15,2)
def repeat2(X):
    return K.repeat_elements(X,description_num,1)
def dot(X,Y):
    return K.dot(X,Y)
def sum_item(X):
    return K.sum(X,axis=2)
def sqrt_item(X):
    return K.sqrt(X)
def cal_denominator(X):
    return 1/(X)
def expand_item(X):
    return K.expand_dims(X,2)
def expand_rate(X):
    return K.expand_dims(X,1)
def sum_user(X):
    return K.sum(X,1)
def sum_rate(X):
    return K.sum(X,1)
def single_exp(X):
    return K.exp(X)
def sum_exp_denominator(X):
    return 1/(K.sum(K.exp(X),1))


def _data_generator(filename):
    while 1:
        f=open(filename)
        for line in f:
            line = eval(line)
            des_list=line[2]
            Des=np.array(des_list)
            Des=np.reshape(Des,(1,description_num,50))

            i=randint(0,description_num-1)
            x_train=line[2][i]
            X_train=np.array(x_train)
            X_train=np.reshape(X_train,(1,50))

            y_train=line[3][i]
            Y_train=np.array(y_train)
            Y_train=np.reshape(Y_train,(1,1))

            yield ([Des,X_train],Y_train)
        f.close()


def _validation_generator(filename,filename1):
    while 1:
        f=open(filename)
        f1=open(filename1)
        for line,line1 in zip(f,f1):
            line = eval(line)
            line1= eval(line1)
            des_list = line[2]
            Des = np.array(des_list)
            Des = np.reshape(Des, (1, description_num, 50))

            v_train = line1[2][0]
            V_train = np.array(v_train)
            V_train = np.reshape(V_train, (1, 50))

            y_train = line1[3][0]
            Y_train = np.array(y_train)
            Y_train = np.reshape(Y_train, (1, 1))
            yield ([Des,V_train],Y_train)
        f.close()

def parse(path):
  g = open(path, 'r')
  for l in g:
    yield eval(l)

def _evaluate_generator(filename,filename1):
        f=open(filename)
        f1=open(filename1)
        for line,line1 in zip(f,f1):
            line = eval(line)
            line1= eval(line1)
            des_list = line[2]
            Des = np.array(des_list)
            Des = np.reshape(Des, (1, description_num, 50))

            for i in range(len(line1[2])):
                v_train = line1[2][i]
                V_train = np.array(v_train)
                V_train = np.reshape(V_train, (1, 50))

                y_train = line1[3][i]
                Y_train = np.array(y_train)
                Y_train = np.reshape(Y_train, (1, 1))
                yield ([Des, V_train], Y_train)



# def _test_generator(filename):

middle_output_final = Lambda(change_dim_2)(middle_output)#?*10*15
middle_validation_final = Lambda(repeat)(middle_validation)#?*10*15
molecule = merge([middle_output_final,middle_validation_final],mode='mul',dot_axes=2)
molecule = Lambda(sum_item)(molecule)

denominator1 = merge([middle_output_final,middle_output_final],mode='mul',dot_axes=2)
denominator1 = Lambda(sum_item)(denominator1)
denominator1 = Lambda(sqrt_item)(denominator1)

denominator2 = merge([middle_validation_final,middle_validation_final],mode='mul',dot_axes=2)
denominator2 = Lambda(sum_item)(denominator2)
denominator2 = Lambda(sqrt_item)(denominator2)


denominator = merge([denominator1,denominator2],mode='mul',dot_axes=1)
denominator = Lambda(cal_denominator)(denominator)

similarity = merge([molecule,denominator],mode='mul',dot_axes=1)
similarity = Lambda(expand_item)(similarity)
similarity = Lambda(repeat1)(similarity)



user = merge([similarity,middle_output_final],mode='mul')
user = Lambda(sum_user)(user)


item = Lambda(change_dim_1)(middle_validation)


rate_hat = merge([user,item],mode='mul')
rate_hat = Lambda(sum_rate)(rate_hat)
rate_hat = Lambda(expand_rate)(rate_hat)


model=Model(inputs=[input_1,input_2],outputs=rate_hat)

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mse'])
hist = model.fit_generator(_data_generator(current_path+"train_whole_list_5.txt"),
                    steps_per_epoch=256,
                    epochs=100,
                    validation_data=_validation_generator(current_path+"train_whole_list_5.txt",
                                                           current_path+"validation_whole_list_5.txt"),
                    validation_steps=256, verbose=2)


# scores = model.evaluate_generator(_evaluate_generator('train_whole_list_5s.txt','test_whole_list_5s.txt'),steps=134103)
# scores = model.evaluate_generator(_evaluate_generator('train_whole_list_10.txt','test_whole_list_10.txt'),steps=34266)
# scores = model.evaluate_generator(_evaluate_generator('train_whole_list_6s.txt','test_whole_list_6s.txt'),steps=134103)
test_number = test_count(current_path+"test_whole_list_5.txt")
scores = model.evaluate_generator(_evaluate_generator(current_path+'train_whole_list_5.txt',current_path+'test_whole_list_5.txt'),steps=test_number)
print(test_number)
print(scores)