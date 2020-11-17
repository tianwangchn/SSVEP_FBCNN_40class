import os
import tensorflow as tf
from tensorflow.keras.layers import Input,Conv2D,BatchNormalization,Dense,Dropout,Flatten,Activation,GlobalAveragePooling2D,Multiply,Add,Concatenate,Average
from tensorflow.keras import models
from tensorflow.keras.regularizers import l2
from tensorflow.keras import initializers,optimizers

import numpy as np

from data_process_11_band_UI import data_batch_FFT_one_tester_real_UI,data_batch_FFT_one_tester_abs_UI,data_batch_FFT_one_tester_complex_UI

from sklearn.preprocessing import LabelBinarizer

epoch = 50
batch_size = 256

def cnn_model(input_shape,decay):
    
    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)
    input3 = Input(shape=input_shape)
    #input1
    input1_x = Conv2D(filters=18, kernel_size = (3,3),use_bias=False,padding='valid',kernel_regularizer=l2(l=decay),kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(input1)
    input1_x = BatchNormalization()(input1_x)
    input1_x = Activation('relu')(input1_x)
    input1_x = Dropout(0.25)(input1_x)

    input1_x = Conv2D(filters=18, kernel_size = (9,1),use_bias=False,padding='valid',kernel_regularizer=l2(l=decay),kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(input1_x)
    input1_x = BatchNormalization()(input1_x)
    input1_x = Activation('relu')(input1_x)
    input1_x = Dropout(0.25)(input1_x)
    
    input1_x = Conv2D(filters=18, kernel_size = (1,25),use_bias=False,padding='valid',kernel_regularizer=l2(l=decay),kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(input1_x)
    input1_x = BatchNormalization()(input1_x)
    input1_x = Activation('relu')(input1_x)
    input1_x = Dropout(0.25)(input1_x)

    input1_x_2 = GlobalAveragePooling2D()(input1_x)
    input1_x_2 = Dense(units=2,activation='relu',use_bias=False,kernel_regularizer=l2(l=decay))(input1_x_2)
    input1_x_2 = Dense(units=18, activation='sigmoid', use_bias=False,kernel_regularizer=l2(l=decay))(input1_x_2)
    input1_x = Multiply()([input1_x,input1_x_2])
    #input2
    input2_x = Conv2D(filters=18, kernel_size = (3,3),use_bias=False,padding='valid',kernel_regularizer=l2(l=decay),kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(input2)
    input2_x = BatchNormalization()(input2_x)
    input2_x = Activation('relu')(input2_x)
    input2_x = Dropout(0.25)(input2_x)

    input2_x = Conv2D(filters=18, kernel_size = (9,1),use_bias=False,padding='valid',kernel_regularizer=l2(l=decay),kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(input2_x)
    input2_x = BatchNormalization()(input2_x)
    input2_x = Activation('relu')(input2_x)
    input2_x = Dropout(0.25)(input2_x)
    
    input2_x = Conv2D(filters=18, kernel_size = (1,25),use_bias=False,padding='valid',kernel_regularizer=l2(l=decay),kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(input2_x)
    input2_x = BatchNormalization()(input2_x)
    input2_x = Activation('relu')(input2_x)
    input2_x = Dropout(0.25)(input2_x)

    input2_x_2 = GlobalAveragePooling2D()(input2_x)
    input2_x_2 = Dense(units=2,activation='relu',use_bias=False,kernel_regularizer=l2(l=decay))(input2_x_2)
    input2_x_2 = Dense(units=18, activation='sigmoid', use_bias=False,kernel_regularizer=l2(l=decay))(input2_x_2)
    input2_x = Multiply()([input2_x,input2_x_2])
    #input3
    input3_x = Conv2D(filters=18, kernel_size = (3,3),use_bias=False,padding='valid',kernel_regularizer=l2(l=decay),kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(input3)
    input3_x = BatchNormalization()(input3_x)
    input3_x = Activation('relu')(input3_x)
    input3_x = Dropout(0.25)(input3_x)

    input3_x = Conv2D(filters=18, kernel_size = (9,1),use_bias=False,padding='valid',kernel_regularizer=l2(l=decay),kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(input3_x)
    input3_x = BatchNormalization()(input3_x)
    input3_x = Activation('relu')(input3_x)
    input3_x = Dropout(0.25)(input3_x)
    
    input3_x = Conv2D(filters=18, kernel_size = (1,25),use_bias=False,padding='valid',kernel_regularizer=l2(l=decay),kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(input3_x)
    input3_x = BatchNormalization()(input3_x)
    input3_x = Activation('relu')(input3_x)
    input3_x = Dropout(0.25)(input3_x)

    input3_x_2 = GlobalAveragePooling2D()(input3_x)
    input3_x_2 = Dense(units=2,activation='relu',use_bias=False,kernel_regularizer=l2(l=decay))(input3_x_2)
    input3_x_2 = Dense(units=18, activation='sigmoid', use_bias=False,kernel_regularizer=l2(l=decay))(input3_x_2)
    input3_x = Multiply()([input3_x,input3_x_2])

    input1_x = Flatten()(input1_x)
    input2_x = Flatten()(input2_x)
    input3_x = Flatten()(input3_x)

    input1_x = Add()([input1_x,input2_x])
    input2_x = Add()([input2_x,input3_x])
    input3_x = Add()([input3_x,input1_x])

    input1_x = Dense(units=240, activation='relu', use_bias=False,kernel_regularizer=l2(l=decay),kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(input1_x)
    input2_x = Dense(units=240, activation='relu', use_bias=False,kernel_regularizer=l2(l=decay),kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(input2_x)
    input3_x = Dense(units=240, activation='relu', use_bias=False,kernel_regularizer=l2(l=decay),kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(input3_x)
    
    x = Concatenate()([input1_x,input2_x,input3_x])
    x = Dense(units=40, activation='softmax', use_bias=False,kernel_regularizer=l2(l=decay),kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(x)
    
    model = tf.keras.Model(inputs=[input1,input2,input3],outputs=x)
    
    return model

'''
Main
'''

if __name__ == "__main__":

    all_trailers = ['S1', 'S2', 'S3', 'S4', 'S5',
                    'S6', 'S7', 'S8', 'S9', 'S10',
                    'S11','S12','S13','S14','S15',
                    'S16','S17','S18','S19','S20',
                    'S21','S22','S23','S24','S25',
                    'S26','S27','S28','S29','S30',
                    'S31','S32','S33','S34','S35']

    acc_per = [0]*10
    acc_avg_total = [0]*35

    for i in range(len(all_trailers)):
        print('.............')
        print('Test trailers is',all_trailers[i])
        train_datas_1,train_datas_2,train_datas_3,train_labels,test_datas_1,test_datas_2,test_datas_3,test_labels = data_batch_FFT_one_tester_complex_UI(all_trailers[i],time_length=1)

        for j in range(len(acc_per)): #每个trailers 10次取平均值

            EEG_model = cnn_model((11,502,1),decay = 0.001) #complex: 11,502,1 or abs: 11,251,1
            
            label_binarizer = LabelBinarizer() #标签二值化
            train_labels_one_hot = label_binarizer.fit_transform(train_labels)
            test_labels_one_hot = label_binarizer.fit_transform(test_labels)
            
            sgd = optimizers.SGD(lr=0.001, decay=0, momentum=0.9, nesterov=False)
            EEG_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

            EEG_model.fit([train_datas_1,train_datas_2,train_datas_3], train_labels_one_hot, epochs=epoch, batch_size=batch_size, verbose=0)

            test_loss, test_acc = EEG_model.evaluate([test_datas_1,test_datas_2,test_datas_3], test_labels_one_hot,verbose=0)
            print("%s第%d次准确率: %.4f，共测试了%d组数据 " % (all_trailers[i],(j+1),test_acc, len(test_labels_one_hot)))
            acc_per[j] = test_acc

        print(all_trailers[i],acc_per)
        print(all_trailers[i]+' avg is',np.mean(acc_per))
        acc_avg_total[i] = np.mean(acc_per)
        
    print('.............')
    print('Total Accuracy is',acc_avg_total)
    print('Total Average Accuracy is',np.mean(acc_avg_total),'STD is',np.std(acc_avg_total))

