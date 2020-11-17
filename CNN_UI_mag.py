import os
import tensorflow as tf
from tensorflow.keras.layers import Input,Conv2D,BatchNormalization,Dense,Dropout,Flatten,Activation,GlobalAveragePooling2D,Multiply,Add,Concatenate,Average
from tensorflow.keras import models
from tensorflow.keras.regularizers import l2
from tensorflow.keras import initializers,optimizers

import numpy as np

from data_process_UI import data_batch_FFT_one_tester_real_UI,data_batch_FFT_one_tester_abs_UI,data_batch_FFT_one_tester_complex_UI

from sklearn.preprocessing import LabelBinarizer

epoch = 50
batch_size = 512

def cnn_model(input_shape,decay):
    
    model = models.Sequential()
    model.add(Conv2D(filters=input_shape[0]*2, kernel_size = (input_shape[0],1),input_shape = input_shape,use_bias=False,padding='valid',kernel_regularizer=l2(l=decay),kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=input_shape[0]*2, kernel_size = (1,15),use_bias=False,padding='valid',kernel_regularizer=l2(l=decay),kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(units=40, activation='softmax', use_bias=False,kernel_regularizer=l2(l=decay),kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
    
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
        train_datas,train_labels,test_datas,test_labels = data_batch_FFT_one_tester_abs_UI(all_trailers[i],time_length=1)

        for j in range(len(acc_per)): #每个trailers 10次取平均值

            EEG_model = cnn_model((9,251,1),decay = 0.0001) #complex: 11,502,1 or abs: 11,251,1
            
            label_binarizer = LabelBinarizer() #标签二值化
            train_labels_one_hot = label_binarizer.fit_transform(train_labels)
            test_labels_one_hot = label_binarizer.fit_transform(test_labels)
            
            sgd = optimizers.SGD(lr=0.001, decay=0, momentum=0.9, nesterov=False)
            EEG_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) #adam的momentum = 0.9

            EEG_model.fit(train_datas, train_labels_one_hot, epochs=epoch, batch_size=batch_size, verbose=0)

            test_loss, test_acc = EEG_model.evaluate(test_datas, test_labels_one_hot,verbose=0)
            print("%s第%d次准确率: %.4f，共测试了%d组数据 " % (all_trailers[i],(j+1),test_acc, len(test_labels_one_hot)))
            acc_per[j] = test_acc

        print(all_trailers[i],acc_per)
        print(all_trailers[i]+' avg is',np.mean(acc_per))
        acc_avg_total[i] = np.mean(acc_per)
        
    print('.............')
    print('Total Accuracy is',acc_avg_total)
    print('Total Average Accuracy is',np.mean(acc_avg_total),'STD is',np.std(acc_avg_total))

