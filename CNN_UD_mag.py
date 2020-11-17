import os
import tensorflow as tf
from tensorflow.keras.layers import Input,Conv2D,BatchNormalization,Dense,Dropout,Flatten,Activation,GlobalAveragePooling2D,Multiply,Add,Concatenate,Average
from tensorflow.keras import models
from tensorflow.keras.regularizers import l2
from tensorflow.keras import initializers,optimizers

import numpy as np
from sklearn.model_selection import KFold

from data_process_UD import data_batch_FFT_one_tester_real_UD,data_batch_FFT_one_tester_abs_UD,data_batch_FFT_one_tester_complex_UD

from sklearn.preprocessing import LabelBinarizer

epoch = 150
batch_size = 128

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

    acc_avg_total = [0]*35
    num_folds = 10

    for i in range(len(all_trailers)):
        print('.............')
        print('Test trailers is',all_trailers[i])
        train_datas,train_labels = data_batch_FFT_one_tester_abs_UD(all_trailers[i],time_length=1)
        kf = KFold(n_splits=num_folds, shuffle=True)
        kf.get_n_splits(train_datas)
        
        label_binarizer = LabelBinarizer() #标签二值化
        train_labels_one_hot = label_binarizer.fit_transform(train_labels)
        
        acc_per = [0]*10
        per_index = 0
        
        for train_index, test_index in kf.split(train_datas):
            x_tr, x_ts = train_datas[train_index], train_datas[test_index]
            label_tr, label_ts = train_labels_one_hot[train_index], train_labels_one_hot[test_index]
            
            EEG_model = cnn_model((9,251,1),decay = 0.0001) #complex: 11,502,1 or abs: 11,251,1
            
            sgd = optimizers.SGD(lr=0.001, decay=0, momentum=0.9, nesterov=False)
            EEG_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

            EEG_model.fit(x_tr, label_tr, epochs=epoch, batch_size=batch_size, verbose=0)

            test_loss, test_acc = EEG_model.evaluate(x_ts, label_ts,verbose=0)
            print("%s第%d次准确率: %.4f，共测试了%d组数据 " % (all_trailers[i],(per_index+1),test_acc, len(label_ts)))
            acc_per[per_index] = test_acc
            per_index+= 1

        print(all_trailers[i],acc_per)
        print(all_trailers[i]+' avg is',np.mean(acc_per))
        acc_avg_total[i] = np.mean(acc_per)
        
    print('.............')
    print('Total Accuracy is',acc_avg_total)
    print('Total Average Accuracy is',np.mean(acc_avg_total),'STD is',np.std(acc_avg_total))

