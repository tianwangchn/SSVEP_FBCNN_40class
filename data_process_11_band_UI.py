from scipy.io import loadmat
import pandas as pd
import numpy as np
from scipy import signal

'''
[64, 1500, 40, 6]
电极位置、采样点数(6s,刺激前500ms,刺激后5.5s即0.5s target cue + 5s target stimuli + 0.5s screen blank)、频率目标traget(40个)、block个数

刺激频率为8~15.8Hz，间隔0.2Hz，相位由0、0.5π、π和1.5π构成
采样率为1000Hz,降采样到250Hz,通带0.15hz到200hz、ground位于Fz和FPz之间的中间、参考电极Cz
分6个block，每个block包含40个traget

频率
[8.  , 9. ,  10. , 11. , 12. , 13.,  14.,  15. , 8.2,  9.2,  10.2,
 11.2, 12.2, 13.2, 14.2, 15.2, 8.4,  9.4,  10.4, 11.4, 12.4, 13.4,
 14.4, 15.4, 8.6 , 9.6 , 10.6, 11.6, 12.6, 13.6, 14.6, 15.6,  8.8,
 9.8,  10.8, 11.8, 12.8, 13.8, 14.8, 15.8]

相位
[0.        ,1.57079633, 3.14159265, 4.71238898, 0.        ,
1.57079633, 3.14159265, 4.71238898, 1.57079633, 3.14159265,
4.71238898, 0.        , 1.57079633, 3.14159265, 4.71238898,
0.        , 3.14159265, 4.71238898, 0.        , 1.57079633,
3.14159265, 4.71238898, 0.        , 1.57079633, 4.71238898,
0.        , 1.57079633, 3.14159265, 4.71238898, 0.        ,
1.57079633, 3.14159265, 0.        , 1.57079633, 3.14159265,
4.71238898, 0.        , 1.57079633, 3.14159265, 4.71238898]

通道位置
61          O1
62          Oz
63          O2

'''

def band_filter_process(input):
    
    b_HPF_1,a_HPF_1 = signal.butter(4, 0.048, btype='highpass') #4阶，6Hz ， 6/(sampling_rate/2) 和iirfilter效果一样
    b_LPF_1,a_LPF_1 = signal.butter(4, 0.144, btype='lowpass') #4阶，18Hz ，18/(sampling_rate/2) 和iirfilter效果一样
    
    b_HPF_2,a_HPF_2 = signal.butter(4, 0.112, btype='highpass') #4阶，14Hz ， 14/(sampling_rate/2) 和iirfilter效果一样
    b_LPF_2,a_LPF_2 = signal.butter(4, 0.25, btype='lowpass') #4阶，36Hz ，36/(sampling_rate/2) 和iirfilter效果一样
    
    b_HPF_3,a_HPF_3 = signal.butter(4, 0.224, btype='highpass') #4阶，28Hz ， 28/(sampling_rate/2) 和iirfilter效果一样
    b_LPF_3,a_LPF_3 = signal.butter(4, 0.512, btype='lowpass') #4阶，64Hz ，64/(sampling_rate/2) 和iirfilter效果一样
    
    data_HF_1 = signal.filtfilt(b_HPF_1, a_HPF_1, input) #零相位滤波，滤两次，反向多滤了一次
    band1 = signal.filtfilt(b_LPF_1, a_LPF_1, data_HF_1) #6-16
    
    data_HF_2 = signal.filtfilt(b_HPF_2, a_HPF_2, input) #零相位滤波，滤两次，反向多滤了一次
    band2 = signal.filtfilt(b_LPF_2, a_LPF_2, data_HF_2) #16-32
    
    data_HF_3 = signal.filtfilt(b_HPF_3, a_HPF_3, input) #零相位滤波，滤两次，反向多滤了一次
    band3 = signal.filtfilt(b_LPF_3, a_LPF_3, data_HF_3) #32-64
    
    return band1,band2,band3

def data_batch_FFT_one_tester_abs_UI(test_trainer,time_length):
    # print('Preparing dataset.....')
    
    all_trailers = ['S1', 'S2', 'S3', 'S4', 'S5',
                    'S6', 'S7', 'S8', 'S9', 'S10',
                    'S11','S12','S13','S14','S15',
                    'S16','S17','S18','S19','S20',
                    'S21','S22','S23','S24','S25',
                    'S26','S27','S28','S29','S30',
                    'S31','S32','S33','S34','S35']
    
    simpling_rate = 250
    window_length = int(250*time_length) #窗口长度1s
    data_length = 1250 #5s
    fft_scale = int(data_length/window_length) #5s则scale=5
    
    all_datas_1 = []
    all_datas_2 = []
    all_datas_3 = []
    all_labels = []
    
    test_datas_1 = []
    test_datas_2 = []
    test_datas_3 = []
    test_labels = []
    
    test_trainer_position = all_trailers.index(test_trainer)
    
    all_trailers.pop(test_trainer_position) #删除test_trainer
    trailer = all_trailers
    # print('Train trailers are ',trailer)
    
    for i in range(len(trailer)):
        # print(trailer[i])
        rawdata = loadmat('./database/' + trailer[i] + '.mat') #1号被试
        rawdata = rawdata['data']
        
        for j in range(6): #6个block
        
            block_index = j
            
            for k in range(40): #40个trail
                
                target_index = k
                O1_data = rawdata[60,160:1410,target_index,block_index] #取中间5s,去除140ms latency
                Oz_data = rawdata[61,160:1410,target_index,block_index]
                O2_data = rawdata[62,160:1410,target_index,block_index]
                PO3_data = rawdata[54,160:1410,target_index,block_index]
                POz_data = rawdata[55,160:1410,target_index,block_index]
                PO4_data = rawdata[56,160:1410,target_index,block_index]
                PO5_data = rawdata[53,160:1410,target_index,block_index]
                Pz_data = rawdata[47,160:1410,target_index,block_index]
                PO6_data = rawdata[57,160:1410,target_index,block_index]
                
                yd4_1,yd3_1,yd2_1 = band_filter_process(O1_data)
                yd4_2,yd3_2,yd2_2 = band_filter_process(Oz_data)
                yd4_3,yd3_3,yd2_3 = band_filter_process(O2_data)
                yd4_4,yd3_4,yd2_4 = band_filter_process(PO3_data)
                yd4_5,yd3_5,yd2_5 = band_filter_process(POz_data)
                yd4_6,yd3_6,yd2_6 = band_filter_process(PO4_data)
                yd4_7,yd3_7,yd2_7 = band_filter_process(PO5_data)
                yd4_8,yd3_8,yd2_8 = band_filter_process(Pz_data)
                yd4_9,yd3_9,yd2_9 = band_filter_process(PO6_data)
                
                for m in range(fft_scale):
                    
                    all_datas_1.append(np.array([(np.abs(np.fft.rfft(yd4_1[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.abs(np.fft.rfft(yd4_2[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.abs(np.fft.rfft(yd4_3[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.abs(np.fft.rfft(yd4_4[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.abs(np.fft.rfft(yd4_5[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.abs(np.fft.rfft(yd4_6[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.abs(np.fft.rfft(yd4_7[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.abs(np.fft.rfft(yd4_8[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.abs(np.fft.rfft(yd4_9[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.abs(np.fft.rfft(yd4_1[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.abs(np.fft.rfft(yd4_2[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1]]))
                    
                    all_datas_2.append(np.array([(np.abs(np.fft.rfft(yd3_1[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.abs(np.fft.rfft(yd3_2[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.abs(np.fft.rfft(yd3_3[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.abs(np.fft.rfft(yd3_4[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.abs(np.fft.rfft(yd3_5[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.abs(np.fft.rfft(yd3_6[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.abs(np.fft.rfft(yd3_7[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.abs(np.fft.rfft(yd3_8[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.abs(np.fft.rfft(yd3_9[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.abs(np.fft.rfft(yd3_1[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.abs(np.fft.rfft(yd3_2[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1]]))
                    
                    all_datas_3.append(np.array([(np.abs(np.fft.rfft(yd2_1[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.abs(np.fft.rfft(yd2_2[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.abs(np.fft.rfft(yd2_3[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.abs(np.fft.rfft(yd2_4[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.abs(np.fft.rfft(yd2_5[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.abs(np.fft.rfft(yd2_6[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.abs(np.fft.rfft(yd2_7[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.abs(np.fft.rfft(yd2_8[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.abs(np.fft.rfft(yd2_9[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.abs(np.fft.rfft(yd2_1[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.abs(np.fft.rfft(yd2_2[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1]]))
                    
                    all_labels.append(target_index)
    
    all_datas_1 = np.expand_dims(np.array(all_datas_1),axis=-1)
    all_datas_2 = np.expand_dims(np.array(all_datas_2),axis=-1)
    all_datas_3 = np.expand_dims(np.array(all_datas_3),axis=-1)
    all_labels = np.array(all_labels)
    
    # print('Test trailer is ',test_trainer)
    rawdata = loadmat('./database/' + test_trainer + '.mat') #1号被试
    rawdata = rawdata['data']
    
    for j in range(6): #6个block
    
        block_index = j
        
        for k in range(40): #40个trail
            
            target_index = k
           
            O1_data = rawdata[60,160:1410,target_index,block_index] #取中间5s,去除140ms latency
            Oz_data = rawdata[61,160:1410,target_index,block_index]
            O2_data = rawdata[62,160:1410,target_index,block_index]
            PO3_data = rawdata[54,160:1410,target_index,block_index]
            POz_data = rawdata[55,160:1410,target_index,block_index]
            PO4_data = rawdata[56,160:1410,target_index,block_index]
            PO5_data = rawdata[53,160:1410,target_index,block_index]
            Pz_data = rawdata[47,160:1410,target_index,block_index]
            PO6_data = rawdata[57,160:1410,target_index,block_index]
            
            yd4_1,yd3_1,yd2_1 = band_filter_process(O1_data)
            yd4_2,yd3_2,yd2_2 = band_filter_process(Oz_data)
            yd4_3,yd3_3,yd2_3 = band_filter_process(O2_data)
            yd4_4,yd3_4,yd2_4 = band_filter_process(PO3_data)
            yd4_5,yd3_5,yd2_5 = band_filter_process(POz_data)
            yd4_6,yd3_6,yd2_6 = band_filter_process(PO4_data)
            yd4_7,yd3_7,yd2_7 = band_filter_process(PO5_data)
            yd4_8,yd3_8,yd2_8 = band_filter_process(Pz_data)
            yd4_9,yd3_9,yd2_9 = band_filter_process(PO6_data)
            
            for m in range(fft_scale):
                
                test_datas_1.append(np.array([(np.abs(np.fft.rfft(yd4_1[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.abs(np.fft.rfft(yd4_2[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.abs(np.fft.rfft(yd4_3[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.abs(np.fft.rfft(yd4_4[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.abs(np.fft.rfft(yd4_5[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.abs(np.fft.rfft(yd4_6[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.abs(np.fft.rfft(yd4_7[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.abs(np.fft.rfft(yd4_8[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.abs(np.fft.rfft(yd4_9[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.abs(np.fft.rfft(yd4_1[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.abs(np.fft.rfft(yd4_2[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1]]))
                
                test_datas_2.append(np.array([(np.abs(np.fft.rfft(yd3_1[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.abs(np.fft.rfft(yd3_2[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.abs(np.fft.rfft(yd3_3[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.abs(np.fft.rfft(yd3_4[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.abs(np.fft.rfft(yd3_5[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.abs(np.fft.rfft(yd3_6[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.abs(np.fft.rfft(yd3_7[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.abs(np.fft.rfft(yd3_8[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.abs(np.fft.rfft(yd3_9[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.abs(np.fft.rfft(yd3_1[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.abs(np.fft.rfft(yd3_2[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1]]))
                
                test_datas_3.append(np.array([(np.abs(np.fft.rfft(yd2_1[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.abs(np.fft.rfft(yd2_2[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.abs(np.fft.rfft(yd2_3[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.abs(np.fft.rfft(yd2_4[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.abs(np.fft.rfft(yd2_5[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.abs(np.fft.rfft(yd2_6[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.abs(np.fft.rfft(yd2_7[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.abs(np.fft.rfft(yd2_8[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.abs(np.fft.rfft(yd2_9[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.abs(np.fft.rfft(yd2_1[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.abs(np.fft.rfft(yd2_2[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1]]))
                
                test_labels.append(target_index)
    
    test_datas_1 = np.expand_dims(np.array(test_datas_1),axis=-1)
    test_datas_2 = np.expand_dims(np.array(test_datas_2),axis=-1)
    test_datas_3 = np.expand_dims(np.array(test_datas_3),axis=-1)
    test_labels = np.array(test_labels)

    return all_datas_1,all_datas_2,all_datas_3,all_labels,test_datas_1,test_datas_2,test_datas_3,test_labels

def data_batch_FFT_one_tester_real_UI(test_trainer,time_length):
    # print('Preparing dataset.....')

    all_trailers = ['S1', 'S2', 'S3', 'S4', 'S5',
                    'S6', 'S7', 'S8', 'S9', 'S10',
                    'S11','S12','S13','S14','S15',
                    'S16','S17','S18','S19','S20',
                    'S21','S22','S23','S24','S25',
                    'S26','S27','S28','S29','S30',
                    'S31','S32','S33','S34','S35']

    simpling_rate = 250
    window_length = int(250*time_length) #窗口长度1s
    data_length = 1250 #5s
    fft_scale = int(data_length/window_length) #5s则scale=5

    all_datas_1 = []
    all_datas_2 = []
    all_datas_3 = []
    all_labels = []
    
    test_datas_1 = []
    test_datas_2 = []
    test_datas_3 = []
    test_labels = []
    
    test_trainer_position = all_trailers.index(test_trainer)
    
    all_trailers.pop(test_trainer_position) #删除test_trainer
    trailer = all_trailers
    # print('Train trailers are ',trailer)
    
    for i in range(len(trailer)):
        # print(trailer[i])
        rawdata = loadmat('./database/' + trailer[i] + '.mat') #1号被试
        rawdata = rawdata['data']
        
        for j in range(6): #6个block
        
            block_index = j
            
            for k in range(40): #40个trail
                
                target_index = k
                O1_data = rawdata[60,160:1410,target_index,block_index] #取中间5s,去除140ms latency
                Oz_data = rawdata[61,160:1410,target_index,block_index]
                O2_data = rawdata[62,160:1410,target_index,block_index]
                PO3_data = rawdata[54,160:1410,target_index,block_index]
                POz_data = rawdata[55,160:1410,target_index,block_index]
                PO4_data = rawdata[56,160:1410,target_index,block_index]
                PO5_data = rawdata[53,160:1410,target_index,block_index]
                Pz_data = rawdata[47,160:1410,target_index,block_index]
                PO6_data = rawdata[57,160:1410,target_index,block_index]
                
                yd4_1,yd3_1,yd2_1 = band_filter_process(O1_data)
                yd4_2,yd3_2,yd2_2 = band_filter_process(Oz_data)
                yd4_3,yd3_3,yd2_3 = band_filter_process(O2_data)
                yd4_4,yd3_4,yd2_4 = band_filter_process(PO3_data)
                yd4_5,yd3_5,yd2_5 = band_filter_process(POz_data)
                yd4_6,yd3_6,yd2_6 = band_filter_process(PO4_data)
                yd4_7,yd3_7,yd2_7 = band_filter_process(PO5_data)
                yd4_8,yd3_8,yd2_8 = band_filter_process(Pz_data)
                yd4_9,yd3_9,yd2_9 = band_filter_process(PO6_data)
                
                for m in range(fft_scale):
                    
                    all_datas_1.append(np.array([(np.real(np.fft.rfft(yd4_1[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.real(np.fft.rfft(yd4_2[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.real(np.fft.rfft(yd4_3[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.real(np.fft.rfft(yd4_4[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.real(np.fft.rfft(yd4_5[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.real(np.fft.rfft(yd4_6[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.real(np.fft.rfft(yd4_7[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.real(np.fft.rfft(yd4_8[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.real(np.fft.rfft(yd4_9[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.real(np.fft.rfft(yd4_1[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.real(np.fft.rfft(yd4_2[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1]]))
                    
                    all_datas_2.append(np.array([(np.real(np.fft.rfft(yd3_1[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.real(np.fft.rfft(yd3_2[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.real(np.fft.rfft(yd3_3[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.real(np.fft.rfft(yd3_4[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.real(np.fft.rfft(yd3_5[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.real(np.fft.rfft(yd3_6[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.real(np.fft.rfft(yd3_7[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.real(np.fft.rfft(yd3_8[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.real(np.fft.rfft(yd3_9[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.real(np.fft.rfft(yd3_1[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.real(np.fft.rfft(yd3_2[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1]]))
                    
                    all_datas_3.append(np.array([(np.real(np.fft.rfft(yd2_1[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.real(np.fft.rfft(yd2_2[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.real(np.fft.rfft(yd2_3[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.real(np.fft.rfft(yd2_4[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.real(np.fft.rfft(yd2_5[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.real(np.fft.rfft(yd2_6[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.real(np.fft.rfft(yd2_7[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.real(np.fft.rfft(yd2_8[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.real(np.fft.rfft(yd2_9[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.real(np.fft.rfft(yd2_1[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                                 (np.real(np.fft.rfft(yd2_2[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1]]))
                    
                    all_labels.append(target_index)
    
    all_datas_1 = np.expand_dims(np.array(all_datas_1),axis=-1)
    all_datas_2 = np.expand_dims(np.array(all_datas_2),axis=-1)
    all_datas_3 = np.expand_dims(np.array(all_datas_3),axis=-1)
    all_labels = np.array(all_labels)
    
    # print('Test trailer is ',test_trainer)
    rawdata = loadmat('./database/' + test_trainer + '.mat') #1号被试
    rawdata = rawdata['data']
    
    for j in range(6): #6个block
    
        block_index = j
        
        for k in range(40): #40个trail
            
            target_index = k
           
            O1_data = rawdata[60,160:1410,target_index,block_index] #取中间5s,去除140ms latency
            Oz_data = rawdata[61,160:1410,target_index,block_index]
            O2_data = rawdata[62,160:1410,target_index,block_index]
            PO3_data = rawdata[54,160:1410,target_index,block_index]
            POz_data = rawdata[55,160:1410,target_index,block_index]
            PO4_data = rawdata[56,160:1410,target_index,block_index]
            PO5_data = rawdata[53,160:1410,target_index,block_index]
            Pz_data = rawdata[47,160:1410,target_index,block_index]
            PO6_data = rawdata[57,160:1410,target_index,block_index]
            
            yd4_1,yd3_1,yd2_1 = band_filter_process(O1_data)
            yd4_2,yd3_2,yd2_2 = band_filter_process(Oz_data)
            yd4_3,yd3_3,yd2_3 = band_filter_process(O2_data)
            yd4_4,yd3_4,yd2_4 = band_filter_process(PO3_data)
            yd4_5,yd3_5,yd2_5 = band_filter_process(POz_data)
            yd4_6,yd3_6,yd2_6 = band_filter_process(PO4_data)
            yd4_7,yd3_7,yd2_7 = band_filter_process(PO5_data)
            yd4_8,yd3_8,yd2_8 = band_filter_process(Pz_data)
            yd4_9,yd3_9,yd2_9 = band_filter_process(PO6_data)
            
            for m in range(fft_scale):
                
                test_datas_1.append(np.array([(np.real(np.fft.rfft(yd4_1[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.real(np.fft.rfft(yd4_2[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.real(np.fft.rfft(yd4_3[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.real(np.fft.rfft(yd4_4[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.real(np.fft.rfft(yd4_5[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.real(np.fft.rfft(yd4_6[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.real(np.fft.rfft(yd4_7[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.real(np.fft.rfft(yd4_8[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.real(np.fft.rfft(yd4_9[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.real(np.fft.rfft(yd4_1[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.real(np.fft.rfft(yd4_2[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1]]))
                
                test_datas_2.append(np.array([(npreals(np.fft.rfft(yd3_1[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.real(np.fft.rfft(yd3_2[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.real(np.fft.rfft(yd3_3[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.real(np.fft.rfft(yd3_4[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.real(np.fft.rfft(yd3_5[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.real(np.fft.rfft(yd3_6[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.real(np.fft.rfft(yd3_7[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.real(np.fft.rfft(yd3_8[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.real(np.fft.rfft(yd3_9[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.real(np.fft.rfft(yd3_1[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.real(np.fft.rfft(yd3_2[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1]]))
                
                test_datas_3.append(np.array([(np.real(np.fft.rfft(yd2_1[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.real(np.fft.rfft(yd2_2[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.real(np.fft.rfft(yd2_3[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.real(np.fft.rfft(yd2_4[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.real(np.fft.rfft(yd2_5[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.real(np.fft.rfft(yd2_6[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.real(np.fft.rfft(yd2_7[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.real(np.fft.rfft(yd2_8[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.real(np.fft.rfft(yd2_9[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.real(np.fft.rfft(yd2_1[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1],
                                             (np.real(np.fft.rfft(yd2_2[m*window_length:(m+1)*window_length],1250))/window_length)[round(5/0.2):round(55/0.2) + 1]]))
                
                test_labels.append(target_index)
    
    test_datas_1 = np.expand_dims(np.array(test_datas_1),axis=-1)
    test_datas_2 = np.expand_dims(np.array(test_datas_2),axis=-1)
    test_datas_3 = np.expand_dims(np.array(test_datas_3),axis=-1)
    test_labels = np.array(test_labels)

    return all_datas_1,all_datas_2,all_datas_3,all_labels,test_datas_1,test_datas_2,test_datas_3,test_labels

def data_batch_FFT_one_tester_complex_UI(test_trainer,time_length):
    # print('Preparing dataset.....')

    all_trailers = ['S1', 'S2', 'S3', 'S4', 'S5',
                    'S6', 'S7', 'S8', 'S9', 'S10',
                    'S11','S12','S13','S14','S15',
                    'S16','S17','S18','S19','S20',
                    'S21','S22','S23','S24','S25',
                    'S26','S27','S28','S29','S30',
                    'S31','S32','S33','S34','S35']

    simpling_rate = 250
    window_length = int(250*time_length) #窗口长度1s
    data_length = 1250 #5s
    fft_scale = int(data_length/window_length) #5s则scale=5

    all_datas_1 = []
    all_datas_2 = []
    all_datas_3 = []
    all_labels = []
    
    test_datas_1 = []
    test_datas_2 = []
    test_datas_3 = []
    test_labels = []
    
    test_trainer_position = all_trailers.index(test_trainer)
    
    all_trailers.pop(test_trainer_position) #删除test_trainer
    trailer = all_trailers
    # print('Train trailers are ',trailer)
    
    for i in range(len(trailer)):
        # print(trailer[i])
        rawdata = loadmat('./database/' + trailer[i] + '.mat') #1号被试
        rawdata = rawdata['data']
        
        for j in range(6): #6个block
        
            block_index = j
            
            for k in range(40): #40个trail
                
                target_index = k
                O1_data = rawdata[60,160:1410,target_index,block_index] #取中间5s,去除140ms latency
                Oz_data = rawdata[61,160:1410,target_index,block_index]
                O2_data = rawdata[62,160:1410,target_index,block_index]
                PO3_data = rawdata[54,160:1410,target_index,block_index]
                POz_data = rawdata[55,160:1410,target_index,block_index]
                PO4_data = rawdata[56,160:1410,target_index,block_index]
                PO5_data = rawdata[53,160:1410,target_index,block_index]
                Pz_data = rawdata[47,160:1410,target_index,block_index]
                PO6_data = rawdata[57,160:1410,target_index,block_index]
                
                yd4_1,yd3_1,yd2_1 = band_filter_process(O1_data)
                yd4_2,yd3_2,yd2_2 = band_filter_process(Oz_data)
                yd4_3,yd3_3,yd2_3 = band_filter_process(O2_data)
                yd4_4,yd3_4,yd2_4 = band_filter_process(PO3_data)
                yd4_5,yd3_5,yd2_5 = band_filter_process(POz_data)
                yd4_6,yd3_6,yd2_6 = band_filter_process(PO4_data)
                yd4_7,yd3_7,yd2_7 = band_filter_process(PO5_data)
                yd4_8,yd3_8,yd2_8 = band_filter_process(Pz_data)
                yd4_9,yd3_9,yd2_9 = band_filter_process(PO6_data)
                
                for m in range(fft_scale):
                    
                    yd4_1_fft = np.fft.rfft(yd4_1[m*window_length:(m+1)*window_length],1250)/window_length
                    yd4_2_fft = np.fft.rfft(yd4_2[m*window_length:(m+1)*window_length],1250)/window_length
                    yd4_3_fft = np.fft.rfft(yd4_3[m*window_length:(m+1)*window_length],1250)/window_length
                    yd4_4_fft = np.fft.rfft(yd4_4[m*window_length:(m+1)*window_length],1250)/window_length
                    yd4_5_fft = np.fft.rfft(yd4_5[m*window_length:(m+1)*window_length],1250)/window_length
                    yd4_6_fft = np.fft.rfft(yd4_6[m*window_length:(m+1)*window_length],1250)/window_length
                    yd4_7_fft = np.fft.rfft(yd4_7[m*window_length:(m+1)*window_length],1250)/window_length
                    yd4_8_fft = np.fft.rfft(yd4_8[m*window_length:(m+1)*window_length],1250)/window_length
                    yd4_9_fft = np.fft.rfft(yd4_9[m*window_length:(m+1)*window_length],1250)/window_length
                    
                    yd3_1_fft = np.fft.rfft(yd3_1[m*window_length:(m+1)*window_length],1250)/window_length
                    yd3_2_fft = np.fft.rfft(yd3_2[m*window_length:(m+1)*window_length],1250)/window_length
                    yd3_3_fft = np.fft.rfft(yd3_3[m*window_length:(m+1)*window_length],1250)/window_length
                    yd3_4_fft = np.fft.rfft(yd3_4[m*window_length:(m+1)*window_length],1250)/window_length
                    yd3_5_fft = np.fft.rfft(yd3_5[m*window_length:(m+1)*window_length],1250)/window_length
                    yd3_6_fft = np.fft.rfft(yd3_6[m*window_length:(m+1)*window_length],1250)/window_length
                    yd3_7_fft = np.fft.rfft(yd3_7[m*window_length:(m+1)*window_length],1250)/window_length
                    yd3_8_fft = np.fft.rfft(yd3_8[m*window_length:(m+1)*window_length],1250)/window_length
                    yd3_9_fft = np.fft.rfft(yd3_9[m*window_length:(m+1)*window_length],1250)/window_length
                    
                    yd2_1_fft = np.fft.rfft(yd2_1[m*window_length:(m+1)*window_length],1250)/window_length
                    yd2_2_fft = np.fft.rfft(yd2_2[m*window_length:(m+1)*window_length],1250)/window_length
                    yd2_3_fft = np.fft.rfft(yd2_3[m*window_length:(m+1)*window_length],1250)/window_length
                    yd2_4_fft = np.fft.rfft(yd2_4[m*window_length:(m+1)*window_length],1250)/window_length
                    yd2_5_fft = np.fft.rfft(yd2_5[m*window_length:(m+1)*window_length],1250)/window_length
                    yd2_6_fft = np.fft.rfft(yd2_6[m*window_length:(m+1)*window_length],1250)/window_length
                    yd2_7_fft = np.fft.rfft(yd2_7[m*window_length:(m+1)*window_length],1250)/window_length
                    yd2_8_fft = np.fft.rfft(yd2_8[m*window_length:(m+1)*window_length],1250)/window_length
                    yd2_9_fft = np.fft.rfft(yd2_9[m*window_length:(m+1)*window_length],1250)/window_length
                    
                    
                    all_datas_1.append(np.array([np.concatenate((np.real(yd4_1_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd4_1_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                                 np.concatenate((np.real(yd4_2_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd4_2_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                                 np.concatenate((np.real(yd4_3_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd4_3_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                                 np.concatenate((np.real(yd4_4_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd4_4_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                                 np.concatenate((np.real(yd4_5_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd4_5_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                                 np.concatenate((np.real(yd4_6_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd4_6_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                                 np.concatenate((np.real(yd4_7_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd4_7_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                                 np.concatenate((np.real(yd4_8_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd4_8_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                                 np.concatenate((np.real(yd4_9_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd4_9_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                                 np.concatenate((np.real(yd4_1_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd4_1_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                                 np.concatenate((np.real(yd4_2_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd4_2_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0)]))
                    
                    all_datas_2.append(np.array([np.concatenate((np.real(yd3_1_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd3_1_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                                 np.concatenate((np.real(yd3_2_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd3_2_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                                 np.concatenate((np.real(yd3_3_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd3_3_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                                 np.concatenate((np.real(yd3_4_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd3_4_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                                 np.concatenate((np.real(yd3_5_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd3_5_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                                 np.concatenate((np.real(yd3_6_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd3_6_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                                 np.concatenate((np.real(yd3_7_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd3_7_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                                 np.concatenate((np.real(yd3_8_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd3_8_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                                 np.concatenate((np.real(yd3_9_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd3_9_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                                 np.concatenate((np.real(yd3_1_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd3_1_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                                 np.concatenate((np.real(yd3_2_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd3_2_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0)]))
                    
                    all_datas_3.append(np.array([np.concatenate((np.real(yd2_1_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd2_1_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                                 np.concatenate((np.real(yd2_2_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd2_2_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                                 np.concatenate((np.real(yd2_3_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd2_3_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                                 np.concatenate((np.real(yd2_4_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd2_4_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                                 np.concatenate((np.real(yd2_5_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd2_5_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                                 np.concatenate((np.real(yd2_6_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd2_6_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                                 np.concatenate((np.real(yd2_7_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd2_7_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                                 np.concatenate((np.real(yd2_8_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd2_8_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                                 np.concatenate((np.real(yd2_9_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd2_9_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                                 np.concatenate((np.real(yd2_1_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd2_1_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                                 np.concatenate((np.real(yd2_2_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd2_2_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0)]))
                    
                    all_labels.append(target_index)
    
    all_datas_1 = np.expand_dims(np.array(all_datas_1),axis=-1)
    all_datas_2 = np.expand_dims(np.array(all_datas_2),axis=-1)
    all_datas_3 = np.expand_dims(np.array(all_datas_3),axis=-1)
    all_labels = np.array(all_labels)
    
    # print('Test trailer is ',test_trainer)
    rawdata = loadmat('./database/' + test_trainer + '.mat') #1号被试
    rawdata = rawdata['data']
    
    for j in range(6): #6个block
    
        block_index = j
        
        for k in range(40): #40个trail
            
            target_index = k
           
            O1_data = rawdata[60,160:1410,target_index,block_index] #取中间5s,去除140ms latency
            Oz_data = rawdata[61,160:1410,target_index,block_index]
            O2_data = rawdata[62,160:1410,target_index,block_index]
            PO3_data = rawdata[54,160:1410,target_index,block_index]
            POz_data = rawdata[55,160:1410,target_index,block_index]
            PO4_data = rawdata[56,160:1410,target_index,block_index]
            PO5_data = rawdata[53,160:1410,target_index,block_index]
            Pz_data = rawdata[47,160:1410,target_index,block_index]
            PO6_data = rawdata[57,160:1410,target_index,block_index]
            
            yd4_1,yd3_1,yd2_1 = band_filter_process(O1_data)
            yd4_2,yd3_2,yd2_2 = band_filter_process(Oz_data)
            yd4_3,yd3_3,yd2_3 = band_filter_process(O2_data)
            yd4_4,yd3_4,yd2_4 = band_filter_process(PO3_data)
            yd4_5,yd3_5,yd2_5 = band_filter_process(POz_data)
            yd4_6,yd3_6,yd2_6 = band_filter_process(PO4_data)
            yd4_7,yd3_7,yd2_7 = band_filter_process(PO5_data)
            yd4_8,yd3_8,yd2_8 = band_filter_process(Pz_data)
            yd4_9,yd3_9,yd2_9 = band_filter_process(PO6_data)
            
            for m in range(fft_scale):
                
                yd4_1_fft = np.fft.rfft(yd4_1[m*window_length:(m+1)*window_length],1250)/window_length
                yd4_2_fft = np.fft.rfft(yd4_2[m*window_length:(m+1)*window_length],1250)/window_length
                yd4_3_fft = np.fft.rfft(yd4_3[m*window_length:(m+1)*window_length],1250)/window_length
                yd4_4_fft = np.fft.rfft(yd4_4[m*window_length:(m+1)*window_length],1250)/window_length
                yd4_5_fft = np.fft.rfft(yd4_5[m*window_length:(m+1)*window_length],1250)/window_length
                yd4_6_fft = np.fft.rfft(yd4_6[m*window_length:(m+1)*window_length],1250)/window_length
                yd4_7_fft = np.fft.rfft(yd4_7[m*window_length:(m+1)*window_length],1250)/window_length
                yd4_8_fft = np.fft.rfft(yd4_8[m*window_length:(m+1)*window_length],1250)/window_length
                yd4_9_fft = np.fft.rfft(yd4_9[m*window_length:(m+1)*window_length],1250)/window_length
                
                yd3_1_fft = np.fft.rfft(yd3_1[m*window_length:(m+1)*window_length],1250)/window_length
                yd3_2_fft = np.fft.rfft(yd3_2[m*window_length:(m+1)*window_length],1250)/window_length
                yd3_3_fft = np.fft.rfft(yd3_3[m*window_length:(m+1)*window_length],1250)/window_length
                yd3_4_fft = np.fft.rfft(yd3_4[m*window_length:(m+1)*window_length],1250)/window_length
                yd3_5_fft = np.fft.rfft(yd3_5[m*window_length:(m+1)*window_length],1250)/window_length
                yd3_6_fft = np.fft.rfft(yd3_6[m*window_length:(m+1)*window_length],1250)/window_length
                yd3_7_fft = np.fft.rfft(yd3_7[m*window_length:(m+1)*window_length],1250)/window_length
                yd3_8_fft = np.fft.rfft(yd3_8[m*window_length:(m+1)*window_length],1250)/window_length
                yd3_9_fft = np.fft.rfft(yd3_9[m*window_length:(m+1)*window_length],1250)/window_length
                
                yd2_1_fft = np.fft.rfft(yd2_1[m*window_length:(m+1)*window_length],1250)/window_length
                yd2_2_fft = np.fft.rfft(yd2_2[m*window_length:(m+1)*window_length],1250)/window_length
                yd2_3_fft = np.fft.rfft(yd2_3[m*window_length:(m+1)*window_length],1250)/window_length
                yd2_4_fft = np.fft.rfft(yd2_4[m*window_length:(m+1)*window_length],1250)/window_length
                yd2_5_fft = np.fft.rfft(yd2_5[m*window_length:(m+1)*window_length],1250)/window_length
                yd2_6_fft = np.fft.rfft(yd2_6[m*window_length:(m+1)*window_length],1250)/window_length
                yd2_7_fft = np.fft.rfft(yd2_7[m*window_length:(m+1)*window_length],1250)/window_length
                yd2_8_fft = np.fft.rfft(yd2_8[m*window_length:(m+1)*window_length],1250)/window_length
                yd2_9_fft = np.fft.rfft(yd2_9[m*window_length:(m+1)*window_length],1250)/window_length
                
                
                test_datas_1.append(np.array([np.concatenate((np.real(yd4_1_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd4_1_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                             np.concatenate((np.real(yd4_2_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd4_2_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                             np.concatenate((np.real(yd4_3_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd4_3_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                             np.concatenate((np.real(yd4_4_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd4_4_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                             np.concatenate((np.real(yd4_5_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd4_5_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                             np.concatenate((np.real(yd4_6_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd4_6_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                             np.concatenate((np.real(yd4_7_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd4_7_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                             np.concatenate((np.real(yd4_8_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd4_8_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                             np.concatenate((np.real(yd4_9_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd4_9_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                             np.concatenate((np.real(yd4_1_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd4_1_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                             np.concatenate((np.real(yd4_2_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd4_2_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0)]))
                
                test_datas_2.append(np.array([np.concatenate((np.real(yd3_1_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd3_1_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                             np.concatenate((np.real(yd3_2_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd3_2_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                             np.concatenate((np.real(yd3_3_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd3_3_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                             np.concatenate((np.real(yd3_4_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd3_4_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                             np.concatenate((np.real(yd3_5_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd3_5_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                             np.concatenate((np.real(yd3_6_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd3_6_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                             np.concatenate((np.real(yd3_7_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd3_7_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                             np.concatenate((np.real(yd3_8_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd3_8_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                             np.concatenate((np.real(yd3_9_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd3_9_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                             np.concatenate((np.real(yd3_1_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd3_1_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                             np.concatenate((np.real(yd3_2_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd3_2_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0)]))
                
                test_datas_3.append(np.array([np.concatenate((np.real(yd2_1_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd2_1_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                             np.concatenate((np.real(yd2_2_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd2_2_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                             np.concatenate((np.real(yd2_3_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd2_3_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                             np.concatenate((np.real(yd2_4_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd2_4_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                             np.concatenate((np.real(yd2_5_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd2_5_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                             np.concatenate((np.real(yd2_6_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd2_6_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                             np.concatenate((np.real(yd2_7_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd2_7_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                             np.concatenate((np.real(yd2_8_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd2_8_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                             np.concatenate((np.real(yd2_9_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd2_9_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                             np.concatenate((np.real(yd2_1_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd2_1_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0),
                                             np.concatenate((np.real(yd2_2_fft)[round(5/0.2):round(55/0.2) + 1],np.imag(yd2_2_fft)[round(5/0.2):round(55/0.2) + 1]),axis=0)]))
                
                
                
                test_labels.append(target_index)
    
    test_datas_1 = np.expand_dims(np.array(test_datas_1),axis=-1)
    test_datas_2 = np.expand_dims(np.array(test_datas_2),axis=-1)
    test_datas_3 = np.expand_dims(np.array(test_datas_3),axis=-1)
    test_labels = np.array(test_labels)

    return all_datas_1,all_datas_2,all_datas_3,all_labels,test_datas_1,test_datas_2,test_datas_3,test_labels

'''
Main
'''

if __name__ == '__main__':
    
    # data_batch_FFT(True)
    data_batch_FFT_one_tester('S1')


















