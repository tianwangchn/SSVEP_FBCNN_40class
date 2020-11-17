from scipy.io import loadmat
import pandas as pd
import numpy as np
from scipy import signal

'''
[64, 1500, 40, 6]
电极位置、采样点数(6s,刺激前500ms,刺激后5.5s即0.5s target cue + 5s target stimuli + 0.5s screen blank)、频率目标trail(40个)、block个数

刺激频率为8~15.8Hz，间隔0.2Hz，相位由0、0.5π、π和1.5π构成
采样率为1000Hz,降采样到250Hz,通带0.15hz到200hz、ground位于Fz和FPz之间的中间、参考电极Cz
分6个block，每个block包含40个trial

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

def band_filter(input):
    
    b_HPF,a_HPF = signal.butter(4, 0.048, btype='highpass') #4阶，6Hz ， 6/(sampling_rate/2) 和iirfilter效果一样
    b_LPF,a_LPF = signal.butter(4, 0.512, btype='lowpass') #4阶，64Hz ，64/(sampling_rate/2) 和iirfilter效果一样
    
    data_HF = signal.filtfilt(b_HPF, a_HPF, input) #零相位滤波，滤两次，反向多滤了一次
    data_LF = signal.filtfilt(b_LPF, a_LPF, data_HF)
    
    return data_LF

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

    all_datas = []
    all_labels = []
    test_datas = []
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
                
                O1_data_filtered = band_filter(O1_data)
                Oz_data_filtered = band_filter(Oz_data)
                O2_data_filtered = band_filter(O2_data)
                PO3_data_filtered = band_filter(PO3_data)
                POz_data_filtered = band_filter(POz_data)
                PO4_data_filtered = band_filter(PO4_data)
                PO5_data_filtered = band_filter(PO5_data)
                Pz_data_filtered = band_filter(Pz_data)
                PO6_data_filtered = band_filter(PO6_data)
                
                for m in range(fft_scale):
                    
                    O1_FFT_list  = np.abs(np.fft.rfft( O1_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                    Oz_FFT_list  = np.abs(np.fft.rfft( Oz_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                    O2_FFT_list  = np.abs(np.fft.rfft( O2_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                    PO3_FFT_list = np.abs(np.fft.rfft(PO3_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                    POz_FFT_list = np.abs(np.fft.rfft(POz_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                    PO4_FFT_list = np.abs(np.fft.rfft(PO4_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                    PO5_FFT_list = np.abs(np.fft.rfft(PO5_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                    Pz_FFT_list  = np.abs(np.fft.rfft( Pz_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                    PO6_FFT_list = np.abs(np.fft.rfft(PO6_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                    
                    O1_FFT_list  = ( O1_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                    Oz_FFT_list  = ( Oz_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                    O2_FFT_list  = ( O2_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                    PO3_FFT_list = (PO3_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                    POz_FFT_list = (POz_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                    PO4_FFT_list = (PO4_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                    PO5_FFT_list = (PO5_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                    Pz_FFT_list  = ( Pz_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                    PO6_FFT_list = (PO6_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                    
                    all_datas.append(np.array([O1_FFT_list,Oz_FFT_list,O2_FFT_list,PO3_FFT_list,POz_FFT_list,PO4_FFT_list,PO5_FFT_list,Pz_FFT_list,PO6_FFT_list]))
                    all_labels.append(target_index)
    
    all_datas = np.expand_dims(np.array(all_datas),axis=-1)
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
            
            O1_data_filtered = band_filter(O1_data)
            Oz_data_filtered = band_filter(Oz_data)
            O2_data_filtered = band_filter(O2_data)
            PO3_data_filtered = band_filter(PO3_data)
            POz_data_filtered = band_filter(POz_data)
            PO4_data_filtered = band_filter(PO4_data)
            PO5_data_filtered = band_filter(PO5_data)
            Pz_data_filtered = band_filter(Pz_data)
            PO6_data_filtered = band_filter(PO6_data)
            
            for m in range(fft_scale):
                
                O1_FFT_list  = np.abs(np.fft.rfft( O1_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                Oz_FFT_list  = np.abs(np.fft.rfft( Oz_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                O2_FFT_list  = np.abs(np.fft.rfft( O2_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                PO3_FFT_list = np.abs(np.fft.rfft(PO3_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                POz_FFT_list = np.abs(np.fft.rfft(POz_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                PO4_FFT_list = np.abs(np.fft.rfft(PO4_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                PO5_FFT_list = np.abs(np.fft.rfft(PO5_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                Pz_FFT_list  = np.abs(np.fft.rfft( Pz_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                PO6_FFT_list = np.abs(np.fft.rfft(PO6_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                
                O1_FFT_list  = ( O1_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                Oz_FFT_list  = ( Oz_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                O2_FFT_list  = ( O2_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                PO3_FFT_list = (PO3_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                POz_FFT_list = (POz_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                PO4_FFT_list = (PO4_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                PO5_FFT_list = (PO5_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                Pz_FFT_list  = ( Pz_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                PO6_FFT_list = (PO6_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                
                test_datas.append(np.array([O1_FFT_list,Oz_FFT_list,O2_FFT_list,PO3_FFT_list,POz_FFT_list,PO4_FFT_list,PO5_FFT_list,Pz_FFT_list,PO6_FFT_list]))
                test_labels.append(target_index)
    
    test_datas = np.expand_dims(np.array(test_datas),axis=-1)
    test_labels = np.array(test_labels)

    return all_datas,all_labels,test_datas,test_labels

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

    all_datas = []
    all_labels = []
    test_datas = []
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
                
                O1_data_filtered = band_filter(O1_data)
                Oz_data_filtered = band_filter(Oz_data)
                O2_data_filtered = band_filter(O2_data)
                PO3_data_filtered = band_filter(PO3_data)
                POz_data_filtered = band_filter(POz_data)
                PO4_data_filtered = band_filter(PO4_data)
                PO5_data_filtered = band_filter(PO5_data)
                Pz_data_filtered = band_filter(Pz_data)
                PO6_data_filtered = band_filter(PO6_data)
                
                for m in range(fft_scale):
                    
                    O1_FFT_list  = np.real(np.fft.rfft( O1_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                    Oz_FFT_list  = np.real(np.fft.rfft( Oz_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                    O2_FFT_list  = np.real(np.fft.rfft( O2_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                    PO3_FFT_list = np.real(np.fft.rfft(PO3_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                    POz_FFT_list = np.real(np.fft.rfft(POz_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                    PO4_FFT_list = np.real(np.fft.rfft(PO4_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                    PO5_FFT_list = np.real(np.fft.rfft(PO5_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                    Pz_FFT_list  = np.real(np.fft.rfft( Pz_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                    PO6_FFT_list = np.real(np.fft.rfft(PO6_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                    
                    O1_FFT_list  = ( O1_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                    Oz_FFT_list  = ( Oz_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                    O2_FFT_list  = ( O2_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                    PO3_FFT_list = (PO3_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                    POz_FFT_list = (POz_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                    PO4_FFT_list = (PO4_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                    PO5_FFT_list = (PO5_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                    Pz_FFT_list  = ( Pz_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                    PO6_FFT_list = (PO6_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                    
                    all_datas.append(np.array([O1_FFT_list,Oz_FFT_list,O2_FFT_list,PO3_FFT_list,POz_FFT_list,PO4_FFT_list,PO5_FFT_list,Pz_FFT_list,PO6_FFT_list]))
                    all_labels.append(target_index)
    
    all_datas = np.expand_dims(np.array(all_datas),axis=-1)
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
            
            O1_data_filtered = band_filter(O1_data)
            Oz_data_filtered = band_filter(Oz_data)
            O2_data_filtered = band_filter(O2_data)
            PO3_data_filtered = band_filter(PO3_data)
            POz_data_filtered = band_filter(POz_data)
            PO4_data_filtered = band_filter(PO4_data)
            PO5_data_filtered = band_filter(PO5_data)
            Pz_data_filtered = band_filter(Pz_data)
            PO6_data_filtered = band_filter(PO6_data)
            
            for m in range(fft_scale):
                
                O1_FFT_list  = np.real(np.fft.rfft( O1_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                Oz_FFT_list  = np.real(np.fft.rfft( Oz_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                O2_FFT_list  = np.real(np.fft.rfft( O2_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                PO3_FFT_list = np.real(np.fft.rfft(PO3_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                POz_FFT_list = np.real(np.fft.rfft(POz_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                PO4_FFT_list = np.real(np.fft.rfft(PO4_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                PO5_FFT_list = np.real(np.fft.rfft(PO5_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                Pz_FFT_list  = np.real(np.fft.rfft( Pz_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                PO6_FFT_list = np.real(np.fft.rfft(PO6_data_filtered[m*window_length:(m+1)*window_length],1250))/window_length #DFT (sampling_rate/resolution) = 1250
                
                O1_FFT_list  = ( O1_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                Oz_FFT_list  = ( Oz_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                O2_FFT_list  = ( O2_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                PO3_FFT_list = (PO3_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                POz_FFT_list = (POz_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                PO4_FFT_list = (PO4_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                PO5_FFT_list = (PO5_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                Pz_FFT_list  = ( Pz_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                PO6_FFT_list = (PO6_FFT_list)[round(5/0.2):round(55/0.2) + 1]
                
                test_datas.append(np.array([O1_FFT_list,Oz_FFT_list,O2_FFT_list,PO3_FFT_list,POz_FFT_list,PO4_FFT_list,PO5_FFT_list,Pz_FFT_list,PO6_FFT_list]))
                test_labels.append(target_index)
    
    test_datas = np.expand_dims(np.array(test_datas),axis=-1)
    test_labels = np.array(test_labels)

    return all_datas,all_labels,test_datas,test_labels

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

    all_datas = []
    all_labels = []
    test_datas = []
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
                
                O1_data_filtered = band_filter(O1_data)
                Oz_data_filtered = band_filter(Oz_data)
                O2_data_filtered = band_filter(O2_data)
                PO3_data_filtered = band_filter(PO3_data)
                POz_data_filtered = band_filter(POz_data)
                PO4_data_filtered = band_filter(PO4_data)
                PO5_data_filtered = band_filter(PO5_data)
                Pz_data_filtered = band_filter(Pz_data)
                PO6_data_filtered = band_filter(PO6_data)
                
                for m in range(fft_scale):
                    
                    O1_FFT_list  = np.fft.rfft( O1_data_filtered[m*window_length:(m+1)*window_length],1250)/window_length #DFT (sampling_rate/resolution) = 1250
                    Oz_FFT_list  = np.fft.rfft( Oz_data_filtered[m*window_length:(m+1)*window_length],1250)/window_length #DFT (sampling_rate/resolution) = 1250
                    O2_FFT_list  = np.fft.rfft( O2_data_filtered[m*window_length:(m+1)*window_length],1250)/window_length #DFT (sampling_rate/resolution) = 1250
                    PO3_FFT_list = np.fft.rfft(PO3_data_filtered[m*window_length:(m+1)*window_length],1250)/window_length #DFT (sampling_rate/resolution) = 1250
                    POz_FFT_list = np.fft.rfft(POz_data_filtered[m*window_length:(m+1)*window_length],1250)/window_length #DFT (sampling_rate/resolution) = 1250
                    PO4_FFT_list = np.fft.rfft(PO4_data_filtered[m*window_length:(m+1)*window_length],1250)/window_length #DFT (sampling_rate/resolution) = 1250
                    PO5_FFT_list = np.fft.rfft(PO5_data_filtered[m*window_length:(m+1)*window_length],1250)/window_length #DFT (sampling_rate/resolution) = 1250
                    Pz_FFT_list  = np.fft.rfft( Pz_data_filtered[m*window_length:(m+1)*window_length],1250)/window_length #DFT (sampling_rate/resolution) = 1250
                    PO6_FFT_list = np.fft.rfft(PO6_data_filtered[m*window_length:(m+1)*window_length],1250)/window_length #DFT (sampling_rate/resolution) = 1250
                    
                    O1_FFT_list  = np.concatenate(((np.real( O1_FFT_list))[round(5/0.2):round(55/0.2) + 1],(np.imag( O1_FFT_list))[round(5/0.2):round(55/0.2) + 1]),axis=0)
                    Oz_FFT_list  = np.concatenate(((np.real( Oz_FFT_list))[round(5/0.2):round(55/0.2) + 1],(np.imag( Oz_FFT_list))[round(5/0.2):round(55/0.2) + 1]),axis=0)
                    O2_FFT_list  = np.concatenate(((np.real( O2_FFT_list))[round(5/0.2):round(55/0.2) + 1],(np.imag( O2_FFT_list))[round(5/0.2):round(55/0.2) + 1]),axis=0)
                    PO3_FFT_list = np.concatenate(((np.real(PO3_FFT_list))[round(5/0.2):round(55/0.2) + 1],(np.imag(PO3_FFT_list))[round(5/0.2):round(55/0.2) + 1]),axis=0)
                    POz_FFT_list = np.concatenate(((np.real(POz_FFT_list))[round(5/0.2):round(55/0.2) + 1],(np.imag(POz_FFT_list))[round(5/0.2):round(55/0.2) + 1]),axis=0)
                    PO4_FFT_list = np.concatenate(((np.real(PO4_FFT_list))[round(5/0.2):round(55/0.2) + 1],(np.imag(PO4_FFT_list))[round(5/0.2):round(55/0.2) + 1]),axis=0)
                    PO5_FFT_list = np.concatenate(((np.real(PO5_FFT_list))[round(5/0.2):round(55/0.2) + 1],(np.imag(PO5_FFT_list))[round(5/0.2):round(55/0.2) + 1]),axis=0)
                    Pz_FFT_list  = np.concatenate(((np.real( Pz_FFT_list))[round(5/0.2):round(55/0.2) + 1],(np.imag( Pz_FFT_list))[round(5/0.2):round(55/0.2) + 1]),axis=0)
                    PO6_FFT_list = np.concatenate(((np.real(PO6_FFT_list))[round(5/0.2):round(55/0.2) + 1],(np.imag(PO6_FFT_list))[round(5/0.2):round(55/0.2) + 1]),axis=0)
                    
                    all_datas.append(np.array([O1_FFT_list,Oz_FFT_list,O2_FFT_list,PO3_FFT_list,POz_FFT_list,PO4_FFT_list,PO5_FFT_list,Pz_FFT_list,PO6_FFT_list]))
                    all_labels.append(target_index)
    
    all_datas = np.expand_dims(np.array(all_datas),axis=-1)
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
            
            O1_data_filtered = band_filter(O1_data)
            Oz_data_filtered = band_filter(Oz_data)
            O2_data_filtered = band_filter(O2_data)
            PO3_data_filtered = band_filter(PO3_data)
            POz_data_filtered = band_filter(POz_data)
            PO4_data_filtered = band_filter(PO4_data)
            PO5_data_filtered = band_filter(PO5_data)
            Pz_data_filtered = band_filter(Pz_data)
            PO6_data_filtered = band_filter(PO6_data)
            
            for m in range(fft_scale):
                
                O1_FFT_list  = np.fft.rfft( O1_data_filtered[m*window_length:(m+1)*window_length],1250)/window_length #DFT (sampling_rate/resolution) = 1250
                Oz_FFT_list  = np.fft.rfft( Oz_data_filtered[m*window_length:(m+1)*window_length],1250)/window_length #DFT (sampling_rate/resolution) = 1250
                O2_FFT_list  = np.fft.rfft( O2_data_filtered[m*window_length:(m+1)*window_length],1250)/window_length #DFT (sampling_rate/resolution) = 1250
                PO3_FFT_list = np.fft.rfft(PO3_data_filtered[m*window_length:(m+1)*window_length],1250)/window_length #DFT (sampling_rate/resolution) = 1250
                POz_FFT_list = np.fft.rfft(POz_data_filtered[m*window_length:(m+1)*window_length],1250)/window_length #DFT (sampling_rate/resolution) = 1250
                PO4_FFT_list = np.fft.rfft(PO4_data_filtered[m*window_length:(m+1)*window_length],1250)/window_length #DFT (sampling_rate/resolution) = 1250
                PO5_FFT_list = np.fft.rfft(PO5_data_filtered[m*window_length:(m+1)*window_length],1250)/window_length #DFT (sampling_rate/resolution) = 1250
                Pz_FFT_list  = np.fft.rfft( Pz_data_filtered[m*window_length:(m+1)*window_length],1250)/window_length #DFT (sampling_rate/resolution) = 1250
                PO6_FFT_list = np.fft.rfft(PO6_data_filtered[m*window_length:(m+1)*window_length],1250)/window_length #DFT (sampling_rate/resolution) = 1250
                
                O1_FFT_list  = np.concatenate(((np.real( O1_FFT_list))[round(5/0.2):round(55/0.2) + 1],(np.imag( O1_FFT_list))[round(5/0.2):round(55/0.2) + 1]),axis=0)
                Oz_FFT_list  = np.concatenate(((np.real( Oz_FFT_list))[round(5/0.2):round(55/0.2) + 1],(np.imag( Oz_FFT_list))[round(5/0.2):round(55/0.2) + 1]),axis=0)
                O2_FFT_list  = np.concatenate(((np.real( O2_FFT_list))[round(5/0.2):round(55/0.2) + 1],(np.imag( O2_FFT_list))[round(5/0.2):round(55/0.2) + 1]),axis=0)
                PO3_FFT_list = np.concatenate(((np.real(PO3_FFT_list))[round(5/0.2):round(55/0.2) + 1],(np.imag(PO3_FFT_list))[round(5/0.2):round(55/0.2) + 1]),axis=0)
                POz_FFT_list = np.concatenate(((np.real(POz_FFT_list))[round(5/0.2):round(55/0.2) + 1],(np.imag(POz_FFT_list))[round(5/0.2):round(55/0.2) + 1]),axis=0)
                PO4_FFT_list = np.concatenate(((np.real(PO4_FFT_list))[round(5/0.2):round(55/0.2) + 1],(np.imag(PO4_FFT_list))[round(5/0.2):round(55/0.2) + 1]),axis=0)
                PO5_FFT_list = np.concatenate(((np.real(PO5_FFT_list))[round(5/0.2):round(55/0.2) + 1],(np.imag(PO5_FFT_list))[round(5/0.2):round(55/0.2) + 1]),axis=0)
                Pz_FFT_list  = np.concatenate(((np.real( Pz_FFT_list))[round(5/0.2):round(55/0.2) + 1],(np.imag( Pz_FFT_list))[round(5/0.2):round(55/0.2) + 1]),axis=0)
                PO6_FFT_list = np.concatenate(((np.real(PO6_FFT_list))[round(5/0.2):round(55/0.2) + 1],(np.imag(PO6_FFT_list))[round(5/0.2):round(55/0.2) + 1]),axis=0)
                
                test_datas.append(np.array([O1_FFT_list,Oz_FFT_list,O2_FFT_list,PO3_FFT_list,POz_FFT_list,PO4_FFT_list,PO5_FFT_list,Pz_FFT_list,PO6_FFT_list]))
                test_labels.append(target_index)
    
    test_datas = np.expand_dims(np.array(test_datas),axis=-1)
    test_labels = np.array(test_labels)

    return all_datas,all_labels,test_datas,test_labels

'''
Main
'''

if __name__ == '__main__':
    
    # data_batch_FFT(True)
    data_batch_FFT_one_tester('S1')


















