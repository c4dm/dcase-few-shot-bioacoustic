import librosa
import librosa.display
import numpy as np
import sklearn
import skimage
import os
import warnings
import csv
import pickle
from skimage.feature import match_template
from scipy.signal import find_peaks

folder_path = './Development_Set/Validation_Set/'
folders = os.listdir(folder_path)

to_write = []
for folder in folders:
    if folder == 'HV':
        sr = 6000
    elif folder == 'PB':
        sr = 44100
    
    files = os.listdir(folder_path+folder)
    if 1:
        for file in files:
            #print(file[-4:], file[:-4])
            #input()
            if file[-4:] == '.wav':
                audio = file
                annotation = file[:-4]+'.csv'

                waveform, sr = librosa.load(folder_path+folder+'/'+audio, sr = sr)
                stft = np.abs(librosa.stft(waveform, n_fft=1024, hop_length=512, win_length=1024, window='hann', pad_mode='reflect'))
                #stft_db = librosa.amplitude_to_db(stft, ref=np.max, amin=1e-10, top_db=80.0)
                stft_median = np.median(stft, axis=-1, keepdims=True)
                stft_time_median = np.median(stft, axis=0, keepdims=True)
                norm_stft = stft - stft_median
               # norm_stft = norm_stft - stft_time_median

                events = []
                gt = []
                with open(folder_path+folder+'/'+annotation) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    for row in csv_reader:
                        if row[-1] == 'POS' and len(events) < 5:
                            events.append(row)
                        if row[-1] == 'POS' and len(events) >= 5:
                            gt.append(row)

                endtime = float(events[-1][-2])

                to_predict = norm_stft[:, int(np.ceil(endtime*sr/512 + 1)):]
                print(to_predict.shape)

                aligned_gt = np.zeros((to_predict.shape[1],1))
                for event in gt:
                    starttime_gt = float(event[1])
                    endtime_gt = float(event[2])
                    aligned_gt[int(np.floor(starttime_gt*sr/512 + 1))-int(np.ceil(endtime*sr/512 + 1)):int(np.ceil(endtime_gt*sr/512 + 1))-int(np.ceil(endtime*sr/512 + 1))] = 1

                result = []
                for event in events:
                    starttime = float(event[1])
                    endtime = float(event[2])
                    event_stft = norm_stft[2:-2,int(np.floor(starttime*sr/512 + 1)):int(np.ceil(endtime*sr/512 + 1))]

                    result.append(match_template(to_predict, event_stft))
                    #ij = np.unravel_index(np.argmax(result), result.shape)
                    #x, y = ij[::-1]
                mr = []
                for i in range(len(events)):
                    event = events[i]
                    starttime = float(event[1])
                    endtime = float(event[2])
                    event_stft = norm_stft[2:-2,int(np.floor(starttime*sr/512 + 1)):int(np.ceil(endtime*sr/512 + 1))]

                    r=[]
                    for j in range(len(events)):
                        if j != i:
                            inner_event = events[j]
                            inner_starttime = float(inner_event[1])
                            inner_endtime = float(inner_event[2])
                            inner_event_stft = norm_stft[2:-2,int(np.floor(inner_starttime*sr/512 + 1)):int(np.ceil(inner_endtime*sr/512 + 1))]

                            if inner_event_stft.shape[1] >= event_stft.shape[1]:
                                r.append(np.max(match_template(inner_event_stft, event_stft)))
                    if r:
                        mr.append(np.median(r))
                #print('threshold: ', np.median(mr))
                threshold = np.median(mr)

                R = []
                for i in range(len(result)):
                    rmax = np.zeros((result[i].shape[1], ))
                    peaks, _ = find_peaks(np.max(result[i], axis=0), height=threshold)
                    rmax[peaks] = 1
                    R.append(rmax)


                for i in range(len(R)):
                    #print('original len of R: ', len(R[i]))
                    event_len = int(np.ceil(np.floor(float(events[i][2])*sr/512 + 1))) - int(np.floor(float(events[i][1])*sr/512))
                    lpad = int(np.floor(event_len/2))
                    rpad = int(np.ceil(event_len/2))
                    indeces = np.where(R[i]==1)[0]
                    for index in indeces:
                        R[i][int(index)-lpad:int(index)+rpad] = 1
                    R[i] = np.pad(R[i], (lpad, rpad))
                    #print('fixed len of R: ', len(R[i]))
                    R[0] += R[i]
                finalR = R[0]
                finalR[np.where(finalR>0)] = 1
                
                startind = np.where(finalR[:-1] - finalR[1:] == -1)[0]
                endind = np.where(finalR[:-1] - finalR[1:] == 1)[0]
                for i in range(len(startind)):
                    to_write.append([audio, float(events[-1][2])+startind[i]*512/sr, float(events[-1][2])+endind[i]*512/sr])
                print(audio, ' threshold: ', threshold, ' predictions: ', len(startind))
                
pickle.dump(to_write, open('to_write.pckl', 'wb'))              
