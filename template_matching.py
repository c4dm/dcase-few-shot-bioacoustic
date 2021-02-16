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

# Path to valisation set folder with all audio files and annotations
folder_path = './Development_Set/Validation_Set/'

# list of all sub-folders of validation set folder
folders = os.listdir(folder_path)

to_write = []

# set sr based on folder
for folder in folders:
    if folder == 'HV':
        sr = 6000
    elif folder == 'PB':
        sr = 44100
    
    # list of all contents of each sub-folder (audio and annotations)
    files = os.listdir(folder_path+folder)
    for file in files:
        if file[-4:] == '.wav':
            audio = file
            annotation = file[:-4]+'.csv'

            # load audio 
            waveform, sr = librosa.load(folder_path+folder+'/'+audio, sr = sr)
            
            # STFT of audio with subtraction of bin median per bin
            stft = np.abs(librosa.stft(waveform, n_fft=1024, hop_length=512, win_length=1024, window='hann', pad_mode='reflect'))
            stft_median = np.median(stft, axis=-1, keepdims=True)
            norm_stft = stft - stft_median

            # separate the first 5 positive (POS) events to be used as templates
            events = []
            with open(folder_path+folder+'/'+annotation) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if row[-1] == 'POS' and len(events) < 5:
                        events.append(row)
               

            # the end time of the last template event that will be used as startime for the predictions 
            endtime = float(events[-1][-2])
            
            # audiofile to use for prediction - starts after endtime
            to_predict = norm_stft[:, int(np.ceil(endtime*sr/512 + 1)):]

            result = []
            for event in events:
                starttime = float(event[1])
                endtime = float(event[2])
                # template STFT is offseted by 2 bins on the top and bottom compared to the full STFT to allow more variability in the vocalisation position
                event_stft = norm_stft[2:-2,int(np.floor(starttime*sr/512 + 1)):int(np.ceil(endtime*sr/512 + 1))]
                # append the predictions of match_template for each event to the result list
                result.append(match_template(to_predict, event_stft))
            
            # a different prediction threshold is set for each audio file
            # based of the similarity of the 5 templates
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
            # threshold set as the median of similarities between the 5 templates
            threshold = np.median(mr)

            # peak picking on the match_template result vector to get the centre of the predictions
            R = []
            for i in range(len(result)):
                rmax = np.zeros((result[i].shape[1], ))
                peaks, _ = find_peaks(np.max(result[i], axis=0), height=threshold)
                rmax[peaks] = 1
                R.append(rmax)

            # creating segments based on centre of predictions and templates length
            # and then overlapping all 5 predictions into one final vector with binary segments or event predictions
            for i in range(len(R)):
                event_len = int(np.ceil(np.floor(float(events[i][2])*sr/512 + 1))) - int(np.floor(float(events[i][1])*sr/512))
                lpad = int(np.floor(event_len/2))
                rpad = int(np.ceil(event_len/2))
                indeces = np.where(R[i]==1)[0]
                for index in indeces:
                    R[i][int(index)-lpad:int(index)+rpad] = 1
                R[i] = np.pad(R[i], (lpad, rpad))
                R[0] += R[i]
            finalR = R[0]
            finalR[np.where(finalR>0)] = 1
            
            # transforming time frames to seconds in predictions
            startind = np.where(finalR[:-1] - finalR[1:] == -1)[0]
            endind = np.where(finalR[:-1] - finalR[1:] == 1)[0]
            for i in range(len(startind)):
                to_write.append([audio, float(events[-1][2])+startind[i]*512/sr, float(events[-1][2])+endind[i]*512/sr])
            

# creating csv with predictions
with open('baseline_template_val_predictions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(to_write)
