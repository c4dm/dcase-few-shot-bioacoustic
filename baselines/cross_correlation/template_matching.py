import librosa
import numpy as np
import sklearn
import skimage
import os
import csv
import pickle
from skimage.feature import match_template
from scipy.signal import find_peaks


def fewshot_match_template(folder_path='./Development_Set/Validation_Set/', shots=5, output_file='output'):
    """Cross-correlation template matching for the validation set of DCASE 2021 task 5: Few-shot Bioacoustic Event Detection.
    Saves predicted events for class of interest for each audiofile in the validation set, in a scv file.

    Parameters
    ----------
    folder_path: str
        Path to folder with audio *.wav files and corresponding annotation *.csv files.
    shots: int 
        Number of shots available for few-shot template matching.
    output_file: str
        Filename for output csv.
    """

    # list of all sub-folders of validation set folder
    folders = os.listdir(folder_path)
    # csv header
    to_write = [['Audiofilename', 'Starttime', 'Endtime']]
    
    for folder in folders:
        
        # list of all contents of each sub-folder (audio and annotations)
        files = os.listdir(folder_path+folder)
        
        for file in files:
            if file[-4:] == '.wav':
                audio = file
                annotation = file[:-4]+'.csv'
                
                # load audio and compute STFT
                waveform, sr = librosa.load(folder_path+folder+'/'+audio, sr = None)
                nfft=int(sr/10)
                hop_len = int(nfft/4)
                stft = np.abs(librosa.stft(waveform, n_fft=nfft, hop_length=hop_len, window='hann', pad_mode='reflect'))
                
                # noise reduction
                # subtraction of frequency bin median value per bin and time frame median value per time frame
                stft_median = np.median(stft, axis=-1, keepdims=True)
                stft_time_median = np.median(stft, axis=0, keepdims=True)
                norm_stft = stft - stft_median
                norm_stft = norm_stft - stft_time_median

                 # separate the first number of shots, positive (POS) events to be used as templates
                events = []
                with open(folder_path+folder+'/'+annotation) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    for row in csv_reader:
                        if row[-1] == 'POS' and len(events) < shots:
                            events.append(row)
                        
                # section of STFT to use for prediction; starts after end time of last template event
                to_predict = norm_stft[:, int(np.ceil(float(events[-1][-2])*sr/hop_len + 1)):]

                result = []
                for event in events:
                    starttime = float(event[1])
                    endtime = float(event[2])
                    # template STFT is offseted by 4 bins (2 on the top and 2 on the bottom) compared to the full STFT to allow more variability in the vocalisation position
                    event_stft = norm_stft[2:-2,int(np.floor(starttime*sr/hop_len + 1)):int(np.ceil(endtime*sr/hop_len + 1))]
                    # append the predictions of match_template for each event to the result list
                    # result list len = number of events
                    # result[i].shape = (offset bins + 1, to_predict.shape[1]-event_stft.shape[1]+1)
                    result.append(match_template(to_predict, event_stft))

                # a different prediction threshold is set for each audio file
                # based of the similarity of the 5 templates
                mr = []
                for i in range(len(events)):
                    event = events[i]
                    starttime = float(event[1])
                    endtime = float(event[2])
                    event_stft = norm_stft[2:-2,int(np.floor(starttime*sr/hop_len + 1)):int(np.ceil(endtime*sr/hop_len + 1))]

                    r=[]
                    for j in range(len(events)):
                        if j != i:
                            inner_event = events[j]
                            inner_starttime = float(inner_event[1])
                            inner_endtime = float(inner_event[2])
                            inner_event_stft = norm_stft[2:-2,int(np.floor(inner_starttime*sr/hop_len + 1)):int(np.ceil(inner_endtime*sr/hop_len + 1))]

                            if inner_event_stft.shape[1] >= event_stft.shape[1]:
                                r.append(np.max(match_template(inner_event_stft, event_stft)))
                    if r:
                        mr.append(np.max(r))
                        
                # threshold set as the max of similarities between the 5 templates
                threshold = np.max(mr)

                # peak picking on the match_template result vector turning peaks into 1 and rest into 0
                binary_result = []
                for i in range(len(result)):
                    event_len = int(np.ceil(np.floor(float(events[i][2])*sr/hop_len + 1))) - int(np.floor(float(events[i][1])*sr/hop_len))
                    rmax = np.zeros((result[i].shape[1], ))
                    peaks, _ = find_peaks(np.max(result[i], axis=0), height=threshold, distance=event_len)
                    rmax[peaks] = 1
                    binary_result.append(rmax)

                # creating segments based on centre of predictions and template length then overlapping all predictions
                for i in range(len(binary_result)):
                    starttime = float(events[i][1])
                    endtime = float(events[i][2])
                    event_stft = norm_stft[2:-2,int(np.floor(starttime*sr/hop_len + 1)):int(np.ceil(endtime*sr/hop_len + 1))]
                    event_len = int(event_stft.shape[1])
                    lpad = int(np.floor(event_len/2))
                    rpad = int(event_len-lpad-1)
                    indeces = np.where(binary_result[i]==1)[0]

                    for index in indeces:
                        binary_result[i][int(index)-lpad:int(index)+rpad] = 1
                    
                    # padding binary_result back to full audio STFT length
                    binary_result[i] = np.pad(binary_result[i], (lpad, rpad))
                    binary_result[i] = np.pad(binary_result[i], (norm_stft.shape[1]-to_predict.shape[1],0))
                    # overlapping binary_result predictions
                    binary_result[0] += binary_result[i]

                # final result vector with binary segments of predicted events
                final_result = binary_result[0]
                final_result[np.where(final_result>0)] = 1

                # transforming time frames to seconds in predictions
                startind = np.where(finalR[:-1] - finalR[1:] == -1)[0]
                endind = np.where(finalR[:-1] - finalR[1:] == 1)[0]
                
                # filling to_write list with predicted results
                for i in range(len(startind)):
                    to_write.append([audio, startind[i]*hop_len/sr, endind[i]*hop_len/sr])
    
    # write csv output file with predictions for all audio files           
    with open(output_file+'.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(to_write)
                    
    return
                
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-folder_path', type=str, help='path to Validation_Set folder')
    args = parser.parse_args()
    
    # folder_path = './Development_Set/Validation_Set/'
    fewshot_match_template(folder_path=args.folder_path)
    
