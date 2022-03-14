import os
import librosa
import h5py
import pandas as pd
import numpy as np
from scipy import signal
from glob import glob
from itertools import chain

pd.options.mode.chained_assignment = None



def create_dataset(df_pos,pcen,glob_cls_name,file_name,hf,seg_len,hop_seg,fps):

    '''Chunk the time-frequecy representation to segment length and store in h5py dataset

    Args:
        -df_pos : dataframe
        -log_mel_spec : log mel spectrogram
        -glob_cls_name: Name of the class used in audio files where only one class is present
        -file_name : Name of the csv file
        -hf: h5py object
        -seg_len : fixed segment length
        -fps: frame per second
    Out:
        - label_list: list of labels for the extracted mel patches'''

    label_list = []
    if len(hf['features'][:]) == 0:
        file_index = 0
    else:
        file_index = len(hf['features'][:])


    start_time,end_time = time_2_frame(df_pos,fps)


    'For csv files with a column name Call, pick up the global class name'

    if 'CALL' in df_pos.columns:
        cls_list = [glob_cls_name] * len(start_time)
    else:
        cls_list = [df_pos.columns[(df_pos == 'POS').loc[index]].values for index, row in df_pos.iterrows()]
        cls_list = list(chain.from_iterable(cls_list))

    assert len(start_time) == len(end_time)
    assert len(cls_list) == len(start_time)

    for index in range(len(start_time)):

        str_ind = start_time[index]
        end_ind = end_time[index]
        label = cls_list[index]

        'Extract segment and move forward with hop_seg'

        if end_ind - str_ind > seg_len:
            shift = 0
            while end_ind - (str_ind + shift) > seg_len:

                pcen_patch = pcen[int(str_ind + shift):int(str_ind + shift + seg_len)]

                hf['features'].resize((file_index + 1, pcen_patch.shape[0], pcen_patch.shape[1]))
                hf['features'][file_index] = pcen_patch
                label_list.append(label)
                file_index += 1
                shift = shift + hop_seg

            pcen_patch_last = pcen[end_ind - seg_len:end_ind]



            hf['features'].resize((file_index+1 , pcen_patch.shape[0], pcen_patch.shape[1]))
            hf['features'][file_index] = pcen_patch_last
            label_list.append(label)
            file_index += 1
        else:

            'If patch length is less than segment length then tile the patch multiple times till it reaches the segment length'

            pcen_patch = pcen[str_ind:end_ind]
            if pcen_patch.shape[0] == 0:
                print(pcen_patch.shape[0])
                print("The patch is of 0 length")
                continue

            repeat_num = int(seg_len / (pcen_patch.shape[0])) + 1
            pcen_patch_new = np.tile(pcen_patch, (repeat_num, 1))
            pcen_patch_new = pcen_patch_new[0:int(seg_len)]
            hf['features'].resize((file_index+1, pcen_patch_new.shape[0], pcen_patch_new.shape[1]))
            hf['features'][file_index] = pcen_patch_new
            label_list.append(label)
            file_index += 1

    
    print("Total files created : {}".format(file_index))
    return label_list

class Feature_Extractor():

       def __init__(self, conf):
           self.sr =conf.features.sr
           self.n_fft = conf.features.n_fft
           self.hop = conf.features.hop_mel
           self.n_mels = conf.features.n_mels
           self.fmax = conf.features.fmax
           #self.win_length = conf.features.win_length
       def extract_feature(self,audio):

           mel_spec = librosa.feature.melspectrogram(audio,sr=self.sr, n_fft=self.n_fft,
                                                     hop_length=self.hop,n_mels=self.n_mels,fmax=self.fmax)
           pcen = librosa.core.pcen(mel_spec,sr=22050)
           pcen = pcen.astype(np.float32)

           return pcen

def extract_feature(audio_path,feat_extractor,conf):

    y,fs = librosa.load(audio_path,sr=conf.features.sr)

    'Scaling audio as per suggestion in librosa documentation'

    y = y * (2**32)
    pcen = feat_extractor.extract_feature(y)
    return pcen.T



def time_2_frame(df,fps):


    'Margin of 25 ms around the onset and offsets'

    df.loc[:,'Starttime'] = df['Starttime'] - 0.025
    df.loc[:,'Endtime'] = df['Endtime'] + 0.025

    'Converting time to frames'

    start_time = [int(np.floor(start * fps)) for start in df['Starttime']]

    end_time = [int(np.floor(end * fps)) for end in df['Endtime']]

    return start_time,end_time

def feature_transform(conf=None,mode=None):
    '''
       Training:
          Extract mel-spectrogram/PCEN and slice each data sample into segments of length conf.seg_len.
          Each segment inherits clip level label. The segment length is kept same across training
          and validation set.
       Evaluation:
           Currently using the validation set for evaluation.
           
           For each audio file, extract time-frequency representation and create 3 subsets:
           a) Positive set - Extract segments based on the provided onset-offset annotations.
           b) Negative set - Since there is no negative annotation provided, we consider the entire
                         audio file as the negative class and extract patches of length conf.seg_len
           c) Query set - From the end time of the 5th annotation to the end of the audio file.
                          Onset-offset prediction is made on this subset.

       Args:
       - config: config object
       - mode: train/valid

       Out:
       - Num_extract_train/Num_extract_valid - Number of samples in training/validation set
                                                                                              '''


    label_tr = []
    pcen_extractor = Feature_Extractor(conf)

    fps =  conf.features.sr / conf.features.hop_mel
    'Converting fixed segment legnth to frames'

    seg_len = int(round(conf.features.seg_len * fps))
    hop_seg = int(round(conf.features.hop_seg * fps))
    extension = "*.csv"


    if mode == 'train':

        print("=== Processing training set ===")
        meta_path = conf.path.train_dir
        all_csv_files = [file
                         for path_dir, subdir, files in os.walk(meta_path)
                         for file in glob(os.path.join(path_dir, extension))]
        all_csv_files = all_csv_files[:100]
        hdf_tr = os.path.join(conf.path.feat_train,'Mel_train.h5')
        hf = h5py.File(hdf_tr,'w')
        hf.create_dataset('features', shape=(0, seg_len, conf.features.n_mels),
                          maxshape=(None, seg_len, conf.features.n_mels))
        num_extract = 0
        for file in all_csv_files:

            split_list = file.split('/')
            glob_cls_name = split_list[split_list.index('Training_Set') + 1]
            file_name = split_list[split_list.index('Training_Set') + 2]
            df = pd.read_csv(file, header=0, index_col=False)
            audio_path = file.replace('csv', 'wav')
            print("Processing file name {}".format(audio_path))
            pcen = extract_feature(audio_path, pcen_extractor,conf)
            df_pos = df[(df == 'POS').any(axis=1)]
            label_list = create_dataset(df_pos,pcen,glob_cls_name,file_name,hf,seg_len,hop_seg,fps)
            label_tr.append(label_list)
        print(" Feature extraction for training set complete")
        num_extract = len(hf['features'])
        flat_list = [item for sublist in label_tr for item in sublist]
        hf.create_dataset('labels', data=[s.encode() for s in flat_list], dtype='S20')
        data_shape = hf['features'].shape
        hf.close()
        return num_extract,data_shape

    else:

        print("=== Processing Validation set ===")

        meta_path = conf.path.eval_dir

        all_csv_files = [file
                         for path_dir, subdir, files in os.walk(meta_path)
                         for file in glob(os.path.join(path_dir, extension))]

        num_extract_eval = 0

        for file in all_csv_files:

            idx_pos = 0
            idx_neg = 0
            start_neg = 0
            hop_neg = 0
            idx_query = 0
            hop_query = 0
            strt_index = 0

            split_list = file.split('/')
            name = str(split_list[-1].split('.')[0])
            feat_name = name + '.h5'
            audio_path = file.replace('csv', 'wav')
            feat_info = []
            hdf_eval = os.path.join(conf.path.feat_eval,feat_name)
            hf = h5py.File(hdf_eval,'w')
            

            df_eval = pd.read_csv(file, header=0, index_col=False)
            Q_list = df_eval['Q'].to_numpy()

            start_time,end_time = time_2_frame(df_eval,fps)

            index_sup = np.where(Q_list == 'POS')[0][:conf.train.n_shot]

            difference = []
            for index in index_sup:
                difference.append(end_time[index] - start_time[index])
            
            # Adaptive segment length based on the audio file. 
            max_len = max(difference)
            
            # Choosing the segment length based on the maximum size in the 5-shot.
            # Logic was based on fitment on 12GB GPU since some segments are quite long. 
            if max_len < 100:

                seg_len = max_len
            elif max_len > 100 and max_len < 500 :
                seg_len = max_len//4
            else:
                seg_len = max_len//8
                

            
            print(f"Segment length for file is {seg_len}")
            hop_seg = seg_len//2

            hf.create_dataset('feat_pos', shape=(0, seg_len, conf.features.n_mels),
                              maxshape= (None, seg_len, conf.features.n_mels))
            hf.create_dataset('feat_query',shape=(0,seg_len,conf.features.n_mels),maxshape=(None,seg_len,conf.features.n_mels))
            hf.create_dataset('feat_neg',shape=(0,seg_len,conf.features.n_mels),maxshape=(None,seg_len,conf.features.n_mels))
            hf.create_dataset('start_index_query',shape=(1,),maxshape=(None))

            

            
            hf.create_dataset('seg_len',shape=(1,), maxshape=(None))
            hf.create_dataset('hop_seg',shape=(1,), maxshape=(None))
            pcen = extract_feature(audio_path, pcen_extractor,conf)
            mean = np.mean(pcen)
            std = np.mean(pcen)
            hf['seg_len'][:] = seg_len
            hf['hop_seg'][:] = hop_seg

            strt_indx_query = end_time[index_sup[-1]]
            end_idx_neg = pcen.shape[0] - 1
            hf['start_index_query'][:] = strt_indx_query

            print("Creating negative dataset")

            while end_idx_neg - (strt_index + hop_neg) > seg_len:

                patch_neg = pcen[int(strt_index + hop_neg):int(strt_index + hop_neg + seg_len)]

                hf['feat_neg'].resize((idx_neg + 1, patch_neg.shape[0], patch_neg.shape[1]))
                hf['feat_neg'][idx_neg] = patch_neg
                idx_neg += 1
                hop_neg += hop_seg

            last_patch = pcen[end_idx_neg - seg_len:end_idx_neg]
            hf['feat_neg'].resize((idx_neg + 1, last_patch.shape[0], last_patch.shape[1]))
            hf['feat_neg'][idx_neg] = last_patch

            print("Creating Positive dataset")
            for index in index_sup:

                str_ind = int(start_time[index])
                end_ind = int(end_time[index])

                if end_ind - str_ind > seg_len:

                    shift = 0
                    while end_ind - (str_ind + shift) > seg_len:

                        patch_pos = pcen[int(str_ind + shift):int(str_ind + shift + seg_len)]

                        hf['feat_pos'].resize((idx_pos + 1, patch_pos.shape[0], patch_pos.shape[1]))
                        hf['feat_pos'][idx_pos] = patch_pos
                        idx_pos += 1
                        shift += hop_seg
                    last_patch_pos = pcen[end_ind - seg_len:end_ind]
                    hf['feat_pos'].resize((idx_pos + 1, patch_pos.shape[0], patch_pos.shape[1]))
                    hf['feat_pos'][idx_pos] = last_patch_pos
                    idx_pos += 1

                else:
                    patch_pos = pcen[str_ind:end_ind]

                    if patch_pos.shape[0] == 0:
                        print(patch_pos.shape[0])
                        print("The patch is of 0 length")
                        continue
                    repeat_num = int(seg_len / (patch_pos.shape[0])) + 1

                    patch_new = np.tile(patch_pos, (repeat_num, 1))
                    patch_new = patch_new[0:int(seg_len)]
                    hf['feat_pos'].resize((idx_pos + 1, patch_new.shape[0], patch_new.shape[1]))
                    hf['feat_pos'][idx_pos] = patch_new
                    idx_pos += 1



            print("Creating query dataset")

            while end_idx_neg - (strt_indx_query + hop_query) > seg_len:

                patch_query = pcen[int(strt_indx_query + hop_query):int(strt_indx_query + hop_query + seg_len)]
                hf['feat_query'].resize((idx_query + 1, patch_query.shape[0], patch_query.shape[1]))
                hf['feat_query'][idx_query] = patch_query
                idx_query += 1
                hop_query += hop_seg


            last_patch_query = pcen[end_idx_neg - seg_len:end_idx_neg]

            hf['feat_query'].resize((idx_query + 1, last_patch_query.shape[0], last_patch_query.shape[1]))
            hf['feat_query'][idx_query] = last_patch_query
            num_extract_eval += len(hf['feat_query'])

            hf.close()

        return num_extract_eval
