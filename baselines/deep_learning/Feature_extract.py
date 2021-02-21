import os
import librosa
import h5py
import pandas as pd
import numpy as np
from scipy import signal
from glob import glob
from itertools import chain



def create_dataset(df_pos,log_mel_spec,prefix_cls_name,file_name,hf,seg_len,hop_seg,fps):

    '''Chunk the time-frequecy representation to segment length and store in h5py dataset

    Args:
        -df_pos : dataframe
        -log_mel_spec : log mel spectrogram
        -prefix_cls_name: Name of the class used in audio files where only one class is present
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



    if 'CALL' in df_pos.columns:
        cls_list = [prefix_cls_name] * len(start_time)
    else:
        cls_list = [df_pos.columns[(df_pos == 'POS').loc[index]].values for index, row in df_pos.iterrows()]
        cls_list = list(chain.from_iterable(cls_list))

    assert len(start_time) == len(end_time)
    assert len(cls_list) == len(start_time)

    print("Processing file name:{}".format(file_name))
    for index in range(len(start_time)):

        str_ind = start_time[index]
        end_ind = end_time[index]
        label = cls_list[index]


        if end_ind - str_ind > seg_len:
            print("Chunking file to length {}".format(seg_len))
            shift = 0
            while end_ind - (str_ind + shift) > seg_len:

                mel_patch = log_mel_spec[int(str_ind + shift):int(str_ind + shift + seg_len)]

                print("Mel_shape {}".format(mel_patch.shape))
                hf['features'].resize((file_index + 1, mel_patch.shape[0], mel_patch.shape[1]))
                hf['features'][file_index] = mel_patch
                label_list.append(label)
                file_index += 1
                shift = shift + hop_seg

            mel_patch_last = log_mel_spec[end_ind - seg_len:end_ind]



            hf['features'].resize((file_index+1 , mel_patch.shape[0], mel_patch.shape[1]))
            hf['features'][file_index] = mel_patch_last
            label_list.append(label)
            file_index += 1
        else:

            mel_patch = log_mel_spec[str_ind:end_ind]
            if mel_patch.shape[0] == 0:
                print(mel_patch.shape[0])
                print("The patch is of 0 length")
                continue

            repeat_num = int(seg_len / (mel_patch.shape[0])) + 1
            print("Replicating audio chunks {} times ".format(repeat_num))
            mel_patch_new = np.tile(mel_patch, (repeat_num, 1))
            mel_patch_new = mel_patch_new[0:int(seg_len)]
            print(mel_patch_new.shape)
            hf['features'].resize((file_index+1, mel_patch_new.shape[0], mel_patch_new.shape[1]))
            hf['features'][file_index] = mel_patch_new
            label_list.append(label)
            file_index += 1

    
    print("Total files created : {}".format(file_index))
    return label_list

class Log_mel_Extractor():

       def __init__(self, conf):
           self.sr =conf.features.sr
           self.n_fft = conf.features.n_fft
           self.hop = conf.features.hop_mel
           self.n_mels = conf.features.n_mels
           self.fmax = conf.features.fmax
           
       def extract_mel(self,audio):

           mel_spec = librosa.feature.melspectrogram(audio,sr=self.sr, n_fft=self.n_fft,
                                                     hop_length=self.hop,n_mels=self.n_mels,fmax=self.fmax)
           log_mel_spec = librosa.power_to_db(mel_spec,ref=np.max)
           log_mel_spec = log_mel_spec.astype(np.float32)
           return log_mel_spec

def extract_feature(audio_path,feat_extractor,conf):

    y,fs = librosa.load(audio_path,sr=conf.features.sr)
    log_mel_spec = feat_extractor.extract_mel(y)
    return log_mel_spec.T



def time_2_frame(df,fps):


    'Margin of 50 ms around the onset and offsets'

    df.loc[:,'Starttime'] = df['Starttime'] - 0.05
    df.loc[:,'Endtime'] = df['Endtime'] - 0.05

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
    mel_extractor = Log_mel_Extractor(conf)

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
        hdf_tr = os.path.join(conf.path.feat_train,'Mel_train.h5')
        hf = h5py.File(hdf_tr,'w')
        hf.create_dataset('features', shape=(0, seg_len, conf.features.n_mels),
                          maxshape=(None, seg_len, conf.features.n_mels))
        num_extract = 0
        for file in all_csv_files:

            split_list = file.split('/')
            prefix_cls_name = split_list[split_list.index('Training_Set') + 1]
            file_name = split_list[split_list.index('Training_Set') + 2]
            df = pd.read_csv(file, header=0, index_col=False)
            audio_path = file.replace('csv', 'wav')
            log_mel_spec = extract_feature(audio_path, mel_extractor,conf)

            df_pos = df[(df == 'POS').any(axis=1)]
            label_list = create_dataset(df_pos,log_mel_spec,prefix_cls_name,file_name,hf,seg_len,hop_seg,fps)
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
            print("Processing file : {}".format(split_list[-1]))
            name = str(split_list[-1].split('.')[0])
            feat_name = name + '.h5'
            audio_path = file.replace('csv', 'wav')

            hdf_eval = os.path.join(conf.path.feat_eval,feat_name)
            hf = h5py.File(hdf_eval,'w')
            hf.create_dataset('feat_pos', shape=(0, seg_len, conf.features.n_mels),
                              maxshape= (None, seg_len, conf.features.n_mels))
            hf.create_dataset('feat_query',shape=(0,seg_len,conf.features.n_mels),maxshape=(None,seg_len,conf.features.n_mels))
            hf.create_dataset('feat_neg',shape=(0,seg_len,conf.features.n_mels),maxshape=(None,seg_len,conf.features.n_mels))
            hf.create_dataset('start_time_query',shape=(1),maxshape=(None))

            df_eval = pd.read_csv(file, header=0, index_col=False)
            Q_list = df_eval['Q'].to_numpy()
            start_time,end_time = time_2_frame(df_eval,fps)

            index_sup = np.where(Q_list == 'POS')[0][:conf.train.n_shot]

            log_mel_spec = extract_feature(audio_path, mel_extractor,conf)
            strt_indx_query = end_time[index_sup[-1]]
            end_idx_neg = log_mel_spec.shape[0] - 1

            hf['start_time_query'][:] = strt_indx_query

            print("Creating negative dataset")

            while end_idx_neg - (strt_index + hop_neg) > seg_len:

                patch_neg = log_mel_spec[int(strt_index + hop_neg):int(strt_index + hop_neg + seg_len)]
                print("Shape of negative patch {}".format(patch_neg.shape))
                hf['feat_neg'].resize((idx_neg + 1, patch_neg.shape[0], patch_neg.shape[1]))
                hf['feat_neg'][idx_neg] = patch_neg
                idx_neg += 1
                hop_neg += hop_seg

            last_patch = log_mel_spec[end_idx_neg - seg_len:end_idx_neg]
            print("Shape of negative patch {}".format(last_patch.shape))
            hf['feat_neg'].resize((idx_neg + 1, last_patch.shape[0], last_patch.shape[1]))
            hf['feat_neg'][idx_neg] = last_patch

            print("Creating Positive dataset")
            for index in index_sup:

                str_ind = int(start_time[index])
                end_ind = int(end_time[index])

                if end_ind - str_ind > seg_len:

                    shift = 0
                    while end_ind - (str_ind + shift) > seg_len:

                        patch_pos = log_mel_spec[int(str_ind + shift):int(str_ind + shift + seg_len)]
                        hf['feat_pos'].resize((idx_pos + 1, patch_pos.shape[0], patch_pos.shape[1]))
                        hf['feat_pos'][idx_pos] = patch_pos
                        print("Shape of positive patch {}".format(patch_pos.shape))
                        idx_pos += 1
                        shift += hop_seg
                    last_patch_pos = log_mel_spec[end_ind - seg_len:end_ind]
                    print("Shape of positive patch {}".format(last_patch_pos.shape))
                    hf['feat_pos'].resize((idx_pos + 1, patch_pos.shape[0], patch_pos.shape[1]))
                    hf['feat_pos'][idx_pos] = last_patch_pos
                    idx_pos += 1

                else:
                    patch_pos = log_mel_spec[str_ind:end_ind]

                    if patch_pos.shape[0] == 0:
                        print(patch_pos.shape[0])
                        print("The patch is of 0 length")
                        continue
                    repeat_num = int(seg_len / (patch_pos.shape[0])) + 1

                    patch_new = np.tile(patch_pos, (repeat_num, 1))
                    print("Shape of positive patch {}".format(patch_new.shape))
                    patch_new = patch_new[0:int(seg_len)]
                    hf['feat_pos'].resize((idx_pos + 1, patch_new.shape[0], patch_new.shape[1]))
                    hf['feat_pos'][idx_pos] = patch_new
                    idx_pos += 1



            print("Creating query dataset")

            while end_idx_neg - (strt_indx_query + hop_query) > seg_len:

                patch_query = log_mel_spec[int(strt_indx_query + hop_query):int(strt_indx_query + hop_query + seg_len)]
                print("Shape of query patch {}".format(patch_query.shape))
                hf['feat_query'].resize((idx_query + 1, patch_query.shape[0], patch_query.shape[1]))
                hf['feat_query'][idx_query] = patch_query
                idx_query += 1
                hop_query += hop_seg


            last_patch_query = log_mel_spec[end_idx_neg - seg_len:end_idx_neg]
            print("Shape of query patch {}".format(last_patch_query.shape))
            hf['feat_query'].resize((idx_query + 1, last_patch_query.shape[0], last_patch_query.shape[1]))
            hf['feat_query'][idx_query] = last_patch_query
            num_extract_eval += len(hf['feat_query'])

            hf.close()

        return num_extract_eval









