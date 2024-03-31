import sys

sys.path.append("/vol/research/dcase2022/project/hhlab")


import os
from glob import glob

import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from src.datamodules.components.pcen import Feature_Extractor
from src.datamodules.components.Datagenerator import Datagen_test


def time_2_frame(df, fps):
    "Margin of 25 ms around the onset and offsets"

    df.loc[:, "Starttime"] = df["Starttime"] - 0.025
    df.loc[:, "Endtime"] = df["Endtime"] + 0.025

    "Converting time to frames"

    start_time = [int(np.floor(start * fps)) for start in df["Starttime"]]

    end_time = [int(np.floor(end * fps)) for end in df["Endtime"]]

    return start_time, end_time


class PrototypeAdaSeglenTestSet(Dataset):
    def __init__(
        self,
        path: dict = {},
        features: dict = {},
        train_param: dict = {},
        eval_param: dict = {},
    ):
        self.path = path
        self.features = features
        self.train_param = train_param
        self.eval_param = eval_param

        self.fps = self.features.sr / self.features.hop_mel
        self.fe = Feature_Extractor(self.features)

        extension = "*.csv"
        self.all_csv_files = [
            file
            for path_dir, _, _ in os.walk(self.path.eval_dir)
            for file in glob(os.path.join(path_dir, extension))
        ]
        self.all_csv_files = sorted(self.all_csv_files)

    def __len__(self):
        return len(self.all_csv_files)

    def __getitem__(self, idx):
        feat_file = self.all_csv_files[idx]
        X_pos, X_neg, X_query, strt_index_query, audio_path, hop_seg = self.read_file(
            feat_file
        )
        return (
            (
                X_pos.astype(np.float32),
                X_neg.astype(np.float32),
                X_query.astype(np.float32),
                hop_seg,
            ),
            strt_index_query,
            audio_path,
        )

    def find_positive_label(self, df):
        for col in df.columns:
            if "Q" in col:
                return col
        else:
            raise ValueError(
                "Error: Expect you change the validation set event name to Q_x"
            )

    def read_file(self, file):
        hop_neg = 0
        hop_query = 0
        strt_index = 0

        audio_path = file.replace("csv", "wav")
        df_eval = pd.read_csv(file, header=0, index_col=False)
        key = self.find_positive_label(df_eval)
        Q_list = df_eval[key].to_numpy()
        start_time, end_time = time_2_frame(df_eval, self.fps)
        index_sup = np.where(Q_list == "POS")[0][: self.train_param.n_shot]
        #################################Adaptive hop_seg#########################################
        difference = []
        for index in index_sup:
            difference.append(end_time[index] - start_time[index])
        # Adaptive segment length based on the audio file.
        max_len = max(difference)
        # Choosing the segment length based on the maximum size in the 5-shot.
        # Logic was based on fitment on 12GB GPU since some segments are quite long.
        if max_len < 100:
            seg_len = max_len
        elif max_len > 100 and max_len < 500:
            seg_len = max_len // 4
        else:
            seg_len = max_len // 8
        print(f"Adaptive segment length for %s is {seg_len}" % (file))
        hop_seg = seg_len // 2
        #################################################################################
        pcen = self.fe.extract_feature(audio_path)
        strt_indx_query = end_time[index_sup[-1]]
        end_idx_neg = pcen.shape[0] - 1

        feat_neg, feat_pos, feat_query = [], [], []

        while end_idx_neg - (strt_index + hop_neg) > seg_len:
            patch_neg = pcen[
                int(strt_index + hop_neg) : int(strt_index + hop_neg + seg_len)
            ]
            feat_neg.append(patch_neg)
            hop_neg += hop_seg

        last_patch = pcen[end_idx_neg - seg_len : end_idx_neg]
        feat_neg.append(last_patch)

        # print("Creating Positive dataset")
        for index in index_sup:
            str_ind = int(start_time[index])
            end_ind = int(end_time[index])

            if end_ind - str_ind > seg_len:
                shift = 0
                while end_ind - (str_ind + shift) > seg_len:
                    patch_pos = pcen[
                        int(str_ind + shift) : int(str_ind + shift + seg_len)
                    ]
                    feat_pos.append(patch_pos)
                    shift += hop_seg
                last_patch_pos = pcen[end_ind - seg_len : end_ind]
                feat_pos.append(last_patch_pos)

            else:
                patch_pos = pcen[str_ind:end_ind]
                if patch_pos.shape[0] == 0:
                    print(patch_pos.shape[0])
                    print("The patch is of 0 length")
                    continue
                repeat_num = int(seg_len / (patch_pos.shape[0])) + 1
                patch_new = np.tile(patch_pos, (repeat_num, 1))
                patch_new = patch_new[0 : int(seg_len)]
                feat_pos.append(patch_new)

        # print("Creating query dataset")

        while end_idx_neg - (strt_indx_query + hop_query) > seg_len:
            patch_query = pcen[
                int(strt_indx_query + hop_query) : int(
                    strt_indx_query + hop_query + seg_len
                )
            ]
            feat_query.append(patch_query)
            hop_query += hop_seg

        last_patch_query = pcen[end_idx_neg - seg_len : end_idx_neg]
        feat_query.append(last_patch_query)
        return (
            np.stack(feat_pos),
            np.stack(feat_neg),
            np.stack(feat_query),
            strt_indx_query,
            audio_path,
            hop_seg,
        )  # [n, seg_len, 128]


if __name__ == "__main__":
    import torch
    from omegaconf import OmegaConf
    from tqdm import tqdm

    from src.datamodules.components.identity_sampler import IdentityBatchSampler

    data = []

    conf = OmegaConf.load("/vol/research/dcase2022/project/hhlab/configs/train.yaml")
    testset = PrototypeTestSet(conf.path, conf.features, conf.train_param)
    for (X_pos, X_neg, X_query, hop_seg), strt_index_query, audio_path in tqdm(testset):
        # import ipdb; ipdb.set_trace()
        pass
