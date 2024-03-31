import sys

sys.path.append("/vol/research/dcase2022/project/hhlab")

import itertools as it
import os
from glob import glob
from itertools import chain

import h5py
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from src.datamodules.components.Datagenerator import Datagen_test
from src.datamodules.components.pcen import Feature_Extractor


class PrototypeDynamicDataSet(Dataset):
    def __init__(self, path: dict = {}, features: dict = {}, train_param: dict = {}):
        """_summary_
            Dynamically select audio from the wave files
        Args:
            path (dict, optional): _description_. Defaults to {}.
            features (dict, optional): _description_. Defaults to {}.
        """
        self.path = path
        self.features = features
        self.train_param = train_param
        self.samples_per_cls = train_param.n_shot * 2
        self.seg_len = train_param.seg_len
        self.fe = Feature_Extractor(self.features)
        self.fps = self.features.sr / self.features.hop_mel
        self.all_csv_files = self.get_all_csv_files()
        self.length = int(3600 * 8 / self.train_param.hop_seg)

        self.meta = {}
        """meta dic structure
        {
            <class-name>: {
                "info":[(<start-time>, <end-time>), ...],
                "duration": [duration1, duration2, ...]
            }
        }
        """
        self.build_meta()
        self.classes = list(self.meta.keys())
        self.classes2int = self.get_class2int()
        self.classes_list = [x for x in it.combinations(self.classes, 5)]
        self.classes_duration = self.get_class_durations()

    def __len__(self):
        # Every two hours of positive data
        return self.length

    def __getitem__(self, idx):
        class_name = self.classes[idx]
        segment = self.select_file(class_name)
        segment = segment * (2**32)
        pcen = self.fe.pcen(segment).T
        return pcen, self.classes2int[class_name]

    def select_file(self, class_name):
        segment_idx = np.random.randint(len(self.meta[class_name]["info"]))
        start, end = self.meta[class_name]["info"][segment_idx]
        fname = self.meta[class_name]["file"][segment_idx]
        return self.select_segment(start, end, fname)

    def select_segment(self, start, end, fname):
        total_duration = end - start
        if total_duration < self.seg_len:
            x, _ = librosa.load(
                fname, sr=None, mono=True, offset=start, duration=total_duration
            )
            tile_times = np.ceil(self.seg_len / total_duration)
            x = np.tile(x, int(tile_times))
            x = x[: int(self.train_param.sr * self.seg_len)]
        else:
            rand_start = np.random.uniform(low=start, high=end - self.seg_len)
            x, _ = librosa.load(
                fname, sr=None, mono=True, offset=rand_start, duration=self.seg_len
            )
        assert 10 > abs(
            x.shape[0] - int(self.train_param.sr * self.seg_len)
        ), "%s %s %s %s %s %s" % (
            fname,
            rand_start,
            start,
            end,
            str(x.shape),
            str(int(self.train_param.sr * self.seg_len)),
        )
        return x

    def build_meta(self):
        # Main function for building up meta data
        for file in self.all_csv_files:
            glob_cls_name = self.get_glob_cls_name(file)
            df_pos = self.get_df_pos(file)
            start_time, end_time = self.get_time(df_pos)
            cls_list = self.get_cls_list(df_pos, glob_cls_name, start_time)
            self.update_meta(start_time, end_time, cls_list, file)

    def update_meta(self, start_time, end_time, cls_list, csv_file):
        wav_file = csv_file.replace(".csv", ".wav")
        for start, end, cls in zip(start_time, end_time, cls_list):
            if cls not in self.meta.keys():
                self.meta[cls] = {}
                self.meta[cls]["info"] = []
                self.meta[cls]["duration"] = []
                self.meta[cls]["file"] = []
            self.meta[cls]["info"].append((start, end))
            self.meta[cls]["duration"].append(end - start)
            self.meta[cls]["file"].append(wav_file)

    def get_class_durations(self):
        durations = []
        for cls in self.classes:
            durations.append(np.sum(self.meta[cls]["duration"]))
        return durations

    def get_all_csv_files(self):
        extension = "*.csv"
        return [
            file
            for path_dir, _, _ in os.walk(self.path.train_dir)
            for file in glob(os.path.join(path_dir, extension))
        ]

    def get_glob_cls_name(self, file):
        split_list = file.split("/")
        return split_list[split_list.index("Training_Set") + 1]

    def get_df_pos(self, file):
        df = pd.read_csv(file, header=0, index_col=False)
        return df[(df == "POS").any(axis=1)]

    def get_cls_list(self, df_pos, glob_cls_name, start_time):
        if "CALL" in df_pos.columns:
            cls_list = [glob_cls_name] * len(start_time)
        else:
            cls_list = [
                df_pos.columns[(df_pos == "POS").loc[index]].values
                for index, row in df_pos.iterrows()
            ]
            cls_list = list(chain.from_iterable(cls_list))
        return cls_list

    def get_time(self, df):
        """Margin of 25 ms around the onset and offsets."""

        # TODO hardcode
        df.loc[:, "Starttime"] = df["Starttime"] - 0.025
        df.loc[:, "Endtime"] = df["Endtime"] + 0.025

        "Converting time to frames"

        start_time = [start for start in df["Starttime"]]
        end_time = [end for end in df["Endtime"]]

        return start_time, end_time

    def time2frame(self, t, fps):
        return int(np.floor(t * fps))

    def get_class2int(self):

        """Convert class label to integer
        Args:
        -label_array: label array
        -class_set: unique classes in label_array

        Out:
        -y: label to index values
        """
        label2indx = {label: index for index, label in enumerate(self.classes)}
        y = np.array([label2indx[label] for label in self.classes])
        return {c: idx for c, idx in zip(self.classes, y)}


def calculate_mean_std():
    import torch
    from omegaconf import OmegaConf
    from tqdm import tqdm

    from src.datamodules.components.identity_sampler import IdentityBatchSampler

    data = []

    conf = OmegaConf.load("/vol/research/dcase2022/project/hhlab/configs/train.yaml")
    dataset = PrototypeDynamicDataSet(
        path=conf.path, features=conf.features, train_param=conf.train_param
    )
    sampler = IdentityBatchSampler(
        conf.train_param,
        len(dataset.classes),
        batch_size=conf.train_param.n_shot * 2,
        n_episode=int(
            len(dataset) / (conf.train_param.k_way * conf.train_param.n_shot * 2)
        ),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=16)
    # mean: 1.4421, std: 1.2201
    for each in tqdm(loader):
        x, y = each


if __name__ == "__main__":
    calculate_mean_std()
