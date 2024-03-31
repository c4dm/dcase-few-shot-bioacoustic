import sys

sys.path.append("/vol/research/dcase2022/project/hhlab")

import itertools as it
import os
from glob import glob
from itertools import chain
import time
import h5py
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from src.datamodules.components.Datagenerator import Datagen_test
from src.datamodules.components.pcen import Feature_Extractor


class PrototypeDynamicArrayDataSet(Dataset):
    def __init__(self, path: dict = {}, features: dict = {}, train_param: dict = {}):
        """_summary_
            Load array from the assigned suffix
        Args:
            path (dict, optional): _description_. Defaults to {}.
            features (dict, optional): _description_. Defaults to {}.
        """
        self.path = path
        self.features = features
        self.train_param = train_param
        self.samples_per_cls = train_param.n_shot * 2
        self.seg_len = train_param.seg_len
        self.fe = Feature_Extractor(
            self.features, audio_path=[path.train_dir, path.eval_dir]
        )  # TODO here only training set

        print(
            "Build the training dataset with suffix",
            self.features.feature_types.split("@"),
        )

        self.all_csv_files = self.get_all_csv_files()

        self.length = int(3600 * 8 / self.train_param.seg_len)
        self.fps = self.features.sr / self.features.hop_mel

        self.meta = {}
        self.pcen = {}
        self.cnt = 0
        self.batchsize = self.train_param.k_way * self.train_param.n_shot * 2

        """meta dic structure
        {
            <class-name>: {
                "info":[(<start-time>, <end-time>), ...],
                "duration": [duration1, duration2, ...]
            }
        }
        """
        self.build_meta()
        # self.remove_short_negative_duration()

        self.classes = list(self.meta.keys())

        self.classes2int = self.get_class2int()
        self.classes_duration = self.get_class_durations()
        self.train_eval_class_idxs = [self.classes2int[x] for x in self.classes]
        self.extra_train_class_idxs = []
        # The positive segment within the batch should come from the same segment
        self.segment_buffer = {}

    def __len__(self):
        # Every two hours of positive data
        return self.length

    def __getitem__(self, idx):
        # Random segment length
        # if(self.cnt % self.batchsize == 0): self._update_seglen()
        # if(self.cnt % self.batchsize == 0): self.segment_buffer = {}
        class_name = self.classes[idx]

        segment = self.select_positive(class_name)

        ########################## Augmentation ################################
        # 48.9
        # aug_segment_neg = self.select_negative(class_name) # Try using negative data from other kinds of sounds
        # mix_ratio = float(np.random.uniform(0.0, 0.3))
        # segment = (segment + mix_ratio*aug_segment_neg) / (1+mix_ratio)
        ########################################################################

        self.cnt += 1

        if not self.train_param.negative_train_contrast:
            return segment.astype(np.float32), self.classes2int[class_name], class_name
        else:
            segment_neg = self.select_negative(class_name)
            # Positive segment, negative segment, positive class label, negative class label
            return (
                segment.astype(np.float32),
                segment_neg.astype(np.float32),
                self.classes2int[class_name] * 2,
                self.classes2int[class_name] * 2 + 1,
                class_name,
            )

    def _update_seglen(self):
        self.seg_len = float(np.random.uniform(0.2, 0.5))

    def select_negative(self, class_name):
        segment_idx = np.random.randint(len(self.meta[class_name]["neg_info"]))
        start, end = self.meta[class_name]["neg_info"][segment_idx]
        while end - start < 0.2:  # TODO the value here
            segment_idx = np.random.randint(len(self.meta[class_name]["neg_info"]))
            start, end = self.meta[class_name]["neg_info"][segment_idx]
        segment = self.select_segment(
            start,
            end,
            self.pcen[self.meta[class_name]["file"][segment_idx]],
            seg_len=int(self.seg_len * self.fps),
        )
        return segment

    def select_positive(self, class_name):
        # TODO this is to cope with the unreliable labeling
        # if(class_name not in self.segment_buffer.keys()):
        #     segment_idx = np.random.randint(len(self.meta[class_name]["info"]))
        #     self.segment_buffer[class_name] = segment_idx
        # else:
        #     segment_idx = self.segment_buffer[class_name]

        segment_idx = np.random.randint(len(self.meta[class_name]["info"]))
        start, end = self.meta[class_name]["info"][segment_idx]

        segment = self.select_segment(
            start,
            end,
            self.pcen[self.meta[class_name]["file"][segment_idx]],
            seg_len=int(self.seg_len * self.fps),
        )
        return segment

    def select_segment(self, start, end, pcen, seg_len=17):
        start, end = int(start * self.fps), int(end * self.fps)
        if start < 0:
            start = 0  # This is due to the function time_2_frame
        total_duration = end - start
        if total_duration < seg_len:
            x = pcen[start:end]
            tile_times = np.ceil(seg_len / total_duration)
            x = np.tile(x, (int(tile_times), 1))
            x = x[:seg_len]
        else:
            rand_start = np.random.uniform(low=start, high=end - seg_len)
            x = pcen[int(rand_start) : int(rand_start) + seg_len]
        if x.shape[0] != seg_len:
            print(
                "Shape error, padded.",
                "%s %s %s %s %s %s"
                % (x.shape, seg_len, start, end, rand_start, pcen.shape),
            )
            x = np.pad(x, ((0, seg_len - x.shape[0]), (0, 0)))
        assert x.shape[0] == seg_len, "%s %s %s %s %s %s" % (
            x.shape,
            seg_len,
            start,
            end,
            rand_start,
            pcen.shape,
        )
        return x

    def build_meta(self):
        from tqdm import tqdm

        print("Preparing meta data...")
        # Main function for building up meta data
        for file in tqdm(self.all_csv_files):
            glob_cls_name = self.get_glob_cls_name(file)
            df_pos = self.get_df_pos(file)
            start_time, end_time = self.get_time(df_pos)
            cls_list = self.get_cls_list(df_pos, glob_cls_name, start_time)
            self.update_meta(start_time, end_time, cls_list, file)

    def update_meta(self, start_time, end_time, cls_list, csv_file):
        audio_path = csv_file.replace("csv", "wav")
        for start, end, cls in zip(start_time, end_time, cls_list):
            if cls not in self.meta.keys():
                self.meta[cls] = {}
                self.meta[cls]["neg_start_time"] = 0  # temp variable
                self.meta[cls]["info"] = []  # positive segment onset and offset
                self.meta[cls]["duration"] = []  # duration of positive segments
                self.meta[cls]["file"] = []  # filename
                self.meta[cls]["neg_info"] = []  # negative segment onset and offset
            self.meta[cls]["neg_info"].append((self.meta[cls]["neg_start_time"], start))
            self.meta[cls]["info"].append((start, end))
            self.meta[cls]["duration"].append(end - start)
            self.meta[cls]["file"].append(audio_path)
            self.meta[cls]["neg_start_time"] = end

            if audio_path not in self.pcen.keys():
                self.pcen[audio_path] = self.fe.extract_feature(audio_path)

    def remove_short_negative_duration(self):
        delete_keys = []
        for k in self.meta.keys():
            neg_info = self.meta[k]["neg_info"]
            end_time = [x[1] for x in neg_info]
            start_time = [x[0] for x in neg_info]
            duration = max([x - y for x, y in zip(end_time, start_time)])
            if duration < 0.3:
                delete_keys.append(k)
        print("Delete class due to short negative length", len(delete_keys))
        for k in delete_keys:
            del self.meta[k]

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
        # import ipdb; ipdb.set_trace()
        # return split_list[split_list.index("Training_Set") + 1]
        return split_list[-2]

    def get_df_pos(self, file):
        df = pd.read_csv(file, header=0, index_col=False)  # TODO
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
    dataset = PrototypeDynamicArrayDataSet(
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
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    # mean: 1.4421, std: 1.2201
    for each in tqdm(loader):
        x, x_neg, y, y_neg, class_name = each
        print("here")
        # import ipdb; ipdb.set_trace()
        # print(torch.mean(x), torch.std(x))
        # print(x.size(), y.size())


if __name__ == "__main__":
    calculate_mean_std()
