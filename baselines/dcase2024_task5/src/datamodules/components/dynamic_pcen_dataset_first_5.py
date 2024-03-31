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
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.datamodules.components.Datagenerator import Datagen_test
from src.datamodules.components.pcen import Feature_Extractor

# logmel.npy: log mel spectrogram
# .npy: PCEN spectrogram
class PrototypeDynamicArrayDataSetWithEval(Dataset):
    def __init__(self, path: dict = {}, features: dict = {}, train_param: dict = {}):
        """_summary_
        Use the first five annotation in the validation files to construct the training dataset
        """
        print(
            "Building training dataset. Use the first 5 samples from the evaluation set for training"
        )
        self.path = path
        self.features = features
        self.train_param = train_param
        self.samples_per_cls = train_param.n_shot * 2
        self.seg_len = train_param.seg_len

        if path.test_dir is not None:
            self.fe = Feature_Extractor(
                self.features, audio_path=[path.train_dir, path.eval_dir, path.test_dir]
            )
        else:
            self.fe = Feature_Extractor(
                self.features, audio_path=[path.train_dir, path.eval_dir]
            )

        print(
            "Build the training dataset with suffix",
            self.features.feature_types.split("@"),
        )

        (
            self.all_train_csv_files,
            self.all_eval_csv_files,
            self.extra_csv_files,
        ) = self.get_all_csv_files()
        self.all_csv_files = (
            self.all_train_csv_files + self.all_eval_csv_files + self.extra_csv_files
        )
        self.train_classes = []
        self.eval_classes = []
        self.extra_train_classes = []

        self.length = int(3600 * 8 / self.train_param.seg_len)
        self.fps = self.features.sr / self.features.hop_mel

        self.meta = {}
        self.pcen = {}
        self.mel = {}
        self.mask = {}

        # Store the un_normalized mel spectrograms
        """meta dic structure
        {
            <class-name>: {
                "info":[(<start-time>, <end-time>), ...],
                "duration": [duration1, duration2, ...]
            }
        }
        """
        # The positive segment within the batch should come from the same segment
        self.segment_buffer = {}
        self.start_end_buffer = {}
        self.cnt = 0
        self.segment_level_training_length = 1.5
        self.batchsize = self.train_param.k_way * self.train_param.n_shot * 2

        self.build_meta()
        # self.load_mask()
        self.build_buffer()
        self.remove_short_negative_duration()

        self.classes = list(self.meta.keys())
        self.classes2int = self.get_class2int()
        print("Training: ", self.classes2int)
        self.classes_duration = self.get_class_durations()

        self.train_eval_class_idxs = [
            self.classes2int[x]
            for x in self.train_classes + self.eval_classes
            if (x in self.classes)
        ]
        self.extra_train_class_idxs = [
            self.classes2int[x] for x in self.extra_train_classes if (x in self.classes)
        ]

    def __len__(self):
        # Every two hours of positive data
        return self.length

    def __getitem__(self, idx):
        if self.cnt % self.batchsize == 0:
            self.segment_buffer = {}
            self.start_end_buffer = {}

        class_name = self.classes[idx]
        segment = self.select_positive(class_name)
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

    def load_mask(self):
        print("Loading frequency masks")
        for clss in tqdm(self.meta.keys()):
            self.mask[clss] = []

            if not os.path.exists(os.path.join(self.path.mask_dir, clss)):
                self.build_mask()

            for file in os.listdir(os.path.join(self.path.mask_dir, clss)):
                if ".npy" not in file:
                    continue
                self.mask[clss].append(
                    np.load(os.path.join(self.path.mask_dir, clss, file))
                )

    def build_mask(self):
        for file in tqdm(self.all_csv_files):
            audio_path = file.replace("csv", "wav")
            if audio_path not in self.pcen.keys():
                self.mel[audio_path] = self.fe.extract_feature(
                    audio_path, feature_types=["mel"], normalized=False
                )
        self.build_freq_energy_mask()

    def build_buffer(self):
        for file in tqdm(self.all_csv_files):
            audio_path = file.replace("csv", "wav")
            if audio_path not in self.pcen.keys():
                self.pcen[audio_path] = self.fe.extract_feature(audio_path)

    def calculate_feature_mean_std(self):
        A = [self.pcen[k] for k in self.pcen.keys()]
        N = float(sum([i.size for i in A]))
        mean_ = sum([i.sum() for i in A]) / N
        std_ = np.sqrt(sum([((i - mean_) ** 2).sum() for i in A]) / N)
        print("Mean %s; Std %s;" % (mean_, std_))

    def select_negative(self, class_name):
        # Choose the negative
        segment_idx = np.random.randint(len(self.meta[class_name]["neg_info"]))
        start, end = self.meta[class_name]["neg_info"][segment_idx]

        while end - start < 0.1:
            segment_idx = np.random.randint(len(self.meta[class_name]["neg_info"]))
            start, end = self.meta[class_name]["neg_info"][segment_idx]

        segment = self.select_segment(
            start,
            end,
            self.pcen[self.meta[class_name]["neg_file"][segment_idx]],
            seg_len=int(self.seg_len * self.fps),
            class_name=class_name,
        )
        return segment

    def select_positive(self, class_name):
        if class_name in self.extra_train_classes:
            # New class in a batch
            if class_name not in self.segment_buffer.keys():
                segment_idx = np.random.randint(len(self.meta[class_name]["info"]))
                self.segment_buffer[class_name] = segment_idx
                start, end = self.meta[class_name]["info"][segment_idx]
                # If the segment length is long enough, restrict the range
                if end - start > self.segment_level_training_length:
                    start = np.random.uniform(
                        0, end - self.segment_level_training_length
                    )
                    end = start + self.segment_level_training_length
                self.start_end_buffer[class_name] = (start, end)
            else:
                segment_idx = self.segment_buffer[class_name]
                start, end = self.start_end_buffer[class_name]
        else:
            segment_idx = np.random.randint(len(self.meta[class_name]["info"]))
            start, end = self.meta[class_name]["info"][segment_idx]

        segment = self.select_segment(
            start,
            end,
            self.pcen[self.meta[class_name]["file"][segment_idx]],
            seg_len=int(self.seg_len * self.fps),
            class_name=class_name,
        )
        return segment

    def select_segment(self, start, end, pcen, seg_len=17, class_name=None):
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
        # Freq mask
        # self.fe.apply_mask(x, mask=mask)
        # Validation set and test set use the averaged mask prompt
        # mask_idx = np.random.choice(np.arange(0, len(self.mask[class_name])))
        # x = self.concate_mask(x, self.mask[class_name][mask_idx])
        return x

    def concate_mask(self, x, mask):
        pad_length = x.shape[1] - mask.shape[0]
        mask = np.pad(mask, pad_width=(0, pad_length))
        mask = np.tile(mask[None, ...], (x.shape[0], 1))
        return np.concatenate(
            [x[None, ...], mask[None, ...]], axis=0
        )  # concatenate on the channel dimension

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

        for clss in set(cls_list):
            if clss in self.meta.keys():
                self.meta[clss]["neg_start_time"] = 0

        for start, end, clss in zip(start_time, end_time, cls_list):
            if csv_file in self.all_train_csv_files and clss not in self.train_classes:
                self.train_classes.append(clss)

            if csv_file in self.all_eval_csv_files and clss not in self.eval_classes:
                self.eval_classes.append(clss)

            if (
                csv_file in self.extra_csv_files
                and clss not in self.extra_train_classes
            ):
                self.extra_train_classes.append(clss)

            if clss not in self.meta.keys():
                self.meta[clss] = {}
                self.meta[clss]["neg_start_time"] = 0  # temp variable
                self.meta[clss]["info"] = []  # positive segment onset and offset
                self.meta[clss]["neg_info"] = []  # negative segment onset and offset
                self.meta[clss]["duration"] = []  # duration of positive segments
                self.meta[clss]["neg_duration"] = []  # duration of negative segments
                self.meta[clss]["total_audio_duration"] = []  # duration
                self.meta[clss]["file"] = []  # filename
                self.meta[clss]["neg_file"] = []  # filename

            self.meta[clss]["total_audio_duration"].append(
                librosa.get_duration(filename=audio_path, sr=None)
            )
            neg_start, neg_end = np.clip(
                self.meta[clss]["neg_start_time"] - 0.025, a_min=0, a_max=None
            ), np.clip(
                start + 0.025,
                a_min=None,
                a_max=self.meta[clss]["total_audio_duration"][-1],
            )
            self.meta[clss]["neg_info"].append((neg_start, neg_end))
            self.meta[clss]["neg_duration"].append(neg_end - neg_start)
            self.meta[clss]["info"].append((start, end))
            self.meta[clss]["duration"].append(end - start)
            self.meta[clss]["file"].append(audio_path)
            self.meta[clss]["neg_file"].append(audio_path)  # filename
            self.meta[clss]["neg_start_time"] = end

        # Add thess lines if the data in the validation set is sparse
        # for clss in cls_list:
        #     if(np.sum(self.meta[clss]["neg_duration"]) < 2.0):
        #         print("The annotated negative sample of %s is less then 2.0 seconds, use all remaining part as negative training set" % audio_path)
        #         neg_start, neg_end = np.clip(self.meta[clss]["neg_start_time"]-0.025, a_min=0, a_max=None), self.meta[clss]["total_audio_duration"][-1]
        #         self.meta[clss]["neg_info"].append((neg_start, neg_end))
        #         self.meta[clss]["neg_duration"].append(neg_end - neg_start)
        #         self.meta[clss]["neg_file"].append(audio_path)

        # for clss in cls_list:
        #     # Add thess lines if the data in the validation set is sparse
        #     if(np.sum(self.meta[clss]["neg_duration"]) < 30.0):
        #         print("The annotated negative sample of %s is less then 30.0 seconds, use all remaining part as negative training set" % audio_path)
        #         # neg_start, neg_end = np.clip(self.meta[clss]["neg_start_time"]-0.025, a_min=0, a_max=None), self.meta[clss]["total_audio_duration"][-1]
        #         # self.meta[clss]["neg_info"].append((neg_start, neg_end))
        #         # self.meta[clss]["neg_duration"].append(neg_end - neg_start)
        #         # self.meta[clss]["info"].append(self.meta[clss]["info"][-1])
        #         # self.meta[clss]["duration"].append(self.meta[clss]["duration"][-1])
        #         # self.meta[clss]["total_audio_duration"].append(librosa.get_duration(filename=audio_path,sr=None))
        #         # self.meta[clss]["file"].append(audio_path)
        #         self.build_negative_based_on_energy(self.fe.extract_feature(audio_path,"logmel"), clss, audio_path)

    # def build_negative_based_on_energy(self, logmel, clss, audio_path):
    #     import ipdb; ipdb.set_trace()
    #     self.meta[clss]["total_audio_duration"].append(librosa.get_duration(filename=audio_path,sr=None))
    #     self.meta[clss]["file"].append(audio_path)

    def get_class_durations(self):
        durations = []
        for cls in self.classes:
            durations.append(np.sum(self.meta[cls]["duration"]))
        return durations

    def get_all_csv_files(self):
        extension = "*.csv"
        eval_csv = [
            file
            for path_dir, _, _ in os.walk(self.path.eval_dir)
            for file in glob(os.path.join(path_dir, extension))
        ]
        if self.path.test_dir is not None:
            eval_csv += [
                file
                for path_dir, _, _ in os.walk(self.path.test_dir)
                for file in glob(os.path.join(path_dir, extension))
            ]
        return (
            [
                file
                for path_dir, _, _ in os.walk(self.path.train_dir)
                for file in glob(os.path.join(path_dir, extension))
            ],
            eval_csv,
            [
                file
                for path_dir, _, _ in os.walk(self.path.extra_train_dir)
                for file in glob(os.path.join(path_dir, extension))
            ]
            if (self.path.extra_train_dir is not None)
            else [],
        )

    # def get_glob_cls_name(self, file, key="Training_Set"):
    #     split_list = file.split("/")
    #     return split_list[split_list.index(key) + 1]

    def get_glob_cls_name(self, file):
        split_list = file.split("/")
        return split_list[-2]

    def get_df_pos(self, file):
        if file in self.all_eval_csv_files:
            df = pd.read_csv(file, header=0, index_col=False)[: self.train_param.n_shot]
        else:
            df = pd.read_csv(file, header=0, index_col=False)
        return df[(df == "POS").any(axis=1)]

    def get_cls_list(self, df_pos, glob_cls_name, start_time):
        if "Q" == df_pos.columns[3]:  # For HB
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

    def get_freq_mask(self, info_prev, info, info_next, mel):
        # mel: [128, T]
        def get_feature(mel, start, end):
            assert np.sum(mel < 0) == 0, "The feature should have all positive values"
            if start == end:
                print("start == end", start, end)
                return None
            feat = mel[:, start:end]
            feat[:3, :] *= 0
            segment = np.mean(feat, axis=1)
            # segment = nn.MaxPool1d(kernel_size=4, stride=4)(torch.tensor(segment)[None,...]).numpy()[0,...]
            return segment / np.max(segment)

        def time2frame(s, e):
            return max(int(self.fps * s), 0), int(self.fps * e)

        def identify_changed_freq(pos_freq, neg_freq_prev, neg_freq_next):
            diff = np.abs(pos_freq - neg_freq_prev) + np.abs(pos_freq - neg_freq_next)
            diff = diff / np.max(diff)
            return pos_freq

        s_prev, e_prev = time2frame(*info_prev)
        s, e = time2frame(*info)
        s_next, e_next = time2frame(*info_next)

        current_pos_freq = get_feature(mel, s, e)
        neg_length_prev = get_feature(mel, s - min(s - e_prev, e - s), s)
        neg_length_next = get_feature(mel, e, e + min(s_next - e, e - s))

        if neg_length_prev is None:
            neg_length_prev = current_pos_freq
        if neg_length_next is None:
            neg_length_next = current_pos_freq

        dist = identify_changed_freq(current_pos_freq, neg_length_prev, neg_length_next)
        # dist = dist[None,...]
        return dist

    def build_freq_energy_mask(self):
        # Set feature to mel
        # With all the information from the key 'info'
        for clss in tqdm(self.meta.keys()):

            self.meta[clss]["freq_mask"] = None

            os.makedirs(os.path.join(self.path.mask_dir, clss), exist_ok=True)
            # Has already been processed
            if len(os.listdir(os.path.join(self.path.mask_dir, clss))) != 0:
                continue
            print(clss)
            for i, (s, e) in tqdm(enumerate(self.meta[clss]["info"])):
                # Use the negative between each positive infomation block to construct the energy mask
                if i == 0:
                    info_prev = (0.0, 0.0)
                    info_next = (
                        self.meta[clss]["info"][i + 1]
                        if (len(self.meta[clss]["info"]) > 1)
                        else self.meta[clss]["info"][0]
                    )
                elif i == len(self.meta[clss]["info"]) - 1:
                    info_prev = self.meta[clss]["info"][i - 1]
                    total_duration = self.meta[clss]["total_audio_duration"][0]
                    info_next = (total_duration, total_duration)
                else:
                    info_prev = self.meta[clss]["info"][i - 1]
                    info_next = self.meta[clss]["info"][i + 1]

                audiofile = self.meta[clss]["file"][i]
                mel = self.mel[audiofile]
                if self.meta[clss]["freq_mask"] is None:
                    self.meta[clss]["freq_mask"] = []
                self.meta[clss]["freq_mask"].append(
                    self.get_freq_mask(info_prev, (s, e), info_next, mel.T)
                )
            # self.meta[clss]['freq_mask'] = np.concatenate(self.meta[clss]['freq_mask'], axis=0)
            # self.meta[clss]['freq_mask'] = np.mean(self.meta[clss]['freq_mask'], axis=0)
            for i, each in enumerate(self.meta[clss]["freq_mask"]):
                assert each.shape[0] == 128
                os.makedirs(os.path.join(self.path.mask_dir, clss), exist_ok=True)
                plt.plot(each)
                plt.savefig(
                    os.path.join(self.path.mask_dir, "%s/visualize_%s.png" % (clss, i))
                )
                np.save(
                    os.path.join(self.path.mask_dir, "%s/mask_%s.npy" % (clss, i)), each
                )
                plt.close()


def main():
    import torch
    from omegaconf import OmegaConf
    from tqdm import tqdm

    from src.datamodules.components.identity_sampler import IdentityBatchSampler

    data = []

    conf = OmegaConf.load("/vol/research/dcase2022/project/hhlab/configs/train.yaml")
    dataset = PrototypeDynamicArrayDataSetWithEval(
        path=conf.path, features=conf.features, train_param=conf.train_param
    )
    sampler = IdentityBatchSampler(
        conf.train_param,
        dataset.train_eval_class_idxs,
        dataset.extra_train_class_idxs,
        batch_size=conf.train_param.n_shot * 2,
        n_episode=int(
            len(dataset) / (conf.train_param.k_way * conf.train_param.n_shot * 2)
        ),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    # mean: 1.4421, std: 1.2201
    for each in tqdm(loader):
        x, x_neg, y, y_neg, class_name = each
        import ipdb

        ipdb.set_trace()


if __name__ == "__main__":
    main()
