# Use the first few labeled segments to construct negative samples

from operator import neg
import sys
from turtle import pos

from transformers import MegatronBertForSequenceClassification

sys.path.append("/vol/research/dcase2022/project/hhlab")
import torch
import os
from glob import glob
import librosa
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from src.datamodules.components.pcen import Feature_Extractor
from src.datamodules.components.Datagenerator import Datagen_test
from scipy.signal import chirp, find_peaks, peak_widths, peak_prominences


def time_2_frame(df, fps, padding=0.025):
    "Margin of 25 ms around the onset and offsets"

    df.loc[:, "Starttime"] = df["Starttime"] - padding
    df.loc[:, "Endtime"] = df["Endtime"] + padding

    "Converting time to frames"
    start_time = [int(np.floor(start * fps)) for start in df["Starttime"]]
    end_time = [int(np.floor(end * fps)) for end in df["Endtime"]]
    return np.clip(start_time, a_min=0, a_max=None), np.clip(
        end_time, a_min=0, a_max=None
    )


class PrototypeAdaSeglenBetterNegTestSetV2(Dataset):
    def __init__(
        self,
        path: dict = {},
        features: dict = {},
        train_param: dict = {},
        eval_param: dict = {},
    ):
        """
        Load data for model testing.

        Strategy applied:
        1. Adaptive segment length
        2. Sample negative data from the labelled part only, unless:
            2.1 The negative data in the labelled part is too short, in this case, we will use energy to determine which part in the query segment should be negative.
        3. Sample two sets of data for inference, one with adaptive segment length, another with short segment length.
            3.1 Adaptive segment length is used for identify the positive segment in the query set.
            3.2 Short segment length is used for identify the possible negative segments, which are used later for post-processing.

        Args:
            path (dict, optional): _description_. Defaults to {}.
            features (dict, optional): _description_. Defaults to {}.
            train_param (dict, optional): _description_. Defaults to {}.
            eval_param (dict, optional): _description_. Defaults to {}.
        """
        self.path = path
        self.features = features
        self.train_param = train_param
        self.eval_param = eval_param

        self.test_seglen_len_lim = features.test_seglen_len_lim
        self.test_hoplen_fenmu = features.test_hoplen_fenmu

        self.fps = self.features.sr / self.features.hop_mel

        if path.test_dir is not None:
            self.fe = Feature_Extractor(
                self.features, audio_path=[path.train_dir, path.eval_dir, path.test_dir]
            )
        else:
            self.fe = Feature_Extractor(
                self.features, audio_path=[path.train_dir, path.eval_dir]
            )

        extension = "*.csv"
        self.all_csv_files = [
            file
            for path_dir, _, _ in os.walk(self.path.eval_dir)
            for file in glob(os.path.join(path_dir, extension))
        ]
        if path.test_dir is not None:
            self.all_csv_files += [
                file
                for path_dir, _, _ in os.walk(self.path.test_dir)
                for file in glob(os.path.join(path_dir, extension))
            ]

        # interested = ["BUK1_20181013_023504"]
        # print("!!!!!!!!!!!!!!", interested)
        # temp = []
        # for each in self.all_csv_files:
        #     for interest in interested:
        #         if(interest not in each): continue
        #         else:
        #             temp.append(each)
        #             break
        # self.all_csv_files = temp

        self.all_csv_files = sorted(self.all_csv_files)

    def __len__(self):
        return len(self.all_csv_files)

    def __getitem__(self, idx):
        feat_file = self.all_csv_files[idx]
        # print(feat_file)
        (
            X_pos,
            X_neg,
            X_query,
            X_pos_neg,
            X_neg_neg,
            X_query_neg,
            max_len,
            neg_min_length,
            strt_index_query,
            audio_path,
            hop_seg,
            hop_seg_neg,
            mask,
            seg_len,
        ) = self.read_file(feat_file)
        return (
            (
                X_pos.astype(np.float32),
                X_neg.astype(np.float32),
                X_query.astype(np.float32),
                X_pos_neg.astype(np.float32),
                X_neg_neg.astype(np.float32),
                X_query_neg.astype(np.float32),
                hop_seg,
                hop_seg_neg,
                max_len,
                neg_min_length,
            ),
            strt_index_query,
            audio_path,
            seg_len,
        )

    def load_mask(self, class_name):
        if self.path.mask_dir is None:
            return None
        return np.load(os.path.join(self.path.mask_dir, "mask_%s.npy" % class_name))

    def find_positive_label(self, df):
        for col in df.columns:
            if "Q" in col or "E_" in col:
                return col.strip()
        else:
            raise ValueError(
                "Error: Expect you change the validation set event name to Q_x or E_x"
            )

    def segment(
        self,
        pcen,
        hop_seg,
        index_sup,
        start_time,
        end_time,
        seg_len,
        negative_onset_offset,
        negative_pcen=None,
    ):
        hop_neg = 0
        hop_query = 0
        strt_index = 0

        strt_indx_query = end_time[index_sup[-1]]
        end_idx_neg = pcen.shape[0] - 1

        feat_neg, feat_pos, feat_query = [], [], []
        ####################################BUILD NEG DATASET########################################
        # Segment out all the negative samples, and then re-segment them with a fixed segment length
        negative_segments = []
        # concatenate negative segments
        for s_neg, e_neg in negative_onset_offset:
            negative_segments.append(pcen[s_neg:e_neg])
        negative_segments = np.concatenate(negative_segments, axis=0)

        if negative_pcen is not None:
            negative_segments = np.concatenate(
                [negative_segments, negative_pcen], axis=0
            )
        # Re-segmentation
        while negative_segments.shape[0] - hop_neg > seg_len:
            patch_neg = negative_segments[int(hop_neg) : int(hop_neg + seg_len)]
            feat_neg.append(patch_neg)
            assert (hop_seg // 2) > 0
            hop_neg += hop_seg // 2  # TODO 4

        last_patch = negative_segments[-seg_len:]
        feat_neg.append(last_patch)
        #############################################################################################

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
                    assert hop_seg > 0
                    shift += hop_seg
                last_patch_pos = pcen[end_ind - seg_len : end_ind]
                feat_pos.append(last_patch_pos)
            else:
                patch_pos = pcen[str_ind:end_ind]
                if patch_pos.shape[0] == 0:
                    print("The patch is of 0 length", str_ind, end_ind)
                    continue
                repeat_num = int(seg_len / (patch_pos.shape[0])) + 1
                patch_new = np.tile(patch_pos, (repeat_num, 1))
                patch_new = patch_new[0 : int(seg_len)]
                feat_pos.append(patch_new)
        while end_idx_neg - (strt_indx_query + hop_query) > seg_len:
            patch_query = pcen[
                int(strt_indx_query + hop_query)
                - seg_len // 2 : int(strt_indx_query + hop_query + seg_len)
                - seg_len // 2
            ]
            feat_query.append(patch_query)
            assert hop_seg > 0
            hop_query += hop_seg

        last_patch_query = pcen[end_idx_neg - seg_len : end_idx_neg]
        feat_query.append(last_patch_query)
        return feat_pos, feat_neg, feat_query, strt_indx_query

    def negative_onset_offset_estimate(self, audio_path, start_time, end_time):
        x, sr = librosa.load(audio_path, sr=None)
        assert sr == 22050
        rms = librosa.feature.rms(
            y=x, frame_length=1024, hop_length=256
        )  # TODO hard code here
        rms = rms[0, ...]

        # Try range limit
        pos_segments = []
        for s, e in zip(start_time, end_time):
            pos_segments.append(rms[s:e])
        pos_segments = np.concatenate(pos_segments)
        min_val, max_val = np.min(pos_segments), np.max(pos_segments)

        mean_val = np.mean(pos_segments)
        mask = (rms < (min_val + mean_val) / 2) | (rms > max_val)

        if np.sum(mask) < 435:
            print("Warning: The negative segment found is too short", np.sum(mask))
            mask = rms < np.max(rms) / 6
        print("Mask length: ", np.sum(mask))
        return mask

    def negative_onset_offset_estimate_freq_mask(
        self, audio_path, key, start_time, end_time, neg_min_length, hop_seg
    ):
        from src.datamodules.components.sisnr import si_snr_new
        from scipy.signal import medfilt

        mask = np.load(os.path.join(self.path.mask_dir, "mask_%s.npy" % key))
        mel = np.load(audio_path.replace(".wav", "_mel.npy"))
        mel = mel * mask[None, ...]
        mel = (mel - np.mean(mel)) / (np.std(mel) + 1e-6)

        # Try range limit
        pos_segments = []
        for s, e in zip(start_time, end_time):
            pos_segments.append(np.mean(mel[s:e, :], axis=0, keepdims=True))
        pos_segments = np.concatenate(pos_segments, axis=0)
        anchor = np.mean(pos_segments, axis=0)
        mel = torch.tensor(mel)
        anchor = torch.tensor(anchor)
        score = []

        query_padding = hop_seg

        # if(query_padding > 87):
        #     query_padding = 87
        if query_padding < 2:
            query_padding = 2

        for i in range(mel.size(0)):
            start, end = i - query_padding, i + query_padding

            if start < 0:
                start = 0
            if end > mel.size(0) - 1:
                end = mel.size(0) - 1

            query = mel[start:end, :]
            query = torch.mean(query, dim=0)
            score.append(si_snr_new(query, anchor))
        score = medfilt(np.array(score), kernel_size=9)

        # Calculate scores
        pos_scores = []
        for s, e in zip(start_time, end_time):
            pos_scores.append(score[s:e])
        pos_scores = np.concatenate(pos_scores)
        pos_scores_min, pos_scores_max = np.min(pos_scores), np.max(pos_scores)
        threshold = pos_scores_min

        mask = score < threshold
        mask = self.remove_short_mask(mask, minlen=neg_min_length // 2)
        # import ipdb; ipdb.set_trace()
        pic_name = os.path.basename(audio_path.replace(".wav", ".png"))
        self.plot_score_and_mask(score, mask, name=pic_name)

        return np.array(mask)

    def plot_score_and_mask(self, scores, mask, name="temp.png"):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 1))
        plt.plot(mask[1200:2000])
        print(name)
        plt.savefig(name)
        plt.close()

        # plt.figure(figsize=(8, 5))
        # plt.subplot(211)
        # plt.plot(scores[1200:2000])
        # plt.subplot(212)
        # plt.plot(mask[1200:2000])
        # print(name)
        # plt.savefig(name+"123.png")
        # plt.close()

    def remove_short_mask(self, mask, minlen=2):
        """_summary_
        If negative, True; Remove short True
        data = [True, True, True, False, True,False, True, True, True, True,]
        """
        if minlen < 1:
            return mask

        ret = []
        cnt = 0
        for i in range(len(mask)):
            if mask[i] != True:
                if cnt != 0:
                    for i in range(cnt):
                        if cnt >= minlen:
                            ret.append(True)
                        else:
                            ret.append(False)
                    cnt = 0
                ret.append(False)
            else:
                cnt += 1
        if cnt != 0:
            for i in range(cnt):
                if cnt >= minlen:
                    ret.append(True)
                else:
                    ret.append(False)
            cnt = 0
        return ret

    def read_file(self, file):
        audio_path = file.replace("csv", "wav")
        df_eval = pd.read_csv(file, header=0, index_col=False)
        # Take the mask here
        key = self.find_positive_label(df_eval)
        mask = self.load_mask(key)

        Q_list = df_eval[key].to_numpy()

        index_sup = np.where(Q_list == "POS")[0][: self.train_param.n_shot]

        pcen = self.fe.extract_feature(audio_path)

        # Calculate negative segments
        start_time, end_time = time_2_frame(
            df_eval, self.fps, padding=0.0
        )  # When calculating negative samples we do not want to use padding in order to maximize the negative sample we can get
        start_times_neg = [int(start_time[index]) for index in index_sup]
        end_times_neg = [int(end_time[index]) for index in index_sup]
        negative_onset_offset = []
        negative_seg_length = []
        start = 0
        # The start and end of negative segments
        for (
            s,
            e,
        ) in zip(start_times_neg, end_times_neg):
            end = s
            if end > start:
                negative_onset_offset.append((start, end))
                negative_seg_length.append(end - start)
            else:
                print("Error: end and start", end, start)
            start = e

        #################################Adaptive hop_seg#########################################
        start_time, end_time = time_2_frame(df_eval, self.fps, padding=0.025)
        difference = []
        for index in index_sup:
            difference.append(end_time[index] - start_time[index])
        # Adaptive segment length based on the audio file.
        max_len = max(difference)
        # Choosing the segment length based on the maximum size in the 5-shot.
        # Logic was based on fitment on 12GB GPU since some segments are quite long.
        if max_len < 8:
            seg_len = 8
        elif max_len < self.test_seglen_len_lim:
            seg_len = max_len
        elif (
            max_len >= self.test_seglen_len_lim
            and max_len <= self.test_seglen_len_lim * 2
        ):
            seg_len = max_len // 2
        elif max_len > self.test_seglen_len_lim * 2 and max_len < 500:
            seg_len = max_len // 4
        else:
            seg_len = max_len // 8
        # print(f"Adaptive segment length for %s is {seg_len}" % (file))
        hop_seg = seg_len // self.test_hoplen_fenmu
        #################################################################################

        neg_min_length = np.min([x for x in negative_seg_length])
        if neg_min_length > 100:  # TODO hard code here
            hop_seg_neg = 50
            seg_len_neg = 100
        elif neg_min_length < 8:
            hop_seg_neg = 4
            seg_len_neg = 8
        else:
            hop_seg_neg = neg_min_length // 4
            seg_len_neg = neg_min_length

        labelled_negative_data_length = np.sum(
            [b - a for a, b in negative_onset_offset]
        )
        neg_pcen = None

        # We have too little negative data
        if labelled_negative_data_length < 435:  # TODO hard code here
            print("The negative part of the %s is too short" % file)
            if self.eval_param.negative_estimate == "rms":
                index = self.negative_onset_offset_estimate(
                    audio_path, start_time, end_time
                )
            elif self.eval_param.negative_estimate == "freq_mask":
                index = self.negative_onset_offset_estimate_freq_mask(
                    audio_path, key, start_time, end_time, neg_min_length, hop_seg
                )
            else:
                raise ValueError("Bad parameters", self.eval_param.negative_estimate)

            assert pcen.shape[0] == index.shape[0], (pcen.shape, index.shape)

            neg_pcen = pcen[end_time[index_sup[-1]] :][index[end_time[index_sup[-1]] :]]
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", neg_pcen.shape)
            if neg_pcen.shape[0] < 20:
                neg_pcen = None
        else:
            neg_pcen = None

        feat_pos, feat_neg, feat_query, strt_indx_query = self.segment(
            pcen,
            hop_seg,
            index_sup,
            start_time,
            end_time,
            seg_len,
            negative_onset_offset,
            negative_pcen=neg_pcen,
        )
        feat_pos_neg, feat_neg_neg, feat_query_neg, _ = self.segment(
            pcen,
            hop_seg_neg,
            index_sup,
            start_time,
            end_time,
            seg_len_neg,
            negative_onset_offset,
            negative_pcen=neg_pcen,
        )

        feat_pos = np.stack(feat_pos)
        feat_neg = np.stack(feat_neg)
        feat_query = np.stack(feat_query)
        feat_pos_neg = np.stack(feat_pos_neg)
        feat_neg_neg = np.stack(feat_neg_neg)
        feat_query_neg = np.stack(feat_query_neg)

        return (
            feat_pos,
            feat_neg,
            feat_query,
            feat_pos_neg,
            feat_neg_neg,
            feat_query_neg,
            max_len,
            neg_min_length,
            strt_indx_query,
            audio_path,
            hop_seg,
            hop_seg_neg,
            mask,
            seg_len,
        )  # [n, seg_len, 128]


if __name__ == "__main__":
    import torch
    from omegaconf import OmegaConf
    from tqdm import tqdm

    from src.datamodules.components.identity_sampler import IdentityBatchSampler

    data = []

    conf = OmegaConf.load("/vol/research/dcase2022/project/hhlab/configs/train.yaml")
    testset = PrototypeAdaSeglenBetterNegTestSetV2(
        conf.path, conf.features, conf.train_param, conf.eval_param
    )

    for (
        (
            X_pos,
            X_neg,
            X_query,
            X_pos_neg,
            X_neg_neg,
            X_query_neg,
            hop_seg,
            hop_seg_neg,
            max_len,
            neg_min_length,
        ),
        strt_index_query,
        audio_path,
        mask,
        _,
    ) in tqdm(testset):
        # import ipdb; ipdb.set_trace()
        pass
        # print(X_pos.shape, X_neg.shape, X_query.shape, X_pos_neg.shape, X_neg_neg.shape, X_query_neg.shape, hop_seg,hop_seg_neg, max_len, strt_index_query, audio_path)
