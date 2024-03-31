import numpy as np
import os
from glob import glob
from tqdm import tqdm
import torch


def recursive_glob(path, suffix):
    return (
        glob(os.path.join(path, "*" + suffix))
        + glob(os.path.join(path, "*/*" + suffix))
        + glob(os.path.join(path, "*/*/*" + suffix))
        + glob(os.path.join(path, "*/*/*/*" + suffix))
        + glob(os.path.join(path, "*/*/*/*/*" + suffix))
        + glob(os.path.join(path, "*/*/*/*/*/*" + suffix))
    )


class Feature_Extractor:
    mean_std = {}

    def __init__(self, features, audio_path=[]):
        self.sr = features.sr
        self.n_fft = features.n_fft
        self.hop = features.hop_mel
        self.n_mels = features.n_mels
        self.fmax = features.fmax
        self.feature_types = features.feature_types.split("@")
        self.files = []
        for each in audio_path:
            if(each is not None):
                assert os.path.exists(each), "Path not found: %s" % each
            print("Looking for data in: ", os.path.abspath(each))
            data = list(recursive_glob(each, ".wav"))
            self.files += data
            print("Find %s audio files" % (len(data)))

        self.files = np.random.permutation(self.files)
        self.update_mean_std()
        self.feature_lens = []

    def update_mean_std(self, feature_types=None):
        if len(list(Feature_Extractor.mean_std.keys())) != 0:
            return
        print("Calculating mean and std")
        for suffix in self.feature_types if (feature_types is None) else feature_types:
            print("Calculating: ", suffix)
            features = []
            for audio_path in tqdm(self.files[:1000]):
                feature_path = audio_path.replace(".wav", "_%s.npy" % suffix)
                features.append(np.load(feature_path).flatten())
            all_data = np.concatenate(features)
            Feature_Extractor.mean_std[suffix] = [np.mean(all_data), np.std(all_data)]
        print(Feature_Extractor.mean_std)

    def extract_feature(self, audio_path, feature_types=None, normalized=True):
        features = []
        for suffix in self.feature_types if (feature_types is None) else feature_types:
            feature_path = audio_path.replace(".wav", "_%s.npy" % suffix)
            if not normalized:
                features.append(np.load(feature_path))
            else:
                mean, std = Feature_Extractor.mean_std[suffix]
                features.append((np.load(feature_path) - mean) / std)
            self.feature_lens.append(features[-1].shape[1])
        return np.concatenate(features, axis=1)

    # def apply_mask(self, features, mask):
    #     start = 0
    #     import ipdb; ipdb.set_trace()
    #     for len, suffix in zip(self.feature_lens, self.feature_types):
    #         if(len != 128):
    #             continue
    #         else:
    #             mean, std = Feature_Extractor.mean_std[suffix]
    #             features[:, start:start +len] = (features[:, start:start +len] * std + mean) * mask
    #             features[:, start:start +len] = (features[:, start:start +len] - mean) / std
    #         start += len
