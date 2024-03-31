import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

from glob import glob
from tqdm import tqdm
import os
import argparse


EPS = 1e-8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--data-dir", default="/import/c4dm-datasets/jinhua-tmp2May/DCASE_2022_FSBioSED")

    return parser.parse_args()


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
    def __init__(self):
        """
        Extract various features from wav files and save them in .npy files
        """

        self.sr = 22050
        self.n_fft = 1024
        self.hop = 256
        self.n_mels = 128
        self.n_mfcc = 32
        self.fmax = 11025

    def norm(self, y):
        return y / np.max(np.abs(y))

    def mel(self, y):
        assert np.max(y) <= 1, np.max(y)
        mel_spec = librosa.feature.melspectrogram(
            y,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop,
            n_mels=self.n_mels,
            fmax=self.fmax,
        )
        # mel_spec = np.log(mel_spec + 1e-8)
        mel_spec = mel_spec.astype(np.float32)
        return mel_spec

    def logmel(self, mel_spec):
        mel_spec = np.log(mel_spec + 1e-8)
        mel_spec = mel_spec.astype(np.float32)
        return mel_spec

    def pcen(self, y):
        assert np.max(y) <= 1
        mel_spec = librosa.feature.melspectrogram(
            y * (2**32),
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop,
            n_mels=self.n_mels,
            fmax=self.fmax,
        )
        pcen = librosa.core.pcen(mel_spec, sr=self.sr)
        pcen = pcen.astype(np.float32)
        return pcen

    def rms(self, S):
        return librosa.feature.rms(S=S, frame_length=self.n_fft)

    def mfcc(self, mel):
        return librosa.feature.mfcc(S=mel)

    def spectral_centroid(self, S):
        return librosa.feature.spectral_centroid(S=S, sr=self.sr)

    def spectral_bandwidth(self, S):
        return librosa.feature.spectral_bandwidth(S=S, sr=self.sr)

    def spectral_contrast(self, S, n_bands=6):
        return librosa.feature.spectral_contrast(S=S, sr=self.sr, n_bands=n_bands)

    def spectral_flatness(self, S):
        return librosa.feature.spectral_flatness(S=S)

    def spectral_bandwidth(self, S):
        return librosa.feature.spectral_bandwidth(S=S, sr=self.sr)

    def spectral_rolloff(self, S, roll_percent=0.9):
        return librosa.feature.spectral_rolloff(
            S=S, sr=self.sr, roll_percent=roll_percent
        )

    def poly_features(self, S, order=1):
        return librosa.feature.poly_features(S=S, sr=self.sr, order=order)

    def zero_crossing_rate(self, y):
        assert np.max(y) <= 1
        return librosa.feature.zero_crossing_rate(
            y, frame_length=self.n_fft, hop_length=self.hop
        )

    def delta_mfcc(self, mfcc, order=1, width=9):
        return librosa.feature.delta(mfcc, order=order, width=width)

    def draw_spec(self, matrix, save_path=None, name="temp"):
        plt.imshow(matrix, aspect="auto", interpolation="none")
        plt.title(name)
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    def draw_plot(self, line, save_path=None, name="temp"):
        plt.plot(line)
        plt.title(name)
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    def visualize_result(self, result):
        for k in result.keys():
            if result[k].shape[1] == 1:
                self.draw_plot(result[k], name=k)
            elif result[k].shape[1] > 1:
                self.draw_spec(result[k], name=k)

    def extract_feature(self, audio_path):
        result = {}
        result["waveform"], _ = librosa.load(audio_path, sr=self.sr)
        result["waveform"] = self.norm(result["waveform"])

        result["spec"], result["phase"] = librosa.magphase(
            librosa.stft(result["waveform"], hop_length=256, n_fft=1024)
        )

        result["mel"] = self.mel(result["waveform"]).astype(np.float32)
        result["mel_un_normalized"] = result["mel"]
        result["logmel"] = self.logmel(result["mel"]).astype(np.float32)
        result["pcen"] = self.pcen(result["waveform"]).astype(np.float32)
        result["mfcc"] = self.mfcc(result["logmel"]).astype(np.float32)
        # result["rms"] = self.rms(result["spec"]).astype(np.float32)
        # result["spectral_centroid"] = self.spectral_centroid(result["spec"]).astype(np.float32)
        # result["spectral_bandwidth"] = self.spectral_bandwidth(result["spec"]).astype(np.float32)
        # result["spectral_contrast"] = self.spectral_contrast(result["spec"], n_bands=6).astype(np.float32)
        # result["spectral_flatness"] = self.spectral_flatness(result["spec"]).astype(np.float32)
        # result["spectral_bandwidth"] = self.spectral_bandwidth(result["spec"]).astype(np.float32)
        # result["spectral_rolloff"] = self.spectral_rolloff(result["spec"]).astype(np.float32)
        # result["poly_features"] = self.poly_features(result["spec"], order=1).astype(np.float32)
        # result["zero_crossing_rate"] = self.zero_crossing_rate(result["waveform"]).astype(np.float32)
        result["delta_mfcc"] = self.delta_mfcc(result["mfcc"], order=1, width=9).astype(
            np.float32
        )

        # result["mel"] = result["mel"].T
        # result["mfcc"] = result["mfcc"].T

        # del result['mel'] # TODO

        # del result['spec']
        del result["phase"]
        del result["waveform"]

        for k in result.keys():
            result[k] = result[k].T

        # return {"mel_un_normalized": result["mel_un_normalized"]}
        return result


def process(fpath):
    print(fpath)
    error_files = []
    for file in tqdm(fpath):
        try:
            features = fe.extract_feature(file)
            for k in features.keys():
                npy_path = file.replace(".wav", "_%s.npy" % k)
                np.save(npy_path, features[k])
        except:
            os.remove(file)
            error_files.append(file)
            continue
    print("Encounter error in these files:", error_files)


def calculate_feature_mean_std(A):
    N = max(float(sum([i.size for i in A])), EPS)
    mean_ = sum([i.sum() for i in A]) / N
    std_ = np.sqrt(sum([((i - mean_) ** 2).sum() for i in A]) / N)
    print("Mean %s; Std %s;" % (mean_, std_))
    return mean_, std_


def main(data_path: str):
    r"""Prepare features for further developement."""
    PATH = data_path

    r"""Data Preparation"""
    print(f"Checking dataset in the local...")
    if not os.path.exists(PATH):
        zipfile_path = os.path.join(*PATH.split("/")[:-1], "Development_Set.zip")
        print(f"Downloading data to {zipfile_path}")
        os.system(f"wget https://zenodo.org/record/6482837/files/Development_Set.zip?download=1 -O {zipfile_path}")
        os.system(f"unzip {zipfile_path}")
    print(f"Dataset is now ready!")
    
    
    r"""Feature extraction"""
    features = ["mel", "logmel", "pcen", "mfcc", "delta_mfcc"]
    suffix = ".wav"
    print(f"Extracting features: {features}")
    # SAMPLE_RATE = 22050
    fe = Feature_Extractor()
    files = recursive_glob(PATH, suffix)

    process(files)

    files = recursive_glob(PATH, ".wav")


    r"""Feature normalization"""
    print("Preparing the normalized features...")
    for k in features:
        if "un_normalized" in features:
            continue
        print(k)
        array_list = []
        for file in tqdm(files):
            npy_path = file.replace(".wav", "_%s.npy" % k)
            array = np.load(npy_path)
            array_list.append(array)
            # try:
            #     npy_path = file.replace(".wav", "_%s.npy" % k)
            #     array = np.load(npy_path)
            #     array_list.append(array)
            # except:
            #     print("no such file", file)
            #     continue
        mean, std = calculate_feature_mean_std(array_list)
        for file in tqdm(files):
            try:
                npy_path = file.replace(".wav", "_%s.npy" % k)
                array = np.load(npy_path)
                array = (array - mean) / std
                np.save(npy_path, array)
                array_list.append(array)
            except:
                print("no such file", file)
                continue
        del array_list
    print("Features normalization is finished!")


if __name__ == "__main__":
    flags = parse_args()
    data_path = os.path.join(flags.data_dir, "Development_Set")
    main(data_path)