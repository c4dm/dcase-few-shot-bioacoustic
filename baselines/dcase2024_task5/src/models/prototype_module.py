import sys

sys.path.append("/vol/research/dcase2022/project/t5_open_source/DCASE_2022_Task_5")
from email.mime import audio
from re import L
from typing import Any, List

import numpy as np
import pandas as pd
from sqlalchemy import over
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from tqdm import tqdm
import librosa
import pandas as pd
import wandb
from scipy.signal import chirp, find_peaks, peak_widths, peak_prominences
import torch.nn.functional as F

from src.datamodules.components.batch_sampler import *
from src.datamodules.components.Datagenerator import *
from src.models.components.simple_dense_net import SimpleDenseNet
import pickle

# Jinhua: pinpoint the dir of eval meta
import os
_FOLDER_PATH_ = os.path.dirname(__file__)

def save_pickle(obj, fname):
    print("Save pickle at " + fname)
    with open(fname, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(fname):
    print("Load pickle at " + fname)
    with open(fname, "rb") as f:
        res = pickle.load(f)
    return res


class PrototypeModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        train: dict = {},
        path: dict = {},
        eval: dict = {},
        features: dict = {},
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.net = net
        # wandb.init()
        import torch.nn as nn

        self.name_arr = np.array([])
        self.onset_arr = np.array([])
        self.offset_arr = np.array([])
        self.max_len_arr = np.array([])
        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.fps = features.sr / features.hop_mel
        self.result = {}
        self.negative_result = {}

        self.onset_offset = {}

    def forward(self, x: torch.Tensor):
        # with torch.no_grad():
        # x = self.net(x)
        # return self.post_net(x)
        return self.net(x)

    def step(self, batch, batch_idx):
        if self.hparams.train.negative_train_contrast:
            (
                x,
                x_neg,
                y,
                y_neg,
                class_name,
            ) = batch  # torch.Size([50, 17, 128]), torch.Size([50])
            x = torch.cat([x, x_neg], dim=0)
            y = torch.cat([y, y_neg], dim=0)
        else:
            x, y, class_name = batch
        x_out = self.forward(x)

        if self.hparams.train.negative_train_contrast:
            from src.utils.loss import prototypical_loss_filter_negative as loss_fn
        else:
            from src.utils.loss import prototypical_loss as loss_fn

        tr_loss, tr_acc, tr_supcon = loss_fn(x_out, y, self.hparams.train.n_shot)
        return tr_loss, tr_acc, tr_supcon

    def training_step(self, batch: Any, batch_idx: int):
        tr_loss, tr_acc, tr_supcon = self.step(batch, batch_idx)
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", cur_lr, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/loss", tr_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", tr_acc, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/supcon", tr_supcon, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": tr_loss + tr_supcon, "acc": tr_acc}

    def validation_step(self, batch: Any, batch_idx: int):
        val_loss, val_acc, val_supcon = self.step(batch, batch_idx)
        self.log("val/loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/acc", val_acc, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/supcon", val_supcon, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": val_loss + val_supcon, "val_acc": val_acc}

    def test_step(self, batch: Any, batch_idx: int):
        ###############################PREPROCESS DATA###########################################
        def transform(a, b, c):
            # a,b,c = self.feature_scale_eval(a), self.feature_scale_eval(b), self.feature_scale_eval(c)
            return a[0, ...], b[0, ...], c[0, ...]

        (
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
            audio_name,
            seg_len,
        ) = batch
        X_pos, X_neg, X_query = transform(X_pos, X_neg, X_query)
        X_pos_neg, X_neg_neg, X_query_neg = transform(X_pos_neg, X_neg_neg, X_query_neg)
        hop_seg, hop_seg_neg, strt_index_query, max_len, neg_min_length, seg_len = (
            hop_seg[0, ...].item(),
            hop_seg_neg[0, ...].item(),
            strt_index_query[0, ...].item(),
            max_len[0, ...].item(),
            neg_min_length[0, ...].item(),
            seg_len[0, ...].item(),
        )
        ##########################################################################################
        # onset, offset = self.evaluate_prototypes(X_pos, X_neg, X_query, hop_seg, strt_index_query, mask)
        # onset_neg, offset_neg = self.evaluate_prototypes(X_pos_neg, X_neg_neg, X_query_neg, hop_seg_neg, strt_index_query, mask)
        padding_len = seg_len // 2
        onset_offset = self.evaluate_prototypes(
            X_pos, X_neg, X_query, hop_seg, strt_index_query, audio_name[0]
        )

        if self.hparams.train.negative_seg_search:
            onset_offset_neg = self.evaluate_prototypes(
                X_pos_neg,
                X_neg_neg,
                X_query_neg,
                hop_seg_neg,
                strt_index_query,
                audio_name[0],
            )

        audio_name = os.path.basename(audio_name[0])

        for k in onset_offset.keys():
            if k not in self.onset_offset.keys():
                self.onset_offset[k] = {}
                self.onset_offset[k]["name_arr"] = np.array([])
                self.onset_offset[k]["onset_arr"] = np.array([])
                self.onset_offset[k]["offset_arr"] = np.array([])

            if self.hparams.train.negative_seg_search:
                neg_onset_offset = []
                start = 0
                for on, off in zip(onset_offset_neg[k][0], onset_offset_neg[k][1]):
                    end = on
                    neg_onset_offset.append((start, end))
                    start = off

            # Use the detected negative samples to perform post-processing (splitting)
            if self.hparams.train.negative_seg_search:
                onset_offset[k][0], onset_offset[k][1] = self.splitting_segment(
                    [(a, b) for a, b in zip(onset_offset[k][0], onset_offset[k][1])],
                    neg_onset_offset,
                    max_len,
                )

            # Remove long segment
            if self.hparams.train.remove_long_segment:
                onset_offset[k][0], onset_offset[k][1] = self.remove_long_segment(
                    [(a, b) for a, b in zip(onset_offset[k][0], onset_offset[k][1])],
                    neg_min_length,
                    max_len,
                )

            # Use the detected negative samples to perform post-processing (merging)
            if self.hparams.train.merging_segment:
                onset_offset[k][0], onset_offset[k][1] = self.merging_segment(
                    [(a, b) for a, b in zip(onset_offset[k][0], onset_offset[k][1])],
                    neg_min_length,
                    max_len,
                )

            # Remove long segment
            if self.hparams.train.remove_long_segment:  # TODO
                onset_offset[k][0], onset_offset[k][1] = self.remove_long_segment(
                    [(a, b) for a, b in zip(onset_offset[k][0], onset_offset[k][1])],
                    neg_min_length,
                    max_len,
                )

            # Padding Tail
            if self.hparams.train.padd_tail:  # TODO
                onset_offset[k][0], onset_offset[k][1] = self.padding_tail(
                    [(a, b) for a, b in zip(onset_offset[k][0], onset_offset[k][1])],
                    padding_len / self.fps,
                )

            name = np.repeat(audio_name, len(onset_offset[k][0]))

            self.onset_offset[k]["name_arr"] = np.append(
                self.onset_offset[k]["name_arr"], name
            )
            self.onset_offset[k]["onset_arr"] = np.append(
                self.onset_offset[k]["onset_arr"], onset_offset[k][0]
            )
            self.onset_offset[k]["offset_arr"] = np.append(
                self.onset_offset[k]["offset_arr"], onset_offset[k][1]
            )

        import time; time.sleep(0.1)  # TODO: remove it if CPU resource is not bottleneck
        return {}

    def convert_single_file(self, file_path, save_path):
        def get_class(fname):
            fname = os.path.basename(fname)
            if "ME" in fname:
                return "ME"
            elif "BUK" in fname:
                return "PB"
            else:
                return "HB"

        def generate_a_line(class_name, start, end, filename):
            return "%s\t%s\t%s\t%s\n" % (class_name, start, end, filename)

        content = "event_label\tonset\toffset\tfilename\n"
        raw_result = pd.read_csv(file_path)

        for i, row in raw_result.iterrows():
            fname, start, end = row["Audiofilename"], row["Starttime"], row["Endtime"]
            # class_name = "VAL@%s" % (os.path.splitext(os.path.basename(csvfile))[0])
            class_name = "VAL@%s" % (get_class(fname))
            line = generate_a_line(
                class_name=class_name,
                start=start,
                end=end,
                filename=os.path.basename(fname).replace(".wav", ".csv"),
            )
            content = content + line

        with open(save_path, "w") as f:
            f.write(content)

    def convert_eval_val(self):
        from glob import glob

        for file in glob("*/Eval_VAL_*.csv"):
            self.convert_single_file(
                file,
                os.path.join(os.path.dirname(file), "PSDS_" + os.path.basename(file)),
            )

    def calculate_psds(self):
        from glob import glob
        from psds_eval import PSDSEval, plot_psd_roc, plot_per_class_psd_roc

        dtc_threshold = 0.5
        gtc_threshold = 0.5
        cttc_threshold = 0.3
        alpha_ct = 0.0
        alpha_st = 0.0
        max_efpr = 100
        ground_truth_csv = os.path.join(
            _FOLDER_PATH_, "eval_meta/subset_gt.csv"
        )
        metadata_csv = os.path.join(
            _FOLDER_PATH_, "eval_meta/subset_meta.csv"
        )
        gt_table = pd.read_csv(ground_truth_csv, sep="\t")
        meta_table = pd.read_csv(metadata_csv, sep="\t")
        psds_eval = PSDSEval(
            dtc_threshold,
            gtc_threshold,
            cttc_threshold,
            ground_truth=gt_table,
            metadata=meta_table,
        )
        for file in glob("*/PSDS_Eval_*.csv"):
            det_t = pd.read_csv(os.path.join(file), sep="\t")
            psds_eval.add_operating_point(det_t)
        psds = psds_eval.psds(alpha_ct, alpha_st, max_efpr)
        print(f"\nPSDS-Score: {psds.value:.5f}")
        print("Saving pickle!")
        save_pickle(psds, "psds.pkl")
        plot_psd_roc(psds, filename="roc.png")
        tpr_vs_fpr, _, tpr_vs_efpr = psds_eval.psd_roc_curves(alpha_ct=alpha_ct)
        plot_per_class_psd_roc(
            tpr_vs_fpr,
            psds_eval.class_names,
            title="Per-class TPR-vs-FPR PSDROC",
            xlabel="FPR",
            filename="per_class_1.png",
        )
        save_pickle(tpr_vs_fpr, "tpr_vs_fpr.pkl")
        save_pickle(psds_eval.class_names, "class_names.pkl")
        plot_per_class_psd_roc(
            tpr_vs_efpr,
            psds_eval.class_names,
            title="Per-class TPR-vs-eFPR PSDROC",
            xlabel="eFPR",
            filename="per_class_2.png",
        )
        save_pickle(tpr_vs_efpr, "tpr_vs_efpr.pkl")
        self.log("psds", psds.value)

    def test_epoch_end(self, outputs: List[Any]):
        # self.split_long_segments_based_on_energy() # TODO checkout if you need this function
        best_result = None
        best_f_measure = 0.0
        for k in self.onset_offset.keys():
            # import ipdb; ipdb.set_trace()
            df_out = pd.DataFrame(
                {
                    "Audiofilename": [
                        os.path.basename(x) for x in self.onset_offset[k]["name_arr"]
                    ],
                    "Starttime": self.onset_offset[k]["onset_arr"],
                    "Endtime": self.onset_offset[k]["offset_arr"],
                }
            )
            os.makedirs(str(k), exist_ok=False)
            csv_path = "%s/Eval_raw.csv" % k
            df_out.to_csv(csv_path, index=False)
            # Postprocessing and evaluate
            res = self.post_process(alpha=k)
            res_new = self.post_process_new(alpha=k)

            if res["fmeasure"] > best_f_measure:
                best_result = res
                best_f_measure = res["fmeasure"]
            if res_new["fmeasure"] > best_f_measure:
                best_result = res_new
                best_f_measure = res_new["fmeasure"]
            # self.post_process_test(alpha=k)
            # self.post_process_new_test(alpha=k)
        print("Best Result: ")
        print(best_result)
        for k in best_result:
            self.log(str(k), best_result[k])
        # New evaluation method
        self.convert_eval_val()
        self.calculate_psds()

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optim = torch.optim.Adam(
            [{"params": self.net.parameters()}], lr=self.hparams.train.lr_rate
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optim,
            gamma=self.hparams.train.scheduler_gamma,
            step_size=self.hparams.train.scheduler_step_size,
        )
        return [optim], [lr_scheduler]

    # def merging_segment(self, pos_onset_offset, neg_min_length, max_len):
    #     onset, offset = [], []
    #     i = 0
    #     while(i < len(pos_onset_offset)):

    #         if(i >= len(pos_onset_offset) - 1): return onset, offset

    #         on, off = pos_onset_offset[i]
    #         on_next, off_next = pos_onset_offset[i+1]
    #         # on_next_next, off_next_next = pos_onset_offset[i+2]

    #         # if(is_all_negative(on, off)): continue
    #         # Divide big segment into small segments
    #         if((off_next-on) * self.fps < 1.2*max_len or (on_next-off) * self.fps < 0.8 * neg_min_length):
    #             onset.append(on)
    #             offset.append(off_next)
    #             while((off_next-on) * self.fps < 1.2*max_len or (on_next-off) * self.fps < 0.8 * neg_min_length):
    #                 i += 1
    #                 offset[-1] = off_next
    #                 off = off_next
    #                 on_next, off_next = pos_onset_offset[i+1]
    #             print("merge", (off_next-on) * self.fps, (on_next-off) * self.fps, max_len, neg_min_length)

    #         # Do not change this segment
    #         else:
    #             onset.append(on)
    #             offset.append(off)
    #         i += 1

    #     return onset, offset

    def merging_segment(self, pos_onset_offset, neg_min_length, max_len):
        onset, offset = [], []
        i = 0
        # fps = 1
        while i <= len(pos_onset_offset) - 1:  # TODO
            if i >= len(pos_onset_offset) - 1:
                # TODO
                on, off = pos_onset_offset[i]
                onset.append(on)
                offset.append(off)
                return onset, offset

            on, off = pos_onset_offset[i]
            on_next, off_next = pos_onset_offset[i + 1]

            if (off_next - on) * self.fps < max_len or (
                on_next - off
            ) * self.fps < 0.5 * neg_min_length:
                onset.append(on)
                offset.append(off_next)
                while (off_next - on) * self.fps < max_len or (
                    on_next - off
                ) * self.fps < 0.5 * neg_min_length:
                    i += 1
                    if not i < len(pos_onset_offset) - 1:
                        break  # TODO
                    offset[-1] = off_next
                    off = off_next
                    on_next, off_next = pos_onset_offset[i + 1]
                print(
                    "merge",
                    (off_next - on) * self.fps,
                    (on_next - off) * self.fps,
                    max_len,
                    neg_min_length,
                )
            # Do not change this segment
            else:
                onset.append(on)
                offset.append(off)
            i += 1

        return onset, offset

    def remove_long_segment(self, pos_onset_offset, neg_min_length, max_len):
        onset, offset = [], []
        i = 0
        while i < len(pos_onset_offset):
            if i >= len(pos_onset_offset) - 1:
                break
            on, off = pos_onset_offset[i]
            if (off - on) * self.fps > 2 * max_len:
                print(
                    "-remove",
                    (off - on) * self.fps,
                    max_len,
                    (off - on) * self.fps / (2 * max_len),
                )
            else:
                onset.append(on)
                offset.append(off)
            i += 1
        return onset, offset

    def padding_tail(self, pos_onset_offset, padding_len):
        onset, offset = [], []
        i = 0
        while i < len(pos_onset_offset):
            if i == 0:
                i += 1
                continue
            if i >= len(pos_onset_offset) - 1:
                break

            prev_on, prev_off = pos_onset_offset[i - 1]
            on, off = pos_onset_offset[i]
            next_on, next_off = pos_onset_offset[i + 1]

            if (
                next_on - off > 0.1 + 2 * padding_len
                and on - prev_off > 0.1 + 2 * padding_len
            ):
                onset.append(on - padding_len)
                offset.append(off + padding_len)
                print("++padding", on, off, padding_len)
            else:
                onset.append(on)
                offset.append(off)
            i += 1
        return onset, offset

    def splitting_segment(self, pos_onset_offset, neg_onset_offset, max_len):
        def merge(_onset, _offset):
            """Devide the predicted positive segment into smaller ones based on negative segment estimation

            Args:
                _onset (_type_): on set of one predicted positive segment
                _offset (_type_): off set of one predicted positive segment

            Returns:
                _type_: _description_
            """
            ret_onset, ret_offset = [], []
            breakpoints = []
            for neg_on, neg_off in neg_onset_offset:
                if neg_on > _onset and neg_off < _offset:
                    breakpoints.append((neg_on, neg_off))
                elif neg_on > _offset:
                    break
            _on = _onset
            for break_on, break_off in breakpoints:
                _off = break_on
                ret_onset.append(_on)
                ret_offset.append(_off)
                _on = break_off
            return ret_onset, ret_offset

        def is_all_negative(_onset, _offset):
            """(Abandoned) If segment is classified as negative, remove this segment

            Args:
                _onset (_type_): _description_
                _offset (_type_): _description_

            Returns:
                _type_: _description_
            """
            for neg_on, neg_off in neg_onset_offset:
                if neg_on < _onset and neg_off > _offset:
                    print("all negative")
                    return True
                elif neg_on > _offset:
                    break
            return False

        onset, offset = [], []
        for on, off in pos_onset_offset:
            # if(is_all_negative(on, off)): continue
            # Divide big segment into small segments
            if (off - on) * self.fps > max_len * 2:  # TODO hard code here
                print(
                    "> splitting",
                    (off - on) * self.fps,
                    max_len * 2,
                    (off - on) * self.fps / (max_len * 2),
                )
                new_onset, new_offset = merge(on, off)
                onset.extend(new_onset)
                offset.extend(new_offset)
            # Do not change this segment
            else:
                onset.append(on)
                offset.append(off)
        return onset, offset

    def split_long_segments_based_on_energy(self):
        print("Splitting long segmetns!!!!!!!")
        name_arr_temp, onset_arr_temp, offset_arr_temp = [], [], []
        for i in range(self.name_arr.shape[0]):
            name, max_len, onset, offset = (
                self.name_arr[i],
                self.max_len_arr[i],
                self.onset_arr[i],
                self.offset_arr[i],
            )
            # Split long segment based on their energy
            if "ML_126376" in name and (offset - onset) * self.fps > 2 * max_len:
                print((offset - onset) * self.fps / max_len)
                x, sr = librosa.load(
                    name, offset=onset, duration=offset - onset, sr=None
                )
                rms = librosa.feature.rms(y=x)
                rms = rms / np.max(rms)
                rms = rms[0, ...]
                peaks, _ = find_peaks(rms)
                _, weight, start, end = peak_widths(rms, peaks, rel_height=0.5)

                start *= 512 / sr
                end *= 512 / sr
                start += onset
                end += onset

                threshold = np.max(weight) / 2
                start = start[weight > threshold]
                end = end[weight > threshold]
                start[0] = onset
                end[-1] = offset
                for s, e in zip(start, end):
                    name_arr_temp.append(name)
                    onset_arr_temp.append(s)
                    offset_arr_temp.append(e)
            else:
                name_arr_temp.append(name)
                onset_arr_temp.append(onset)
                offset_arr_temp.append(offset)
        self.name_arr, self.onset_arr, self.offset_arr = (
            np.array(name_arr_temp),
            np.array(onset_arr_temp),
            np.array(offset_arr_temp),
        )

    def log_result(self, overall_scores, scores_per_set, scores_per_audiofile, name):
        self.log_final_result(overall_scores, name)
        self.log_result_for_each_set(scores_per_set, name)
        self.log_result_for_each_audio_file(scores_per_audiofile, name)

    def log_final_result(self, overall_scores, name="test"):
        for v in overall_scores.keys():
            overall_scores[v] = overall_scores[v].item()
        for k in overall_scores.keys():
            self.log("%s-overall_scores/%s" % (name, k), overall_scores[k])

    def log_result_for_each_set(self, scores_per_set, name="test"):
        cache = {}
        for dataset in scores_per_set.keys():
            for k in scores_per_set[dataset].keys():
                cache["%s/%s" % (k, dataset)] = scores_per_set[dataset][k]
        for k in cache.keys():
            self.log("%s-each_set_scores/%s" % (name, k), cache[k])

    def log_result_for_each_audio_file(self, scores_per_audio_file, name="test"):
        cache = {}
        for audiofile in scores_per_audio_file.keys():
            for k in scores_per_audio_file[audiofile].keys():
                cache[
                    "%s/%s" % (k, os.path.basename(audiofile))
                ] = scores_per_audio_file[audiofile][k]
        for k in cache.keys():
            self.log("%s-each_audiofile/%s" % (name, k), cache[k])

    def post_process_test(self, dataset="TEST", alpha=0.9):
        from src.utils.evaluation import evaluate
        from src.utils.post_proc import post_processing

        test_path = self.hparams.path.test_dir
        if test_path[-1] != "/":
            test_path += "/"

        evaluation_file = "%s/Eval_raw.csv" % alpha
        save_path = "%s" % alpha

        best_result = None
        for threshold in np.arange(0.1, 0.9, 0.1):
            print("Threshold %s" % threshold)
            team_name = "Baseline" + str(threshold)
            new_evaluation_file = "%s/Eval_%s_threshold_ada_postproc_%s.csv" % (
                alpha,
                dataset,
                threshold,
            )
            post_processing(
                test_path, evaluation_file, new_evaluation_file, threshold=threshold
            )

    def post_process_new_test(self, dataset="TEST", alpha=0.9):
        from src.utils.evaluation import evaluate
        from src.utils.post_proc_new import post_processing

        test_path = self.hparams.path.test_dir
        if test_path[-1] != "/":
            test_path += "/"

        evaluation_file = "%s/Eval_raw.csv" % alpha
        save_path = "%s" % alpha

        best_result = None
        for threshold_length in np.arange(0.05, 0.25, 0.05):
            team_name = "Baseline" + str(threshold_length)
            print("Threshold length %s" % threshold_length)
            new_evaluation_file = "%s/Eval_%s_threshold_fix_length_postproc_%s.csv" % (
                alpha,
                dataset,
                threshold_length,
            )
            post_processing(
                test_path,
                evaluation_file,
                new_evaluation_file,
                threshold_length=threshold_length,
            )

    def post_process(self, dataset="VAL", alpha=0.9):
        from src.utils.evaluation import evaluate
        from src.utils.post_proc import post_processing

        val_path = self.hparams.path.eval_dir
        if val_path[-1] != "/":
            val_path += "/"

        evaluation_file = "%s/Eval_raw.csv" % alpha
        save_path = "%s" % alpha

        print("Before preprocessing: ")
        team_name = "Baseline_unprocessed"

        (
            overall_scores,
            individual_file_result,
            scores_per_set,
            scores_per_audiofile,
        ) = evaluate(evaluation_file, val_path, team_name, dataset, save_path)

        # if(alpha == 0.9):
        #     self.log("fmeasure_no_postprocessing", overall_scores["fmeasure"])
        #     wandb.log({"fmeasure_no_postprocessing":overall_scores["fmeasure"]})

        self.log_result(
            overall_scores,
            scores_per_set=scores_per_set,
            scores_per_audiofile=scores_per_audiofile,
            name="No_Post",
        )

        best_result = None
        for threshold in np.arange(0.2, 0.6, 0.1):
            print("Threshold %s" % threshold)
            team_name = "Baseline" + str(threshold)
            new_evaluation_file = "%s/Eval_%s_threshold_ada_postproc_%s.csv" % (
                alpha,
                dataset,
                threshold,
            )
            post_processing(
                val_path, evaluation_file, new_evaluation_file, threshold=threshold
            )
            (
                overall_scores,
                individual_file_result,
                scores_per_set,
                scores_per_audiofile,
            ) = evaluate(new_evaluation_file, val_path, team_name, dataset, save_path)
            if (
                best_result is None
                or best_result[0]["fmeasure"] < overall_scores["fmeasure"]
            ):
                best_result = (
                    overall_scores,
                    individual_file_result,
                    threshold,
                    scores_per_set,
                    scores_per_audiofile,
                )
        print("******************BEST RESULT*****************")
        for k in best_result[1].keys():
            print(k, best_result[1][k])
        print(best_result[0], best_result[2])
        self.log_result(
            best_result[0],
            best_result[3],
            best_result[4],
            name="proc_thresh_minlen_%.2f" % (threshold),
        )
        return best_result[0]

    def post_process_new(self, dataset="VAL", alpha=0.9):
        from src.utils.evaluation import evaluate
        from src.utils.post_proc_new import post_processing

        val_path = self.hparams.path.eval_dir
        if val_path[-1] != "/":
            val_path += "/"

        evaluation_file = "%s/Eval_raw.csv" % alpha
        save_path = "%s" % alpha

        best_result = None
        for threshold_length in np.arange(0.05, 0.25, 0.05):
            team_name = "Baseline" + str(threshold_length)
            print("Threshold length %s" % threshold_length)
            new_evaluation_file = "%s/Eval_%s_threshold_fix_length_postproc_%s.csv" % (
                alpha,
                dataset,
                threshold_length,
            )
            post_processing(
                val_path,
                evaluation_file,
                new_evaluation_file,
                threshold_length=threshold_length,
            )
            (
                overall_scores,
                individual_file_result,
                scores_per_set,
                scores_per_audiofile,
            ) = evaluate(new_evaluation_file, val_path, team_name, dataset, save_path)
            if (
                best_result is None
                or best_result[0]["fmeasure"] < overall_scores["fmeasure"]
            ):
                best_result = (
                    overall_scores,
                    individual_file_result,
                    threshold_length,
                    scores_per_set,
                    scores_per_audiofile,
                )
        print("******************BEST RESULT*****************")
        for k in best_result[1].keys():
            print(k, best_result[1][k])
        print(best_result[0], best_result[2])

        # self.log_final_result(best_result[0], name="proc_by_length_%.2f" % (threshold_length))
        self.log_result(
            best_result[0],
            best_result[3],
            best_result[4],
            name="proc_by_length_%.2f" % (threshold_length),
        )
        return best_result[0]

    def get_probability(self, proto_pos, neg_proto, query_set_out):

        """Calculates the  probability of each query point belonging to either the positive or negative class
        Args:
        - x_pos : Model output for the positive class
        - neg_proto : Negative class prototype calculated from randomly chosed 100 segments across the audio file
        - query_set_out:  Model output for the first 8 samples of the query set

        Out:
        - Probabiility array for the positive class
        """

        prototypes = torch.stack([proto_pos, neg_proto]).squeeze(1)
        dists = self.euclidean_dist(query_set_out, prototypes)
        # dists = self.cosine_dist(query_set_out, prototypes)
        """  Taking inverse distance for converting distance to probabilities"""
        logits = -dists

        prob = torch.softmax(logits, dim=1)
        inverse_dist = torch.div(1.0, dists)

        # prob = torch.softmax(inverse_dist,dim=1)
        """  Probability array for positive class"""
        prob_pos = prob[:, 0]

        return prob_pos.detach().cpu().tolist()

    def get_probability_old(self, x_pos, neg_proto, query_set_out):
        """Calculates the  probability of each query point belonging to either the positive or negative class
        Args:
        - x_pos : Model output for the positive class
        - neg_proto : Negative class prototype calculated from randomly chosed 100 segments across the audio file
        - query_set_out:  Model output for the first 8 samples of the query set

        Out:
        - Probabiility array for the positive class
        """
        pos_prototype = x_pos.mean(0)
        prototypes = torch.stack([pos_prototype, neg_proto])
        dists = self.euclidean_dist(query_set_out, prototypes)
        # dists = self.cosine_dist(query_set_out,prototypes)
        """  Taking inverse distance for converting distance to probabilities"""
        inverse_dist = torch.div(1.0, dists)
        prob = torch.softmax(inverse_dist, dim=1)
        """  Probability array for positive class"""
        prob_pos = prob[:, 0]
        return prob_pos.detach().cpu().tolist()

    # def concate_mask(self, x, mask):
    #     import ipdb; ipdb.set_trace()
    #     return x

    def concate_mask(self, x, mask):
        # mask: [1, 128]
        # x: [1, 60, 148]
        pad_length = x.size(2) - mask.size(1)
        mask = F.pad(mask, (0, pad_length))
        mask = mask.unsqueeze(1).expand(x.size(0), x.size(1), -1)
        return torch.cat(
            [x.unsqueeze(1), mask.unsqueeze(1)], axis=1
        )  # concatenate on the channel dimension

    def evaluate_prototypes(
        self, X_pos, X_neg, X_query, hop_seg, strt_index_query=None, audio_name=None
    ):
        X_pos = torch.tensor(X_pos)
        Y_pos = torch.LongTensor(np.zeros(X_pos.shape[0]))
        X_neg = torch.tensor(X_neg)
        Y_neg = torch.LongTensor(np.zeros(X_neg.shape[0]))
        X_query = torch.tensor(X_query)
        Y_query = torch.LongTensor(np.zeros(X_query.shape[0]))

        # num_batch_query = len(Y_query) // self.hparams.eval.query_batch_size
        query_dataset = torch.utils.data.TensorDataset(X_query, Y_query)
        q_loader = torch.utils.data.DataLoader(
            dataset=query_dataset,
            batch_sampler=None,
            batch_size=self.hparams.eval.query_batch_size,
            shuffle=False,
        )
        # query_set_feat = torch.zeros(0, 48).cpu()
        # batch_samplr_pos = EpisodicBatchSampler(Y_pos, 2, 1, self.hparams.train.n_shot)
        pos_dataset = torch.utils.data.TensorDataset(X_pos, Y_pos)
        pos_loader = torch.utils.data.DataLoader(
            dataset=pos_dataset, batch_sampler=None
        )
        "List for storing the combined probability across all iterations"
        prob_comb = []
        emb_dim = self.hparams.features.embedding_dim
        pos_set_feat = torch.zeros(0, emb_dim).cpu()

        print("Creating positive prototype")
        for batch in tqdm(pos_loader):
            x, y = batch
            # x = self.concate_mask(x, mask)
            feat = self.net(x.cuda())
            feat = feat.cpu()
            feat_mean = feat.mean(dim=0).unsqueeze(0)
            pos_set_feat = torch.cat((pos_set_feat, feat_mean), dim=0)
        pos_proto = pos_set_feat.mean(dim=0)

        iterations = self.hparams.eval.iterations
        for i in range(iterations):
            prob_pos_iter = []
            neg_indices = torch.randperm(len(X_neg))[: self.hparams.eval.samples_neg]
            # import ipdb; ipdb.set_trace()
            X_neg_ind = X_neg[neg_indices]
            Y_neg_ind = Y_neg[neg_indices]
            # X_neg_ind = self.concate_mask(X_neg_ind, mask)
            feat_neg = self.net(X_neg_ind.cuda())
            feat_neg = feat_neg.detach().cpu()
            proto_neg = feat_neg.mean(dim=0)
            q_iterator = iter(q_loader)

            print("Iteration number {}".format(i))
            for batch in tqdm(q_iterator):
                x_q, y_q = batch
                x_q = x_q
                # x_q = self.concate_mask(x_q, mask)
                x_query = self.net(x_q)

                proto_neg = proto_neg.detach().cpu()
                x_query = x_query.detach().cpu()

                probability_pos = self.get_probability(pos_proto, proto_neg, x_query)
                prob_pos_iter.extend(probability_pos)

            prob_comb.append(prob_pos_iter)

        prob_final = np.mean(np.array(prob_comb), axis=0)
        # Save the probability here to perform model ensemble
        filename = os.path.basename(audio_name).split(".")[0]
        os.makedirs("prob_comb", exist_ok=True)
        np.save("prob_comb/%s.npy" % filename, np.array(prob_comb))

        thresh_list = np.arange(0, 1, 0.05)
        onset_offset_ret = {}
        for thresh in thresh_list:
            krn = np.array([1, -1])
            prob_thresh = np.where(prob_final > thresh, 1, 0)
            # prob_pos_final = prob_final * prob_thresh

            changes = np.convolve(krn, prob_thresh)

            onset_frames = np.where(changes == 1)[0]
            offset_frames = np.where(changes == -1)[0]

            str_time_query = (
                strt_index_query
                * self.hparams.features.hop_mel
                / self.hparams.features.sr
            )

            onset = (
                (onset_frames)
                * (hop_seg)
                * self.hparams.features.hop_mel
                / self.hparams.features.sr
            )
            onset = onset + str_time_query

            offset = (
                (offset_frames)
                * (hop_seg)
                * self.hparams.features.hop_mel
                / self.hparams.features.sr
            )
            offset = offset + str_time_query

            assert len(onset) == len(offset)
            onset_offset_ret[thresh] = [onset, offset]

        # from scipy.signal import medfilt

        # # Use median filtering
        # for thresh in thresh_list:
        #     krn = np.array([1, -1])
        #     prob_thresh = np.where(medfilt(prob_final) > thresh, 1, 0)
        #     # prob_pos_final = prob_final * prob_thresh

        #     changes = np.convolve(krn, prob_thresh)

        #     onset_frames = np.where(changes == 1)[0]
        #     offset_frames = np.where(changes == -1)[0]

        #     str_time_query = (
        #         strt_index_query * self.hparams.features.hop_mel / self.hparams.features.sr
        #     )

        #     onset = (
        #         (onset_frames)
        #         * (hop_seg)
        #         * self.hparams.features.hop_mel
        #         / self.hparams.features.sr
        #     )
        #     onset = onset + str_time_query

        #     offset = (
        #         (offset_frames)
        #         * (hop_seg)
        #         * self.hparams.features.hop_mel
        #         / self.hparams.features.sr
        #     )
        #     offset = offset + str_time_query

        #     assert len(onset) == len(offset)

        #     # Use median filtering (plus 1)
        #     onset_offset_ret[thresh+1] = [onset, offset]

        return onset_offset_ret

    def euclidean_dist(self, x, y):
        """Compute euclidean distance between two tensors."""
        # x: N x D
        # y: M x D

        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        if d != y.size(1):
            raise Exception

        x = x.unsqueeze(1).expand(n, m, d)

        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)

    def cosine_dist(self, x, y):
        """Compute euclidean distance between two tensors."""
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        if d != y.size(1):
            raise Exception

        x = x.unsqueeze(1).expand(n, m, d)

        y = y.unsqueeze(0).expand(n, m, d)

        return -torch.nn.CosineSimilarity(dim=2, eps=1e-6)(x, y)


if __name__ == "__main__":

    def calculate_psds():
        from glob import glob
        from psds_eval import PSDSEval, plot_psd_roc, plot_per_class_psd_roc

        dtc_threshold = 0.5
        gtc_threshold = 0.5
        cttc_threshold = 0.3
        alpha_ct = 0.0
        alpha_st = 0.0
        max_efpr = 100
        ground_truth_csv = os.path.join(
            "/vol/research/dcase2022/project/hhlab/src/models/eval_meta/subset_gt.csv"
        )
        metadata_csv = os.path.join(
            "/vol/research/dcase2022/project/hhlab/src/models/eval_meta/subset_meta.csv"
        )
        gt_table = pd.read_csv(ground_truth_csv, sep="\t")
        meta_table = pd.read_csv(metadata_csv, sep="\t")
        psds_eval = PSDSEval(
            dtc_threshold,
            gtc_threshold,
            cttc_threshold,
            ground_truth=gt_table,
            metadata=meta_table,
        )
        for file in glob(
            "/vol/research/dcase2022/project/t5_open_source/DCASE_2022_Task_5/logs/experiments/runs/final/2022-07-05_19-25-40/*/PSDS_Eval_*.csv"
        ):
            det_t = pd.read_csv(os.path.join(file), sep="\t")
            psds_eval.add_operating_point(det_t)
            break
        psds = psds_eval.psds(alpha_ct, alpha_st, max_efpr)
        print(f"\nPSDS-Score: {psds.value:.5f}")
        print("Saving pickle!")
        save_pickle(psds, "psds.pkl")
        plot_psd_roc(psds, filename="roc.png")
        tpr_vs_fpr, _, tpr_vs_efpr = psds_eval.psd_roc_curves(alpha_ct=alpha_ct)
        plot_per_class_psd_roc(
            tpr_vs_fpr,
            psds_eval.class_names,
            title="Per-class TPR-vs-FPR PSDROC",
            xlabel="FPR",
            filename="per_class_1.png",
        )
        save_pickle(tpr_vs_fpr, "tpr_vs_fpr.pkl")
        save_pickle(psds_eval.class_names, "class_names.pkl")
        plot_per_class_psd_roc(
            tpr_vs_efpr,
            psds_eval.class_names,
            title="Per-class TPR-vs-eFPR PSDROC",
            xlabel="eFPR",
            filename="per_class_2.png",
        )
        save_pickle(tpr_vs_efpr, "tpr_vs_efpr.pkl")

    calculate_psds()
