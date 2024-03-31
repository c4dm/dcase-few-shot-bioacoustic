import sys

sys.path.append("/vol/research/dcase2022/project/hhlab")

import argparse
import copy
import csv
import glob
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

import src.utils.metrics as metrics

MIN_EVAL_VALUE = 0.00001
N_SHOTS = 5
MIN_IOU_TH = 0.3
PRED_FILE_HEADER = ["Audiofilename", "Starttime", "Endtime"]
POS_VALUE = "POS"
UNK_VALUE = "UNK"


def remove_shots_from_ref(ref_df, number_shots=5):

    ref_pos_indexes = select_events_with_value(ref_df, value=POS_VALUE)
    ref_n_shot_index = ref_pos_indexes[number_shots - 1]
    # remove all events (pos and UNK) that happen before this 5th event
    events_to_drop = ref_df.index[
        ref_df["Endtime"] <= ref_df.iloc[ref_n_shot_index]["Endtime"]
    ].tolist()

    return ref_df.drop(events_to_drop)


def select_events_with_value(data_frame, value=POS_VALUE):
    def find_positive_label(df):
        for col in df.columns:
            if "Q" in col:
                return col
        else:
            raise ValueError(
                "Error: Expect you change the validation set event name to Q_x"
            )

    key = find_positive_label(data_frame)
    indexes_list = data_frame.index[data_frame[key] == value].tolist()
    return indexes_list


def build_matrix_from_selected_rows(data_frame, selected_indexes_list):

    matrix_data = np.ones((2, len(selected_indexes_list))) * -1
    for n, idx in enumerate(selected_indexes_list):
        matrix_data[0, n] = data_frame.loc[idx].Starttime  # start time for event n
        matrix_data[1, n] = data_frame.loc[idx].Endtime
    return matrix_data


def compute_tp_fp_fn(pred_events_df, ref_events_df):
    # inputs: dataframe with predicted events, dataframe with reference events and their value (POS, UNK, NEG)
    # output: True positives, False Positives, False negatives counts and total number of pos events in ref.

    # makes one pass with bipartite graph matching between pred events and ref positive events
    # get TP
    # make second pass with remaining pred events and ref Unk events
    # compute FP as the number of remaining predicted events after the two rounds of matches.
    # FN is the remaining unmatched pos events in ref.
    ref_pos_indexes = select_events_with_value(ref_events_df, value=POS_VALUE)

    if "Q" not in pred_events_df.columns:
        pred_events_df["Q"] = POS_VALUE
    pred_pos_indexes = select_events_with_value(pred_events_df, value=POS_VALUE)

    ref_1st_round = build_matrix_from_selected_rows(ref_events_df, ref_pos_indexes)
    pred_1st_round = build_matrix_from_selected_rows(pred_events_df, pred_pos_indexes)

    m_pos = metrics.match_events(ref_1st_round, pred_1st_round, min_iou=MIN_IOU_TH)
    matched_ref_indexes = [ri for ri, pi in m_pos]
    matched_pred_indexes = [pi for ri, pi in m_pos]

    ref_unk_indexes = select_events_with_value(ref_events_df, value=UNK_VALUE)
    ref_2nd_round = build_matrix_from_selected_rows(ref_events_df, ref_unk_indexes)

    unmatched_pred_events = list(
        set(range(pred_1st_round.shape[1])) - set(matched_pred_indexes)
    )
    pred_2nd_round = pred_1st_round[:, unmatched_pred_events]

    m_unk = metrics.match_events(ref_2nd_round, pred_2nd_round, min_iou=MIN_IOU_TH)

    # print("# Positive matches between Ref and Pred :", len(m_pos))
    # print("# matches with Unknown events: ", len(m_unk))

    tp = len(m_pos)
    fp = pred_1st_round.shape[1] - tp - len(m_unk)

    ## compute unmatched pos ref events:
    count_unmached_pos_ref_events = len(ref_pos_indexes) - tp

    fn = count_unmached_pos_ref_events

    total_n_POS_events = len(ref_pos_indexes)
    return tp, fp, fn, total_n_POS_events


def compute_scores_per_class(counts_per_class):

    scores_per_class = {}
    for cl in counts_per_class.keys():
        tp = counts_per_class[cl]["TP"]
        fp = counts_per_class[cl]["FP"]
        fn = counts_per_class[cl]["FN"]

        # to compute the harmonic mean we need to have all entries as non zero
        precision = (
            tp / (tp + fp) if tp + fp != 0 else MIN_EVAL_VALUE
        )  # case where no predictions were made
        if precision < MIN_EVAL_VALUE:
            precision = MIN_EVAL_VALUE
        recall = tp / (fn + tp) if tp != 0 else MIN_EVAL_VALUE
        fmeasure = tp / (tp + 0.5 * (fp + fn)) if tp != 0 else MIN_EVAL_VALUE

        scores_per_class[cl] = {
            "precision": precision,
            "recall": recall,
            "f-measure": fmeasure,
        }

    return scores_per_class


def compute_scores_from_counts(counts):
    tp = counts["TP"]
    fp = counts["FP"]
    fn = counts["FN"]

    # to compute the harmonic mean we need to have all entries as non zero
    precision = (
        tp / (tp + fp) if tp + fp != 0 else MIN_EVAL_VALUE
    )  # case where no predictions were made
    if precision < MIN_EVAL_VALUE:
        precision = MIN_EVAL_VALUE
    recall = tp / (fn + tp) if tp != 0 else MIN_EVAL_VALUE
    fmeasure = tp / (tp + 0.5 * (fp + fn)) if tp != 0 else MIN_EVAL_VALUE

    scores = {"precision": precision, "recall": recall, "f-measure": fmeasure}

    return scores


def build_report(
    main_set_scores,
    scores_per_miniset,
    scores_per_audiofile,
    save_path,
    main_set_name="EVAL",
    team_name="test_team",
    **kwargs
):

    # datetime object containing current date and time
    now = datetime.now()
    date_string = now.strftime("%d%m%Y_%H_%M_%S")
    # print("date and time =", date_string)

    # make dict:
    report = {
        "team_name": team_name,
        "set_name": main_set_name,
        "report_date": date_string,
        "overall_scores": main_set_scores,
        "scores_per_subset": scores_per_miniset,
        "scores_per_audiofile": scores_per_audiofile,
    }
    if "scores_per_class" in kwargs.keys():
        report["scores_per_class"] = kwargs["scores_per_class"]

    with open(
        os.path.join(
            save_path,
            "Evaluation_report_"
            + team_name
            + "_"
            + main_set_name
            + "_"
            + date_string
            + ".json",
        ),
        "w",
    ) as outfile:
        json.dump(report, outfile)
    return


def evaluate(pred_file_path, ref_file_path, team_name, dataset, savepath, metadata=[]):
    individual_file_result = {}
    # print("\nEvaluation for:", team_name, dataset)
    # read Gt file structure: get subsets and paths for ref csvs make an inverted dictionary with audiofilenames as keys and folder as value
    gt_file_structure = {}
    gt_file_structure[dataset] = {}
    inv_gt_file_structure = {}
    list_of_subsets = os.listdir(ref_file_path)
    for subset in list_of_subsets:
        gt_file_structure[dataset][subset] = [
            os.path.basename(fl)[0:-4] + ".wav"
            for fl in glob.glob(os.path.join(ref_file_path, subset, "*.csv"))
        ]
        for audiofile in gt_file_structure[dataset][subset]:
            inv_gt_file_structure[audiofile] = subset

    # read prediction csv
    pred_csv = pd.read_csv(pred_file_path, dtype=str)
    # verify headers:
    if list(pred_csv.columns) != PRED_FILE_HEADER:
        print(
            "Please correct the header of the prediction file. This should be",
            PRED_FILE_HEADER,
        )
        exit(1)
    #  parse prediction csv
    #  split file into lists of events for the same audiofile.
    pred_events_by_audiofile = dict(tuple(pred_csv.groupby("Audiofilename")))

    counts_per_audiofile = {}
    for audiofilename in list(pred_events_by_audiofile.keys()):
        if audiofilename not in inv_gt_file_structure.keys():
            continue  # Testset or validation set
        # for each audiofile, load correcponding GT File (audiofilename.csv)
        ref_events_this_audiofile_all = pd.read_csv(
            os.path.join(
                ref_file_path,
                inv_gt_file_structure[audiofilename],
                audiofilename[0:-4] + ".csv",
            ),
            dtype={"Starttime": np.float64, "Endtime": np.float64},
        )

        # Remove the 5 shots from GT:
        ref_events_this_audiofile = remove_shots_from_ref(
            ref_events_this_audiofile_all, number_shots=N_SHOTS
        )

        # compare and get counts: TP, FP ..
        tp_count, fp_count, fn_count, total_n_events_in_audiofile = compute_tp_fp_fn(
            pred_events_by_audiofile[audiofilename], ref_events_this_audiofile
        )

        counts_per_audiofile[audiofilename] = {
            "TP": tp_count,
            "FP": fp_count,
            "FN": fn_count,
            "total_n_pos_events": total_n_events_in_audiofile,
        }
        # print(audiofilename, counts_per_audiofile[audiofilename])
        individual_file_result[audiofilename] = counts_per_audiofile[audiofilename]

    if metadata:
        # using the key for classes => audiofiles,  # load sets metadata:
        with open(metadata) as metadatafile:
            dataset_metadata = json.load(metadatafile)
    else:
        dataset_metadata = copy.deepcopy(gt_file_structure)

    # include audiofiles for which there were no predictions:
    list_all_audiofiles = []
    for miniset in dataset_metadata[dataset].keys():
        if metadata:
            for cl in dataset_metadata[dataset][miniset].keys():
                list_all_audiofiles.extend(dataset_metadata[dataset][miniset][cl])
        else:
            list_all_audiofiles.extend(dataset_metadata[dataset][miniset])

    for audiofilename in list_all_audiofiles:
        if audiofilename not in counts_per_audiofile.keys():
            ref_events_this_audiofile = pd.read_csv(
                os.path.join(
                    ref_file_path,
                    inv_gt_file_structure[audiofilename],
                    audiofilename[0:-4] + ".csv",
                ),
                dtype=str,
            )
            total_n_pos_events_in_audiofile = len(
                select_events_with_value(ref_events_this_audiofile, value=POS_VALUE)
            )
            counts_per_audiofile[audiofilename] = {
                "TP": 0,
                "FP": 0,
                "FN": total_n_pos_events_in_audiofile,
                "total_n_pos_events": total_n_pos_events_in_audiofile,
            }

    # aggregate the counts per class or subset:
    list_sets_in_mainset = list(dataset_metadata[dataset].keys())

    counts_per_class_per_set = {}
    scores_per_class_per_set = {}
    counts_per_set = {}
    scores_per_set = {}
    scores_per_audiofile = {}
    for data_set in list_sets_in_mainset:
        if "." == dataset[0]:
            continue

        if metadata:
            list_classes_in_set = list(dataset_metadata[dataset][data_set].keys())

            counts_per_class_per_set[data_set] = {}
            tp_set = 0
            fn_set = 0
            fp_set = 0
            total_n_events_set = 0
            for cl in list_classes_in_set:
                # print(cl)
                list_audiofiles_this_class = dataset_metadata[dataset][data_set][cl]
                tp = 0
                fn = 0
                fp = 0
                total_n_pos_events_this_class = 0
                for audiofile in list_audiofiles_this_class:
                    scores_per_audiofile[audiofile] = compute_scores_from_counts(
                        counts_per_audiofile[audiofile]
                    )

                    tp = tp + counts_per_audiofile[audiofile]["TP"]
                    tp_set = tp_set + counts_per_audiofile[audiofile]["TP"]
                    fn = fn + counts_per_audiofile[audiofile]["FN"]
                    fn_set = fn_set + counts_per_audiofile[audiofile]["FN"]
                    fp = fp + counts_per_audiofile[audiofile]["FP"]
                    fp_set = fp_set + counts_per_audiofile[audiofile]["FP"]
                    total_n_pos_events_this_class = (
                        total_n_pos_events_this_class
                        + counts_per_audiofile[audiofile]["total_n_pos_events"]
                    )
                    total_n_events_set = (
                        total_n_events_set
                        + counts_per_audiofile[audiofile]["total_n_pos_events"]
                    )

                # counts_per_class[cl] = {"TP":tp, "FN": fn, "FP": fp, "total_n_pos_events_this_class": total_n_pos_events_this_class}
                counts_per_class_per_set[data_set][cl] = {
                    "TP": tp,
                    "FN": fn,
                    "FP": fp,
                    "total_n_pos_events_this_class": total_n_pos_events_this_class,
                }
                counts_per_set[data_set] = {
                    "TP": tp_set,
                    "FN": fn_set,
                    "FP": fp_set,
                    "total_n_pos_events_this_set": total_n_events_set,
                }

            #  compute scores per subset.
            scores_per_set[data_set] = compute_scores_from_counts(
                counts_per_set[data_set]
            )
            #  compute scores per class
            scores_per_class_per_set[data_set] = compute_scores_per_class(
                counts_per_class_per_set[data_set]
            )

        else:
            list_audiofiles_in_set = dataset_metadata[dataset][data_set]
            tp = 0
            fn = 0
            fp = 0
            total_n_pos_events_this_set = 0
            for audiofile in list_audiofiles_in_set:

                scores_per_audiofile[audiofile] = compute_scores_from_counts(
                    counts_per_audiofile[audiofile]
                )
                tp = tp + counts_per_audiofile[audiofile]["TP"]
                fn = fn + counts_per_audiofile[audiofile]["FN"]
                fp = fp + counts_per_audiofile[audiofile]["FP"]
                total_n_pos_events_this_set = (
                    total_n_pos_events_this_set
                    + counts_per_audiofile[audiofile]["total_n_pos_events"]
                )
                counts_per_set[data_set] = {
                    "TP": tp,
                    "FN": fn,
                    "FP": fp,
                    "total_n_pos_events_this_set": total_n_pos_events_this_set,
                }

            #  compute scores per subset
            scores_per_set[data_set] = compute_scores_from_counts(
                counts_per_set[data_set]
            )

    overall_scores = {
        "precision": stats.hmean(
            [scores_per_set[dt]["precision"] for dt in scores_per_set.keys()]
        ),
        "recall": stats.hmean(
            [scores_per_set[dt]["recall"] for dt in scores_per_set.keys()]
        ),
        "fmeasure": np.round(
            stats.hmean(
                [scores_per_set[dt]["f-measure"] for dt in scores_per_set.keys()]
            )
            * 100,
            3,
        ),
        "precision-avg": np.mean(
            [scores_per_set[dt]["precision"] for dt in scores_per_set.keys()]
        ),
        "recall-avg": np.mean(
            [scores_per_set[dt]["recall"] for dt in scores_per_set.keys()]
        ),
        "fmeasure-avg": np.round(
            np.mean([scores_per_set[dt]["f-measure"] for dt in scores_per_set.keys()])
            * 100,
            3,
        ),
    }

    print("\nOverall_scores:", overall_scores)
    # print("\nwriting report")
    if metadata:
        build_report(
            overall_scores,
            scores_per_set,
            scores_per_audiofile,
            savepath,
            dataset,
            team_name,
            scores_per_class=scores_per_class_per_set,
        )
    else:
        build_report(
            overall_scores,
            scores_per_set,
            scores_per_audiofile,
            savepath,
            dataset,
            team_name,
        )

    return overall_scores, individual_file_result, scores_per_set, scores_per_audiofile


if __name__ == "__main__":
    import pandas as pd

    pred_file = "/vol/research/dcase2022/project/hhlab/src/utils/pred.csv"
    ref_files_path = "/vol/research/dcase2022/project/hhlab/src/utils/ref.csv"
    team_name, dataset, savepath = "TEST", "VAL", "."
    res = compute_tp_fp_fn(pd.read_csv(pred_file), pd.read_csv(ref_files_path))
    print(res)
