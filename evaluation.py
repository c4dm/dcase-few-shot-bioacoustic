import metrics
import numpy as np
import pandas as pd
import argparse
import os
from glob import glob

def select_unmatched_events_with_value(data_frame, class_name, value = 'POS',  matched_events_index=[]):

    indexes_list = data_frame.index[data_frame[class_name] == value].tolist()

    return indexes_list

def build_matrix_from_selected_rows(data_frame, selected_indexes_list ):

    matrix_data = np.ones((2, len(selected_indexes_list)))* -1
    for n, idx in enumerate(selected_indexes_list):
        matrix_data[0, n] =  data_frame.loc[idx].Starttime # start time for event n
        matrix_data[1, n] =  data_frame.loc[idx].Endtime
    return matrix_data


if __name__ == "__main__":

    # # I'm assuming one team produces one predictions csv per ref csv"
    # # filenames must follow a convention, here i'm using only an example, but assuming there will be a team name there.


    # parser = argparse.ArgumentParser()
    # parser.add_argument('-path', type=str, help='path to folder with Predictions csv files of all teams')
    # parser.add_argument('-ref_file_path', type=str, help='path to the reference events file')
    # args = parser.parse_args()
    # print(args)

    # # list prediction files in folder args.path:
    # pred_filenames_list = glob(args.path+'*_PRED_example.csv')  # TODO agree on a filename convention and change it here,
    
    # #read ref csv
    # ref_csv = pd.read_csv(args.ref_file_path)


    # results_per_team = {}
    # for team_pred_filename in pred_filenames_list:
    #     team = team_pred_filename.split("_PRED_example.csv")[0] # TODO agree on a filename convention and change it here, it must constain the team name

    #     # read pred csv:
    #     pred_csv = pd.read_csv(team_pred_filename)


    ref_csv = pd.read_csv('/mnt/c/Users/madzi/Dropbox/QMUL/PHD/DCASE2021_few-shot_bioacoustics_challenge/TEAMS_results/GT_example.csv')
    pred_csv = pd.read_csv('/mnt/c/Users/madzi/Dropbox/QMUL/PHD/DCASE2021_few-shot_bioacoustics_challenge/TEAMS_results/T1_PRED_example.csv')

    class_names = list(ref_csv.columns)[3:]
    print(class_names)



    # select events(rows) that have POS on given class:
    # ref_per_class = {}
    # pred_per_class = {}
    # ref_UNK_per_class = {}
    metric_results_per_class= {}
        
    for cl in class_names:
        # if cl != 'CalltypeX':
        #     break
        print(cl)
        ref_per_class = select_unmatched_events_with_value(ref_csv, class_name=cl, value = 'POS')
        # ref_csv.index[ref_csv[cl] == "POS"].tolist() # indexes of events where this label is pos
        ref_UNK_per_class = select_unmatched_events_with_value(ref_csv, class_name=cl, value = 'UNK')
        # ref_UNK_per_class[cl] = ref_csv.index[ref_csv[cl] == "UNK"].tolist()
        pred_per_class = select_unmatched_events_with_value(pred_csv, class_name=cl, value = 'POS')        
        # pred_per_class[cl] = pred_csv.index[pred_csv[cl] == "POS"].tolist()


        print("ref events for class", cl, " : ", ref_csv.loc[ref_per_class])
        print("ref events UNK for class", cl, " : ", ref_csv.loc[ref_UNK_per_class])
        print("pred events for class", cl, " : ", pred_csv.loc[pred_per_class])

        ref = build_matrix_from_selected_rows(ref_csv, ref_per_class)
        pred = build_matrix_from_selected_rows(pred_csv, pred_per_class)

       # 1st round of matches:  POS in REF with POS in PRED
        m_pos = metrics.match_events(ref, pred)
        matched_ref_indexes = [ri for ri,pi in m_pos]  #(indexes from dataframe are different from the match_events function)
        matched_pred_indexes = [pi for ri,pi in m_pos]



        ref_unk = build_matrix_from_selected_rows(ref_csv, ref_UNK_per_class)
    
        remaining = list(set(range(pred.shape[1])) - set(matched_pred_indexes))
        pred_unk = pred[: , remaining]

        m_unk = metrics.match_events(ref_unk, pred_unk)

        print("Positive matches between Ref and Pred :", m_pos)
        print("matches with Unknown events: ", m_unk)
        print("\n")

        TP = len(m_pos)
        FP = pred.shape[1] - TP - len(m_unk)
        # compute unmatched ref events:
        matched_events_ref = TP + len(m_unk)
        FN = len(ref_csv)- matched_events_ref
       

        # precision = sed_eval.metric.precicion(TP, pred.shape[1])
        # recall = sed_eval.metric.recall(TP,ref.shape[1])
        # # Fmeasure =
        metric_results_per_class[cl] = {"TP": len(m_pos), "FP": pred.shape[1] - len(m_pos) - len(m_unk), "FN": FN  }
        print(metric_results_per_class)


    

    ######    IF a predicrion is made on an Unknown event, we donkt wanf to penalize that
    # cases with UNK in more than one class?

    print("stop")

    # compute precision and such per class

    # sed_eval.metric.precision()




