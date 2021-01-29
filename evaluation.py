
import pandas as pd
import argparse
import os
import json
import numpy as np
import metrics



# v.0 use one single prediction file with all audios from eval sets together.
# def read_parse_single_pred_file(pred_csv_path, )
def select_unmatched_events_with_value(data_frame, value = 'POS'):

    indexes_list = data_frame.index[data_frame["Q"] == value].tolist()

    return indexes_list

def build_matrix_from_selected_rows(data_frame, selected_indexes_list ):

    matrix_data = np.ones((2, len(selected_indexes_list)))* -1
    for n, idx in enumerate(selected_indexes_list):
        matrix_data[0, n] =  data_frame.loc[idx].Starttime # start time for event n
        matrix_data[1, n] =  data_frame.loc[idx].Endtime
    return matrix_data


def compute_TP_FP_FN(pred_events_df, ref_events_df):
    # make one pass with bipartite graph matching between pred events and ref positive events
    # get TP
    

    ref_pos_indexes = select_unmatched_events_with_value(ref_events_df, value = 'POS')

    if "Q" not in pred_events_df.columns:
        pred_events_df["Q"] = "POS"
    pred_pos_indexes = select_unmatched_events_with_value(pred_events_df, value="POS")

    ref_1st_round = build_matrix_from_selected_rows(ref_events_df, ref_pos_indexes)
    pred_1st_round = build_matrix_from_selected_rows(pred_events_df, pred_pos_indexes)

    m_pos = metrics.match_events(ref_1st_round, pred_1st_round)
    matched_ref_indexes = [ri for ri,pi in m_pos]  # TODO correct the indexes: causes cnfusion! indexes from dataframe are different from the match_events function
    matched_pred_indexes = [pi for ri,pi in m_pos]


    ref_unk_indexes = select_unmatched_events_with_value(ref_events_df, value = 'UNK')
    ref_2nd_round = build_matrix_from_selected_rows(ref_events_df, ref_unk_indexes)

    unmatched_pred_events = list(set(range(pred_1st_round.shape[1])) - set(matched_pred_indexes))
    pred_2nd_round = pred_1st_round[: , unmatched_pred_events]

    m_unk = metrics.match_events(ref_2nd_round, pred_2nd_round)

    print("Positive matches between Ref and Pred :", m_pos)
    print("matches with Unknown events: ", m_unk)
    
    TP = len(m_pos)
    FP = pred_1st_round.shape[1] - TP - len(m_unk)
    
    # compute unmatched ref events:
    matched_events_ref = TP + len(m_unk)
    count_unmached_ref_events = len(ref_events_df)- matched_events_ref
    FN = count_unmached_ref_events

    
    # normalize these numbers by number of events?
    # and its total number of POS events or POs and unk?
    total_n_events = len(ref_events_df)
    return TP, FP, FN, total_n_events

def compute_scores_per_class_and_average_scores_per_set(counts_per_class):

    scores_per_class = {}
    cumulative_fmeasure = 0
    cumulative_precision = 0
    cumulative_recall = 0
    for cl in counts_per_class.keys():
        TP = counts_per_class[cl]["TP"]
        FP = counts_per_class[cl]["FP"]
        FN = counts_per_class[cl]["FN"]
        if np.sum([TP,FP,FN])==0:  # no predictions for this class
            continue
        
        precision = TP/(TP+FP)
        recall = TP/(FN+TP)
        fmeasure = TP/(TP+0.5*(FP+FN))
        scores_per_class[cl] = {"precision": precision, "recall": recall, "f-measure":fmeasure }

        cumulative_fmeasure = cumulative_fmeasure +fmeasure 
        cumulative_precision = cumulative_precision+precision
        cumulative_recall = cumulative_recall +recall
    
    n_classes = len(list(counts_per_class.keys())) 
    # average scores in this set:
    av_scores_set = {"av_precision": cumulative_precision/n_classes, "av_recall": cumulative_recall/n_classes, "av_fmeasure": cumulative_fmeasure/n_classes }
    return scores_per_class, av_scores_set
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-pred_file_path', type=str, help='path to folder with prediction csv')
    parser.add_argument('-ref_file_path', type=str, help='path to the reference events csvs')
    parser.add_argument('-metadata', type=str, help="path for metadata json (map between audiofiles and classes")
    # parser.add_argument('-team_name', type=str, help='team identification') # make this optional?
    args = parser.parse_args()
    print(args)

    #read prediction csv
    pred_csv = pd.read_csv(args.pred_file_path, dtype=str)
    #  parse prediction csv
    # split file into lists of events for the same audiofile.
    pred_events_by_audiofile = dict(tuple(pred_csv.groupby('Audiofilename')))

    counts_per_audiofile = {}
    for audiofilename in list(pred_events_by_audiofile.keys()):
        
        # for each audiofile list, load correcponding GT File (audiofilename.csv)
        ref_events_this_audiofile = pd.read_csv(os.path.join(args.ref_file_path, audiofilename[0:-4]+'.csv'), dtype=str)
        # compare and get counts: TP, FP .. 
        TP, FP, FN , total_n_events_in_audiofile= compute_TP_FP_FN(pred_events_by_audiofile[audiofilename], ref_events_this_audiofile )

        counts_per_audiofile[audiofilename]={"TP": TP, "FP": FP, "FN": FN}
    
    # using the key for classes => audiofiles,  # load sets metadata:
    with open(args.metadata) as metadatafile:
            dataset_metadata = json.load(metadatafile)

    # aggregate the counts per class
    list_sets_in_evalset = list(dataset_metadata["EVAL"].keys())
    list_classes_in_eval = []
    counts_per_class = {}
    counts_per_class_per_set = {}
    scores_per_class_per_set={}
    av_scores_per_set={}
    for data_set in list_sets_in_evalset:
        list_classes_in_eval.extend(list(dataset_metadata["EVAL"][data_set].keys()))        
        list_classes_in_set = list(dataset_metadata["EVAL"][data_set].keys())
   
        counts_per_class_per_set[data_set] = {}
    
        for cl in list_classes_in_set:
            list_audiofiles_this_class = dataset_metadata["EVAL"][data_set][cl]
            tp = 0
            fn = 0
            fp = 0
            for audiofile in list_audiofiles_this_class:
                try:
                    tp = tp + counts_per_audiofile[audiofile+".wav"]["TP"]
                    fn = fn + counts_per_audiofile[audiofile+".wav"]["FN"]
                    fp = fp + counts_per_audiofile[audiofile+".wav"]["FP"]
                except KeyError:
                    # there were no detections for this audiofile:
                    continue
            counts_per_class[cl] = {"TP":tp, "FN": fn, "FP": fp}
            counts_per_class_per_set[data_set][cl] = {"TP":tp, "FN": fn, "FP": fp}

        #  compute scores per class. # aggregate scores per sets: will this be an average of scores across classes of each set?
        scores_per_class_per_set[data_set], av_scores_per_set[data_set] = compute_scores_per_class_and_average_scores_per_set(counts_per_class_per_set[data_set])  
            
    
    #average scores per all sets in the eval set 
    # av(dc_scores, ME_scores, ML_scores)
    Overall_scores = {"precision" : sum([av_scores_per_set[dt]["av_precision"] for dt in av_scores_per_set.keys()])/ len(list(av_scores_per_set.keys())) , 
                    "recall": sum([av_scores_per_set[dt]["av_recall"] for dt in av_scores_per_set.keys()])/ len(list(av_scores_per_set.keys())) ,
                    "fmeasure": sum([av_scores_per_set[dt]["av_fmeasure"] for dt in av_scores_per_set.keys()])/ len(list(av_scores_per_set.keys())) ,
                    }
    
    

    ## Build report

    # Team name: from args or default
    # main set name: from args (it needs to read the metadata key that maps classes to audiofiles)
    # overall scores: 
        # precision, recall, fmeasure
    # scores per subset:
        # set1
            #precision, recall, fmeasure
        # set2
             #...
        # set3
            # ...       
    # scores per class: 
        #class 1
            # prcision ...
        #class 2
        #class 3




    # v.1 add possibility for several prediction files
# v.2 adapt code for csvs of training set where each audiofile has more than one class present.



### backbone functions?:

print('stop')