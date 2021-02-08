
import pandas as pd
import argparse
import os
import json
import numpy as np
import csv
import metrics
from datetime import datetime
import copy
from scipy import stats


# v.0 use one single prediction file with all audios from eval sets together.
# def read_parse_single_pred_file(pred_csv_path, )
def select_events_with_value(data_frame, value = 'POS'):

    indexes_list = data_frame.index[data_frame["Q"] == value].tolist()

    return indexes_list

def build_matrix_from_selected_rows(data_frame, selected_indexes_list ):

    matrix_data = np.ones((2, len(selected_indexes_list)))* -1
    for n, idx in enumerate(selected_indexes_list):
        matrix_data[0, n] =  data_frame.loc[idx].Starttime # start time for event n
        matrix_data[1, n] =  data_frame.loc[idx].Endtime
    return matrix_data


def compute_TP_FP_FN(pred_events_df, ref_events_df):
    # inputs: dataframe with predicted events, dataframe with reference events and their value (POS, UNK, NEG)
    # output: True positives, False Positives, False negatives counts and total number of pos events in ref.

    # makes one pass with bipartite graph matching between pred events and ref positive events
    # get TP
    # make second pass with remaining pred events and ref Unk events
    # compute FP as the number of remaining predicted events after the two rounds of matches.
    # FN is the remaining unmatched pos events in ref.

    ref_pos_indexes = select_events_with_value(ref_events_df, value = 'POS')

    if "Q" not in pred_events_df.columns:
        pred_events_df["Q"] = "POS"
    pred_pos_indexes = select_events_with_value(pred_events_df, value="POS")

    ref_1st_round = build_matrix_from_selected_rows(ref_events_df, ref_pos_indexes)
    pred_1st_round = build_matrix_from_selected_rows(pred_events_df, pred_pos_indexes)

    m_pos = metrics.match_events(ref_1st_round, pred_1st_round)
    matched_ref_indexes = [ri for ri,pi in m_pos]  # TODO correct the indexes: causes cnfusion! indexes from dataframe are different from the match_events function
    matched_pred_indexes = [pi for ri,pi in m_pos]


    ref_unk_indexes = select_events_with_value(ref_events_df, value = 'UNK')
    ref_2nd_round = build_matrix_from_selected_rows(ref_events_df, ref_unk_indexes)

    unmatched_pred_events = list(set(range(pred_1st_round.shape[1])) - set(matched_pred_indexes))
    pred_2nd_round = pred_1st_round[: , unmatched_pred_events]

    m_unk = metrics.match_events(ref_2nd_round, pred_2nd_round)

    print("Positive matches between Ref and Pred :", m_pos)
    print("matches with Unknown events: ", m_unk)
    
    TP = len(m_pos)
    FP = pred_1st_round.shape[1] - TP - len(m_unk)
    
    ## compute unmatched pos ref events:
    # matched_events_ref = TP + len(m_unk)
    # count_unmached_ref_events = len(ref_events_df)- matched_events_ref
    count_unmached_pos_ref_events = len(ref_pos_indexes) - TP

    FN = count_unmached_pos_ref_events

    
    # normalize these numbers by number of events?
    # and its total number of POS events or POs and unk?
    total_n_POS_events = len(ref_pos_indexes)
    return TP, FP, FN, total_n_POS_events

def compute_scores_per_class_and_average_scores_per_set(counts_per_class):

    scores_per_class = {}
    cumulative_fmeasure = []
    cumulative_precision = []
    cumulative_recall = []
    for cl in counts_per_class.keys():
        TP = counts_per_class[cl]["TP"]
        FP = counts_per_class[cl]["FP"]
        FN = counts_per_class[cl]["FN"]

            
        precision = TP/(TP+FP) if TP+FP != 0 else 0
        # precision = TP/(TP+FP) 
        recall = TP/(FN+TP)
        fmeasure = TP/(TP+0.5*(FP+FN))

        scores_per_class[cl] = {"precision": precision, "recall": recall, "f-measure":fmeasure }

        cumulative_fmeasure.append(fmeasure)
        cumulative_precision.append(precision)
        cumulative_recall.append(recall)
    
    n_classes = len(list(counts_per_class.keys())) 
    # average scores in this set:
    av_scores_set = {"precision": stats.hmean(cumulative_precision), "recall": stats.hmean(cumulative_recall), "f-measure": stats.hmean(cumulative_fmeasure) }
    return scores_per_class, av_scores_set
    
def compute_scores_from_counts(counts):
    TP = counts["TP"]
    FP = counts["FP"]
    FN = counts["FN"]

    precision = TP/(TP+FP) if TP+FP != 0 else 0
    # precision = TP/(TP+FP) 
    recall = TP/(FN+TP)
    fmeasure = TP/(TP+0.5*(FP+FN))

    scores = {"precision": precision, "recall": recall, "f-measure":fmeasure }
    
    return scores


def build_report(main_set_scores, scores_per_miniset, scores_per_audiofile, save_path, main_set_name="EVAL", team_name="test_team" , **kwargs):
    

    # datetime object containing current date and time
    now = datetime.now()
    date_string = now.strftime("%d%m%Y_%H_%M_%S")
    # print("date and time =", date_string)	

    #make dict:
    report={
            'team_name':team_name,
            "set_name": main_set_name,
            "report_date": date_string,
            "overall_scores": main_set_scores,
            "scores_per_subset": scores_per_miniset,
            "scores_per_audiofile": scores_per_audiofile
    }
    if kwargs['scores_per_class']:
        report["scores_per_class"] =  kwargs['scores_per_class']

    with open(os.path.join(save_path,"Evaluation_report_"+team_name+"_"+main_set_name+'_'+date_string+'.json'), 'w') as outfile:
        json.dump(report, outfile)

    return



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-pred_file_path', type=str, help='path to folder with prediction csv')
    parser.add_argument('-ref_file_path', type=str, help='path to the reference events csvs folder')
    parser.add_argument('-metadata', type=str, help="path for metadata json (map between audiofiles and classes")
    # parser.add_argument('-team_name', type=str, help='team identification') # make this optional?
    parser.add_argument('-dataset', type=str, help="which set to evaluate: EVAL, VAL, TRAIN")
    parser.add_argument('-savepath', type=str, help="path where to save the report to")
    args = parser.parse_args()
    print(args)


    #read Gt file structure: get subsets and paths for ref csvs make an inverted dictionary with audiofilenames as keys and folder as value
    gt_file_structure = {}
    gt_file_structure[args.dataset] = {}
    inv_gt_file_structure = {}
    list_of_subsets = os.listdir(args.ref_file_path)
    for subset in list_of_subsets:
        gt_file_structure[args.dataset][subset] = os.listdir(os.path.join(args.ref_file_path,subset))

        for audiofile in gt_file_structure[args.dataset][subset]:
            inv_gt_file_structure[audiofile] = subset


    #read prediction csv
    pred_csv = pd.read_csv(args.pred_file_path, dtype=str)
    #  parse prediction csv
    # split file into lists of events for the same audiofile.
    pred_events_by_audiofile = dict(tuple(pred_csv.groupby('Audiofilename')))

    counts_per_audiofile = {}
    for audiofilename in list(pred_events_by_audiofile.keys()):
        print(audiofilename)
        
        # find wich subset the audiofile belongs to:
        
        # for each audiofile list, load correcponding GT File (audiofilename.csv)
        ref_events_this_audiofile = pd.read_csv(os.path.join(args.ref_file_path, inv_gt_file_structure[audiofilename], audiofilename[0:-4]+'.csv'), dtype=str)
        # compare and get counts: TP, FP .. 
        TP, FP, FN , total_n_events_in_audiofile= compute_TP_FP_FN(pred_events_by_audiofile[audiofilename], ref_events_this_audiofile )

        counts_per_audiofile[audiofilename]={"TP": TP, "FP": FP, "FN": FN, "total_n_pos_events": total_n_events_in_audiofile}
   

    # this is optional:
    if args.metadata:
        # using the key for classes => audiofiles,  # load sets metadata:
        with open(args.metadata) as metadatafile:
                dataset_metadata = json.load(metadatafile)
    else:
        dataset_metadata = copy.deepcopy(gt_file_structure)

    # include audiofiles for which there were no predictions:
    list_all_audiofiles = []
    for miniset in dataset_metadata[args.dataset].keys():
        if args.metadata:
            for cl in dataset_metadata[args.dataset][miniset].keys():
                list_all_audiofiles.extend(dataset_metadata[args.dataset][miniset][cl] )
        else:
            list_all_audiofiles.extend(dataset_metadata[args.dataset][miniset])

    for audiofilename in list_all_audiofiles:
        if audiofilename+".wav" not in counts_per_audiofile.keys():
            ref_events_this_audiofile = pd.read_csv(os.path.join(args.ref_file_path, inv_gt_file_structure[audiofilename], audiofilename+'.csv'), dtype=str)
            total_n_pos_events_in_audiofile =  len(select_events_with_value(ref_events_this_audiofile, value = 'POS'))
            counts_per_audiofile[audiofilename+".wav"] = {"TP": 0, "FP": 0, "FN": total_n_pos_events_in_audiofile, "total_n_pos_events": total_n_pos_events_in_audiofile}
    


        
    # aggregate the counts per class: 
    list_sets_in_mainset = list(dataset_metadata[args.dataset].keys())
    # list_classes_in_mainset = []
    # counts_per_class = {}
    counts_per_class_per_set = {}
    scores_per_class_per_set={}
    counts_per_set = {}
    scores_per_set = {}
    scores_per_audiofile={}
    for data_set in list_sets_in_mainset:
        print(data_set)


        if args.metadata:
            # list_classes_in_mainset.extend(list(dataset_metadata[args.dataset][data_set].keys()))        
            list_classes_in_set = list(dataset_metadata[args.dataset][data_set].keys())

            counts_per_class_per_set[data_set] = {}
        
            for cl in list_classes_in_set:
                print(cl)
                list_audiofiles_this_class = dataset_metadata[args.dataset][data_set][cl]
                tp = 0
                fn = 0
                fp = 0
                total_n_pos_events_this_class = 0
                for audiofile in list_audiofiles_this_class:
                    scores_per_audiofile[audiofile] = compute_scores_from_counts(counts_per_audiofile)

                    tp = tp + counts_per_audiofile[audiofile+".wav"]["TP"]
                    fn = fn + counts_per_audiofile[audiofile+".wav"]["FN"]
                    fp = fp + counts_per_audiofile[audiofile+".wav"]["FP"]
                    total_n_pos_events_this_class = total_n_pos_events_this_class + counts_per_audiofile[audiofile+".wav"]["total_n_pos_events"]
                
                # counts_per_class[cl] = {"TP":tp, "FN": fn, "FP": fp, "total_n_pos_events_this_class": total_n_pos_events_this_class}
                counts_per_class_per_set[data_set][cl] = {"TP":tp, "FN": fn, "FP": fp, "total_n_pos_events_this_class": total_n_pos_events_this_class}

            #  compute scores per class. # aggregate scores per sets: will this be an average of scores across classes of each set?
            scores_per_class_per_set[data_set], scores_per_set[data_set] = compute_scores_per_class_and_average_scores_per_set(counts_per_class_per_set[data_set])  
        
        
        else:
            list_audiofiles_in_set = dataset_metadata[args.dataset][data_set]
            tp = 0
            fn = 0
            fp = 0
            total_n_pos_events_this_set = 0
            for audiofile in  list_audiofiles_in_set:

                scores_per_audiofile[audiofile] = compute_scores_from_counts(counts_per_audiofile)
                tp = tp + counts_per_audiofile[audiofile+".wav"]["TP"]
                fn = fn + counts_per_audiofile[audiofile+".wav"]["FN"]
                fp = fp + counts_per_audiofile[audiofile+".wav"]["FP"]
                total_n_pos_events_this_set = total_n_pos_events_this_set + counts_per_audiofile[audiofile+".wav"]["total_n_pos_events"]
                counts_per_set[data_set] = {"TP":tp, "FN": fn, "FP": fp, "total_n_pos_events_this_set": total_n_pos_events_this_class}
            
            #  compute scores per class. # aggregate scores per sets: will this be an average of scores across classes of each set?
            scores_per_set[data_set]= compute_scores_from_counts(counts_per_set[data_set])
                    

    #average scores per all sets in the eval set 
    # av(dc_scores, ME_scores, ML_scores)
    
    Overall_scores = {"precision" : stats.hmean([scores_per_set[dt]["precision"] for dt in scores_per_set.keys()]) , 
                    "recall":  stats.hmean([scores_per_set[dt]["recall"] for dt in scores_per_set.keys()]) ,
                    "fmeasure":  stats.hmean([scores_per_set[dt]["f-measure"] for dt in scores_per_set.keys()])
                    }
   
    
    if args.metadata:
        build_report(Overall_scores, scores_per_set, scores_per_audiofile,
                scores_per_class=scores_per_class_per_set,
                save_path="/mnt/c/Users/madzi/Dropbox/QMUL/PHD/DCASE2021_few-shot_bioacoustics_challenge/dcase-few-shot-bioacoustic/", 
                main_set_name = args.dataset )
    else:
        build_report(Overall_scores, scores_per_set, scores_per_audiofile,
                save_path="/mnt/c/Users/madzi/Dropbox/QMUL/PHD/DCASE2021_few-shot_bioacoustics_challenge/dcase-few-shot-bioacoustic/", 
                main_set_name=args.dataset  )



# v.1 add possibility for several prediction files
# v.2 adapt code for csvs of training set where each audiofile has more than one class present.

