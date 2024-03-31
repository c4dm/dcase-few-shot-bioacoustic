import argparse
import csv
import os
from statistics import mean

import numpy as np
from sklearn.preprocessing import minmax_scale


def post_processing(
    val_path, evaluation_file, new_evaluation_file, n_shots=5, threshold_length=0.200
):
    """Post processing of a prediction file by removing all events that have shorter duration than
    200 ms.

    Parameters
    ----------
    val_path: path to validation set folder containing subfolders with wav audio files and csv annotations
    evaluation_file: .csv file of predictions to be processed
    new_evaluation_file: .csv file to be saved with predictions after post processing
    n_shots: number of available shots
    """

    results = []
    with open(evaluation_file, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader, None)  # skip the headers
        for row in reader:
            results.append(row)

    new_results = [["Audiofilename", "Starttime", "Endtime"]]
    for event in results:
        audiofile = event[0]

        if float(event[2]) - float(event[1]) >= threshold_length:
            new_results.append(event)

    with open(new_evaluation_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(new_results)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-val_path", type=str, help="path to validation folder with wav and csv files"
    )
    parser.add_argument(
        "-evaluation_file", type=str, help="path and name of prediction file"
    )
    parser.add_argument(
        "-new_evaluation_file",
        type=str,
        help="name of prost processed prediction file to be saved",
    )

    args = parser.parse_args()

    post_processing(args.val_path, args.evaluation_file, args.new_evaluation_file)
