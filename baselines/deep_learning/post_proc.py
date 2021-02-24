import csv
import numpy as np
import os

val_path = './Development_Set/Validation_Set/'
evaluation_file = 'Eval_out.csv'
new_evaluation_file = 'new_eval_out.csv'

dict_duration = {}
folders = os.listdir(val_path)
for folder in folders:
    files = os.listdir(val_path+folder)
    for file in files:
        if file[-4:] == '.csv':
            audiofile = file[:-4]+'.wav'
            annotation = file
            events = []
            with open(val_path+folder+'/'+annotation) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    for row in csv_reader:
                        if row[-1] == 'POS' and len(events) < 5:
                            events.append(row)
            min_duration = 10000
            for event in events:
                if float(event[2])-float(event[1]) < min_duration:
                    min_duration = float(event[2])-float(event[1])
            dict_duration[audiofile] = min_duration
            
results = []
with open(evaluation_file, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the headers
    for row in reader:
        results.append(row)
        
new_results = [['Audiofilename', 'Starttime', 'Endtime']]
for event in results:
    audiofile = event[0]
    min_dur = dict_duration[audiofile]
    if float(event[2])-float(event[1]) >= min_dur:
        new_results.append(event)
        
with open(new_evaluation_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(new_results)
