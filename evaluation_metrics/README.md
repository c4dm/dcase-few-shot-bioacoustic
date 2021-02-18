# Description

This is the evaluation metrics for the DCASE 2021 task 5 Few-shot Bioacoustic Event Detection.

**Make sure the folder structure of the ground truth annotations are as downloaded for <a href="">Zenodo</a>** TODO

Example of how to run for the Validation set:

```
python evaluation.py -pred_file=baseline_template_val_predictions.csv -ref_files_path=/Validation_Set/ -team_name=TESTteam -dataset=VAL -savepath=./
 
```

An example prediction file can be downloaded <a href="">here</a>  TODO
