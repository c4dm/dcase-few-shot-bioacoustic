# Description

This is the evaluation metrics for the DCASE 2021 task 5 Few-shot Bioacoustic Event Detection.

We implemented an event-based F-measure, macro-averaged evaluation metric. We use IoU followed by bipartite graph matching. The evalution metric ignores the part of the file that contains the first five positive (POS) events and measures are estimated after the end time of the fitfh positive event for each file. Furthermore, real-world datasets contain a small number of ambiguous or unknown labels marked as UNK in the annotation files provided. This evaluation metrics treats these separately during evaluation, so as not to penalise algorithms that can perform better than a human annotator. Final ranking of methods will be based on the overall F-measure for the whole of the evaluation set.

**Make sure the folder structure of the ground truth annotations are as downloaded from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4543504.svg)](https://doi.org/10.5281/zenodo.4543504)**

Example of how to run for the Validation set:

```
python evaluation.py -pred_file=baseline_template_val_predictions.csv -ref_files_path=./Development_Set/Validation_Set/ -team_name=TESTteam -dataset=VAL -savepath=./
 
```

An example prediction file can be downloaded <a href="https://github.com/c4dm/dcase-few-shot-bioacoustic/blob/main/dcase_2021_fewshot_submission_package.zip">here</a> 
