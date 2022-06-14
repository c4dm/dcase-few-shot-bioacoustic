# Few-shot Bioacoustic Event Detection (DCASE 2022 Task 5) source code
This is the official repository for Few-shot Bioacoustic Event Detection (Task 5 in the DCASE Challenge 2022). This repository contains the source code to run the evaluation metrics, and the baseline systems, along with a detailed description of the task. 

# Task Description
**Few-shot learning is a highly promising paradigm for sound event detection. It is also an extremely good fit to the needs of users in bioacoustics, in which increasingly large acoustic datasets commonly need to be labelled for events of an identified category** (e.g. species or call-type), even though this category might not be known in other datasets or have any yet-known label. While satisfying user needs, this will also benchmark few-shot learning for the wider domain of sound event detection (SED).

<p align="center"><img src="https://github.com/c4dm/dcase-few-shot-bioacoustic/blob/main/VM.png" alt="figure" width="500"/></p>

**Few-shot learning describes tasks in which an algorithm must make predictions given only a few instances of each class, contrary to standard supervised learning paradigm.** The main objective is to find reliable algorithms that are capable of dealing with data sparsity, class imbalance and noisy/busy environments. Few-shot learning is usually studied using N-way-K-shot classification, where N denotes the number of classes and K the number of examples for each class.

**Some reasons why few-shot learning has been of increasing interest:**
+ Scarcity of supervised data can lead to unreliable generalisations of machine learning models; 
+ Explicitly labeling a huge dataset can be costly both in time and resources;
+ Fixed ontologies or class labels used in SED and other DCASE tasks are often a poor fit to a given user’s goal.

# Development Set
The development set is pre-split into training and validation sets. The training set consists of five subfolders deriving from a different source each. Along with the audio files multi-class annotations are provided for each. The validation set consists of three sub-folders deriving from a different source each, with a single-class (class of interest) annoation file provided for each audio file. 

## Training Set 
The training set contains five different sub-folders (BV, HV, JD, MT, WMW). Statistics are given overall and specific for each sub-folder. 

### Overall

| Statistics | Values |
| --- | --- |
|Number of audio recordings		|	174|
|Total duration					|	21 hours|
|Total classes					|	47|
|Total events					|	14229|

### BV

The BirdVox-DCASE-10h (BV for short) contains five audio files from four different autonomous recording units, each lasting two hours. These autonomous recording units are all located in Tompkins County, New York, United States. Furthermore, they follow the same hardware specification: the Recording and Observing Bird Identification Node (ROBIN) developed by the Cornell Lab of Ornithology. Andrew Farnsworth, an expert ornithologist, has annotated these recordings for the presence of flight calls from migratory passerines, namely: American sparrows, cardinals, thrushes, and warblers. In total, the annotator found 2,662 positive events from 11 different species. We estimate these flight calls to have a duration of 150 milliseconds and a fundamental frequency between 2 kHz and 10 kHz.

| Statistics | Values |
| --- | --- |
| Number of audio recordings		|	5 |
| Total duration					|	10 hours |
| Total events 					|	9026|
|Ratio event/duration			|	0.04|
|Sampling rate					|	24000 Hz |

### HT

Spotted hyenas are a highly social species that live in "fission-fusion" groups where group members range alone or in smaller subgroups that split and merge over time. Hyenas use a variety of types of vocalizations to coordinate with one another over both short and long distances. Spotted hyena vocalization data were recorded on custom-developed audio tags designed by Mark Johnson and integrated into combined GPS / acoustic collars (Followit Sweden AB) by Frants Jensen and Mark Johnson. Collars were deployed on female hyenas of the Talek West hyena clan at the <a href="https://www.holekamplab.org/">MSU-Mara Hyena Project</a> (directed by Kay Holekamp) in the Masai Mara, Kenya as part of a multi-species <a href="https://www.movecall.group/">study on communication and collective behavior</a>. Field work was carried out by Kay Holekamp, Andrew Gersick, Frants Jensen, Ariana Strandburg-Peshkin, and  Benson Pion; labeling was done by Kenna Lehmann and colleagues.

| Statistics | Values |
| --- | --- |
| Number of audio recordings		|	5 |
| Total duration					|	5 hours |
| Total events 					|	611|
|Ratio event/duration			|	0.05|
|Sampling rate					|	6000 Hz |

### JD

Jackdaws are corvid songbirds which usually breed, forage and sleep in large groups, but form a pair bond with the same partner for life. They produce thousands of vocalisations per day, but many aspects of their vocal behaviour remained unexplored due to the difficulty in recording and assigning vocalisations to specific individuals, especially in natural settings. In a multi-year field study (Max-Planck-Institute for Ornithology, Seewiesen, Germany), wild jackdaws were equipped with small backpacks containing miniature voice recorders (Edic Mini Tiny A31, TS-Market Ltd., Russia) to investigate the vocal behaviour of individuals interacting normally with their group, and behaving freely in their natural environment. The jackdaw training dataset contains a 10-minute on-bird sound recording of one male jackdaw during the breeding season 2015. Field work was conducted by Lisa Gill, Magdalena Pelayo van Buuren and Magdalena Maier. Sound files were annotated by Lisa Gill, based on a previously established video-validation in a captive setting.

| Statistics | Values |
| --- | --- |
| Number of audio recordings		|	1 |
| Total duration					|	10 mins |
| Total classes 					|	1 |
| Total events 						|	357 |
| Ratio event/duration			|	0.06 |
| Sampling rate					|	22,050 Hz |

### MT

Meerkats are a highly social mongoose species that live in stable social groups and use a variety of distinct vocalizations to communicate and coordinate with one another. Meerkat vocalization data were recorded at the <a href="https://kalahari-meerkats.com/kmp/">Kalahari Meerkat Project</a> (Kuruman River Reserve, South Africa; directed by Marta Manser and Tim Clutton-Brock), as part of a multi-species <a href="https://www.movecall.group/">study on communication and collective behavior</a>. Data in the training set were recorded on small audio devices (TS Market, Edic Mini Tiny+ A77, 8 kHz) integrated into combined GPS/audio collars which were deployed on multiple members of meerkat groups to monitor their movements and vocalizations simultaneously. Recordings were carried out during daytime hours while meerkats were primarily foraging (digging in the ground for small prey items). Field work was carried out by Ariana Strandburg-Peshkin, Baptiste Averly, Vlad Demartsev, Gabriella Gall, Rebecca Schaefer and Marta Manser. Audio recordings were labeled by Baptiste Averly, Vlad Demartsev, Ariana Strandburg-Peshkin, and colleagues.


| Statistics | Values |
| --- | --- | 
| Number of audio recordings		|	2 |
| Total duration					|	1 hour and 10 mins |
| Total classes 					|	4 |
| Total events 						|	1294 |
| Ratio event/duration			|	0.04 |
| Sampling rate					|	8,000 Hz |

### WMW

WMW consist on a selection of recordings from the <a href="https://zenodo.org/record/5093173?token=eyJhbGciOiJIUzUxMiIsImV4cCI6MTYzOTc4MTk5OSwiaWF0IjoxNjM3MTY3Nzc3fQ.eyJkYXRhIjp7InJlY2lkIjo1MDkzMTczfSwiaWQiOjE4NDAxLCJybmQiOiI5ZjBjODY3ZCJ9.Jbxn_ia64IvfYAfOvet0IBHoacyvMAasfXUatUSqBKa339Xqeo0Ee5Ccg2Lf8QoGhEjqy5NZ_6D1dQijRT0xVw#.Yh9t5OjP1D9"> Western Mediterranean Wetlands Bird dataset</a>. The recordings are taken from the Xeno-Canto portal. The present selection consists in 161 audio recordings of different lengths that have at least 10 positive events. These have been annotated for 26 different classes of 20 species of birds.


| Statistics | Values |
| --- | --- | 
| Number of audio recordings		|	161 |
| Total duration					|	4 hours and 40 mins |
| Total classes 					|	26 |
| Total events 						|	2941 |
| Ratio event/duration			|	0.24 |
| Sampling rate					|	various |

### Training annotation format
Annotation files have the same name as their corresponding audiofiles with extension `*.csv`. For the training set multi-class annotations are provided, with positive (POS), negative (NEG) and unknown (UNK) values for each class. UNK indicates uncertainty about a class and participants can choose to ignore it. 

Example of an annotation file for `audio.wav`:
```
Audiofilename,Starttime,Endtime,CLASS_1,CLASS_2,...,CLASS_N
audio.wav,1.1,2.2,POS,NEG,...,NEG
.
.
.
audio.wav,99.9,100.0,UNK,UNK,...,NEG
```

## Validation Set
The validation set comprises of three sub-folders (HB, PB, ME). Specific information about the source of the recordings and target classes should not be used to help development of the submitted systems, since the corresponding info for the Evaluation set is not going to be provided for the participants for the duration of the challenge.
Participants should treat the validation set in a similar way to the evaluation set.

**There is no overlap between the training set and validation set classes.** 


### Overall

| Statistics | Values |
| --- | --- |
| Number of audio recordings		|	18 |
| Total duration					|	5 hours and 57 minutes |
| Total classes 					|	5 |
| Total events 						|	972 |


### HB

| Statistics | Values |
| --- | --- |
| Number of audio recordings		|	10 |
| Total duration					|	2 hours and 38 minutes |
| Total classes 					|	1 |
| Total events 						|	607 |
|Ratio event/duration			|	0.7 |
| Sampling rate					|	44100 Hz |

### PB

| Statistics | Values |
| --- | --- |
| Number of audio recordings		|	6 |
| Total duration					|	3 hours |
| Total classes 	|	2 |
| Total events 		|	292 |
| Ratio event/duration			|	0.003|
| Sampling rate					|	44100 Hz |

### ME

| Statistics | Values |
| --- | --- |
| Number of audio recordings		|	2 |
| Total duration					|	20 minutes |
| Total classes 	|	2 |
| Total events 	|	73 |
| Ratio event/duration			|	0.01|
| Sampling rate					|	48000 Hz |


### Validation annotation format
Annotation files have the same name as their corresponding audiofiles with extension `*.csv`. For the validation set single-class (class of interest) annotations are provided, with positive (POS), unkwown (UNK) values. UNK indicates uncertainty about a class and participants can choose to ignore it. Each audio file should be treated separately of the rest, as there is possible overlap between the classes of the evaluation set across different audio files.

**Participants must treat the task as a 5-shot setting and only use the first five POS annotations for the class of interest for each file, when trying to predict the rest.**

Example of an annotation file for `audio_val.wav`:
```
Audiofilename,Starttime,Endtime,Q
audio_val.wav,1.1,2.2,POS
.
.
.
audio_val.wav,99.9,100.0,UNK
```

# Evaluation Set
The evaluation set has been released! 

it consists of 46 audio files acquired from different bioacoustic sources organized by 6 subsets (DC, CHE, MGE, MS, QU)
At this time only the first 5 annotations are provided for each file, with events marked as positive (POS) for the class of interest. 
The annotation files follow the same format as for the validation set. 
This dataset is to be used for evaluation purposes during the task and the rest of the annotations will be released after the end of the DCASE 2022 challenge (July 1st) together with some extra information related to target classes.

# Downloads

### DCASE 2022 Task 5: Few-shot Bioacoustic Event Detection Development Set 
(last release 25th April)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6012309.svg)](https://doi.org/10.5281/zenodo.6012309)

### DCASE 2022 Task 5: Few-shot Bioacoustic Event Detection Evaluation Set 
(released 1st June 2022 - the complete annotations wil be released after the challenge ends)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6517413.svg)](https://doi.org/10.5281/zenodo.6517413)


**There is no overlap between the development set and evaluation set classes.** 

Each audio file will be accompanied by an single-class (class of interest) annotation file that will only contain the first five events of the class, making this a 5-shot problem. Participants will be asked to perform sound event detection for the class of interest for the rest of the audio file. Each audio file should be treated separately of the rest, as there is possible overlap between the classes of the evaluation set across different audio files.


# Task setup

**This few-shot task will run as a 5-shot task.** Hence, five annotated calls from each recording in the evaluation set will be provided to the participants. Each recording of the evaluation set will have a single class of interest which the participants will then need to detect through the recording. Each recording can have multiple types of calls or species present in it, as well as background noise, however only the label of interest needs to be detected.

**During the develpoment period the partcipants are required to treat the validation set in the same way as the evaluation set by using the first five positive (POS) events for their models.** Participants should keep in mind that our evaluation metric ignores anything before the end time of the fifth positive event, hence using randomly selected events from the validation set may lead to incorrect performance values.

# Task rules
+ Use of external data (e.g. audio files, annotations) is **allowed only after approval** from the task coordinators (contact: `i.dealmeidanolasco@qmul.ac.uk`). Typically these
external datasets should be public, open datasets.
+ Use of pre-trained models is **allowed only after approval** from the task coordinators (contact: `i.dealmeidanolasco@qmul.ac.uk`).
+ The development dataset (i.e. training and validation) can be augmented **without** the use of external data.
+ Participants are **not allowed** to make subjective judgments of the evaluation data, nor to annotate it.
+ Participants are **not allowed** to use extra annotations for the provided data.
+ Participants are **only allowed to use the first five positive (POS) annotations from each validation set annotation file** and use the rest for evaluation of their method.
+ Participants **must treat each file in the validation set independently** of the others (e.g. for prototypical networks do not save prototypes between audio files). This is due to the fact that there is possible overlap between them inside the validation set.

# Submission

Official challenge submission consists of:

+ System output file (`*.csv`)
+ Metadata file (`*.yaml`)
+ Technical report explaining in sufficient detail the method (`*.pdf`)

System output should be presented as a **single** text-file (in CSV format, with a header row as in the submission zip example below). 

For each system, meta information should be provided in a separate file, containing the task-specific information. This meta information enables fast processing of the submissions and analysis of submitted systems. Participants are advised to fill the meta information carefully while making sure all information is correctly provided.

We allow up to 4 system output submissions per participant/team. For each system, metadata should be provided in a separate file, containing the task specific information. All files should be packaged into a zip file for submission. Please make a clear connection between the system name in the submitted metadata (the `*.yaml` file), submitted system output (the `*.csv` file), and the technical report. The detailed information regarding the challenge information can be found in the Submission page.
Finally, for supporting reproducible research, we kindly ask from each participant/team to consider making available the code of their method (e.g. in GitHub) and pre-trained models.

**Please note:** automated procedures will be used for the evaluation of the submitted results. Therefore, the column names should be exactly as indicated in the example `*.csv` in the submission zip below and events in each file should be in order of start time.

<a href="https://github.com/c4dm/dcase-few-shot-bioacoustic/blob/main/dcase_2022_fewshot_submission_package.zip">Submission zip example</a>
  
# Evaluation Metric

We implemented an event-based F-measure, macro-averaged evaluation metric. We use IoU followed by bipartite graph matching. The evalution metric ignores the part of the file that contains the first five positive (POS) events and measure are estimated after the end time of the fitfh positive event for each file. Furthermore, real-world datasets contain a small number of ambiguous or unknown labels marked as UNK in the annotation files provided. This evaluation metrics treats these separately during evaluation, so as not to penalise algorithms that can perform better than a human annotator. **Final ranking of methods will be based on the overall F-measure for the whole of the evaluation set.**

<a href="https://github.com/c4dm/dcase-few-shot-bioacoustic/tree/main/evaluation_metrics">Access the evaluation metrics code</a>.

# Baseline Systems

Two baselines are provided:

+ Template matching with normalized cross-correlation in the spectrogram domain, common in bioacoustics
+ Prototypical networks, a deep learning approach designed for few-shot scenarios

For more details on our template matching baseline, please refer to:

LeBien, J., Zhong, M., Campos-Cerqueira, M., Velev, J. P., Dodhia, R., Ferres, J. L., & Aide, T. M. (2020). A pipeline for identification of bird and frog species in tropical soundscape recordings using a convolutional neural network. Ecological Informatics, 59, 101113.

For more details on prototypical networks, please refer to:

Snell, J., Swersky, K., & Zemel, R. S. (2017). Prototypical networks for few-shot learning. Advances in Neural Information Processing Systems.

Lastly, note that a recent publication has applied prototypical networks to few-shot speech recognition:

Wang, Y., Salamon, J., Bryan, N. J., & Bello, J. P. (2020). Few-shot sound event detection. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 81-85.


<div class="brepository-item" data-item="dcase2021-task5-baseline"></div>

## Baseline Performance

| System | F-measure | Precision | Recall |
| --- | --- | --- | --- |
| Template Matching		|	4.28% | 2.42% | 18.32% |
| Prototypical Network	|	  29.59% | 36.34% | 24.96%| 

Sound event detection via few-shot learning is a novel and challenging task, as reflected in the performance of the baseline systems. There is thus lots of scope for improving on these scores, and making a significant contribution to animal monitoring.

# Contact

Participants can contact the task organisers via email (i.dealmeidanolasco@qmul.ac.uk) or in the slack channel: <a href="https://join.slack.com/t/dcase/shared_invite/zt-12zfa5kw0-dD41gVaPU3EZTCAw1mHTCA">task-fewshot-bio-sed</a>

# Cite:
This task, the baselines and results for the 2021 edition are described in the paper: <a href="https://dcase.community/documents/workshop2021/proceedings/DCASE2021Workshop_Morfi_52.pdf">Morfi, Veronica, et al. "Few-Shot Bioacoustic Event Detection: A New Task at the DCASE 2021 Challenge." DCASE. 2021.</a>

