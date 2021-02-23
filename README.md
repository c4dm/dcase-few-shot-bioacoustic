# Repo Description
This is the official repository for Few-shot Bioacoustic Event Detection (Task 5 in the DCASE Challenge 2021). This repository contains the source code to run the evaluation metrics, and the baseline systems, along with a detailed description of the task. 

# Task Description
**Few-shot learning is a highly promising paradigm for sound event detection. It is also an extremely good fit to the needs of users in bioacoustics, in which increasingly large acoustic datasets commonly need to be labelled for events of an identified category** (e.g. species or call-type), even though this category might not be known in other datasets or have any yet-known label. While satisfying user needs, this will also benchmark few-shot learning for the wider domain of sound event detection (SED).

<p align="center"><img src="https://github.com/c4dm/dcase-few-shot-bioacoustic/blob/main/VM.png" alt="figure" width="500"/></p>

**Few-shot learning describes tasks in which an algorithm must make predictions given only a few instances of each class, contrary to standard supervised learning paradigm.** The main objective is to find reliable algorithms that are capable of dealing with data sparsity, class imbalance and noisy/busy environments. Few-shot learning is usually studied using N-way-K-shot classification, where N denotes the number of classes and K the number of examples for each class.

**Some reasons why few-shot learning has been of increasing interest:**
+ Scarcity of supervised data can lead to unreliable generalisations of machine learning models; 
+ Explicitly labeling a huge dataset can be costly both in time and resources;
+ Fixed ontologies or class labels used in SED and other DCASE tasks are often a poor fit to a given user’s goal.

# Development Set
The development set is pre-split into training and validation sets. The training set consists of four subfolders deriving from a different source each. Along with the audio files multi-class annotations are provided for each. The validation set consists of two sub-folders deriving from a different source each, with a single-class (class of interest) annoation file provided for each audio file. 

## Training Set 
The training set contains four different sub-folders (BV, HV, JD, MT). Statistics are given overall and specific for each sub-folder. 

### Overall

| Statistics | Values |
| --- | --- |
| Number of audio recordings		|	11 |
|Total duration					|	14 hours and 20 mins |
| Total classes (excl. UNK)		|	19 |
| Total events (excl. UNK)		|	4,686 |

### BV

The BirdVox-DCASE-10h (BV for short) contains five audio files from four different autonomous recording units, each lasting two hours. These autonomous recording units are all located in Tompkins County, New York, United States. Furthermore, they follow the same hardware specification: the Recording and Observing Bird Identification Node (ROBIN) developed by the Cornell Lab of Ornithology. Andrew Farnsworth, an expert ornithologist, has annotated these recordings for the presence of flight calls from migratory passerines, namely: American sparrows, cardinals, thrushes, and warblers. In total, the annotator found 2,662 from 11 different species. We estimate these flight calls to have a duration of 150 milliseconds and a fundamental frequency between 2 kHz and 10 kHz.

| Statistics | Values |
| --- | --- |
| Number of audio recordings		|	5 |
| Total duration					|	10 hours |
| Total classes (excl. UNK)		|	11 |
| Total events (excl. UNK)		|	2,662 |
| Sampling rate					|	24,000 Hz |

### HT

Spotted hyenas are a highly social species that live in "fission-fusion" groups where group members range alone or in smaller subgroups that split and merge over time. Hyenas use a variety of types of vocalizations to coordinate with one another over both short and long distances. Spotted hyena vocalization data were recorded on custom-developed audio tags designed by Mark Johnson and integrated into combined GPS / acoustic collars (Followit Sweden AB) by Frants Jensen and Mark Johnson. Collars were deployed on female hyenas of the Talek West hyena clan at the <a href="https://www.holekamplab.org/">MSU-Mara Hyena Project</a> (directed by Kay Holekamp) in the Masai Mara, Kenya as part of a multi-species <a href="https://www.movecall.group/">study on communication and collective behavior</a>. Field work was carried out by Kay Holekamp, Andrew Gersick, Frants Jensen, Ariana Strandburg-Peshkin, and  Benson Pion; labeling was done by Kenna Lehmann and colleagues.

| Statistics | Values |
| --- | --- |
| Number of audio recordings		|	3 |
| Total duration					|	3 hours |
| Total classes (excl. UNK)		|	3 |
| Total events (excl. UNK)		|	435 |
| Sampling rate					|	6,000 Hz |

### JD

Jackdaws are corvid songbirds which usually breed, forage and sleep in large groups, but form a pair bond with the same partner for life. They produce thousands of vocalisations per day, but many aspects of their vocal behaviour remained unexplored due to the difficulty in recording and assigning vocalisations to specific individuals, especially in natural settings. In a multi-year field study (Max-Planck-Institute for Ornithology, Seewiesen, Germany), wild jackdaws were equipped with small backpacks containing miniature voice recorders (Edic Mini Tiny A31, TS-Market Ltd., Russia) to investigate the vocal behaviour of individuals interacting normally with their group, and behaving freely in their natural environment. The jackdaw training dataset contains a 10-minute on-bird sound recording (44100 Hz) of one male jackdaw during the breeding season 2015. Field work was conducted by Lisa Gill, Magdalena Pelayo van Buuren and Magdalena Maier. Sound files were annotated by Lisa Gill, based on a previously established video-validation in a captive setting.

| Statistics | Values |
| --- | --- |
| Number of audio recordings		|	1 |
| Total duration					|	10 mins |
| Total classes (excl. UNK)		|	1 |
| Total events (excl. UNK)		|	355 |
| Sampling rate					|	22,050 Hz |

### MT

Meerkats are a highly social mongoose species that live in stable social groups and use a variety of distinct vocalizations to communicate and coordinate with one another. Meerkat vocalization data were recorded at the <a href="https://kalahari-meerkats.com/kmp/">Kalahari Meerkat Project</a> (Kuruman River Reserve, South Africa; directed by Marta Manser and Tim Clutton-Brock), as part of a multi-species <a href="https://www.movecall.group/">study on communication and collective behavior</a>. Data in the training set were recorded on small audio devices (TS Market, Edic Mini Tiny+ A77, 8 kHz) integrated into combined GPS/audio collars which were deployed on multiple members of meerkat groups to monitor their movements and vocalizations simultaneously. Data in the test set were recorded by an observer following a focal meerkat with a Sennheiser ME66 directional microphone (44.1 kHz) from a distance of < 1 m. Recordings were carried out during daytime hours while meerkats were primarily foraging (digging in the ground for small prey items). Field work was carried out by Ariana Strandburg-Peshkin, Baptiste Averly, Vlad Demartsev, Gabriella Gall, Rebecca Schaefer and Marta Manser. Audio recordings were labeled by Baptiste Averly, Vlad Demartsev, Ariana Strandburg-Peshkin, and colleagues.


| Statistics | Values |
| --- | --- | 
| Number of audio recordings		|	2 |
| Total duration					|	1 hour and 10 mins |
| Total classes (excl. UNK)		|	4 |
| Total events (excl. UNK)		|	1,234 |
| Sampling rate					|	8,000 Hz |

### Training annotation format
Annotation files have the same name as their corresponding audiofiles with extension `*.csv`. For the training set multi-class annotations are provided, with positive (POS), negative (NEG) and unkwown (UNK) values for each class. UNK indicates uncertainty about a class and participants can choose to ignore it. 

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
The validation set comprises of two sub-folders (HV, PB). Specific information about the source of the recordings are not provided for the participants for the duration of the challenge, as to make information available for the validation set as similar to the evaluation set (once that is also published). More information about both will be made available after the end of the challenge.

**There is no overlap between the training set and validation set classes.** 


### Overall

| Statistics | Values |
| --- | --- |
| Number of audio recordings		|	8 |
| Total duration					|	5 hours |
| Total classes (excl. UNK)		|	4 |
| Total events (excl. UNK)		|	310 |


### HV

| Statistics | Values |
| --- | --- |
| Number of audio recordings		|	2 |
| Total duration					|	2 hours |
| Total classes (excl. UNK)		|	2 |
| Total events (excl. UNK)		|	50 |
| Sampling rate					|	6,000 Hz |

### PB

| Statistics | Values |
| --- | --- |
| Number of audio recordings		|	6 |
| Total duration					|	3 hours |
| Total classes (excl. UNK)		|	2 |
| Total events (excl. UNK)		|	260 |
| Sampling rate					|	44,100 Hz |

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

**The evaluation set will be made available on June 1st 2021.** 

**There is no overlap between the development set and evaluation set classes.** 

Each audio file will be accompanied by an single-class (class of interest) annotation file that will only contain the first five events of the class, making this a 5-shot problem. Participants will be asked to perform sound event detection for the class of interest for the rest of the audio file. Each audio file should be treated separately of the rest, as there is possible overlap between the classes of the evaluation set across different audio files.

## Download

### DCASE 2021 Task 5: Few-shot Bioacoustic Event Detection Development Set

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4543504.svg)](https://doi.org/10.5281/zenodo.4543504)

# Task setup

**This few-shot task will run as a 5-shot task.** Hence, five annotated calls from each recording in the evaluation set will be provided to the participants. Each recording of the evaluation set will have a single class of interest which the participants will then need to detect through the recording. Each recording can have multiple types of calls or species present in it, as well as background noise, however only the label of interest needs to be detected.

**During the develpoment period the partcipants are required to treat the validation set in the same way as the evaluation set by using the first five positive (POS) events for their models.* Participants should keep in mind that our evaluation metric ignores anything before the end time of the fifth positive event, hence using randomly selected events from the validation set may lead to incorrect performance values.

# Task rules
+ Use of external data (e.g. audio files, annotations) is **allowed only after approval** from the task coordinators (contact: `g.v.morfi@qmul.ac.uk`)
+ Use of pre-trained models is **allowed only after approval** from the task coordinators (contact: `g.v.morfi@qmul.ac.uk`).
+ The development dataset (i.e. training and validation) can be augmented **without** the use of external data.
+ Participants are **not allowed** to make subjective judgments of the evaluation data, nor to annotate it.
+ Participants are **not allowed** to use extra annotations for the provided data.
+ Participants are **only allowed to use the first five positive (POS) annotations from each validation set annotation file** and use the rest for evaluation of their method.
+ Participants **must treat each file in the validation set independently** of the others (e.g. for prototypical networks do not save prototypes between audio files). This is due to the fact that the classes of the validation set are hidden and there is possible overlap between them inside the validation set.

# Submission

Official challenge submission consists of:

+ System output file (`*.csv`)
+ Metadata file (`*.yaml`)
+ Technical report explaining in sufficient detail the method (`*.pdf`)

System output should be presented as a **single** text-file (in CSV format, with a header row as in the submission zip example below). 

For each system, meta information should be provided in a separate file, containing the task-specific information. This meta information enables fast processing of the submissions and analysis of submitted systems. Participants are advised to fill the meta information carefully while making sure all information is correctly provided.

We allow up to 4 system output submissions per participant/team. For each system, metadata should be provided in a separate file, containing the task specific information. All files should be packaged into a zip file for submission. Please make a clear connection between the system name in the submitted metadata (the `*.yaml` file), submitted system output (the `*.csv` file), and the technical report. The detailed information regarding the challenge information can be found in the Submission page.
Finally, for supporting reproducible research, we kindly ask from each participant/team to consider making available the code of their method (e.g. in GitHub) and pre-trained models, after the challenge is over.

**Please note:** automated procedures will be used for the evaluation of the submitted results. Therefore, the column names should be exactly as indicated in the example `*.csv` in the submission zip below and events in each file should be in order of start time.

<a href="https://github.com/c4dm/dcase-few-shot-bioacoustic/blob/main/dcase_2021_fewshot_submission_package.zip">Submission zip example</a>
  
# Evaluation Metric

We implemented an event-based F-measure, macro-averaged evaluation metric. We use IoU followed by bipartite graph matching. The evalution metric ignores the part of the file that contains the first five positive (POS) events and measure are estimated after the end time of the fitfh positive event for each file. Furthermore, real-world datasets contain a small number of ambiguous or unknown labels marked as UNK in the annotation files provided. This evaluation metrics treats these separately during evaluation, so as not to penalise algorithms that can perform better than a human annotator. **Final ranking of methods will be based on the overall F-measure for the whole of the evaluation set.**

<a href="https://github.com/c4dm/dcase-few-shot-bioacoustic/tree/main/evaluation_metrics">Access the evaluation metrics code</a>.

# Baseline Systems

Two baselines are provided:
+ Spectrogram correlation template matching (common in bioacoustics)
+ Deep learning prototypical network (a modern machine-learning approach designed for few-shot scenarios)

<div class="brepository-item" data-item="dcase2021-task5-baseline"></div>

