# Prototypical_Network

This is the deep learning baseline code for DCASE task 5. Prototypical networks were introduced by <a href="https://arxiv.org/abs/1703.05175">Snell et. al in 2017</a>. The core idea of the methodology is to learn an emebedding space where points cluster around a single prototype representation of each class. A non-linear mapping from the input space to embedding space is learnt using a convolutional neural network. Class prototype is calculated by taking a mean of its support set in the embedding space. Classification of a query point is conducted by finding the nearest class prototype.

# Episodic training

Prototypical networks adopt an episodic training procedure where in each episode, a mini-batch is sampled from the dataset ensuring that each class has an equal representation, post which a subset of the mini batch is used as the support set to train the model and the remaining data is used as query set. The intention of episodic training is to replicate a few-shot learning task.

The positive annotations in the training data are of unequal duration, hence we extract equal length patches from the annotated segments, where each patch inherits the label of its corresponding annotation. The training set is heavily imbalanced in terms of class distribution, hence we balance the dataset using oversampling. 

# Evaluation

In evaluation stage, each audio file is split in the same manner as done during training stage. Since there is only one class per audio file in the validation set, we adopt a binary classification strategy inspired from <a href="https://arxiv.org/abs/2008.02791">Wang el. al</a>. We use the 5 first positive (POS) annotations for calculation of positive class prototype and consider the entire audio file as negative class based on the assumption that the positive class is relatively sparse as compared to the entire track. 

We randomly sample from the negative class to calculate the negative prototype. Each query sample is assigned a probability based on the distance from the positive and negative prototype. Onset and offset prediction is made based on thresholding the probabilities across the query set. Since samples are selected randomly for calculating the negative prototype, the prediction process for each file is repeated 5 times to negate some amount of randomness. The final prediction probability for each query frame is the average of predictions across all iterations. 

# Files

1) Features_extract.py: For extracting the features

2) Datagenerator.py: For creating training, validation and evaluation set

3) batch_sample.py: Batch sampler

4) Model.py: Prototypical network

5) util.py: This file contains the prototypical loss function and prototype evaluation function. The evaluation function is used for calculating negative and positive prototypes during evaluation stage, post which onset offset predictions are made. 

6) config.yaml: Consists of all the control parameters from feature extraction and training. 

# Running the code

We use <a href="https://hydra.cc/docs/intro/">hydra framework</a> for configuration management. To run the code:

### Feature Extraction:

1) We use config.yaml file to store the configuration parameters.
2) To set the root directory and begin the feature extraction run the following command in terminal:
```
python main.py path.root_dir= root_dir set.features=true

e.g. python main.py path.root_dir=/Bird_dev_train set.features=true
```
The training, evaluation, model and feature directories have been set relative to the root directory path. You can choose to set them based on your preference.

### Training:

Run the following command 

```
python main.py set.train=true
```
### Evaluation:

For evaluation, either place the evaluation_metric code in the same folder as the rest of the code or include the path of the evaluation code in the eval section of the config file. Run the following command for evaluation:

```
python main.py set.eval=true
```
### Configuration for baseline results:

The reported results for the prototypical networks was achieved with the following configuration 

| Parameter | Value | 
| --- | --- | 
| Sampling rate		|	22050 | 
| n_fft	|	1024 (samples)|
| hop_length	|	256 (samples) |
| Segment length	|	0.2s |
| Hop length for segment	|	0.05s |
| Feature type	|	PCEN |
| N_way	|	10|
| K_shot	|	5|
| Training episodes	|12000|
| Number of samples for negative prototype	|	650|
| Number of iterations	|	5|

+ Per channel energy normalisation (PCEN) <a href="https://arxiv.org/abs/1607.05666">Wang el. al</a>. is conducted on mel frequency spectrogram and used as input
  feature. Raw audio is scaled to the range [-2**31; 2**31-1 ] before mel transformation. PCEN is performed using librosa (default parameters).  
+ Segment length refers to the equal length patches extracted from the time frequency representation. 
+ N_way - Number of classes used in support set for each episode during training. The configuration for query set is same as support set. 
+ K_shot - Number of samples per class in the support set.
+ Number of samples for negative prototype - The number of random samples selected from the entire audio file to calculate the negative prototype.
+ Number of iterations - Number of iterarations for calculating the final prediction/per audio file. 
# Post Processing

After predictions are produced, post processing is performed on the events. For each audio file,  There are two post processing methodologies - adaptive and fixed. In adaptive predicted events with shorter duration than 60% of the duration shortest shot provided for that file are removed. In fixed, any event less than 200 ms are removed. Code for adaptive post processing is in post_proc.py and code for fixed is in post_proc_new.py. The results on the DCASE page are from post_proc_new.py.
Run the following command for post processing on a .csv file:

```
python post_proc_new.py -val_path=./Development_Set/Validation_Set/ -evaluation_file=eval_output.csv -new_evaluation_file=new_eval_output.csv
```

### Config parameters:

#set

| Parameter | Value | 
| --- | --- | 
| Sampling rate		|	22050 | 
| n_fft	|	1024 (samples)|
| hop_length	|	256 (samples) |
| Segment length	|	0.2s |
| Hop length for segment	|	0.05s |
| Feature type	|	PCEN |
| N_way	|	10|
| K_shot	|	5|
| Training episodes	|12000|
| Number of samples for negative prototype	|	650|
| Number of iterations	|	5|
