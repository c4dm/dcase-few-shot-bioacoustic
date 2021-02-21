# Prototypical_Network

This is the deep learning baseline code for DCASE task 5. Prototypical networks were introduced by <a href="https://arxiv.org/abs/1703.05175">Snell et. al in 2017</a>. The core idea of the methodology is to learn an emebedding space where points cluster around a single prototype representation of each class. A non-linear mapping from the input space to embedding space is learnt using a convolutional neural network. Class prototype is calculated by taking a mean of its support set in the embedding space. Classification of a query point is conducted by finding the nearest class prototype.

# Episodic training

Prototypical networks adopt an episodic training procedure where in each episode, a mini-batch is sampled from the dataset ensuring that each class has an equal representation, post which the a subset of the mini batch is used as the support set to train the model and the remainder data is used as query set. The intention of episodic training is to replicate a few-shot learning task.

The positive annotations in the training data are of unequal duration, hence we extract equal length patches from the annotated segments, where each patch inherits the label of its corresponding annotation. The training set is heavily imbalanced in terms of class distribution, hence we balance the dataset using oversampling. 

# Evaluation

In evaluation stage, each audio file is split in the same manner as done during training stage. Since there is only one class per audio file in the validation set, we adopt a binary classification strategy inspired from <a href="https://arxiv.org/abs/2008.02791">Wang el. al</a>. We use the 5 first positive (POS) annotations for calculation of positive class prototype and consider the entire audio file as negative class based on the assumption that the positive class is relatively sparse as compared to the entire track. 

We randomly sample from the negative class to calculate the negative prototype. Each query sample is assigned a probability based on the distance from the positive and negative prototype. Onset and offset prediction is made based on thresholding the probabilities across the query set. 

# Files

1) Features_extract.py: For extracting the features

2) Datagenerator.py: For creating training, validation and evaluation set

3) batch_sample1.py : Batch sampler

4) Model.py : Prototypical network

5) util.py : This file contains the prototypical loss function and prototype evaluation function. The evaluation function is used for calculating negative and positive prototypes during evaluation stage, post which onset offset predictions are made. 

6) config.yaml : Consists of all the control parameters from feature extraction and training. 

# Running the code

We use hydra framework (https://hydra.cc/docs/intro/) for configuration management. To run the code:

Feature Extraction:

1) We use config.yaml file to store the configuration parameters.
2) To set the root directory and begin the feature extraction run the following command in terminal:
```
python main.py path.root_dir= root_dir set.features=true

e.g. python main.py path.root_dir=/Bird_dev_train set.features=true
```
The training, evaluation, model and feature directories have been set relative to the root directory path. You can choose to set them based on your preference.

Training:

Run the following command 

```
python main.py set.train=true
```
Evaluation:

For evaluation, either place the evaluation_metric code in the same folder as the rest of the code or include the path of the evaluation code in the eval section of the config file. Run the following command for evaluation:

```
python main.py set.eval=true
