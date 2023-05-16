# Constrained Potential Games 

This includes the code for the experimental section of the paper entitled: "Computing Nash Equilibria in Potential Games with Private
Uncoupled Constraints"

Experiments are in the form of jupyter notebooks.



## Table of Contents
* [Google Colab ](#file-structure)
* [Download Dataset and/or Trained Model](#download-dataset)
* [Sequence of operation](#modes-of-operation)
    * [Create Docker Container](#ins-create-docker-container-ins)
    * [Run Essential Tests](#ins-run-essential-tests-ins)
    * [Model Training](#ins-model-training-ins)
    * [Model Predictions](#ins-model-predictions-ins)
---
### Theoretical Section

* [What is a protein classifier](#classifier-preliminaries)
* [Data understanding](#data-understanding)
* [Data preprocessing](#data-preprocessing)
* [Training](#training)
    * [Neural Network Architecture](#neural-network-architecture)
    * [Hyperparameter Tuning](#hyperparameter-sweep)
* [Further Improvements](#further-improvements)
* [References](#references)
---
## Repository Understanding
Below we find important information about the structure of the repository as well as how to interact with it.

### **File Structure**


    .
    ├── src                     # Source folder 
    │   ├── main                # Main function
    │   ├── test_main           # Tests to check main
    │   ├── train               # Training part
    │   └── predict             # Predictor part
    │      
    ├── checkpoint              # Path to save/load checkpoints
    │   └── ...                 
    │      
    ├── data                    # Folder to save image/other data
    │   └── ...          
    └── ...

### **Download Dataset and/or Trained Model**

Please follow the [link](https://www.kaggle.com/datasets/googleai/pfam-seed-random-split
) to download the dataset.

A checkpoint with a trained model can be found [here](https://drive.google.com/drive/folders/1m6UNxWYvFjQLccaF9oRSagqhEpA5gloa?usp=sharing)

Once downloaded, insert it into the current folder.

    .
    ├── src                             # Source folder 
    │      
    ├── checkpoint                      # Checkpoint path
    │   └── epoch=16-step=18053.ckpt                
    │      
    ├── random_split                      # Dataset
    │   ├── dev     
    │   ├── random_split     
    │   ├── test     
    │   └── train     
    │      
    └── ...


### **Sequence of operation**

The sequence in which we advise usage is:

1. Create a Docker container to isolate the execution of the classifier
2. Run tests to make sure the program works properly
3. Train the model
4. Predict the Pfam of any protein of your choice :+1:

### <ins> Create Docker Container </ins>

Make sure to [install docker](https://docs.docker.com/get-docker/) and build a docker image by using the following command:

`docker build -t pfam .`

After that you can run `./start-docker.sh` in the terminal to jump into a container. Feel free to change the arguments accordingly.

### <ins> Run Essential Tests </ins>

First, before running any code, it is advised to run the three custom tests created, following the health checks of the jupyter notebook provided in the problem description. There are three different tests:

* Test 1: **Check the dataset**
* Test 2: **Check the dataloader** 
* Test 3: **Check that the network works on a single mini-batch** 

To run the test, navigate to the `src` folder and execute the following command:

`pytest test_main.py`

### <ins> Model Training </ins>

A classifier has two distinct parts, a training and an inference part. Choose to either train or evaluate the network by mentioning one of the flags in the CLI. For example:

`python main.py --train-flag --checkpoint-path {$path/to/checkpoint} ` 

This will flip the training flag `on` which will start training the classifier, given the [PFAM dataset](https://www.kaggle.com/datasets/googleai/pfam-seed-random-split). Incorporate the `--checkpoint-path` for previously trained models. If no checkpoints are available, do not include the checkpoint-path argument. 

### <ins> Model Predictions </ins>

To make predictions using the test set provided run the following command.

`python main.py --test-flag --checkpoint-path {$path/to/checkpoint}` 

The predictor will notify the user in the terminal about the accuracy of the model saved in the `--chechpoint-path` against the test set.

---
## Classifier Preliminaries

The classifier described in [ProtCNN](https://www.biorxiv.org/content/10.1101/626507v3.full) maps the relationship between unaligned amino acid sequences and their functional classification, as defined in the Pfam database. It learns how to classify by using labeled data for training. After training, the classifier should be able to generalize and correctly classify previously unseen proteins. Below is a block diagram of how this multiclass classification works.
<p align="center">
    <img src="./data/readme_pics/protein.jpg" alt="drawing" width="700"/>
</p>

## Data Understanding

### Dataset Structure

The labeled data used to train the classifier comes from the [PFAM dataset](https://www.kaggle.com/datasets/googleai/pfam-seed-random-split). We first fetch the labeled data and get 1086741 `(sequence,family_association)` pairs. Check out the reader function [here](https://github.com/SS8295/ml_files/blob/be83b27f0705b39b8e939bde5af4775483b3262d/lib_data.py#L6) that reads the data from the dataset and saves it as a [pandas.Series](https://pandas.pydata.org/docs/reference/api/pandas.Series.html). Below is a pictorial representation of the dataset.

| Index | Sequence | Pfam |
| --- | --- | --- |
| 0 | `INIWK...FFYGGP` | `PF03080.15` |
| 1 | `AVQVQ...ARWAH` | `PF01804.18` |
| ... | ... | ... |
| 1086740 | `QGHVE...NIEVD` | `PF00677.17` |

### Imbalanced Dataset

The total number of samples greatly outweighs the number of possible classes, `1086741 >> 17929`. This means we will have multiple samples for each family class. Ideally, we would like to have samples uniformly distributed across the families and having a relatively equal number of samples for each family class. Unfortunately, if we plot the distribution of family sizes, we see that we are dealing with a **heavily imbalanced** dataset.

<p align="center">
    <img src="./data/readme_pics/family_sizes.png" alt="drawing" width="550"/>
</p>

### Variable Length Sequences

Moreover, we observe that the sequences are **variable in length**. Let us now plot a histogram of the sequence length to verify this.

<p align="center">
    <img src="./data/readme_pics/sequence_length.png" alt="drawing" width="550"/>
</p>

### Amino Acid Frequencies

Each (variable) sequence is a collection of multiple amino acids (AA). We track the total number of amino acids in the dataset and plot, yet again, a histogram to see the distribution of amino acids in the dataset.

<p align="center">
    <img src="./data/readme_pics/AA_freq.png" alt="drawing" width="550"/>
</p>

Some amino acids are pretty infrequent compared to others. Recall, the vertical axis is a log scale, henceforth, the `L` amino acid's frequency is orders of magnitude larger than the frequency of the `O` amino acid.

It makes sense to collect all infrequently occurring amino acids and group them into one distinct group called the `<unk>` group. We arbitrarily select to include in this group the 5 most infrequent characters: `{'X','U','B','O','Z'}`.

This is done to augment the classifier since it can now differentiate between a rare and a non-rare amino acid without the need of further differentiation between rare amino acids.

### Amino Acid Vocabulary

Grouping all infrequent amino acids into one group allows for the creation of an amino acid dictionary, with a size of 22. In the dictionary we also have a key-value pair for padding called `<pad>`. The dictionary is shown below:

| Key | Value |
| --- | --- |
|`<pad>`|`0`|
|`<unk>`|`1`|
| `A` | `2` |
| `B` | `3` |
| ... | ... |
| `Y` | `21` |

The code of the following function can be found [here]().

---
## Data Preprocessing

After understanding the data in hand, the next step in the machine learning loop is to separate the dataset into:
* Train dataset
* Dev dataset
* Test dataset
---
## Training

### Neural Network Architecture

For training, we use the [ProtCNN](https://www.biorxiv.org/content/10.1101/626507v3.full) architecture. The architecture is shown below:

```
ProtCNN(
  (model): Sequential(
    (0): Conv1d(22, 128, kernel_size=(1,), stride=(1,), bias=False)
    (1): ResidualBlock(
      (skip): Sequential()
      (bn1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv1): Conv1d(128, 128, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,), bias=False)
      (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv1d(128, 128, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
    )
    (2): ResidualBlock(
      (skip): Sequential()
      (bn1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv1): Conv1d(128, 128, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,), bias=False)
      (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv1d(128, 128, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
    )
    (3): MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (4): Lambda()
    (5): Linear(in_features=7680, out_features=17930, bias=True)
  )
  (train_acc): Accuracy()
  (valid_acc): Accuracy()
)
```

The network contains five stages and its block-representation is more clearly shown in the [paper](https://www.biorxiv.org/content/10.1101/626507v3.full).

<p align="center">
    <img src="./data/readme_pics/nn_arch.png" alt="drawing" width="550"/>
</p>

During training, [Weights & Biases](https://wandb.ai/site) were used to monitor and visualize training. Here is a snapshot of the accuracy and loss during training:

<p align="center">
    <img src="./data/readme_pics/train_acc.png" alt="drawing" width="550"/>
</p>

The whole run is publicly accessible and can be found [here](https://wandb.ai/stelios/uncategorized/reports/Pfam-Training--VmlldzoyNDMzNTU4?accessToken=dydsma18w412g32xj2cykq40mexf227fz2dhaet8y5hlyaspiotfar36y5oh2fvz). For more information about how to integrate PyTorch Lightning and Weights & Biases, check [this](https://wandb.ai/site/articles/pytorch-lightning-with-weights-biases) illustration from Weights & Biases.

<p align="center">
    <img src="./data/readme_pics/wandb_report.png" alt="drawing" width="600"/>
</p>

Saving and loading from a checkpoint is extremely useful as we can pause and resume training anytime. The previous run was paused and resumed due to a computer restart. Run can be found [here](https://wandb.ai/stelios/uncategorized/reports/Pfam-Training-Continued--VmlldzoyNDMzNTcx?accessToken=cyn6wjmuuxw0725vx2s757pjof4j61spsc98wqabjekky0q4ibjt0pj1gjuu0ri6)

### Hyperparameter Sweep

Hyperparameter tuning was omitted due to time constraints but can certainly be a pathway towards better performance or faster convergence. Weights & Biases has a native hyperparameter sweep that uses three different methods of sweeping:

1. Random Search
2. Bayesian Search
3. Grid Search (exhaustive)

The reader is encouraged to further investigate hyperparameter sweeps with various hyperparameters such as maximum sequence length, batch size, number of epochs, etc. These can all be part of a `config_dict` that is passed into the sweep.

---

## Further Improvements

Next we will discuss potential ways to improve ProtCNN for even better results. Before that, we look into some pros and cons of using the ProtCNN architecture.

1. Some notable advantages of using ProtCNN is its <ins>invariance in translation</ins> [1]. This works really well because the protein sequence data does not have a specific alignment.

2. ProtCNN allows information to be <ins>locally transmitted</ins> and creates local correlations of data without greatly increasing the number of model parameters.

3. Moreover, the <ins>dilated convolutions</ins> enable larger receptive field sizes to be obtained without an explosion in the number of model parameters. Therefore, an increase in the max_seq_length would (in principle) yield better performance. This is one of the main innovations of the ProtCNN architecture.

4. ProtCNN also enjoys high parallelizability due to the way the batches are created. Training can be parallelized both in the sequence length dimension as well as in the number of sequences.

### <ins>Ways to potentially improve ProtCNN</ins>

* Despite all these advantages, ProtCNN could benefit from replacing the CNN models with <ins>transformers</ins> instead of RNNs/LSTMs that suffer from problems like exploding gradients, applying gradient clipping, memory, etc.

* Define relationships between amino acids and use <ins>graph neural networks</ins> which are permutation invariant [2].

* Maybe InstaDeep's DeepChain can also help.

---
## References

[[1] Quantifying Translation-Invariance in Convolutional Neural Networks](https://arxiv.org/abs/1801.01450)


[[2] Protein Interface Prediction using Graph Convolutional Networks](https://proceedings.neurips.cc/paper/2017/file/f507783927f2ec2737ba40afbd17efb5-Paper.pdf)
