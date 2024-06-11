English | [简体中文](./README_cn.md) 
# AnomalousSoundDetection-Pytorch-KZ

# Introduction of DCASE2020 Task2
This Project is focuses on Anomalous Sound Detection based on Python(Pytorch), with the aim of detecting abnormal sounds from machines (such as the DCASE Challenge [DCASE2020 Task2](https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds)）.

DCASE2020 Task2: Unsupervised Detection of Anomalous Sounds for Machine Condition Monitoring.

Train an anomaly score calculator A with parameter $\theta$. The input of A is a target machine's operating sound x \in R^L and its machine information including Machine Type and Machine ID. and A outputs one anomaly score for the whole audio-clip x as A^{\theta}(x).
Then x is determined to be anomalous when the anomaly score exceeds the pre-defined threshold value.

Model A needs to be trained so that A takes a large value not only when the whole audio-clip x is anomalous but also when a part of x is anomalous, such as with collision anomalous sounds.

What's more, DCASE's other ASD tasks are as follows, that are closely related to ASD techniques can be focused on when studying ASD:
- 2021 Under Domain Shifted Conditions
- 2022 Applying Domain Generalization Techniques
- 2023 Few-Shot Unsupervised

### overview of this task
- 1 training an anomaly score calculator A, using training data
- 2 using A to calculate anomaly scores of all test samples of certain single type of machine with the same single machine ID.
- 3 repeating this procedure, calculate the AS of all test samples of all Machine Types and All Machine IDs.
- 4 using test datas to calculate the AUC and pAUC as evaluate indices

I implemented some ASD models as follows, which are all Reconstruction-Based models.
# Model Papers:
 - STgram-MFN(PointWiseCNN + CNN): [Anomalous Sound Detection using Spectral-Temporal Information Fusion](https://arxiv.org/abs/2201.05510)
 - AEGAN-AD(CNN+GAN): [Unsupervised anomaly detection and localization of Machine Audio A GAN-based Approach](https://arxiv.org/abs/2303.17949)
 - Anomaly Transformer(PointWiseCNN + Transformer): [Anomaly Transformer Time Series Anomaly Detection with Association Discrepancy](https://arxiv.org/abs/2110.02642)

# Environment
 - conda 4.10.3
 - Python 3.8.5
 - torch 1.13.0+cu115
 - torchaudio 0.13.0+cu116
 - Windows 10

### Module 1
1. ./train_stgrammfn.py
2. ./asdkit/trainer_stgrammfn.py
3. ./asdkit/models/stgrammfn.py
4. Dataset from DCASE2020
5. Dataset: ./asdkit/data_utils/reader.py class ASDDataset

### Preparing Data
The user needs to store the audio dataset in a certain path, for instance, the train dataset list is:
```commandline
 - /root/data/dcase2020-asd-task2/data/dataset/fun/train/
 - /root/data/dcase2020-asd-task2/data/dataset/pump/train/
 - /root/data/dcase2020-asd-task2/data/dataset/slider/train/
 - /root/data/dcase2020-asd-task2/data/dataset/ToyCar/train/
 - /root/data/dcase2020-asd-task2/data/dataset/Conveyor/train/
 - /root/data/dcase2020-asd-task2/data/dataset/valve/train/
```
This dataset consists of part of MIMII and ToyADMOS dataset.
 - [development dataset](https://zenodo.org/record/3678171)
 - [evaluation dataset](https://zenodo.org/record/3841772)
 - [additional training dataset](https://zenodo.org/record/3727685)

As a reference, my configurationa are as follows:
```
ASD-ReconBasedAugmentation-dcase20t2
└─train_rbasd.py
├─asdkit
│    └─models
│    │    └─autoencoder.py
│    │    └─tdnn.py
│    │    └─stgrammfn.py
│    │    └─mobilefacenet.py
│    │    └─trainer_settings.py
│    └─modules
│    └─data_utils
│    │    └─machine_sound_readers.py
│    │    └─audio.py
│    │    └─featurizer.py
│    │    └─dcase2020_fileutils.py
│    └─utils
│    │    └─logger.py
│    │    └─utils.py
│    │    │    └─logger.pt
└─configs
│    └─v3tfs.yaml
│    └─rbasd.yaml
└─datasets
│    └─metadata2label.json
│    └─train_list.txt
```

# Reference
1. https://github.com/yeyupiaoling/AudioClassification-Pytorch
2. https://github.com/liuyoude/STgram-MFN
