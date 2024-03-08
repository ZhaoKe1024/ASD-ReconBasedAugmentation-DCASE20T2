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

# Model Test

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
root
└─DCASE2020Task2ASD
│    └─dataset
│    │    └─dev_data_fun/fan/train/normal_id_**_********.wav
│    │    └─dev_data_fun/fan/train/{normal/anomaly}_id_**_********.wav
│    │    └─dev_data_pump/pump/train/normal_id_**_********.wav
│    │    └─dev_data_pump/pump/train/{normal/anomaly}_id_**_********.wav
│    │    └─......
│    └─eval_dataset/
│    │    └─eval_data_test_fun/fan/test/id_**_********.wav
│    │    └─......
├─PythonFiles
│    └─AnomalousSoundDetection-Pytorch-KZ
│    │    └─train.py
│    │    └─eval.py
│    │    └─create_data.py
│    │    └─configs
│    │    │    └─stgrammfn.yaml
│    │    └─datasets
│    │    │    └─train_list.txt
│    │    │    └─valid_list.txt
│    │    │    └─test_list.txt
│    │    │    └─metadata2label.json
│    │    │    └─label2metadata.json
│    │    └─asdkits
│    │    │    └─trainer_stgrammfn.py
│    │    │    └─models/
│    │    │    │    └─stgrammfn.py
│    │    │    │    └─loss.py
│    │    │    │    └─fc.py
│    │    │    │    └─emb_distance.py
│    │    │    └─utils
│    │    │    │    └─logger.py
│    │    │    │    └─scheduler.py
│    │    │    │    └─utils.py
│    │    │    └─metric
│    │    │    │    └─metrics.py
│    │    │    └─data_utils/
│    │    │    │    └─reader.py
│    │    │    │    └─utils.py
│    │    │    │    └─featurizer.py
│    │    │    │    └─collate.py
│    │    │    ......
```

### Modify the configs.yaml
setting some important parameters as follows（take mine as instance）

About datasets：
- dataset_conf.data_dir: "C:/DATAS/DCASE2020Task2ASD/dataset"
- dataset_conf.eval_dir: "C:/DATAS/DCASE2020Task2ASD/eval_dataset"
- dataset_conf.train_list: "dataset/train_list.txt"
- dataset_conf.valid_list: "dataset/valid_list.txt"
- dataset_conf.test_list: "dataset/test_list.txt"

hyperparameters
- dataset_conf.dataLoder.batch_size: 256  # training loader's batch_size
- train_conf.max_epoch: 10  # the times of training (validation per epoch)

others
- use_model:"STgramMFN"  # model to use
- cuda: True  # whether to use GPU
- gmm_n: False
- result_dir: "./results"  # root directory to save results
- train_conf.train_from_zero: True  # whether train from zero, or load best_model from last time and continue training (if True, it needs to set a larger max_epoch)

### train
training with above configurations, the output results as follows：
```
ToyCar
id,AUC,pAUC
1,0.7191 ,0.5895 
2,0.9471 ,0.8577 
3,0.9544 ,0.8849 
4,0.9995 ,0.9974 
Average,0.9050 ,0.8324 
ToyConveyor
id,AUC,pAUC
1,0.7873 ,0.6443 
2,0.5969 ,0.5281 
3,0.7501 ,0.6228 
Average,0.7114 ,0.5984 
fan
id,AUC,pAUC
0,0.4599 ,0.5162 
2,0.9692 ,0.8815 
4,0.7142 ,0.6962 
6,0.9343 ,0.9029 
Average,0.7694 ,0.7492 
pump
id,AUC,pAUC
0,0.8113 ,0.6776 
2,0.6558 ,0.5529 
4,0.9965 ,0.9816 
6,0.9056 ,0.7358 
Average,0.8423 ,0.7370 
slider
id,AUC,pAUC
0,0.9981 ,0.9907 
2,0.9673 ,0.8407 
4,0.9904 ,0.9494 
6,0.9037 ,0.6458 
Average,0.9649 ,0.8567 
valve
id,AUC,pAUC
0,0.9976 ,0.9876 
2,0.7331 ,0.5219 
4,0.9945 ,0.9711 
6,0.9503 ,0.7684 
Average,0.9189 ,0.8123 
Total Average,0.8520 ,0.7643 
```
### test

run `eval.py` then results will generated to `./results/STgram-MFN/test/*.csv` 



# Reference
1. https://github.com/yeyupiaoling/AudioClassification-Pytorch
2. https://github.com/liuyoude/STgram-MFN
3. https://github.com/jianganbai/AEGAN-AD
4. https://github.com/thuml/Anomaly-Transformer

# Introduction of Other ASD Tasks
- 2020 ASD **Unsupervised Detection** of Anomalous Sounds for Machine Condition Monitoring
- 2021 ASD Unsupervised Anomalous Sound Detection for Machine Condition Monitoring **under Domain Shifted Conditions**
- 2022 ASD Unsupervised Anomalous Sound Detection for Machine Condition Monitoring **Applying Domain Generalization Techniques**
- 2023 ASD **First-Shot Unsupervised** Anomalous Sound Detection for Machine Condition Monitoring
- 2024 ASD **First-Shot Unsupervised** Anomalous Sound Detection for Machine Condition Monitoring

### ASD Under Domain Shifted Conditions
The [DCASE2021T2 Domain Shifted Conditional](https://dcase.community/challenge2021/task-unsupervised-detection-of-anomalous-sounds)
1. we have to detect unknown anomalous sounds that were not observed in the given training data.
2. The task is performed under the conditions that the acoustic characteristics of the training data and the test data are different (i.e., domain shift).


### ASD Applying Domain Generation Techniques
The [DCASE2022T2 Domain Generalization task](https://dcase.community/challenge2022/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring) is a task that requires an efficient training scheme for unsupervised anomaly detection with highly unbalanced training data.(990 normal samples from the source domain and only 10 samples from the target domain are given for training) and that no domain label be given for testing target samples.

- There are seven machine types: ToyCar, ToyTrain, Fan, Gearbox, Bearing, Slide rail(Slider), Valve.
- Each machine type has three sections reflecting the different operating conditions of the machines.
- Each section has source and target domains representing domain shift conditions.
- Labels indicating source/target domains are given only for training. No domain label is given for testing.
- There are 990 files given as training data for the source domain, and only 10 are given for training the target domain. All files contain 10s of normal machine operating sound.
- No anomalous sound sample is given.

For testing systems, the Evaluation dataset of the DCASE2022T2:
- 50 normal and 50 anomaly files for source and target domains; 200 unlabeled files per section in each machine type.

### First-Shot Unsupervised ASD
The [DCASE2023T2 First-shot](https://dcase.community/challenge2023/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring)
In DCASE ASD challenges for machine condition monitoring held in 2020, 2021, 2022, some of the techniques made use of data from different section as anomaly samples. **Besides, some of the techniques relied on machine-type dependent hyperparameter tuning and tool ensembling based on the performance observation with the given anomaly samples**.

The task this year is to develop an ASD system that meets the following four requirements.
1. Unsupervised， Train a model using only normal sound.
2. Domain Generalization. 
3. Train a model for a completely new machine type.
4. Train a model using a limited number of machines from its machine type.

The First-Shot ASD is characterized as follows:
- No use of data from different sections(different machine instances). **Only a single section per each machine type is available; therefore no classification technique is applicable.**
- No hyperparameter tuning applied for dedicated machine type. **Machine types provided with the Evaluation datasets (to rank the sytstems) are completely different from the given machine types for the Development dataset**, therefore, no hyperparameter tuning nor ensemble tool tuning by checking the groundtruth is possible.
- No tool-ensemble applied for dedicated machine type.

Dataset:
- Training data: 990 source domain normal waveform, 10 target domain normal waveform.
- Test data: 50 source normal, 50 source anomaly, 50 target normal, 50 target anomaly.
- evaluation data: 200 clips, normal or anomaly, source or target is not specified.

Official Baseline：
A simple implementation of Mahalanobis metric as the baseline system.
- paper: [AutoEncoder](https://arxiv.org/abs/2303.00455)
- code: [dcase2023_task2_baseline_ae](https://github.com/nttcslab/dcase2023_task2_baseline_ae)