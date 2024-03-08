[English](./README_en.md) | 简体中文
# AnomalousSoundDetection-Pytorch-KZ

# Introduction of DCASE2020 Task2
该项目是基于Python（Pytorch）的异常声音检测（ASD），目的是检测机器的异常声音（例如DCASE挑战赛 [DCASE2020 Task2](https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds)）.

DCASE2020 Task2: Unsupervised Detection of Anomalous Sounds for Machine Condition Monitoring.

训练一个异常分数计算模型（参数 theta）。模型输入为目标机器的运转声音（整段音频）和该机器的Type与ID信息。输出为对应的异常分数（anomaly score)，如果anomaly score超过一个预定义的阈值，就认为是异常，其中，对于一段音频数据，无论是整段均为异常或者其中一个小片段为异常，都为人这条数据是异常的，由此能够囊括类似碰撞声这类很短的异常音频情况，而不仅仅是机器本身出了故障的声音。

评价指标：AUC和pAUC

官方Baseline：[AutoEncoder-Keras](https://github.com/y-kawagu/dcase2020_task2_baseline)

顺带一提，DCASE其他ASD任务如下，研究ASD的时候可以关注关注这些紧密相关的问题：
- 2021 Under Domain Shifted Conditions
- 2022 Applying Domain Generalization Techniques
- 2023 Few-Shot Unsupervised

### overview of this task
- 1 用train dataset训练模型A
- 2 用模型A计算test dataset里某种Machine Type某种Machine ID的数据的anomaly score.
- 3 重复上述过程，计算test dataset中所有Machine Types各个Machine IDs的anomaly score。
- 4 通过test 数据来计算AUC和pAUC作为评估指标。


实现了一些ASD模型，例如STgramMFN[Anomalous Sound Detection using Spectral-Temporal Information Fusion](https://arxiv.org/abs/2201.05510)

# Model Papers:
 - STgram-MFN: [Anomalous Sound Detection using Spectral-Temporal Information Fusion](https://arxiv.org/abs/2201.05510)
 - AEGAN-AD: [Unsupervised anomaly detection and localization of Machine Audio A GAN-based Approach](https://arxiv.org/abs/2303.17949)
 - Anomaly Transformer: [Anomaly Transformer Time Series Anomaly Detection with Association Discrepancy](https://arxiv.org/abs/2110.02642)

# Environment
 - conda 4.10.3
 - Python 3.8.5
 - torch 1.13.0+cu115
 - torchaudio 0.13.0+cu116
 - Windows 10

# Model Test

### Module1
- 1 ./train_stgrammfn.py
- 2 ./asdkit/trainer_stgrammfn.py
- 3 ./asdkit/models/stgrammfn
- 4 数据集来源：DCASE2020
- 5 Dataset对象: ./asdkit/data_utils/reader.py class ASDDataset 


### Preparing Data
首先需要把音频文件存储在某个目录下，例如训练数据列表：
```commandline
- /root/data/dcase2020-asd-task2/data/dataset/fun/train/
- /root/data/dcase2020-asd-task2/data/dataset/pump/train/
- /root/data/dcase2020-asd-task2/data/dataset/slider/train/
- /root/data/dcase2020-asd-task2/data/dataset/ToyCar/train/
- /root/data/dcase2020-asd-task2/data/dataset/Conveyor/train/
- /root/data/dcase2020-asd-task2/data/dataset/valve/train/
```
数据集由 MIMII 和 ToyADMOS 两个数据集组成，下载链接在zenono，如下（来源：DCASE2020）：
 - [development dataset](https://zenodo.org/record/3678171)
 - [evaluation dataset](https://zenodo.org/record/3841772)
 - [additional training dataset](https://zenodo.org/record/3727685)

作为参考，我的项目目录结构如下（实际数据集存在哪里无所谓，在configs.yaml设置正确的路径即可）:
```
root
└─DCASE2020Task2ASD
│    └─dataset
│    │    └─dev_data_fun/
│    │    └─dev_data_pump/
│    │    └─dev_data_slider/
│    │    └─dev_data_ToyCar/
│    │    └─dev_data_ToyConveyor/
│    │    └─dev_data_valve/
│    └─eval_dataset/
│    │    └─eval_data_test_fun/
│    │    ......
├─PythonFiles
│    └─AnomalousSoundDetection-Pytorch-KZ
│    │    └─train.py
│    │    └─eval.py
│    │    └─create_data.py
│    │    └─configs
│    │    │    └─stgrammfn.yaml
│    │    └─asdkits
│    │    │    └─trainer_stgrammfn.py
│    │    │    └─models/
│    │    │    │    └─stgrammfn.py
│    │    │    │    └─aeganad.py
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
重点设置以下参数（以我自己的设置为例）

数据集相关：
- dataset_conf.data_dir: "C:/DATAS/DCASE2020Task2ASD/dataset"
- dataset_conf.eval_dir: "C:/DATAS/DCASE2020Task2ASD/eval_dataset"
- dataset_conf.train_list: "dataset/train_list.txt"
- dataset_conf.valid_list: "dataset/valid_list.txt"
- dataset_conf.test_list: "dataset/test_list.txt"

训练超参数
- dataset_conf.dataLoder.batch_size: 256  # 训练集 batch_size
- train_conf.max_epoch: 10  # 训练轮数（每一轮验证一次）

其他
- use_model:"STgramMFN"  # 采用的模型
- cuda: True  # 是否使用GPU训练
- gmm_n: False
- result_dir: "./results"  # 结果保存的根目录
- train_conf.train_from_zero: True  # 是否从头训练，或者加载上次训练的 best_model 继续训练（需要设置比上次更大的 max_epoch）

### train
根据上面的参数设置，STgram-MFN训练得到结果如下：
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

# Reference
1. https://github.com/yeyupiaoling/AudioClassification-Pytorch
2. https://github.com/liuyoude/STgram-MFN
3. https://github.com/jianganbai/AEGAN-AD
4. https://github.com/thuml/Anomaly-Transformer

# 其他ASD任务介绍科普
- 2023 First-Shot Unsupervised ASD
- 2022 ASD Applying Domain Generalization
- 2021 ASD Under Domain Shifted Conditions
- 2020 Unsupervised Anomalous Sound Detection (ASD)

单纯的 Unsupervised ASD 是理想的条件，在此基础上，后续的比赛延伸出以下问题。

### ASD Under Domain Shifted Conditions
Domain Shifts 是指training data and test data 的声学特征的不同，这可能是因为 运转速度、机器负载、环境噪音的不同所导致的。

由于Domain Shift 的存在，normal sounds 会被错误地判定为 anomalous。在实际场景中，domain shift 意味着在复杂的实际情况中，training and testing phases 处于不同的机器操作条件中。
例如说产线上的传送带，其速度是连续变化的，也就会有无限种状态，不同时段也会有不同的速度限制，ASD系统必须连续监测传送带的所有可能的转速情况，包括各种不同转速的training data。除此之外，环境噪声条件随季节变化而不可控，这也是一种域转换。

给出2个挑战
1. 无监督训练，要检测到训练集中没有出现过的、未知的异常声音。（同2020）
2. 域转换，要在训练集数据和测试集数据的声学特征不同的情况下，表现出很好的性能。

给定的数据集，7种机器：
- dev dataset: 00 01 02， training data 全是normal，包含源域1000条和目标域3条
- dev dataset: 00 01 02， test data，包含源域 normal和anomaly各100条，目标域normal和anomaly各100条。
- eval dataset：03 04 05，包含源域和目标域，无标签。
- additional training data：03 04 05 全是normal data，源域1000条，目标域3条。

官方baseline： 
1. [AutoEncoder](https://github.com/y-kawagu/dcase2021_task2_baseline_ae)
2. [MobileNetV2](https://github.com/y-kawagu/dcase2021_task2_baseline_mobile_net_v2)

### ASD Applying Domain Generalization Techniques
解决 Domain Shifts 问题的一个方法是 track these shifts and use domain adaptation technique to adapt the model using the target domain data。然而，实际情况下采用域适应技术不仅成本高而且impractical。

因此就提出了 Domain Generalization，域泛化是指，use the source domain data to learn common features across different domains，从而模型能够泛化测试数据的源域和目标域。在训练集中，几乎没有目标域的数据。在以下四个场景下，域泛化比域适应技术更好。
1. 机器的物理参数不同所导致的域转换。虽然这很容易追踪，但是如果在很短时间内发生了参数变化，那么就很难去适应每一时刻的值改变了。
2. 环境条件所导致的域转换。因为背景噪音由很多因素导致，因此很难追踪其变化。因此一个不受这些改变影响的模型就很需要了。
3. 维修保养导致的域转换。经过维修或者局部替换之后的机器也会发生域转换。
4. 不同记录设备导致的域转换。不同地方安装的麦克风，不同制造商的麦克风，都有可能导致。

给出3个挑战：
1. Unsupervised, 预测训练集中未出现过的anomalous sounds。
2. Domain Shift, anomalies之外的因素也改变了training和test data之间的特点（domain shift）。
3. Domain Generalization, test data 样本中，不受 domain shifts 的影响的数据(即 source domain data) 和受 domain shifts 影响的数据 (即 target domain data) 混在了一起，每个样本的源域和目标域并不是特定的。因此，模型必须能够在相同的阈值下检测出异常, regardless of the domain。

数据集同2021，一共7种机器。

官方Baseline，同2021，AutoEncoder和MobileNetV2。

### First-Shot Unsupervised ASD
通过测试集微调超参数，对于一个新的机器类型是不可行的，过去几年的比赛设置中，训练集和测试集的机器类型相同，不同在于，不是同一台机器。于是就有参赛者通过测试机去微调到最佳超参数，从而提高某种机器的检测性能。

但实际场景中，这是做不到的，首先机器类型可能是全新的，并且测试数据的数量很少不足以进行微调。因此First-Shot任务的数据集，训练集和测试集的机器类型完全不同，另外每一种机器类型的机器数目都很少。
提出了4个挑战：
1. Unsupervised， Train a model using only normal sound.
2. Domain Generalization.
3. Train a model for a completely new machine type. 针对一个完全新的机器类型，不可以调整已经训练好的模型的超参数。因此，检测系统需要具有不依靠调整额外超参数就训练模型的能力。
4. Train a model using a limited number of machines from its machine type. 同类型的机器越多，采集到的该类型机器的数据，越能够提高检测的性能，但某种类型的机器本身就不多，因此系统需要能够在同类型机器数量较少的情况下训练模型。

官方Baseline：[AutoEncoder](https://arxiv.org/abs/2303.00455)