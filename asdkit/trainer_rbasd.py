#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/1/18 16:37
# @Author: ZhaoKe
# @File : trainer_rbasd.py
# @Software: PyCharm
import time

import matplotlib.pyplot as plt
import random
import yaml
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.utils.data import DataLoader
from asdkit.models.trainer_settings import get_model, get_pretrain_models
from asdkit.data_utils.dcase2020_fileutils import *
from asdkit.data_utils.machine_sound_readers import SpecAllReader
from asdkit.modules.func import accuracy
from asdkit.modules.loss import ContrastiveLoss
from asdkit.utils.utils import logger, save_csv, load_ckpt

from sklearn.manifold import TSNE
from asdkit.utils.plotting import plot_embedding_2D, get_heat_map

# saveimg_params = {"format": "svg", "marker_size": 8, "alpha": 0.8}
saveimg_params = {"format": "svg", "marker_size": 4, "alpha": 0.5}
names = ["ToyCar-id_01", "ToyCar-id_02", "ToyCar-id_03", "ToyCar-id_04",
         "ToyConveyor-id_01", "ToyConveyor-id_02", "ToyConveyor-id_03",
         "fan-id_00", "fan-id_02", "fan-id_04", "fan-id_06",
         "pump-id_00", "pump-id_02", "pump-id_04", "pump-id_06",
         "slider-id_00", "slider-id_02", "slider-id_04", "slider-id_06",
         "valve-id_00", "valve-id_02", "valve-id_04", "valve-id_06"]


def get_wavmel_settings(data_file, m2l_map, configs=None, mode="train", demo_test=False):
    loaders = []
    id_ma_map = {"valve": 5, "slider": 4, "pump": 3, "fan": 2, "ToyConveyor": 1, "ToyCar": 0}
    ano_id_map = {"normal": 0, "anomal": 1}
    ma_id_map = {5: "valve", 4: "slider", 3: "pump", 2: "fan", 1: "ToyConveyor", 0: "ToyCar"}
    print("---------------train dataset-------------")
    file_paths = []
    mtid_list = []
    mtype_list = []
    y_true_list = []
    with open(data_file, 'r') as fin:
        train_path_list = fin.readlines()
        if demo_test:
            train_path_list = random.choices(train_path_list, k=200)
        for item in train_path_list:
            parts = item.strip().split('\t')
            machine_type_id = int(parts[1])
            file_paths.append(parts[0])
            mtype_list.append(machine_type_id)
            machine_id_id = parts[2]
            meta = ma_id_map[machine_type_id] + '-id_' + machine_id_id
            mtid_list.append(m2l_map[meta])
            y_true_list.append(int(item.strip().split('\t')[3]))
    if mode == "train":
        istrain, istest = True, False
    elif mode == "test":
        istrain, istest = True, True
    else:
        istrain, istest = False, True
    dataset = SpecAllReader(file_paths=file_paths, mtype_list=mtype_list, mtid_list=mtid_list,
                            y_true_list=y_true_list,
                            configs=configs,
                            istrain=istrain, istest=istest)
    loader = DataLoader(dataset, batch_size=configs["fit"]["batch_size"],
                        shuffle=True)
    return dataset, loader


def get_wavmel_loader_list(train_dirs, m2l_map, configs=None, mode="train", demo_test=False):
    loaders = []
    id_ma_map = {"valve": 5, "slider": 4, "pump": 3, "fan": 2, "ToyConveyor": 1, "ToyCar": 0}
    ano_id_map = {"normal": 0, "anomal": 1}
    ma_id_map = {5: "valve", 4: "slider", 3: "pump", 2: "fan", 1: "ToyConveyor", 0: "ToyCar"}
    print("---------------train dataset-------------")
    for machine_dir in train_dirs:
        mt = machine_dir.split('\\')[-1].split('_')[-1]
        machine_root_path = machine_dir + '/' + mt + '/' + mode
        file_paths = [machine_root_path + '/' + item for item in os.listdir(machine_root_path)]
        if demo_test:
            file_paths = random.choices(file_paths, k=200)
        mtype_list = [id_ma_map[mt]] * len(file_paths)
        mtid_list = [m2l_map[mt + "-id_" + item.split('_')[2]] for item in os.listdir(machine_root_path)]
        y_true_list = [ano_id_map[item[:6]] for item in os.listdir(machine_root_path)]
        if demo_test:
            mtid_list = random.choices(mtid_list, k=200)
        if mode == "train":
            istrain, istest = True, False
        elif mode == "test":
            istrain, istest = True, True
        else:
            istrain, istest = False, True

        dataset = SpecAllReader(file_paths=file_paths, mtype_list=mtype_list, mtid_list=mtid_list,
                                y_true_list=y_true_list,
                                configs=configs,
                                istrain=istrain, istest=istest)
        loader = DataLoader(dataset, batch_size=configs["fit"]["batch_size"],
                            shuffle=True)
        loaders.append(loader)
    return loaders


class TrainerRBASD(object):
    def __init__(self, configs="./configs/rbasd.yaml", istrain=True):
        self.configs = None
        with open(configs, 'r', encoding='utf-8') as stream:
            self.configs = yaml.safe_load(stream)
        os.makedirs(self.configs["model_directory"], exist_ok=True)
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        # load base_directory list
        self.train_dirs = select_dirs(param=self.configs, mode=True)
        print("---------dirs")
        for d in self.train_dirs:
            print(d)
        self.num_epoch = self.configs["fit"]["epochs"]
        self.timestr = time.strftime("%Y%m%d%H%M", time.localtime())
        self.is_train = istrain
        if istrain:
            self.run_save_dir = self.configs[
                                    "run_save_dir"] + self.timestr + f'_mystgram_tdnn_double_contras/'
            if istrain:
                os.makedirs(self.run_save_dir, exist_ok=True)
        self.w2m = Wave2Mel(16000)

        with open("./datasets/metadata2label.json", 'r', encoding='utf_8') as fp:
            self.meta2label = json.load(fp)
        self.id2map = {5: "valve", 4: "slider", 3: "pump", 2: "fan", 1: "ToyConveyor", 0: "ToyCar"}
        self.mt2id = {"valve": 5, "slider": 4, "pump": 3, "fan": 2, "ToyConveyor": 1, "ToyCar": 0}
        self.m_types = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]
        self.rgb_planning_6 = ["#800000", "#000075", "#dcbeff", "#ffe119", "#000000", "#f58231"]
        self.model = None
        self.train_dataset = None
        self.train_loader = None
        self.test_loader = None
        self.pretrain_encoder = None
        self.pretrain_decoder = None

    def train_all(self, is_load_recon_model=False, is_trans=True, is_double=True):
        if is_load_recon_model:
            self.pretrain_encoder, self.pretrain_decoder = get_pretrain_models(
                encoder_path=f"./run/V3TFS/202401141218_v3tfs_w2m_encoderhome",
                decoder_path=f"./run/V3TFS/202401141234_v3tfs_w2m_decoderhome",
                encoder_epoch=24, decoder_epoch=24, configs=self.configs)
            print("---------load pretrain ae successful------")
        self.model = get_model("rbasd", configs=self.configs, istrain=True).to(self.device)
        class_loss = nn.CrossEntropyLoss().to(self.device)
        contrast_loss = ContrastiveLoss()
        print("All model and loss are on device:", self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=5e-5)

        self.train_dataset, self.train_loader = get_wavmel_settings("./datasets/train_list.txt", self.meta2label,
                                                                    self.configs,
                                                                    mode="train",
                                                                    demo_test=not self.is_train)
        logger.info(
            "setup DataLoader of batch size:{}".format(self.configs["dataset_conf"]["dataLoader"]["batch_size"]))
        # 这四个存储所有epoch内的loss
        history1 = []

        for epoch in range(self.num_epoch):
            print(f"============== START TRAINING {epoch} ==============")
            self.model.train()
            accuracies, loss_sum = [], []
            train_epoch_start_time = time.time()
            for x_idx, (x_wav, x_mel, _, mtid, y_true) in enumerate(tqdm(self.train_loader, desc="Training")):
                latent_feat, anchor_feat, neg_feat, pos_feat = None, None, None, None
                indices, anc_contras_mel, pos_contras_mel, neg_contras_mel = None, None, None, None
                batch_size = x_mel.shape[0]
                sample_num = batch_size // 2
                if is_load_recon_model:
                    latent_feat = self.pretrain_encoder(input_mel=x_mel.unsqueeze(1) / 255., class_vec=None,
                                                        coarse_cls=False,
                                                        fine_cls=False)
                    if not is_double:
                        x_mel = self.pretrain_decoder(latent_feat).squeeze()
                        x_mel = x_mel.transpose(1, 2)  # / 255.
                    # print("shape of encoded latent:", latent_feat.shape)
                    else:
                        recon_mel = self.pretrain_decoder(latent_feat).squeeze()
                        recon_mel = recon_mel  # / 255.
                        x_wav = torch.cat((x_wav, x_wav), dim=0)
                        x_mel = torch.cat((x_mel / 255., recon_mel), dim=0).transpose(1, 2)
                        mtid = torch.cat((mtid, mtid), dim=0)
                    if is_trans:
                        indices = random.choices(list(range(batch_size)), k=sample_num)
                        c, h, w = latent_feat[0, :, ...].shape
                        # anchor最好设置为中心值，才能把对比样本推到周围
                        pos_feat = latent_feat[indices, :, ...] + 0.3 * torch.rand(sample_num, c, h, w,
                                                                                   device=self.device)
                        neg_feat = latent_feat[indices, :, ...] + 1.0 * torch.randn(sample_num, c, h, w,
                                                                                    device=self.device)
                        anchor_feat = latent_feat[indices, :, ...].mean(3).mean(2)[:, :, np.newaxis,
                                      np.newaxis].expand_as(pos_feat)

                        anc_contras_mel = self.pretrain_decoder(anchor_feat).squeeze().transpose(1, 2)
                        pos_contras_mel = self.pretrain_decoder(pos_feat).squeeze().transpose(1, 2)
                        neg_contras_mel = self.pretrain_decoder(neg_feat).squeeze().transpose(1, 2)
                        # neg_mel = self.pretrain_decoder(neg_feat).squeeze().transpose(1, 2)
                        # pos_mel = self.pretrain_decoder(pos_feat).squeeze().transpose(1, 2)
                else:
                    x_mel = x_mel.transpose(1, 2)
                if x_idx == 0:
                    print("----input shape:----")
                    print(x_wav.shape, x_mel.shape, mtid.shape)
                optimizer.zero_grad()
                mtid = torch.tensor(mtid, device=self.device)
                mtid_pred, latent_feat = self.model(x_wav=x_wav, x_mel=x_mel, label=mtid)
                # recon_loss = self.recon_loss(recon_spec, x_mel)
                mtid_pred_loss = class_loss(mtid_pred, mtid)
                if is_trans:
                    _, anchor_feat = self.model(x_wav=x_wav[indices, :], x_mel=anc_contras_mel, label=mtid[indices])
                    _, pos_feat = self.model(x_wav=x_wav[indices, :], x_mel=pos_contras_mel, label=mtid[indices])
                    _, neg_feat = self.model(x_wav=x_wav[indices, :], x_mel=neg_contras_mel, label=mtid[indices])
                    contrast_loss_value = contrast_loss(anchor=anchor_feat, pos=pos_feat, neg=neg_feat)
                    mtid_pred_loss += 0.1 * contrast_loss_value
                mtid_pred_loss.backward()
                optimizer.step()
                history1.append(mtid_pred_loss.item())

                acc = accuracy(mtid_pred, mtid)
                accuracies.append(acc)
                loss_sum.append(mtid_pred_loss.data.cpu().numpy())
                if x_idx % 60 == 0:
                    print(f"Epoch[{epoch}], mtid pred loss:{mtid_pred_loss.item():.4f}")
                    print(f"Epoch[{epoch}], accuracy: {sum(accuracies) / len(accuracies):.5f},")
                    print(f"Epoch[{epoch}], {sum(loss_sum) / len(loss_sum):.5f}")
                    accuracies, loss_sum = [], []
            train_epoch_time = (time.time() - train_epoch_start_time)
            print(f"This epoch cost: {train_epoch_time} s")
            if epoch >= self.configs["model"]["start_scheduler_epoch"]:
                scheduler.step()
            if epoch % 1 == 0:
                os.makedirs(self.run_save_dir + f"model_epoch_{epoch}/", exist_ok=True)
                plt.figure(2)
                plt.plot(range(len(history1)), history1, c="green", alpha=0.7)
                plt.savefig(self.run_save_dir + f'model_epoch_{epoch}/mtid_loss_iter_{epoch}.png')
                # if epoch > 1 and epoch % 2 == 0:
                tmp_model_path = "{model}model_{epoch}.pth".format(
                    model=self.run_save_dir + f"model_epoch_{epoch}/",
                    epoch=epoch)
                torch.save(self.model.state_dict(), tmp_model_path)
                with open(self.run_save_dir + f"model_epoch_{epoch}/loss_value_{epoch}.csv", 'w') as fout:
                    fout.write(",".join([f"{item:.6f}" for item in history1]))
            print(f"============== START TESTING {epoch}==============")
            self.test_auc(epoch_id=epoch, resume_model_path=self.run_save_dir,
                          is_load_recon_model=False, mode="test")

    def train_list(self):
        v3model = get_model("stgram", configs=self.configs, istrain=True).to(self.device)
        class_loss = nn.CrossEntropyLoss().to(self.device)
        print("All model and loss are on device:", self.device)
        optimizer = optim.Adam(v3model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=5e-5)

        train_loader = get_wavmel_loader_list(self.train_dirs, self.meta2label, self.configs, mode="train",
                                              demo_test=not self.is_train)
        # 这四个存储所有epoch内的loss
        history1 = []

        for epoch in range(self.num_epoch):
            print(f"============== START TRAINING {epoch} ==============")
            v3model.train()
            for x_idx, (x_wav, x_mel, _, mtid) in enumerate(tqdm(train_loader, desc="Training")):
                x_mel = x_mel.transpose(1, 2)  # / 255.
                optimizer.zero_grad()
                mtid = torch.tensor(mtid, device=self.device)
                mtid_pred, _ = v3model(x_wav=x_wav, x_mel=x_mel, label=mtid)
                # recon_loss = self.recon_loss(recon_spec, x_mel)
                mtid_pred_loss = class_loss(mtid_pred, mtid)
                mtid_pred_loss.backward()
                optimizer.step()
                history1.append(mtid_pred_loss.item())
                if x_idx % 60 == 0:
                    print(f"Epoch[{epoch}], mtid pred loss:{mtid_pred_loss.item():.4f}")
            if epoch % 1 == 0:
                plt.figure(2)
                plt.plot(range(len(history1)), history1, c="green", alpha=0.7)
                plt.savefig(self.run_save_dir + f'mtid_loss_iter_{epoch}.png')
            if epoch > 6 and epoch % 2 == 0:
                os.makedirs(self.run_save_dir + f"model_epoch_{epoch}/", exist_ok=True)
                tmp_model_path = "{model}model_{epoch}.pth".format(
                    model=self.run_save_dir + f"model_epoch_{epoch}/",
                    epoch=epoch)
                torch.save(v3model.state_dict(), tmp_model_path)

            if epoch >= self.configs["model"]["start_scheduler_epoch"]:
                scheduler.step()

    def test_tsne_all(self, resume_model_path):
        stgram_model = get_model("stgram", configs=self.configs, istrain=False).to(self.device)
        load_epoch = 2
        load_ckpt(stgram_model, resume_model_path + f"/model_epoch_{load_epoch}", load_epoch=load_epoch)
        stgram_model.eval()

        train_loader = get_wavmel_loader_list(self.train_dirs, self.meta2label, self.configs, mode="test",
                                              demo_test=self.is_train)
        with torch.no_grad():
            tsne_input = None
            pred_input = None
            mtids = None
            # istrues = None
            for id_idx, (x_wav, x_mel, _, mtid) in enumerate(tqdm(train_loader, desc="Training")):
                x_mel = x_mel.transpose(1, 2) / 255.
                mtid = torch.tensor(mtid, device=self.device)
                pred, feat = stgram_model(x_wav=x_wav, x_mel=x_mel, label=mtid)
                # recon_loss = self.recon_loss(recon_spec, x_mel)
                # print(pred.shape, feat.shape)
                if id_idx == 0:
                    pred_input = pred
                    tsne_input = feat
                    mtids = mtid
                    # istrues = y_true
                else:
                    pred_input = torch.concatenate([pred_input, pred], dim=0)
                    tsne_input = torch.concatenate([tsne_input, feat], dim=0)
                    mtids = torch.concatenate([mtids, mtid], dim=0)
                    # istrues = torch.concatenate([istrues, y_true], dim=0)
            tsne_input = tsne_input.data.cpu().numpy()
            pred_input = pred_input.data.cpu().numpy()
            mtids = mtids.data.cpu().numpy()
            # istrues = istrues.data.cpu().numpy()
            print("tnse shape:", tsne_input.shape)
            print("pred shape:", pred_input.shape)
            print("mtid shape:", mtids.shape)
            from sklearn.manifold import TSNE
            from asdkit.utils.plotting import plot_embedding_2D, get_heat_map
            tsne_model = TSNE(n_components=2, init="pca", random_state=3407)
            result2D = tsne_model.fit_transform(tsne_input)
            print("TSNE finish.")
            plot_embedding_2D(result2D, mtids, "t-SNT for mtid",
                              savepath=resume_model_path + f"/stgram_{load_epoch}_tsne.png")
            # plot_embedding_2D(result2D, istrues, "t-SNT for anomaly",
            #                   savepath=resume_model_path + f"/{mode}1_{mt}_tsne_anomaly.png")
            get_heat_map(pred_matrix=pred_input, label_vec=mtids,
                         savepath=resume_model_path + f'/stgram_{load_epoch}_heatmap.png')

        print("============== END TRAINING ==============")

    def test_auc(self, epoch_id, resume_model_path, is_load_recon_model=False, mode="test"):
        if not self.model:
            self.model = get_model("stgram", configs=self.configs, istrain=False).to(self.device)
            load_epoch = epoch_id
            load_ckpt(self.model, resume_model_path + f"/model_epoch_{load_epoch}", load_epoch=load_epoch)
        self.model.eval()
        if is_load_recon_model:
            if self.pretrain_encoder is None:
                self.pretrain_encoder, self.pretrain_decoder = get_pretrain_models(
                    encoder_path=f"../run/V3TFS/202401141218_v3tfs_w2m_encoderhome",
                    decoder_path=f"../run/V3TFS/202401141234_v3tfs_w2m_decoderhome",
                    encoder_epoch=24, decoder_epoch=24, configs=self.configs)
            print("---------load pretrain ae successful------")
        if self.train_dataset is None:
            self.train_dataset = SpecAllReader(None, None, None, None,
                                               configs=self.configs)
        m_types = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]

        # train_loader = get_wavmel_loader_list(self.train_dirs, self.meta2label, self.configs, mode="train",
        #                                       demo_test=self.is_train)
        # test_loader = get_wavmel_loader_list(self.train_dirs, self.meta2label, self.configs, mode="test",
        #                                      demo_test=self.is_train)
        csv_lines = []
        sum_auc, sum_pauc, num = 0, 0, 0
        test_dirs = select_dirs(param=self.configs, mode=True)
        for idx, target_dir in enumerate(test_dirs):
            print("\n===========================")
            print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx + 1, total=len(test_dirs)))
            machine_type = os.path.split(target_dir)[1].split('_')[2]
            if machine_type != "pump":
                continue
            print(machine_type)
            # if mode:
            csv_lines.append([machine_type])
            csv_lines.append(
                ["id", "AUC", "pAUC", "precision", "recall", "F1-score", "Accuracy", "tn", "fp", "fn", "tp"])
            performance = []
            machine_id_list = get_machine_id_list_for_test(target_dir + "/" + machine_type, dir_name="test")
            print(machine_id_list)

            tsne_input = []
            pred_input = []
            mtids = []
            istrues = []
            for id_str in machine_id_list:
                # load test file
                meta = machine_type + '-' + id_str
                print("machine id:", meta)
                label = self.meta2label[meta]
                # generate dataset
                # print("============== DATASET_GENERATOR ==============")
                test_files, y_true = test_file_list_generator(target_dir + "/" + machine_type, id_str)
                # print("length of test file and y_true:", len(test_files), len(y_true))

                y_pred = [0. for _ in test_files]

                for file_idx, file_path in enumerate(test_files):
                    x_wav, x_mel = self.train_dataset.get_mel_from_filepath(file_path)

                    x_wav = torch.tensor(x_wav, device=self.device).to(torch.float32).unsqueeze(0)
                    x_mel = x_mel.to(self.device).unsqueeze(0).to(torch.float32)
                    label = torch.tensor([label]).long().to(self.device)

                    with torch.no_grad():
                        if is_load_recon_model:
                            x_mel = self.pretrain_decoder(
                                self.pretrain_encoder(input_mel=x_mel.transpose(1, 2).unsqueeze(1) / 255.,
                                                      class_vec=None,
                                                      coarse_cls=False, fine_cls=False))
                            x_mel = x_mel.transpose(2, 3)[:, 0, :, :]

                        pred, feat = self.model(x_wav, x_mel, label)

                    probs = - torch.log_softmax(pred, dim=1).mean(0).squeeze().cpu().numpy()  # .squeeze()  #
                    y_pred[file_idx] = probs[label]

                    tsne_input.append(feat.data.cpu().numpy())
                    pred_input.append(pred.data.cpu().numpy())
                    mtids.append(label.data.cpu().numpy())
                    istrues.append(y_true[file_idx])

                max_fpr = 0.1
                auc = metrics.roc_auc_score(y_true, y_pred)
                p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
                csv_lines.append([id_str, auc, p_auc])
                performance.append([auc, p_auc])
            averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
            mean_auc, mean_p_auc = averaged_performance[0], averaged_performance[1]
            logger.info(f'\n{machine_type}\tAUC: {mean_auc * 100:.3f}\tpAUC: {mean_p_auc * 100:.3f}')
            csv_lines.append(['Average'] + list(averaged_performance))
            sum_auc += mean_auc
            sum_pauc += mean_p_auc
            num += 1

            # tsne_input = torch.stack(tsne_input, dim=0)
            # pred_input = torch.stack(pred_input, dim=0)
            tsne_input = np.concatenate(tsne_input, axis=0)
            pred_input = np.concatenate(pred_input, axis=0)
            mtids = np.concatenate(mtids, axis=0)
            istrues = np.asarray(istrues, dtype=np.int32)
            # mtids = torch.stack(mtids, dim=0)
            # istrues = torch.stack(istrues, dim=0)
            print("----type and shape----")
            print("feat:", tsne_input.dtype, tsne_input.shape)
            print("pred:", pred_input.dtype, pred_input.shape)
            print("mtids:", len(mtids), mtids[0], )
            print("ytrue:", len(istrues), istrues[0])
            tsne_model = TSNE(n_components=2, init="pca", random_state=3407)
            result2D = tsne_model.fit_transform(tsne_input)
            print("TSNE finish.")
            plot_embedding_2D(result2D, mtids, "t-SNT for mtid",
                              savepath=resume_model_path + f"/model_epoch_{epoch_id}/{mode}{epoch_id}_{machine_type}_tsne.{saveimg_params['format']}",
                              names=names,
                              params=saveimg_params)
            plot_embedding_2D(result2D, istrues, "t-SNT for anomaly",
                              savepath=resume_model_path + f"/model_epoch_{epoch_id}/{mode}{epoch_id}_{machine_type}_tsne_anomaly.{saveimg_params['format']}",
                              names=names,
                              params=saveimg_params)
            get_heat_map(pred_matrix=pred_input, label_vec=mtids,
                         savepath=resume_model_path + f"/model_epoch_{epoch_id}/{mode}{epoch_id}_{machine_type}_heatmap.{saveimg_params['format']}")

        avg_auc, avg_pauc = sum_auc / num, sum_pauc / num
        csv_lines.append(['Total Average', avg_auc, avg_pauc])
        logger.info(f'Total average:\t\tAUC: {avg_auc * 100:.3f}\tpAUC: {avg_pauc * 100:.3f}')
        # save csv to /results/versions/results.csv
        save_csv(resume_model_path + f'/model_epoch_{epoch_id}/test_result_{epoch_id}.csv', csv_lines)

    def test_tsne_list(self, epoch_id, is_trans, resume_model_path=None, mode="test", is_load_recon_model=False):
        if not self.model:
            self.model = get_model("stgram", configs=self.configs, istrain=False).to(self.device)
            load_epoch = epoch_id
            load_ckpt(self.model, resume_model_path + f"/model_epoch_{load_epoch}", load_epoch=load_epoch)
        self.model.eval()
        if is_load_recon_model:
            if self.pretrain_encoder is None:
                self.pretrain_encoder, self.pretrain_decoder = get_pretrain_models(
                    encoder_path=f"../run/V3TFS/202401141218_v3tfs_w2m_encoderhome",
                    decoder_path=f"../run/V3TFS/202401141234_v3tfs_w2m_decoderhome",
                    encoder_epoch=24, decoder_epoch=24, configs=self.configs)
            print("---------load pretrain ae successful------")
        if self.train_dataset is None:
            self.train_dataset = SpecAllReader(None, None, None, None,
                                               configs=self.configs)
        # m_types = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]
        # train_loader = get_wavmel_loader_list(self.train_dirs, self.meta2label, self.configs, mode="train",
        #                                       demo_test=self.is_train)
        # test_loader = get_wavmel_loader_list(self.train_dirs, self.meta2label, self.configs, mode="test",
        #                                      demo_test=self.is_train)
        csv_lines = []
        sum_auc, sum_pauc, num = 0, 0, 0

        test_dirs = select_dirs(param=self.configs, mode=True)
        for idx, target_dir in enumerate(test_dirs):
            print("\n===========================")
            print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx + 1, total=len(test_dirs)))
            machine_type = os.path.split(target_dir)[1].split('_')[2]
            print(machine_type)
            if machine_type != "pump":
                continue
            # if mode:
            csv_lines.append([machine_type])
            csv_lines.append(
                ["id", "AUC", "pAUC", "precision", "recall", "F1-score", "Accuracy", "tn", "fp", "fn", "tp"])
            performance = []
            machine_id_list = get_machine_id_list_for_test(target_dir + "/" + machine_type, dir_name="test")
            print(machine_id_list)

            tsne_input = []
            pred_input = []
            mtids = []
            istrues = []
            for id_str in machine_id_list:
                # load test file
                meta = machine_type + '-' + id_str
                print("machine id:", meta)
                label = self.meta2label[meta]
                # generate dataset
                # print("============== DATASET_GENERATOR ==============")
                test_files, y_true = test_file_list_generator(target_dir + "/" + machine_type, id_str)
                # print("length of test file and y_true:", len(test_files), len(y_true))

                y_pred = [0. for _ in test_files]

                for file_idx, file_path in enumerate(test_files):
                    x_wav, x_mel = self.train_dataset.get_mel_from_filepath(file_path)

                    x_wav = torch.tensor(x_wav, device=self.device).to(torch.float32).unsqueeze(0)
                    x_mel = x_mel.to(self.device).unsqueeze(0).to(torch.float32)
                    label = torch.tensor([label]).long().to(self.device)

                    with torch.no_grad():
                        if is_load_recon_model:
                            x_mel = self.pretrain_decoder(
                                self.pretrain_encoder(input_mel=x_mel.transpose(1, 2).unsqueeze(1) / 255.,
                                                      class_vec=None,
                                                      coarse_cls=False, fine_cls=False))
                            x_mel = x_mel.transpose(2, 3)[:, 0, :, :]

                        pred, feat = self.model(x_wav, x_mel, label)

                    probs = - torch.log_softmax(pred, dim=1).mean(0).squeeze().cpu().numpy()  # .squeeze()  #
                    y_pred[file_idx] = probs[label]

                    tsne_input.append(feat.data.cpu().numpy())
                    pred_input.append(pred.data.cpu().numpy())
                    mtids.append(label.data.cpu().numpy())
                    istrues.append(y_true[file_idx])

                max_fpr = 0.1
                auc = metrics.roc_auc_score(y_true, y_pred)
                p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
                csv_lines.append([id_str, auc, p_auc])
                performance.append([auc, p_auc])
            averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
            mean_auc, mean_p_auc = averaged_performance[0], averaged_performance[1]
            logger.info(f'\n{machine_type}\tAUC: {mean_auc * 100:.3f}\tpAUC: {mean_p_auc * 100:.3f}')
            csv_lines.append(['Average'] + list(averaged_performance))
            sum_auc += mean_auc
            sum_pauc += mean_p_auc
            num += 1

            # tsne_input = torch.stack(tsne_input, dim=0)
            # pred_input = torch.stack(pred_input, dim=0)
            tsne_input = np.concatenate(tsne_input, axis=0)
            pred_input = np.concatenate(pred_input, axis=0)
            mtids = np.concatenate(mtids, axis=0)
            istrues = np.asarray(istrues, dtype=np.int32)
            # mtids = torch.stack(mtids, dim=0)
            # istrues = torch.stack(istrues, dim=0)
            print("----type and shape----")
            print("feat:", tsne_input.dtype, tsne_input.shape)
            print("pred:", pred_input.dtype, pred_input.shape)
            print("mtids:", len(mtids), mtids[0], )
            print("ytrue:", len(istrues), istrues[0])
            tsne_model = TSNE(n_components=2, init="pca", random_state=3407)
            result2D = tsne_model.fit_transform(tsne_input)
            print("TSNE finish.")
            plot_embedding_2D(result2D, mtids, "t-SNT for mtid",
                              savepath=resume_model_path + f"/model_epoch_{epoch_id}_0226/{mode}{epoch_id}_{machine_type}_tsne.png")
            plot_embedding_2D(result2D, istrues, "t-SNT for anomaly",
                              savepath=resume_model_path + f"/model_epoch_{epoch_id}_0226/{mode}{epoch_id}_{machine_type}_tsne_anomaly.png")
            get_heat_map(pred_matrix=pred_input, label_vec=mtids,
                         savepath=resume_model_path + f'/model_epoch_{epoch_id}_0226/{mode}{epoch_id}_{machine_type}_heatmap.png')

        avg_auc, avg_pauc = sum_auc / num, sum_pauc / num
        csv_lines.append(['Total Average', avg_auc, avg_pauc])
        logger.info(f'Total average:\t\tAUC: {avg_auc * 100:.3f}\tpAUC: {avg_pauc * 100:.3f}')
        # save csv to /results/versions/results.csv
        save_csv(resume_model_path + f'/model_epoch_{epoch_id}_0226/test_result_{epoch_id}.csv', csv_lines)
