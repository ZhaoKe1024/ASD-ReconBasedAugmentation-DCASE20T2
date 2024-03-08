#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/1/20 14:33
# @Author: ZhaoKe
# @File : trainer_v3multi_model_inone.py
# @Software: PyCharm
"""
source file for training Encoder and Decoder, respectively.
"""
import yaml
import time
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
from asdkit.models.autoencoder import ConvEncoder, ConvDecoder

from asdkit.models.trainer_settings import get_pretrain_models
from asdkit.modules import func
from asdkit.data_utils.dcase2020_fileutils import *
from asdkit.data_utils.machine_sound_readers import get_former_loader, FormerReader
from asdkit.utils.plotting import plot_embedding_2D
from asdkit.utils.utils import load_ckpt

saveimg_params = {"format": "svg", "marker_size": 8, "alpha": 0.8}
names = ["ToyCar-id_01", "ToyCar-id_02", "ToyCar-id_03", "ToyCar-id_04",
         "ToyConveyor-id_01", "ToyConveyor-id_02", "ToyConveyor-id_03",
         "fan-id_00", "fan-id_02", "fan-id_04", "fan-id_06",
         "pump-id_00", "pump-id_02", "pump-id_04", "pump-id_06",
         "slider-id_00", "slider-id_02", "slider-id_04", "slider-id_06",
         "valve-id_00", "valve-id_02", "valve-id_04", "valve-id_06"]


class TrainerV3MultiModel_inone(object):
    def __init__(self, configs="../configs/v3tfs.yaml", istrain=True):
        self.configs = None
        with open(configs) as stream:
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
        self.istrain = istrain
        if istrain:
            self.run_save_dir = self.configs[
                                    "run_save_dir"] + self.timestr + f'_ae4reconcls_6type/'
            os.makedirs(self.run_save_dir, exist_ok=True)
        self.w2m = Wave2Mel(16000)

        with open("../datasets/metadata2label.json", 'r', encoding='utf_8') as fp:
            self.meta2label = json.load(fp)
        # self.id2map = {5: "valve", 4: "slider", 3: "pump", 2: "fan", 1: "ToyConveyor", 0: "ToyCar"}
        # self.mt2id = {"valve": 5, "slider": 4, "pump": 3, "fan": 2, "ToyConveyor": 1, "ToyCar": 0}
        self.m_types = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]
        self.rgb_planning_6 = ["#800000", "#000075", "#dcbeff", "#ffe119", "#000000", "#f58231"]
        self.model = None
        self.cls = True
        self.k = 3
        self.pretrain_encoder = None
        self.pretrain_decoder = None

    def train_encoder(self, istrain=True, is_load_recon_model=True):
        if is_load_recon_model:
            self.pretrain_encoder, self.pretrain_decoder = get_pretrain_models(
                encoder_path=f"../run/V3TFS/202401141218_v3tfs_w2m_encoderhome",
                decoder_path=f"../run/V3TFS/202401141234_v3tfs_w2m_decoderhome",
                encoder_epoch=24, decoder_epoch=24, configs=self.configs)
            print("---------load pretrain ae successful------")
        self.model = ConvEncoder(input_channel=1, input_length=self.configs["model"]["input_length"],
                                 input_dim=self.configs["feature"]["n_mels"],
                                 class_num=self.configs["model"]["mtid_class_num"],
                                 class_num1=self.configs["model"]["type_class_num"]).to(self.device)
        self.model.apply(func.weight_init)
        self.class_loss = nn.CrossEntropyLoss().to(self.device)
        print("All model and loss are on device:", self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=5e-5)

        train_loader, _ = get_former_loader(istrain=True, istest=True, configs=self.configs,
                                            meta2label=self.meta2label, shuffles=(True, False),
                                            isdemo=not istrain, exclude=None)
        history1 = []
        for epoch in range(self.num_epoch):
            self.model.train()
            for x_idx, (x_mel, mtype, mtid) in enumerate(tqdm(train_loader, desc="Training")):
                if is_load_recon_model:
                    x_mel = self.pretrain_decoder(self.pretrain_encoder(input_mel=x_mel.unsqueeze(1) / 255.,
                                                                        class_vec=None,
                                                                        coarse_cls=False,
                                                                        fine_cls=False)).squeeze()
                    # x_mel = x_mel.transpose(1, 2)  # / 255.
                x_mel = x_mel.unsqueeze(1)
                optimizer.zero_grad()
                mtid = torch.tensor(mtid, device=self.device)
                feat, mtid_pred = self.model(input_mel=x_mel, class_vec=mtid, coarse_cls=False, fine_cls=True)
                # recon_loss = self.recon_loss(recon_spec, x_mel)
                mtid_pred_loss = self.class_loss(mtid_pred, mtid)
                mtid_pred_loss.backward()
                optimizer.step()
                history1.append(mtid_pred_loss.item())
                if x_idx % 60 == 0:
                    print(f"Epoch[{epoch}], mtid pred loss:{mtid_pred_loss.item():.4f}")
            # os.makedirs(self.run_save_dir+f"model_eopch_{epoch}/", exist_ok=True)
            if epoch % 1 == 0:
                plt.figure(2)
                plt.plot(range(len(history1)), history1, c="green", alpha=0.7)
                plt.savefig(self.run_save_dir + f'mtid_loss_iter_{epoch}.png')
            if epoch > 3:
                os.makedirs(self.run_save_dir + f"model_epoch_{epoch}/", exist_ok=True)
                tmp_model_path = "{model}model_{epoch}.pth".format(
                    model=self.run_save_dir + f"model_epoch_{epoch}/",
                    epoch=epoch)
                torch.save(self.model.state_dict(), tmp_model_path)

            if epoch >= self.configs["model"]["start_scheduler_epoch"]:
                scheduler.step()
        print("============== END TRAINING ==============")
        # test_anomaly_loss = []
        # test_normal_loss = []
        # ------------------------------------------------------
        # -------------测试阶段不用dataloader！！！----------------
        # ------------------------------------------------------
        # with torch.no_grad():
        #     for id_idx, (x_mel, mtype, mtid, y_trues, _) in enumerate(tqdm(test_loader, desc="testing")):
        #         # load test file
        #         # generate dataset
        #         # print("============== DATASET_GENERATOR ==============")
        #         x_mel = x_mel.unsqueeze(1)
        #         # mtype = torch.tensor(mtype, device=self.device)
        #         mtid = torch.tensor(mtid, device=self.device)
        #         feat, mtid_pred = self.model(input_mel=x_mel, m_class_vec=mtid, task=3)
        #         mtid_pred = mtid_pred.argmax(axis=1)
        #         cm = metrics.confusion_matrix(mtid_pred.data.cpu().numpy(), mtid.data.cpu().numpy())
        #         acc_list = np.diagonal(cm) / cm.sum(axis=1)
        #         print()

    def test_tSNE(self, resume_model_path=None, load_epoch=None, mode=True, is_load_recon_model=False):
        if is_load_recon_model:
            self.pretrain_encoder, self.pretrain_decoder = get_pretrain_models(
                encoder_path=f"../run/V3TFS/202401141218_v3tfs_w2m_encoderhome",
                decoder_path=f"../run/V3TFS/202401141234_v3tfs_w2m_decoderhome",
                encoder_epoch=24, decoder_epoch=24, configs=self.configs)
            print("---------load pretrain ae successful------")

        # self.model = ConvEncoder(input_channel=1, input_length=self.configs["model"]["input_length"],
        #                          input_dim=self.configs["feature"]["n_mels"],
        #                          class_num=self.configs["model"]["mtid_class_num"],
        #                          class_num1=self.configs["model"]["type_class_num"]).to(self.device)
        # if resume_model_path:
        #     if load_epoch is not None:
        #         load_ckpt(self.model, resume_model_path, load_epoch=load_epoch)
        #     else:
        #         load_ckpt(self.model, resume_model_path)
        self.model, _ = get_pretrain_models(
                encoder_path=f"../run/V3TFS/202401141218_v3tfs_w2m_encoderhome",
                decoder_path=f"../run/V3TFS/202401141234_v3tfs_w2m_decoderhome",
                encoder_epoch=24, decoder_epoch=24, configs=self.configs)
        # self.model.eval()

        ma_id_map = {5: "valve", 4: "slider", 3: "pump", 2: "fan", 1: "ToyConveyor", 0: "ToyCar"}
        print("---------------train dataset-------------")
        # file_paths = []
        # mtid_list = []
        # mtype_list = []
        # with open("../datasets/train_list.txt", 'r') as fin:
        #     train_path_list = fin.readlines()
        #     for item in train_path_list:
        #         parts = item.strip().split('\t')
        #         machine_type_id = int(parts[1])
        #         file_paths.append(parts[0])
        #         mtype_list.append(machine_type_id)
        #         machine_id_id = parts[2]
        #         meta = ma_id_map[machine_type_id] + '-id_' + machine_id_id
        #         mtid_list.append(self.meta2label[meta])
        print("============== DATASET_GENERATOR ==============")
        train_dataset = FormerReader(file_paths=None, mtype_list=None, mtid_list=None,
                                     y_true_list=None,
                                     configs=self.configs,
                                     istrain=True)
        m_types = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]
        test_dirs = select_dirs(param=self.configs, mode=True)
        for idx, target_dir in enumerate(test_dirs):
            print("\n===========================")
            print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx + 1, total=len(test_dirs)))
            machine_type = os.path.split(target_dir)[1].split('_')[2]
            print(machine_type)
            machine_id_list = get_machine_id_list_for_test(target_dir + "/" + machine_type, dir_name="test")
            print(machine_id_list)
            tsne_input = []
            # pred_input = []
            mtids = []
            # istrues = []
            for id_str in machine_id_list:
                # load test file
                meta = machine_type + '-' + id_str
                print("machine id:", meta)
                label = self.meta2label[meta]
                # generate dataset
                # print("============== DATASET_GENERATOR ==============")
                test_files, y_true = test_file_list_generator(target_dir + "/" + machine_type, id_str)
                # print("length of test file and y_true:", len(test_files), len(y_true))
                # y_pred = [0. for _ in test_files]
                for file_idx, file_path in enumerate(test_files):
                    _, x_mel = train_dataset.get_mel_from_filepath(file_path)
                    # x_wav = torch.tensor(x_wav, device=self.device).to(torch.float32).unsqueeze(0)
                    x_mel = x_mel.to(self.device).unsqueeze(0).to(torch.float32)
                    label = torch.tensor([label]).long().to(self.device)
                    with torch.no_grad():
                        if is_load_recon_model:
                            x_mel = self.pretrain_decoder(
                                self.pretrain_encoder(input_mel=x_mel.transpose(1, 2).unsqueeze(1) / 255.,
                                                      class_vec=None,
                                                      coarse_cls=False, fine_cls=False))
                        x_mel = x_mel.transpose(1, 2).unsqueeze(1) / 255.
                        feat = self.model(input_mel=x_mel,
                                          class_vec=None,
                                          coarse_cls=False, fine_cls=False)
                    feat = feat.mean(3).mean(2)
                    tsne_input.append(feat.data.cpu().numpy())
                    mtids.append(label.data.cpu().numpy())
            tsne_input = np.concatenate(tsne_input, axis=0)
            mtids = np.concatenate(mtids, axis=0)
            print("----type and shape----")
            print("feat:", tsne_input.dtype, tsne_input.shape)
            print("mtids:", len(mtids), mtids[0], )
            tsne_model = TSNE(n_components=2, init="pca", random_state=3407)
            result2D = tsne_model.fit_transform(tsne_input)
            print("TSNE finish.")
            plot_embedding_2D(result2D, mtids, "t-SNT for mtid",
                              savepath=resume_model_path + f"{load_epoch}_{machine_type}_tsne.{saveimg_params['format']}",
                              names=names,
                              params=saveimg_params)

    def train_decoder(self):
        encoder_model = ConvEncoder(input_channel=1, input_length=self.configs["model"]["input_length"],
                                    input_dim=self.configs["feature"]["n_mels"],
                                    class_num=self.configs["model"]["mtid_class_num"],
                                    class_num1=self.configs["model"]["type_class_num"]).to(self.device)
        # for param in encoder_model.named_parameters():
        #     param.requires_grad = False

        load_ckpt(encoder_model, "../run/V3TFS/202401141218_v3tfs_w2m_encoderhome/model_epoch_24", 24)
        encoder_model.eval()

        decoder_model = ConvDecoder(input_channel=1, input_length=self.configs["model"]["input_length"],
                                    input_dim=self.configs["feature"]["n_mels"]).to(self.device)
        decoder_model.apply(func.weight_init)
        recon_func = nn.MSELoss().to(self.device)
        print("All model and loss are on device:", self.device)

        optimizer = optim.Adam(decoder_model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-4)
        train_loader, _ = get_former_loader(istest=False, configs=self.configs, meta2label=self.meta2label,
                                            shuffles=(True, False),
                                            isdemo=not self.istrain, exclude=None)
        history2 = []
        for epoch in range(self.num_epoch):
            decoder_model.train()
            # y_pred = []  # 存储当前epoch的训练和测试loss
            train_loss_line = []  # 存储当前epoch的训练loss
            for x_idx, (x_mel, _, _) in enumerate(tqdm(train_loader, desc="Training")):
                optimizer.zero_grad()
                x_mel = x_mel.unsqueeze(1) / 255.
                with torch.no_grad():
                    latent_map = encoder_model(input_mel=x_mel, label_vec=None, ifcls=False)
                recon_spec = decoder_model(latent_map=latent_map)
                # print(f"x_mel:{x_mel.shape}, mtype:{mtype.shape}, mtid:{mtid.shape}")
                recon_loss = recon_func(recon_spec, x_mel)
                recon_loss.backward()
                optimizer.step()
                history2.append(recon_loss.item())
                if x_idx % 60 == 0:
                    print(
                        f"Epoch[{epoch}], recon loss:{recon_loss.item():.4f}")
            if epoch % 1 == 0:
                plt.figure(1)
                plt.plot(range(len(history2) - 25), history2[25:], c="green", alpha=0.7)
                plt.savefig(self.run_save_dir + f'type_loss_iter_{epoch}.png')
                os.makedirs(self.run_save_dir + f"model_epoch_{epoch}/", exist_ok=True)
                tmp_model_path = "{model}model_{epoch}.pth".format(
                    model=self.run_save_dir + f"model_epoch_{epoch}/",
                    epoch=epoch)
                torch.save(decoder_model.state_dict(), tmp_model_path)

            if epoch >= self.configs["model"]["start_scheduler_epoch"]:
                scheduler.step()


if __name__ == '__main__':
    # trainer = TrainerV3MultiModel_inone(istrain=True)
    # trainer.train_encoder(istrain=True, is_load_recon_model=True)
    trainer = TrainerV3MultiModel_inone(istrain=False)
    load_epoch = 14
    trainer.test_tSNE(resume_model_path=f"../run/V3TFS/202403022101_ae4reconcls_6type/model_epoch_{load_epoch}/",
                      is_load_recon_model=False, load_epoch=load_epoch)
