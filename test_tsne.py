#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/2/26 10:31
# @Author: ZhaoKe
# @File : test_tsne.py
# @Software: PyCharm
import os
import yaml
import json
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

from asdkit.data_utils.dcase2020_fileutils import select_dirs, get_machine_id_list_for_test, test_file_list_generator
from asdkit.data_utils.machine_sound_readers import SpecAllReader
from asdkit.models.trainer_settings import get_pretrain_models, get_model
from asdkit.utils.plotting import plot_embedding_2D
from asdkit.utils.utils import load_ckpt

saveimg_params = {"format": "svg", "marker_size": 3, "alpha": 0.7}
# saveimg_params = {"format": "svg", "marker_size": 4, "alpha": 0.5}


def test_mfn_tsne_origin_recon(is_load_recon_model, is_double=False):
    with open("./configs/stgrammfn.yaml", encoding="utf_8") as stream:
        configs = yaml.safe_load(stream)
    use_gpu = True
    device = torch.device("cuda") if use_gpu else torch.device("cpu")
    pretrain_encoder, pretrain_decoder = None, None
    if is_load_recon_model:
        pretrain_encoder, pretrain_decoder = get_pretrain_models(
            encoder_path=f"./run/V3TFS/202401141218_v3tfs_w2m_encoderhome",
            decoder_path=f"./run/V3TFS/202401141234_v3tfs_w2m_decoderhome",
            encoder_epoch=24, decoder_epoch=24, configs=configs, use_gpu=use_gpu)

    eval_model = get_model("stgram", configs=configs, istrain=False).to(device)
    load_epoch = 6
    resume_model_path = "./run/dcase2020_mystgram/202402071605_mystgram_tdnn_double_contras"
    load_ckpt(eval_model,
              "./run/dcase2020_mystgram/202402071605_mystgram_tdnn_double_contras" + f"/model_epoch_{load_epoch}",
              load_epoch=load_epoch)
    eval_model.eval()
    print("---------load pretrain ae successful------")

    m_types = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]
    id_ma_map = {"valve": 5, "slider": 4, "pump": 3, "fan": 2, "ToyConveyor": 1, "ToyCar": 0}
    with open("./datasets/metadata2label.json", 'r', encoding='utf_8') as fp:
        meta2label = json.load(fp)

    test_dirs = select_dirs(param=configs, mode=True)
    file_paths = []
    mtid_list = []
    mtype_list = []
    y_true_list = []
    for idx, target_dir in enumerate(test_dirs):
        machine_type = os.path.split(target_dir)[1].split('_')[2]
        if machine_type != "pump":
            continue
        print("\n===========================")
        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx + 1, total=len(test_dirs)))
        print(machine_type)
        machine_id_list = get_machine_id_list_for_test(target_dir + "/" + machine_type, dir_name="test")
        print(machine_id_list)

        for id_str in machine_id_list:
            # load test file
            meta = machine_type + '-' + id_str
            print("machine id:", meta)
            label = meta2label[meta]
            test_files, y_true = test_file_list_generator(target_dir + "/" + machine_type, id_str)

            # print("length of test file and y_true:", len(test_files), len(y_true))
            # y_pred = [0. for _ in test_files]
            for file_idx, file_path in enumerate(test_files):
                file_paths.append(file_path)
                mtid_list.append(label)
                mtype_list.append(id_ma_map[machine_type])
                y_true_list.append(y_true[file_idx])
    print("mtid set: ", set(mtid_list))
    # return
    print(f"length: files {len(file_paths)}; mtid {len(mtid_list)}; mtype {len(mtype_list)}; y_true {len(y_true_list)}")
    tmp_dataset = SpecAllReader(file_paths=file_paths, mtype_list=mtype_list, mtid_list=mtid_list,
                                y_true_list=y_true_list, configs=configs, istrain=True, istest=False)
    loader = DataLoader(tmp_dataset, batch_size=32, shuffle=False)
    tsne_input = []
    # pred_input = []
    mtids = []
    machine_type = "pump"
    # istrues = []
    for x_idx, (x_wav, x_mel, _, mtid, y_true) in enumerate(tqdm(loader, desc="Training")):
        x_wav = x_wav.to(torch.float32)
        x_mel = x_mel.to(torch.float32)
        # print(mtid)
        # print(mtid, mtid)
        mtid = mtid.long().to(device) - 11
        with torch.no_grad():
            if is_load_recon_model:
                latent_feat = pretrain_encoder(input_mel=x_mel.unsqueeze(1) / 255., class_vec=None,
                                               coarse_cls=False,
                                               fine_cls=False)
                if not is_double:
                    x_mel = pretrain_decoder(latent_feat).squeeze()
                    x_mel = x_mel.transpose(1, 2)  # / 255.
                # print("shape of encoded latent:", latent_feat.shape)
                else:
                    recon_mel = pretrain_decoder(latent_feat).squeeze()
                    recon_mel = recon_mel  # / 255.
                    x_wav = torch.cat((x_wav, x_wav), dim=0)
                    x_mel = torch.cat((x_mel / 255., recon_mel), dim=0).transpose(1, 2)
                    new_mtid = mtid + 4
                    mtid = torch.cat((mtid, new_mtid), dim=0)
            _, feat = eval_model(x_wav, x_mel, mtid)

        tsne_input.append(feat.data.cpu().numpy())
        # pred_input.append(pred.data.cpu().numpy())
        mtids.append(mtid.data.cpu().numpy())
        # istrues.append(y_true[file_idx])
    tsne_input = np.concatenate(tsne_input, axis=0)
    # pred_input = np.concatenate(pred_input, axis=0)
    mtids = np.concatenate(mtids, axis=0)
    print("----type and shape----")
    print("feat:", tsne_input.dtype, tsne_input.shape)
    # print("pred:", pred_input.dtype, pred_input.shape)
    print("mtids:", len(mtids), mtids[0], )
    # print("ytrue:", len(istrues), istrues[0])
    tsne_model = TSNE(n_components=2, init="pca", random_state=3407)
    result2D = tsne_model.fit_transform(tsne_input)
    print("TSNE finish.")
    plot_embedding_2D(result2D, mtids, "t-SNT for mtid",
                      savepath=resume_model_path + f"/model_epoch_{6}_combine_4/stgram_test6_{machine_type}_tsne_s8_3.{saveimg_params['format']}",
                      names=[f"normal_id-{i}" for i in range(4)]+[f"anomaly_id-{i}" for i in range(4)], params=saveimg_params)


def test_convencoder_tsne_origin_recon(is_load_recon_model, is_double=False):
    with open("./configs/stgrammfn.yaml", encoding="utf_8") as stream:
        configs = yaml.safe_load(stream)
    use_gpu = True
    device = torch.device("cuda") if use_gpu else torch.device("cpu")
    pretrain_encoder, pretrain_decoder = get_pretrain_models(
        encoder_path=f"./run/V3TFS/202401141218_v3tfs_w2m_encoderhome",
        decoder_path=f"./run/V3TFS/202401141234_v3tfs_w2m_decoderhome",
        encoder_epoch=24, decoder_epoch=24, configs=configs, use_gpu=use_gpu)

    print("---------load pretrain ae successful------")

    m_types = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]
    id_ma_map = {"valve": 5, "slider": 4, "pump": 3, "fan": 2, "ToyConveyor": 1, "ToyCar": 0}
    with open("./datasets/metadata2label.json", 'r', encoding='utf_8') as fp:
        meta2label = json.load(fp)
    test_dirs = select_dirs(param=configs, mode=True)
    file_paths = []
    mtid_list = []
    mtype_list = []
    y_true_list = []
    for idx, target_dir in enumerate(test_dirs):
        machine_type = os.path.split(target_dir)[1].split('_')[2]
        if machine_type != "pump":
            continue
        print("\n===========================")
        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx + 1, total=len(test_dirs)))
        print(machine_type)
        machine_id_list = get_machine_id_list_for_test(target_dir + "/" + machine_type, dir_name="test")
        print(machine_id_list)

        for id_str in machine_id_list:
            # load test file
            meta = machine_type + '-' + id_str
            print("machine id:", meta)
            label = meta2label[meta]
            test_files, y_true = test_file_list_generator(target_dir + "/" + machine_type, id_str)

            # print("length of test file and y_true:", len(test_files), len(y_true))
            # y_pred = [0. for _ in test_files]
            for file_idx, file_path in enumerate(test_files):
                file_paths.append(file_path)
                mtid_list.append(label)
                mtype_list.append(id_ma_map[machine_type])
                y_true_list.append(y_true[file_idx])
    print("mtid set: ", set(mtid_list))
    # return
    print(f"length: files {len(file_paths)}; mtid {len(mtid_list)}; mtype {len(mtype_list)}; y_true {len(y_true_list)}")
    tmp_dataset = SpecAllReader(file_paths=file_paths, mtype_list=mtype_list, mtid_list=mtid_list,
                                y_true_list=y_true_list, configs=configs, istrain=True, istest=False)
    loader = DataLoader(tmp_dataset, batch_size=32, shuffle=False)
    tsne_input = []
    mtids = []
    machine_type = "pump"

    for x_idx, (_, x_mel, _, mtid, _) in enumerate(tqdm(loader, desc="Training")):
        # x_wav = x_wav.to(torch.float32)
        x_mel = x_mel.to(torch.float32)
        # print(mtid)
        # print(mtid, mtid)
        mtid = mtid.long().to(device) - 11
        with torch.no_grad():
            if is_load_recon_model:
                latent_feat = pretrain_encoder(input_mel=x_mel.unsqueeze(1) / 255., class_vec=None,
                                               coarse_cls=False,
                                               fine_cls=False)
                if not is_double:
                    x_mel = pretrain_decoder(latent_feat).squeeze()
                    x_mel = x_mel.transpose(1, 2)  # / 255.
                # print("shape of encoded latent:", latent_feat.shape)
                else:
                    recon_mel = pretrain_decoder(latent_feat).squeeze()
                    recon_mel = recon_mel  # / 255.
                    # x_wav = torch.cat((x_wav, x_wav), dim=0)
                    x_mel = torch.cat((x_mel / 255., recon_mel), dim=0).unsqueeze(1)  # .transpose(2, 3)
                    new_mtid = mtid + 4
                    mtid = torch.cat((mtid, new_mtid), dim=0)
            print("shape of concat mels:", x_mel.shape)
            feat = pretrain_encoder(input_mel=x_mel, class_vec=None, coarse_cls=False, fine_cls=False)
        feat = feat.mean(3).mean(2)
        tsne_input.append(feat.data.cpu().numpy())
        # pred_input.append(pred.data.cpu().numpy())
        mtids.append(mtid.data.cpu().numpy())
        # istrues.append(y_true[file_idx])
    tsne_input = np.concatenate(tsne_input, axis=0)
    # pred_input = np.concatenate(pred_input, axis=0)
    mtids = np.concatenate(mtids, axis=0)
    print("----type and shape----")
    print("feat:", tsne_input.dtype, tsne_input.shape)
    # print("pred:", pred_input.dtype, pred_input.shape)
    print("mtids:", len(mtids), mtids[0], )
    # print("ytrue:", len(istrues), istrues[0])
    tsne_model = TSNE(n_components=2, init="pca", random_state=3407)
    result2D = tsne_model.fit_transform(tsne_input)
    print("TSNE finish.")
    resume_model_path = "./run/dcase2020_mystgram/202402071605_mystgram_tdnn_double_contras"
    plot_embedding_2D(result2D, mtids, "t-SNT for mtid",
                      savepath=resume_model_path + f"/model_epoch_{6}_combine_3/convenc_test6_{machine_type}_tsne_s8_3.{saveimg_params['format']}",
                      names=[f"origin_id-{i}" for i in range(4)]+[f"recon_id-{i}" for i in range(4)], params=saveimg_params)


if __name__ == '__main__':
    test_convencoder_tsne_origin_recon(is_load_recon_model=True, is_double=True)
    # test_mfn_tsne_origin_recon(is_load_recon_model=True, is_double=True)
