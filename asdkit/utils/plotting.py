#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/1/5 8:43
# @Author: ZhaoKe
# @File : plotting.py
# @Software: PyCharm
"""Visualize audio embeddings."""
import os
import librosa
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
from sklearn.manifold import TSNE
from tqdm import tqdm

from asdkit.data_utils.audio import wav_slice_padding
from asdkit.data_utils.featurizer import Wave2Mel
from asdkit.models.autoencoder import LinearAENet, ConvEncoder, ConvDecoder
from asdkit.data_utils.dcase2020_fileutils import file_to_vector_array
from asdkit.utils.utils import load_ckpt

# 对比度max的配色
rgb_planning_22 = ["#e6194B",  # 0
                   "#3cb44b",
                   "#ffe119",
                   "#4363d8",
                   "#f59231",
                   "#911eb4",  # 5
                   "#42d4f4",
                   "#f032e6",
                   "#bfef45",
                   "#fabed4",
                   "#469990",  # 10
                   "#dcbeff",
                   "#9A6324",
                   "#fffac8",
                   "#800000",
                   "#aaffc3",
                   "#808000",  # 16
                   "#ffd8b1",
                   "#000075",
                   "#a9a9a9",
                   "#ffffff",
                   "#000000",  # 21
                   ]


def t_sne_visualize(data_dirs, checkpoint_path):
    """Visualize high-dimensional embeddings using t-SNE."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearAENet(input_dim=640)
    model.state_dict = torch.load(checkpoint_path + "/model_slider.pth")
    model.eval().to(device)
    print("----model----")
    print(model)
    print("[INFO] model loaded.")

    # paths = []
    # for data_dir in data_dirs+"/train/":
    #     data_dir_path = Path(data_dir)
    #     for spkr_dir in [x for x in data_dir_path.iterdir() if x.is_dir()]:
    #         n_spkrs += 1
    #         audio_paths = find_files(spkr_dir)
    #         spkr_name = spkr_dir.name
    #         for audio_path in audio_paths:
    #             paths.append(audio_path)
    #             spkr_names.append(spkr_name)
    embs = []
    names = []
    for audio_path in tqdm(os.listdir(data_dirs + "/train/"), ncols=0, desc="Preprocess and Embed"):
        names.append(audio_path.split('_')[2])
        mel = file_to_vector_array(data_dirs + "/train/" + audio_path, n_mels=128, frames=5)
        with torch.no_grad():
            # print(mel.shape)  # [309, 640]
            mel = torch.tensor(mel[np.newaxis, np.newaxis, :, :], device=device).to(torch.float32)
            # print(mel.view(-1, 640).shape)
            # print(mel.shape)  # [1, 1, 309, 640]
            _, emb = model(mel)
            # print(emb.flatten().shape)  # [309, 8]
            emb = emb.flatten().detach().cpu().numpy()
        embs.append(emb)

    n_spkrs = len(set(names))
    # for mel in tqdm(mels, ncols=0, desc="Embed"):
    #     :
    #         # 输入[b, 1, length, dim=320]
    #         # 输出[:, 8]  -> reshape()
    #     embs.append(emb)

    embs = np.array(embs)
    print("embs shape", embs.shape)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    transformed = tsne.fit_transform(embs)
    #
    print("[INFO] embeddings transformed.")
    #
    data = {
        "dim-1": transformed[:, 0],
        "dim-2": transformed[:, 1],
        "label": names,
    }
    #
    plt.figure()
    sns.scatterplot(
        x="dim-1",
        y="dim-2",
        hue="label",
        palette=sns.color_palette(n_colors=n_spkrs),
        data=data,
        legend="full",
    )
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(checkpoint_path + "/model_slider_tsne_dev_train.png")


color_map = ['r', 'y', 'k', 'g', 'b', 'm', 'c']  # 7个类，准备7种颜色

labels = ["ToyCar_01", "ToyCar_02", "ToyCar_03", "ToyCar_04",
          "ToyConveyor_01", "ToyConveyor_02", "ToyConveyor_03",
          "fan_00", "fan_02", "fan_04", "fan_06",
          "pump_00", "pump_02", "pump_04", "pump_06",
          "slider_00", "slider_02", "slider_04", "slider_06",
          "valve_00", "valve_02", "valve_04", "valve_06"]

labels1 = ["Ca1", "Ca2", "Ca3", "Ca4",
           "C1", "C2", "C3",
           "f0", "f2", "f4", "f6",
           "p0", "p2", "p4", "p6",
           "s0", "s2", "s4", "s6",
           "v0", "v2", "v4", "v6"]
# 32个对比度最有区分度的颜色
rgb_planning_23 = [
    "#B0C4DE", "#FF00FF", "#1E90FF", "#FA8072",
    "#EEE8AA", "#FF1493", "#7B68EE",
    "#FFC0CB", "#696969", "#556B2F", "#CD853F",
    "#000080", "#32CD32", "#7F007F", "#B03060",
    "#800000", "#483D8B", "#008000", "#3CB371",
    "#008B8B", "#FF0000", "#FF8C00", "#FFD700",
    "#00FF00", "#9400D3", "#00FA9A", "#DC143C", "#00FFFF", "#00BFFF", "#0000FF", "#ADFF2F", "#DA70D6"
]

# names = ["ToyCar", "ToyConveyor", "fan", "pump", "slider", "valve"]
rgb_planning_6 = ["#800000", "#000075", "#dcbeff", "#ffe119", "#000000", "#f58231"]
color_8 = [
    "#000075", "#9A6324", "#000000", "#000075",
    "#800000", "#f58231", "#ffe119", "#f032e6", ]
def save_legend():
    names = [f"origin_id-{i}" for i in range(4)]+[f"recon_id-{i}" for i in range(4)]
    params = {"format": "svg", "marker_size": 8, "alpha": 0.8}
    label = list(range(8))
    plt.figure(0)
    for i in range(8):
        plt.scatter(0, 0, s=params["marker_size"], c=color_8[label[i]], label=names[label[i]], alpha=params["alpha"])
    plt.xticks([])
    plt.yticks([])
    plt.legend(names, loc="upper right")
    plt.savefig("./legend_bar.svg", dpi=600, format=params["format"], bbox_inches="tight")
    plt.close()


def plot_embedding_2D(data, label, title, savepath, names, params=None):
    """
    """
    # x_min, x_max = np.min(data, 0), np.max(data, 0)
    # data = (data - x_min) / (x_max - x_min)
    fig1 = plt.figure()

    label_cnt = [1] * len(names)
    p_list = [None] * len(names)
    for i in range(data.shape[0]):
        p = plt.scatter(data[i, 0], data[i, 1], s=params["marker_size"], c=color_8[label[i]], label=names[label[i]], alpha=params["alpha"])
        if label_cnt[label[i]] == 1:
            p_list[label[i]] = p
            label_cnt[label[i]] = 0
        # plt.plot(data[i, 0], data[i, 1], marker='o', markersize=1, color=color_8[label[i]])
    plt.xticks([])
    plt.yticks([])
    # plt.title(title)
    # plt.legend(names, bbox_to_anchor=(1.05, 1), loc="upper right")
    # plt.legend(p_list, names, loc="lower right")
    # plt.savefig(savepath)
    plt.savefig(savepath, dpi=600, format=params["format"], bbox_inches="tight")
    plt.close()
    # return fig1  # , fig2


import pandas as pd
from sklearn import metrics


def get_heat_map(pred_matrix, label_vec, savepath):
    max_arg = list(pred_matrix.argmax(axis=1))
    conf_mat = metrics.confusion_matrix(max_arg, label_vec)
    df_cm = pd.DataFrame(conf_mat, index=range(conf_mat.shape[0]), columns=range(conf_mat.shape[0]))
    heatmap = sns.heatmap(df_cm, annot=True, fmt='d', cmap='YlGnBu')
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.xlabel("predict label")
    plt.ylabel("true label")
    plt.savefig(savepath)
    # plt.show()


def recon_image(ae_model):
    test_wav_path = "G:/DATAS-DCASE-ASD/DCASE2020Task2ASD/dataset/dev_data_fan/fan/test/normal_id_06_00000090.wav"
    y, sr = librosa.core.load(test_wav_path, sr=16000)
    y = wav_slice_padding(y, 147000)
    w2m = Wave2Mel(16000)
    x_mel = w2m(torch.from_numpy(y.T))
    x_input = x_mel.unsqueeze(0).unsqueeze(0).to(torch.device("cuda")).transpose(2, 3)
    print(x_input.shape)
    # latent_spec = encoder_model(input_mel=x_input, label_vec=None, ifcls=False)
    # print(latent_spec.shape)
    recons_spec, _, _ = ae_model(x_input)
    print(recons_spec.shape)

    # print(z.shape)
    # print()
    plt.figure(0)
    plt.subplot(2, 1, 1)
    plt.imshow(np.asarray(x_input.transpose(2, 3).squeeze().data.cpu().numpy()))

    plt.subplot(2, 1, 2)
    plt.imshow(np.asarray(255 * recons_spec.transpose(2, 3).squeeze().data.cpu().numpy()))
    plt.show()


if __name__ == '__main__':
    # recon_image()
    save_legend()
    # t_sne_visualize(data_dirs="G:/DATAS-DCASE-ASD/DCASE2020Task2ASD/dataset/dev_data_slider/slider",
    #                 checkpoint_path="../../run/LinearAE/202401041213_mse_epoch100", )
    # plt.figure(0)
    # x = [1, 2, 3, 1, 2, 3]
    # y = [2, 2, 2, 1, 1, 1]
    # plt.scatter(x, y, 100, rgb_planning_6)
    # plt.legend(names)
    # plt.show()
    # plt.figure(1)
    # x = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5]
    # y = [4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1]
    # plt.scatter(x, y, 100, rgb_planning_23[:23])
    # plt.legend(labels)
    # plt.show()
