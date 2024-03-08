#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/1/13 15:22
# @Author: ZhaoKe
# @File : trainer_settings.py
# @Software: PyCharm
from asdkit.models.autoencoder import *
from asdkit.models.stgrammfn import STgramMFN
from asdkit.models.mobilefacenet import MobileFaceNet
from asdkit.modules import func
from asdkit.utils.utils import load_ckpt


def get_model(use_model, configs, istrain=True, params=None):
    model = None
    if use_model == "mobilefacenet":
        model = MobileFaceNet(inp_c=1, num_class=configs["model"]["mtid_class_num"])
    elif use_model == "conv_encoder_decoder":
        encoder = ConvEncoder(input_channel=1, input_length=configs["model"]["input_length"],
                              input_dim=configs["feature"]["n_mels"],
                              class_num=configs["model"]["mtid_class_num"],
                              class_num1=configs["model"]["type_class_num"])
        decoder = ConvDecoder(input_channel=1, input_length=configs["model"]["input_length"],
                              input_dim=configs["feature"]["n_mels"])
        model = [encoder, decoder]
    elif use_model == "rbasd":
        model = STgramMFN(num_classes=23, use_arcface=False, m=0.7, s=30, sub=1)
    else:
        raise ValueError("this model is not found!!")
    if istrain:
        if isinstance(model, list):
            for item in model:
                item.apply(func.weight_init)
        else:
            model.apply(func.weight_init)
        # amp_scaler = torch.cuda.amp.GradScaler(init_scale=1024)
    return model


def get_pretrain_models(encoder_path, decoder_path, encoder_epoch, decoder_epoch, configs, use_gpu=True):
    if use_gpu:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")
    ed_models = get_model("conv_encoder_decoder", configs=configs, istrain=False)
    encoder = ed_models[0]
    decoder = ed_models[1]
    load_ckpt(encoder, encoder_path + f"/model_epoch_{encoder_epoch}", load_epoch=encoder_epoch)
    load_ckpt(decoder, decoder_path + f"/model_epoch_{decoder_epoch}", load_epoch=decoder_epoch)
    encoder.to(device).eval()
    decoder.to(device).eval()
    print("Load Pretrain AutoEncoder on Device", device, " Successfully!!")
    return encoder, decoder
