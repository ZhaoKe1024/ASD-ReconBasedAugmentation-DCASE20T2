#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/3/8 14:30
# @Author: ZhaoKe
# @File : train_rbasd.py
# @Software: PyCharm
from asdkit.trainer_rbasd import TrainerRBASD

if __name__ == '__main__':
    is_tr = False
    trainer = TrainerRBASD(istrain=is_tr)
    # recon, recon+origin, contrastive
    reconstruction = True
    is_double = True
    if is_double and not reconstruction:
        exit(-1)
    trainer.train_all(is_load_recon_model=True, is_double=True, is_trans=True)
    # trainer = TrainerSTgram(istrain=False)
    # trainer.test_auc(epoch_id=4,
    #                  resume_model_path="../run/dcase2020_mystgram/202403021528_mystgram_tdnn_double_contras",
    #                  is_load_recon_model=False, mode="test")
    # trainer.test_tsne_list("../run/dcase2020_mystgram/202401211444_mystgram")
