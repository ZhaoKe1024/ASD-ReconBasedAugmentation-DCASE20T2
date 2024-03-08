#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/12/31 16:47
# @Author: ZhaoKe
# @File : func.py
# @Software: PyCharm
import os
import sys
import numpy as np
import random
import torch
import scipy
from sklearn import metrics
import pickle


def get_prec_recall_f1_acc(pred_list, true_list, ifauc=False):
    if ifauc:
        max_value = list(pred_list.data.cpu().numpy().max(axis=1))
        auc = metrics.roc_auc_score(true_list, max_value, multi_class="ovr")
        p_auc = metrics.roc_auc_score(true_list, max_value, max_fpr=0.1, multi_class="ovr")
    max_arg = list(pred_list.data.cpu().numpy().argmax(axis=1))
    # print("shape of max", pred_list.data.cpu().numpy().max(axis=1).shape)
    # print("shape of argmax", pred_list.data.cpu().numpy().argmax(axis=1).shape)
    tn, fp, fn, tp = metrics.confusion_matrix(max_arg, true_list).ravel()
    prec = tp / np.maximum(tp + fp, sys.float_info.epsilon)
    recall = tp / np.maximum(tp + fn, sys.float_info.epsilon)
    f1 = 2.0 * prec * recall / np.maximum(prec + recall, sys.float_info.epsilon)
    acc = (tp + tn) / (tp + tn + fp + fn)
    if ifauc:
        return prec, recall, f1, acc, tn, fp, fn, tp, auc, p_auc
    else:
        return prec, recall, f1, acc, tn, fp, fn, tp


def loss_class(prob_mtid, true_mtid):
    one_hot = torch.zeros(probs.shape, device=probs.device)
    one_hot = one_hot.scatter_(1, mtid.unsqueeze(1).long(), 1)
    return


def set_global_seed(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        # if any(m.bias):
        # torch.nn.init.constant_(m.bias, 0.)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1.)
        # torch.nn.init.constant_(m.bias, 0.)


def loss_reduction_1d(score):
    return torch.mean(score, dim=1)


def loss_reduction(score, n_loss):
    return torch.sum(score) / n_loss


def fit_anomaly_score_distribution(y_pred, score_dist_file_path=None):
    shape_hat, loc_hat, scale_hat = scipy.stats.gamma.fit(y_pred)
    gamma_params = [shape_hat, loc_hat, scale_hat]
    if score_dist_file_path is not None:
        with open(score_dist_file_path, "wb") as f:
            pickle.dump(gamma_params, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        return gamma_params


def calc_decision_threshold(score_distr_file_path=None, decision_threshold=0.9):
    # load anomaly score distribution for determining threshold
    with open(score_distr_file_path, "rb") as f:
        shape_hat, loc_hat, scale_hat = pickle.load(f)
    print(f"shape_hat:{shape_hat}, loc_hat:{loc_hat}, scale_hat:{scale_hat}")
    # determine threshold for decision
    return scipy.stats.gamma.ppf(q=decision_threshold, a=shape_hat, loc=loc_hat, scale=scale_hat)


def loss_function_mahala(recon_x, x, block_size, cov=None, update_cov=False,
                         use_precision=False, reduction=True):
    ### Modified mahalanobis loss###
    if update_cov is False:
        loss = mahalanobis(recon_x.view(-1, block_size), x.view(-1, block_size), cov, use_precision,
                           reduction=reduction)
        return loss
    else:
        diff = x - recon_x
        cov_diff, _ = cov_v_diff(in_v=diff.view(-1, block_size))

        loss = diff ** 2
        if reduction:
            loss = torch.mean(loss, dim=1)

        return loss, cov_diff


def loss_function_mahala_st(recon_x, x, block_size, cov=None, is_source_list=None, is_target_list=None,
                            update_cov=False,
                            use_precision=False, reduction=True):
    ### Modified mahalanobis loss###
    if update_cov is False:
        loss = mahalanobis(recon_x.view(-1, block_size), x.view(-1, block_size), cov, use_precision,
                           reduction=reduction)
        return loss
    else:
        diff = x - recon_x
        cov_diff_source, _ = cov_v_diff(in_v=diff[is_source_list].view(-1, block_size))

        cov_diff_target = None
        is_calc_cov_target = any(is_target_list)
        if is_calc_cov_target:
            cov_diff_target, _ = cov_v_diff(in_v=diff[is_target_list].view(-1, block_size))

        loss = diff ** 2
        if reduction:
            loss = torch.mean(loss, dim=1)

        return loss, cov_diff_source, cov_diff_target


def mahalanobis(u, v, cov_x, use_precision=False, reduction=True):
    num, dim = v.size()
    if use_precision is True:
        inv_cov = cov_x
    else:
        inv_cov = torch.inverse(cov_x)
    delta = torch.sub(u, v)
    m_loss = torch.matmul(torch.matmul(delta, inv_cov), delta.t())

    if reduction:
        return torch.sum(m_loss) / num
    else:
        return m_loss, num


def cov_v(diff, num):
    var = torch.matmul(diff.t(), diff) / num
    return var


def cov_v_diff(in_v):
    in_v_tmp = in_v.clone()
    mu = torch.mean(in_v_tmp.t(), 1)
    diff = torch.sub(in_v, mu)

    return diff, mu


def calc_inv_cov(model, device="cpu"):
    inv_cov = None
    cov_x = model.cov.data
    cov_x = cov_x.to(device).float()
    inv_cov = torch.inverse(cov_x)
    inv_cov = inv_cov.to(device).float()
    return inv_cov

def calc_inv_cov_st(model, device="cpu"):
    inv_cov_source = None
    inv_cov_target = None

    cov_x_source = model.cov_source.data
    cov_x_source = cov_x_source.to(device).float()
    inv_cov_source = torch.inverse(cov_x_source)
    inv_cov_source = inv_cov_source.to(device).float()

    cov_x_target = model.cov_target.data
    cov_x_target = cov_x_target.to(device).float()
    inv_cov_target = torch.inverse(cov_x_target)
    inv_cov_target = inv_cov_target.to(device).float()

    return inv_cov_source, inv_cov_target


def get_onehot(batch_softmax, true_label):
    one_hot = torch.zeros(batch_softmax.shape, device=batch_softmax.device)
    # print(x.device, label.device, one_hot.device)
    one_hot.scatter_(1, true_label.long(), 1)
    # output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
    # print(one_hot)
    return one_hot


def padding_to_max(data_x, max_len=313):
    d, le = data_x.shape
    new_tensor = torch.zeros(d, max_len)
    new_tensor[:, :le] = data_x[:]
    return new_tensor


def accuracy(output, label):
    output = torch.nn.functional.softmax(output, dim=-1)
    output = output.data.cpu().numpy()
    output = np.argmax(output, axis=1)
    label = label.data.cpu().numpy()
    acc = np.mean((output == label).astype(int))
    return acc


if __name__ == '__main__':
    data_x = torch.randn(128, 311)
    print(padding_to_max(data_x, 313).shape)
    # dt = calc_decision_threshold("../run/dcase2020_convae/ConvAE/202401011551/anomaly_dist_fit.pkl", 0.9)
    # print("decision threshold:", dt)

    # x = torch.zeros(8, 1)
    # x[0][0] = 1.
    # x[1][0] = 22.
    # x[2][0] = 15.
    # x[3][0] = 14.
    # x[4][0] = 6.
    # x[5][0] = 0.
    # x[6][0] = 8.
    # x[7][0] = 12.
    # print(x)
    # print(get_onehot(torch.randn(8, 23), x))
