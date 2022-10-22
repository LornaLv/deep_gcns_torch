import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_kd(all_out, teacher_all_out, outputs, labels, teacher_outputs,
            alpha, temperature):
    """
    loss function for Knowledge Distillation (KD)
    """

    T = temperature

    # loss_CE：CE loss，交叉熵损失，计算outputs与labels的二进制交叉熵
    loss_CE = F.cross_entropy(outputs, labels)

    # D_kl：KLDivLoss，KL散度损失，相对熵，用于衡量两个分布（离散分布和连续分布）之间的距离。
    # dim=1，表示对行做归一化；dim=0，表示对列做归一化
    D_KL = nn.KLDivLoss()(F.log_softmax(all_out / T, dim=1),
                          F.softmax(teacher_all_out / T, dim=1)) * (T * T)
    KD_loss = (1. - alpha) * loss_CE + alpha * D_KL

    return KD_loss

def loss_kd_only(all_out, teacher_all_out, temperature):
    T = temperature

    D_KL = nn.KLDivLoss()(F.log_softmax(all_out / T, dim=1),
                          F.softmax(teacher_all_out / T, dim=1)) * (T * T)

    return D_KL
