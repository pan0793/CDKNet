import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.common import pad_size
import torch.distributed as dist
import numpy as np

base=32

def test_kd(model, data_batch, gpu):
    model.eval()
    def f(a): return a.cuda(gpu) if a is not None else a
    L = f(data_batch['left'])
    R = f(data_batch['right'])
    disp_L = f(data_batch['disp'])
    disp_L = torch.squeeze(disp_L, dim=1)
    
    bot_pad = int(
        base-L.shape[2] % base) if int(base-L.shape[2] % base) != base else 0
    right_pad = int(
        base-L.shape[3] % base) if int(base-L.shape[3] % base) != base else 0
    # model.module.Regression.set_full_shape(
    #     (L.shape[2]+bot_pad, L.shape[3]+right_pad))
    L = F.pad(
        L, (0, right_pad, 0, bot_pad),
        "constant", 0
    )
    R = F.pad(
        R, (0, right_pad, 0, bot_pad,),
        "constant", 0
    )
    output = model(L, R)[1][0]
    # output = unpad_img(output, bot_pad, right_pad)
    if bot_pad > 0:
        output = output[:, :-bot_pad, :]
    if right_pad > 0:
        output = output[:, :, :-right_pad]
    mask = (disp_L>0)&(disp_L<192)
    loss_disp = [0,0]
    loss_disp = F.l1_loss(output[mask], disp_L[mask])
    if loss_disp == 0 or not torch.isfinite(loss_disp):
        loss_disp =0
    else:
        loss_disp = loss_disp.item()
    # for idx, l in enumerate(loss_disp):
    #     if l == 0 or not torch.isfinite(l):
    #         l = 0
    #     else:
    #         l = l.item()
    #     loss_disp[idx] = l

    return [loss_disp,1], 0



def train_kd(KD_warper, Smodel, Tmodel, data_batch, gpu, optimizer):
    Smodel.train()
    Tmodel.eval()
    
    def f(a): return a.cuda(gpu) if a is not None else a
    imgL = f(data_batch['left'])
    imgR = f(data_batch['right'])
    disp_L = f(data_batch['disp'])
    disp_L = torch.squeeze(disp_L, dim=1)
    optimizer.zero_grad()
    loss_list = KD_warper(Smodel, Tmodel, imgL, imgR, disp_L)
    
    if(len(loss_list) == 2):
        loss_disp, loss_head = loss_list
        loss = loss_disp
    elif(len(loss_list) == 3):
        loss_disp, loss_head, loss_kd = loss_list
        loss = loss_disp + loss_kd
    loss.backward()
    optimizer.step()

    ans = (
        loss.item() if not isinstance(loss, int) else loss,
        loss_disp.item() if not isinstance(loss_disp, int) else loss_disp, 
        0,
    )
    if len(loss_list) == 3:
        ans += (loss_kd.item() if not (isinstance(loss_kd, float) or isinstance(loss_kd, int)) else loss_kd , )

    return ans



class KD_warper(nn.Module):
    def __init__(self, maxdisp=192, naive=False):
        super().__init__()
        self.maxdisp = maxdisp
        # self.T_model = teacher
        # self.model = student
        # self.naive = naive
        if naive:
            self.ah1 = nn.Sequential(
                nn.ReLU6(True),
                nn.Conv2d(32, 16, 1, 1, 0),
            )
            self.ah2 = nn.Sequential(
                nn.ReLU6(True),
                nn.Conv2d(64, 16, 1, 1, 0),
            )
            self.ah3 = nn.Sequential(
                nn.ReLU6(True),
                nn.Conv2d(128, 32, 1, 1, 0),
            )
            self.ah4 = nn.Sequential(
                nn.ReLU6(True),
                nn.Conv2d(128, 32, 1, 1, 0),
            )
        self.eval_losses=[
            self.loss_disp,
            self.D1_metric,
        ]
        # for param in self.T_model.parameters():
        #     param.requires_grad = False
        # for param in self.model.parameters():
        #     param.requires_grad = True
        # self.Knowledge_distill = Knowledge_distill

    def forward(self, Smodel, T_model, L, R, gt):
        # self.T_model.eval()
        
        # L, R, bot_pad, right_pad = self.pad_img(L, R)
        KL_S, S_output = Smodel(L, R)
        # S_output = self.unpad_img(S_output, bot_pad, right_pad)
        # if self.training:
        #     KL_T, T_output = T_model(L, R)
        #     loss_kd = self.Knowledge_distill(KL_T, KL_S)
        # else:
        #     loss_kd = 0
        loss_kd = 0

        mask = (gt > 0) & (gt < self.maxdisp)
        if S_output[0] is not None:
            if self.training:
                loss_disp = self.loss_disp(S_output[0], gt, mask)
            else:
                loss_disp = [_(S_output[0], gt, mask)
                             for _ in self.eval_losses]

        # loss_head = 0

        return loss_disp, 0, loss_kd

    def Knowledge_distill(self, KL_T, KL_S):
        return 0
        """
        #* KL_T contains:
        (
            (feature_L:(x,l2,l3,l4), feature_R:(x,l2,l3,l4)),
            (Volume),
            (head),
            (out3, pred3)
        )
        #* KL_S contains:
        (
            (feature_L:(x,l2), feature_R:(x,l2)),
            (Volume),
            (head),
            (out3, pred3)
        )
        """
        KL_T_1, KL_T_2, KL_T_3, KL_T_4 = KL_T
        KL_S_1, KL_S_2, KL_S_3, KL_S_4 = KL_S
        loss = 0
        # feature_T_L, feature_T_R = KL_T_1
        # ! ---------------------------------------------------
        # KL_T = (x, l2, l3, l4)
        # KL_S = (x, l2)
        if self.naive:
            for i in range(2):
                KL_T_1_new = [
                    self.ah1(KL_T_1[i][0]),
                    self.ah2(KL_T_1[i][1]),
                    self.ah3(KL_T_1[i][2]),
                    self.ah4(KL_T_1[i][3])
                ]
                KL_S_1_new = [
                    KL_S_1[i][0],
                    KL_S_1[i][1][:, :16, :, :],
                    KL_S_1[i][1][:, 16:48, :, :],
                    KL_S_1[i][1][:, 48:, :, :]
                ]

                for feature_L_T, feature_L_S in zip(KL_T_1_new, KL_S_1_new):
                    # F.avg_pool2d()
                    loss += F.smooth_l1_loss(
                        feature_L_T, feature_L_S, reduction='mean')
        else:
            for feature_L_T, feature_L_S in zip(KL_T_1[0], KL_S_1[0]):
                loss += F.smooth_l1_loss(
                    feature_L_T, feature_L_S, reduction='mean')
            for feature_R_T, feature_R_S in zip(KL_T_1[1], KL_S_1[1]):
                loss += F.smooth_l1_loss(
                    feature_R_T, feature_R_S, reduction='mean')
        # ----------------------------------------------------
        # ! ---------------------------------------------------

        loss += F.smooth_l1_loss(
            KL_S_2, KL_T_2,  reduction='mean')

        loss += F.kl_div(
            F.log_softmax(KL_S_3, dim=1), F.softmax(KL_T_3, dim=1), reduction='batchmean')

        loss += F.smooth_l1_loss(
            KL_S_4[0], KL_T_4[0],  reduction='mean')
        loss += F.kl_div(KL_S_4[1].log(), KL_T_4[1], reduction='batchmean')

        return loss
    
    def loss_disp(self, preds, gt, mask):
        mask = torch.squeeze(mask, dim=1)
        gt = torch.squeeze(gt, dim=1)

        if isinstance(preds, list):
            weights = [0.5, 0.5, 0.7, 1.0]
            loss1 = []
            for weights, output in zip(weights, preds):
                loss1.append(
                    weights*(
                        F.smooth_l1_loss(
                            output[mask], gt[mask], reduction='mean')
                    )
                )
            loss = sum(loss1)
        else:
            loss = F.smooth_l1_loss(preds[mask], gt[mask], reduction='mean')
        return loss
    
    def D1_metric(self, D_es, D_gt, mask):
        D_es, D_gt = D_es[mask], D_gt[mask]
        E = torch.abs(D_gt - D_es)
        err_mask = (E > 3) & (E / D_gt.abs() > 0.05)
        return torch.mean(err_mask.float())

class Loss_warper(nn.Module):
    def __init__(self, model=None, maxdisp=192):
        super().__init__()
        self.maxdisp = maxdisp
        self.model = model
        # self.right_pad = -1
        # self.bot_pad = -1
        self.T_model = None
        self.eval_losses = [
            self.loss_disp,
            self.D1_metric,
        ]

    def forward(self, L, R, gt):
        L, R, bot_pad, right_pad = self.pad_img(L, R)
        output = self.model(L, R)
        output = self.unpad_img(output, bot_pad, right_pad)
        mask = (gt > 0) & (gt < self.maxdisp)
        if output[0] is not None:
            if self.training:
                loss_disp = self.loss_disp(output[0], gt, mask)
            else:
                loss_disp = [_(output[0], gt, mask) for _ in self.eval_losses]
        else:
            loss_disp = 0
        if output[1] is not None:
            loss_head = self.losses_head(output[1], gt, mask)
        else:
            loss_head = 0
        return loss_disp, loss_head
    

    def pad_img(self, L, R, base=32):
        if self.training is True:
            if self.T_model:
                self.T_model.Regression.set_full_shape((L.shape[2], L.shape[3]))
            self.model.Regression.set_full_shape((L.shape[2], L.shape[3]))
            return L, R, 0, 0
        else:
            bot_pad = int(
                base-L.shape[2] % base) if int(base-L.shape[2] % base) != base else 0
            right_pad = int(
                base-L.shape[3] % base) if int(base-L.shape[3] % base) != base else 0
            self.model.Regression.set_full_shape(
                (L.shape[2]+bot_pad, L.shape[3]+right_pad))
            L = F.pad(
                L, (0, right_pad, 0, bot_pad),
                "constant", 0
            )
            R = F.pad(
                R, (0, right_pad, 0, bot_pad,),
                "constant", 0
            )
            return L, R, bot_pad, right_pad

    def unpad_img(self, output, bot_pad, right_pad):
        if self.training is True:
            return output
        else:
            output = list(output)
            for i in range(len(output)):
                if output[i] is not None:
                    output[i] = output[i][:, :-bot_pad,
                                          :] if bot_pad > 0 else output[i]
                    output[i] = output[i][:, :, :-
                                          right_pad] if right_pad > 0 else output[i]

            return output

    def loss_disp(self, preds, gt, mask):
        mask = torch.squeeze(mask, dim=1)
        gt = torch.squeeze(gt, dim=1)

        if isinstance(preds, list):
            weights = [0.5, 0.5, 0.7, 1.0]
            loss1 = []
            for weights, output in zip(weights, preds):
                loss1.append(
                    weights*(
                        F.smooth_l1_loss(
                            output[mask], gt[mask], reduction='mean')
                    )
                )
            loss = sum(loss1)
        else:
            loss = F.smooth_l1_loss(preds[mask], gt[mask], reduction='mean')
        return loss
    
    def eval_loss(self, preds, gt, mask):
        loss = F.l1_loss(preds[mask], gt[mask], reduction='mean')
        return loss
        
    
    def D1_metric(self, D_es, D_gt, mask):
        D_es, D_gt = D_es[mask], D_gt[mask]
        E = torch.abs(D_gt - D_es)
        err_mask = (E > 3) & (E / D_gt.abs() > 0.05)
        return torch.mean(err_mask.float())

