import torch
import torch.nn as nn
import torch.nn.functional as F

class DispLoss(nn.Module):
    def __init__(self):
        super(DispLoss, self).__init__()

    def forward(self, img, gt):

        return loss

def losses_head(self, out, disp_true, mask):
    # out = torch.unsqueeze(out, dim=1)
    # B, C, H, W = out.shape
    out = F.interpolate(
        out, disp_true.shape[-2:], mode='bilinear', align_corners=False)  # up to 1
    #--------loss2--------
    out = F.log_softmax(out, dim=1)
    out2 = disparity_regression(torch.exp(out), self.num_class)
    out2 = torch.unsqueeze(out2, dim=1)
    disp_true = disp_true//(self.maxdisp//self.num_class)
    alarm = torch.abs(out2 - disp_true)
    mmask = (alarm >= 1.2).logical_and(mask)
    if(len(mmask[mmask==True])):
        loss2 = F.smooth_l1_loss(
            out2[mmask], disp_true[mmask], reduction='mean')
    else:
        loss2 = 0

    disp_true = disp_true//1
    disp_true = disp_true.type('torch.cuda.LongTensor')
    disp_true[~mask] = -1

    loss = F.nll_loss(out, disp_true.squeeze(1), ignore_index=-1)

    return loss+loss2

def loss_disp(self, outputs, disp_true, mask, left=None):
    # offset = torch.exp(F.log_softmax(offset, dim=1))
    mask = torch.squeeze(mask, dim=1)
    # if(len(mask) == 0):
    #     return 0
    loss1 = []
    disp_true = torch.squeeze(disp_true, dim=1)
    weights = [0.5, 0.5, 0.7, 1.0]

    for weights, output in zip(weights, outputs):
        loss1.append(
            weights*(
                F.smooth_l1_loss(output[mask], disp_true[mask], reduction='mean') 
            )
        )
    loss = sum(loss1)

    return loss