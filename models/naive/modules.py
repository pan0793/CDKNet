from __future__ import print_function
import math
import numpy as np
from numpy.core.fromnumeric import argmax
from torch.autograd.variable import Variable
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from models.submodule import groupwise_correlation, build_gwc_volume, MobileV2_Residual, convbn, interweave_tensors, disparity_regression, hourglass2D, MobileV1_Residual
from models.model import model_backbone, Regression_bone

group_num = 1

class Volume_construct(nn.Module):
    def __init__(self, volume_size, inchannels=80):
        super().__init__()
        self.num_groups = 1
        self.volume_size = volume_size
        self.preconv11 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(inchannels),
            nn.Conv2d(inchannels, 32, 1, 1, 0, 1)
        )

    def forward(self, features_L, features_R):
        featL = self.preconv11(features_L)
        featR = self.preconv11(features_R)
        B, C, H, W = featL.shape
        volume = featL.new_zeros([B, self.num_groups, self.volume_size, H, W])
        for i in range(self.volume_size):
            if i > 0:
                volume[:, :, i, :, i:] = groupwise_correlation(featL[:, :, :, i:], featR[:, :, :, :-i],
                                                               1)
            else:
                volume[:, :, i, :, :] = groupwise_correlation(featL, featR, 1)

        volume = volume.contiguous()
        volume = torch.squeeze(volume, 1)
        return volume

####################### insert ######################

####################### insert #####################



class Regression(Regression_bone):
    def __init__(self, maxdisp=192, full_shape=None,KL_mode=False, output_disp=48):
        super().__init__(maxdisp=maxdisp,full_shape=full_shape,KL_mode=KL_mode,output_disp=output_disp)

        # self.classif1 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1, groups=group_num),
        #                               nn.ReLU(inplace=True),
        #                               nn.Conv2d(self.hg_size, self.output_disp, kernel_size=3, padding=1, stride=1,
        #                                         bias=False, dilation=1, groups=group_num)
        #                             )

    def forward(self, volume, weight=None):
        cost0 = self.dres0(volume)
        cost0 = self.dres1(cost0) + cost0
        out1 = self.encoder_decoder1(cost0)  # [2, hg_size, 64, 128]
        out1 = torch.mul(out1, weight)
        cost = self.classif1(out1)
        if self.output_disp != self.maxdisp:
            cost = torch.unsqueeze(cost, 1)
            cost = F.interpolate(cost, [
                self.maxdisp, self.full_shape[0], self.full_shape[1]], mode='trilinear', align_corners=False)
            cost = torch.squeeze(cost, 1)
        else:
            cost = F.interpolate(
                cost, [self.full_shape[0], self.full_shape[1]], mode='bilinear')
        cost = F.softmax(cost, dim=1)
        pred = disparity_regression(cost, self.maxdisp)
        if self.KL_mode:
            KL = (out1, cost)
            return KL, pred
        else:
            return pred

  



class feature_extraction(nn.Module):
    def __init__(self, KL_mode=False):
        super(feature_extraction, self).__init__()
        self.KL_mode = KL_mode
        self.expanse_ratio = 3
        self.inplanes = 32

        self.firstconv = nn.Sequential(MobileV2_Residual(3, 32, 2,self.expanse_ratio),# 1/2
                                           nn.ReLU(inplace=True),
                                           MobileV2_Residual(32, 32, 1, self.expanse_ratio),
                                           nn.ReLU(inplace=True),
                                           )


        self.layer1 = self._make_layer(MobileV1_Residual, 16,  1, 1, 1, 1) # 1/2
        self.layer2 = self._make_layer(MobileV1_Residual, 80,  1, 2, 1, 1)# 1/4


    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [block(self.inplanes, planes, stride, downsample, pad, dilation)]
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        if self.KL_mode:
            return (x, l2)
        else:
            return (l2)



class naive_main(model_backbone):
    def __init__(self, maxdisp=192, inchannels=80, head_only=False, freeze_head=False, full_shape=None, KL_mode=False, plain_mode=False, output_disp=48):
        self.KL_mode = False if plain_mode else KL_mode
        super().__init__(maxdisp=maxdisp, inchannels=inchannels, head_only=head_only, freeze_head=freeze_head, KL_mode=self.KL_mode)
        self.feature_extraction = feature_extraction(KL_mode=self.KL_mode)
        self.Volume_construct = Volume_construct(volume_size=maxdisp//4, inchannels=inchannels)
        if not self.head_only:
            self.Regression = Regression(
                maxdisp=maxdisp, full_shape=full_shape,KL_mode=self.KL_mode, output_disp=output_disp)
    
    def get_Knowledge(self, features):
        if self.KL_mode:
            KL = features
            features = torch.cat(KL[1:], dim=1)
            return KL, features
        else:
            return features
