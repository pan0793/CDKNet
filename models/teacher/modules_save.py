from __future__ import print_function
# from torch import Tensor
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from models.submodule import groupwise_correlation, build_gwc_volume, MobileV2_Residual, convbn, interweave_tensors, disparity_regression, hourglass2D, MobileV1_Residual
from models.model import model_backbone, Regression_bone


group_num = 1
class Volume_construct(nn.Module):
    def __init__(self, volume_size, inchannels):
        super().__init__()
        self.num_groups = 1
        self.volume_size = volume_size
        self.preconv11 = nn.Sequential(convbn(inchannels, 256, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(256, 128, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(128, 64, 1, 1, 0, 1),
                                       )
        self.preconv12 = nn.Sequential(
                                       nn.ReLU(inplace=True),
                                       convbn(64, 32, 1, 1, 0, 1),
                                    #    nn.ReLU(inplace=True),
                                    #    convbn(32, 16, 1, 1, 0, 1),
                                       )

        self.volume12 = nn.Sequential(convbn(64, 32, 1, 1, 0, 1),
                                      nn.ReLU(inplace=True),
                                      convbn(32, 16, 1, 1, 0, 1),
                                      nn.ReLU(inplace=True),
                                      convbn(16, 1, 1, 1, 0, 1),
                                      nn.ReLU(inplace=True)
                                      )

    def forward(self, features_L, features_R):
        featL = self.preconv11(features_L)
        featR = self.preconv11(features_R)
        featL2 = self.preconv12(featL)
        featR2 = self.preconv12(featR)
        B, C, H, W = featL.shape
        volume = featL.new_zeros([B, self.num_groups, self.volume_size, H, W])
        for i in range(self.volume_size):
            if i > 0:
                # x = groupwise_correlation(featL[:, :, :, i:], featR[:, :, :, :-i],
                #                           32)
                x = interweave_tensors(featL2[:, :, :, i:], featR2[:, :, :, :-i])
                x = self.volume12(x)
                volume[:, :, i, :, i:] = x
            else:
                # x = groupwise_correlation(featL, featR, 32)
                x = interweave_tensors(featL2, featR2)
                x = self.volume12(x)
                volume[:, :, 0, :, :] = x
        volume = volume.contiguous()
        volume = torch.squeeze(volume, 1)
        
        return volume

####################### insert ######################

####################### insert #####################

class Regression(Regression_bone):
    def __init__(self, maxdisp=192, full_shape=None, KL_mode=False, output_disp=48):
        super().__init__(maxdisp=maxdisp, full_shape=full_shape,
                         KL_mode=KL_mode, output_disp=output_disp)
        self.encoder_decoder2 = hourglass2D(self.hg_size)
        self.encoder_decoder3 = hourglass2D(self.hg_size)

        self.classif0 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1, groups=group_num),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(self.hg_size, self.output_disp, kernel_size=3, padding=1, stride=1,
                                                bias=False, dilation=1, groups=group_num)
                                      )
        # self.classif1 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1, groups=group_num),
        #                               nn.ReLU(inplace=True),
        #                               nn.Conv2d(self.hg_size, self.output_disp, kernel_size=3, padding=1, stride=1,
        #                                         bias=False, dilation=1, groups=group_num))
        self.classif2 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1, groups=group_num),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(self.hg_size, self.output_disp, kernel_size=3, padding=1, stride=1,
                                                bias=False, dilation=1, groups=group_num))
        self.classif3 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1, groups=group_num),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(self.hg_size, self.output_disp, kernel_size=3, padding=1, stride=1,
                                                bias=False, dilation=1, groups=group_num))

    def forward(self, volume, weight=None):
        # volume = torch.mul(volume, weight)
        cost0 = self.dres0(volume)
        cost0 = self.dres1(cost0) + cost0

        out1 = self.encoder_decoder1(cost0)  # [2, hg_size, 64, 128]
        out2 = self.encoder_decoder2(out1)
        out3 = self.encoder_decoder3(out2)

        if self.training:
            cost0 = torch.mul(cost0, weight)
            out1 = torch.mul(out1, weight)
            out2 = torch.mul(out2, weight)
            out3 = torch.mul(out3, weight)
            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)
            cost3 = self.classif3(out3)

            costs = [cost0, cost1, cost2, cost3]
            outputs = []
            num = 0
            for i, cost in enumerate(costs):
                if cost.shape[1] != self.maxdisp:
                    cost = torch.unsqueeze(cost, 1)
                    cost = F.interpolate(cost, [
                        self.maxdisp, self.full_shape[0], self.full_shape[1]], mode='trilinear', align_corners=False)
                    cost = torch.squeeze(cost, 1)
                else:
                    cost = F.interpolate(
                        cost, [self.full_shape[0], self.full_shape[1]], mode='bilinear')

                cost = F.softmax(cost, dim=1)
                if i == 3 and self.KL_mode:
                    KL = (out3, cost)
                pred = disparity_regression(cost, self.maxdisp)
                outputs.append(pred) 
                
            if self.KL_mode:
                return KL, outputs
            else:
                return outputs

        else:
            out3 = torch.mul(out3, weight)
            cost3 = self.classif3(out3)

            if cost3.shape[1] != self.maxdisp:
                cost3 = torch.unsqueeze(cost3, 1)
                cost3 = F.interpolate(cost3, [
                    self.maxdisp, self.full_shape[0], self.full_shape[1]], mode='trilinear', align_corners=False)
                cost3 = torch.squeeze(cost3, 1)
            else:
                cost3 = F.interpolate(
                        cost3, [self.full_shape[0], self.full_shape[1]], mode='bilinear')
            
            cost3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(cost3, self.maxdisp)
            if self.KL_mode:
                KL = (out3, cost3)
                return KL, pred3
            else:
                return pred3




class feature_extraction(nn.Module):
    def __init__(self, KL_mode=False):
        super(feature_extraction, self).__init__()
        self.KL_mode = KL_mode
        self.expanse_ratio = 3
        self.inplanes = 32
        self.firstconv = nn.Sequential(MobileV2_Residual(3, 32, 2, self.expanse_ratio),# 1/2
                                        nn.ReLU(inplace=True),
                                        MobileV2_Residual(32, 32, 1, self.expanse_ratio),
                                        nn.ReLU(inplace=True),
                                        MobileV2_Residual(32, 32, 1, self.expanse_ratio),
                                        nn.ReLU(inplace=True)
                                        )

        self.layer1 = self._make_layer(MobileV1_Residual, 32, 3, 1, 1, 1)# 1/2
        self.layer2 = self._make_layer(MobileV1_Residual, 64, 16, 2, 1, 1)# 1/4
        self.layer3 = self._make_layer(MobileV1_Residual, 128, 3, 1, 1, 1)# 1/4
        self.layer4 = self._make_layer(MobileV1_Residual, 128, 3, 1, 1, 2)# 1/4

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
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        if self.KL_mode:
            KL_teacher = (x, l2, l3, l4)
            return KL_teacher
        else:
            return (l2, l3, l4)


class teacher_main(model_backbone):
    def __init__(self, maxdisp=192, inchannels=320, head_only=False, freeze_head=False, full_shape=None, KL_mode=False, plain_mode=False, output_disp=48, num_class_ratio=3):
        self.KL_mode= False if plain_mode else KL_mode
        # maxdisp=192, inchannels=80, head_only=False, freeze_head=False, onnx_export=False, full_shape=None, KL_mode=False
        super().__init__(maxdisp=maxdisp, inchannels=inchannels, head_only=head_only, freeze_head=freeze_head, KL_mode=self.KL_mode, num_class_ratio=num_class_ratio)
        self.feature_extraction = feature_extraction(KL_mode=self.KL_mode)
        self.Volume_construct = Volume_construct(volume_size=maxdisp//4, inchannels=inchannels)
        if not self.head_only:
            self.Regression = Regression(
                maxdisp=maxdisp, full_shape=full_shape, KL_mode=self.KL_mode, output_disp=output_disp)
    def get_Knowledge(self, features):
        if self.KL_mode:
            KL = features
            features = torch.cat(KL[1:], dim=1)
            return KL, features
        else:
            return torch.cat(features, dim=1)


if __name__ == '__main__':

    model = teacher_main(192).cuda()
    model.eval()
    # torch.Tensor.new_ones(1, 3, 256, 256).zero_().cuda()
    input = torch.FloatTensor(1, 3, 480, 640).fill_(1.0).cuda()
    truth = torch.tensor(torch.randint(
        256, (1, 480, 640)), dtype=torch.long).cuda()
    # while(1):
    with torch.no_grad():
        from thop import clever_format
        from thop import profile

        flops, params = profile(model, inputs=(
            input, input), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        print(flops, params)

