from __future__ import print_function
# from torch import Tensor
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

# import MobileV2_Residual, convbn, interweave_tensors, disparity_regression
from models.submodule import MobileV2_Residual, convbn, interweave_tensors, disparity_regression,hourglass2D

group_num = 1

# def groupwise_correlation(fea1, fea2, num_groups):
#     B, C, H, W = fea1.shape
#     # assert C % num_groups == 0
#     channels_per_group = C // num_groups
#     cost = (fea1 * fea2).view([B, num_groups,
#                                channels_per_group, H, W]).mean(dim=2)
#     # assert cost.shape == (B, num_groups, H, W)
#     return cost


# def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
#     B, C, H, W = refimg_fea.shape
#     volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
#     for i in range(maxdisp):
#         if i > 0:
#             volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
#                                                            num_groups)
#         else:
#             volume[:, :, i, :, :] = groupwise_correlation(
#                 refimg_fea, targetimg_fea, num_groups)
#     volume = volume.contiguous()
#     return volume


# def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
#     B, C, H, W = refimg_fea.shape
#     volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
#     for i in range(maxdisp):
#         if i > 0:
#             volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
#                                                            num_groups)
#         else:
#             volume[:, :, i, :, :] = groupwise_correlation(
#                 refimg_fea, targetimg_fea, num_groups)
#     volume = volume.contiguous()

#     for h in range(H):
#         for w in range(W):
#             volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
#                                                            num_groups)
#         else:
#             volume[:, :, i, :, :] = groupwise_correlation(
#                 refimg_fea, targetimg_fea, num_groups)
#     volume = volume.contiguous()
#     return volume


# class hourglass2D(nn.Module):
#     def __init__(self, in_channels):
#         super(hourglass2D, self).__init__()

#         self.expanse_ratio = 2

#         self.conv1 = MobileV2_Residual(
#             in_channels, in_channels * 2, stride=2, expanse_ratio=self.expanse_ratio)

#         self.conv2 = MobileV2_Residual(
#             in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)

#         self.conv3 = MobileV2_Residual(
#             in_channels * 2, in_channels * 4, stride=2, expanse_ratio=self.expanse_ratio)

#         self.conv4 = MobileV2_Residual(
#             in_channels * 4, in_channels * 4, stride=1, expanse_ratio=self.expanse_ratio)

#         self.conv5 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 3,
#                                padding=1, output_padding=1, stride=2, bias=False),
#             nn.BatchNorm2d(in_channels * 2))

#         self.conv6 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels * 2, in_channels, 3,
#                                padding=1, output_padding=1, stride=2, bias=False),
#             nn.BatchNorm2d(in_channels))

#         self.redir1 = MobileV2_Residual(
#             in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
#         self.redir2 = MobileV2_Residual(
#             in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)

#     def forward(self, x):
#         conv1 = self.conv1(x)
#         conv2 = self.conv2(conv1)
#         conv3 = self.conv3(conv2)
#         conv4 = self.conv4(conv3)

#         conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
#         conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

#         return conv6






# class Volume_construct(nn.Module):
#     def __init__(self, maxdisp, inchannels):
#         super().__init__()
#         self.maxdisp = maxdisp
#         self.num_groups = 1
#         self.volume_size = maxdisp//4
#         self.dres_expanse_ratio = 3
#         self.train_detector_first = True
#         self.preconv11 = nn.Sequential(convbn(inchannels, 256, 1, 1, 0, 1),
#                                        nn.ReLU(inplace=True),
#                                        convbn(256, 128, 1, 1, 0, 1),
#                                        nn.ReLU(inplace=True),
#                                        convbn(128, 64, 1, 1, 0, 1),
#                                        )
#         self.preconv12 = nn.Sequential(
#                                        nn.ReLU(inplace=True),
#                                        convbn(64, 32, 1, 1, 0, 1),
#                                        nn.ReLU(inplace=True),
#                                        convbn(32, 16, 1, 1, 0, 1),
#                                        )

#         self.volume12 = nn.Sequential(convbn(64, 32, 1, 1, 0, 1),
#                                       nn.ReLU(inplace=True),
#                                       convbn(32, 16, 1, 1, 0, 1),
#                                       nn.ReLU(inplace=True),
#                                       convbn(16, 1, 1, 1, 0, 1),
#                                       nn.ReLU(inplace=True)
#                                       )

#     def forward(self, features_L, features_R):
#         featL = self.preconv11(features_L)
#         featR = self.preconv11(features_R)
#         featL2 = self.preconv12(featL)
#         featR2 = self.preconv12(featR)
#         B, C, H, W = featL.shape
#         volume = featL.new_zeros([B, self.num_groups, self.volume_size, H, W])

#         for i in range(self.volume_size):
#             if i > 0:
#                 x = groupwise_correlation(featL[:, :, :, i:], featR[:, :, :, :-i],
#                                           32)
#                 x = torch.cat(
#                     (x, interweave_tensors(featL2[:, :, :, i:], featR2[:, :, :, :-i])), dim=1)
#                 x = self.volume12(x)
#                 volume[:, :, i, :, i:] = x
#             else:
#                 x = groupwise_correlation(featL, featR, 32)
#                 x = torch.cat(
#                     (x, interweave_tensors(featL2, featR2)), dim=1)
#                 x = self.volume12(x)
#                 volume[:, :, 0, :, :] = x
#         volume = volume.contiguous()
#         volume = torch.squeeze(volume, 1)
#         return volume



class detector(nn.Module):
    def __init__(self, maxdisp, num_class=16):
        super(detector, self).__init__()
        self.maxdisp = maxdisp
        self.num_class = num_class
        self.layer1 = self._make_layer(
            Bottleneck, self.maxdisp//4, 24, 1, 1, 1)
        self.layer2 = self._make_layer(Bottleneck, 24, self.num_class, 1, 1, 1)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, groups=1):
        downsample = None
        block.expansion = 1
        self.inplanes = inplanes
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False, groups=groups),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                            stride, downsample, groups=groups))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
# ####################### insert ######################

# ####################### insert #####################

class Regression_bone(nn.Module):
    def __init__(self, maxdisp=192, full_shape=None, KL_mode=False, output_disp=48):
        super(Regression_bone, self).__init__()
        self.maxdisp = maxdisp
        self.KL_mode = KL_mode
        self.volume_size = self.maxdisp//4
        self.hg_size = self.maxdisp//4
        self.full_shape = full_shape
        self.dres_expanse_ratio = 3
        self.output_disp = output_disp if output_disp != maxdisp else maxdisp

        self.dres0 = nn.Sequential(MobileV2_Residual(self.volume_size, self.hg_size, 1, self.dres_expanse_ratio),
                                   nn.ReLU(inplace=True),
                                   MobileV2_Residual(
                                       self.hg_size, self.hg_size, 1, self.dres_expanse_ratio),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(MobileV2_Residual(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio),
                                   nn.ReLU(inplace=True),
                                   MobileV2_Residual(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio)
                                   )
        self.encoder_decoder1 = hourglass2D(self.hg_size)
        self.classif1 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1, groups=group_num),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(self.hg_size, self.output_disp, kernel_size=3, padding=1, stride=1,
                                                bias=False, dilation=1, groups=group_num)
                                    )

    def set_full_shape(self, full_shape):
        if self.full_shape[0] != full_shape[0] or self.full_shape[1] != full_shape[1]:
            self.full_shape = full_shape


class model_backbone(nn.Module):
    def __init__(self, maxdisp=192, inchannels=80, head_only=False, freeze_head=False, onnx_export=False, full_shape=None, KL_mode=False, num_class_ratio=3):
        super(model_backbone, self).__init__()
        self.maxdisp = maxdisp
        self.KL_mode = KL_mode
        self.num_class_ratio = num_class_ratio
        self.num_class = (maxdisp//4)//self.num_class_ratio
        self.detector = detector(maxdisp, self.num_class)
        self.head_only = head_only
        self.freeze_head = freeze_head

    def forward(self, L, R):
        if self.KL_mode:
            return self.forward_knwoledge(L, R)
        else:
            return self.forward_plain(L, R)

    def forward_plain(self, L, R):
        feature_L = self.feature_extraction(L)
        feature_R = self.feature_extraction(R)
        feature_L = self.get_Knowledge(feature_L)
        feature_R = self.get_Knowledge(feature_R)
        Volume = self.Volume_construct(feature_L, feature_R)
        head = self.detector(Volume)
        head_weight = upsampleweight(head, shape=Volume.shape)
        if self.head_only:
            return (None, head)
        disp = self.Regression(Volume, head_weight)
        if (self.freeze_head or self.training is False):
            return (disp, None)
        else:
            return (disp, head)

    def forward_knwoledge(self, L, R):
        # if self.training is False:
        self.Regression.set_full_shape((L.shape[2],L.shape[3]))
        Knowledge = []
        feature_L = self.feature_extraction(L)
        feature_R = self.feature_extraction(R)
        Knowledge_L, feature_L = self.get_Knowledge(feature_L)
        Knowledge_R, feature_R = self.get_Knowledge(feature_R)
        Volume = self.Volume_construct(feature_L, feature_R)
        head = self.detector(Volume)
        Knowledge.append((Knowledge_L, Knowledge_R))
        Knowledge.append(Volume)
        if self.freeze_head:
            Knowledge.append(None)
        else:
            Knowledge.append(head)
        head_weight = upsampleweight(head, shape=Volume.shape)
        if self.head_only:
            Knowledge.append(None)
            return Knowledge, (None, head)
        Knowledge_soft, disp = self.Regression(Volume, head_weight)
        Knowledge.append(Knowledge_soft)
        if (self.freeze_head or self.training is False):
            return Knowledge, (disp, None)
        else:
            return Knowledge, (disp, head)




def upsampleweight(head, shape):
    if head.shape[1] == shape[1]:
        head_weight = head
    else:
        head_weight = torch.cuda.FloatTensor(
            shape[0], shape[1], shape[2], shape[3]).zero_()
        for i in range(1, head.shape[1]):
            head_weight[:, 3*i-1:3*i+2, :, :] = head[:, i:i+1, :, :].repeat(1, 3, 1, 1)
        head_weight[:,  0:2, :, :] = head[:,  0:1, :, :].repeat(1, 2, 1, 1)
        head_weight[:,  -1:, :, :] = head[:,  -1:, :, :]
    head_weight = F.softmax(head_weight, dim=1)
    return head_weight


# class MSNet2D(nn.Module):
#     def __init__(self, maxdisp,full_shape=None):
#         super(MSNet2D, self).__init__()
#         self.maxdisp = maxdisp

#         self.num_groups = 1

#         self.volume_size = self.maxdisp//4

#         self.hg_size = self.maxdisp//4
#         self.full_shape = full_shape
#         self.dres_expanse_ratio = 3
#         # ! 输出的维度
#         # self.output_disp = self.maxdisp
#         self.output_disp = self.maxdisp//4

#         self.dres0 = nn.Sequential(MobileV2_Residual(self.volume_size, self.hg_size, 1, self.dres_expanse_ratio),
#                                    nn.ReLU(inplace=True),
#                                    MobileV2_Residual(
#                                        self.hg_size, self.hg_size, 1, self.dres_expanse_ratio),
#                                    nn.ReLU(inplace=True))

#         self.dres1 = nn.Sequential(MobileV2_Residual(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio),
#                                    nn.ReLU(inplace=True),
#                                    MobileV2_Residual(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio))

#         self.encoder_decoder1 = hourglass2D(self.hg_size)

#         self.encoder_decoder2 = hourglass2D(self.hg_size)

#         self.encoder_decoder3 = hourglass2D(self.hg_size)

#         self.classif0 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1, groups=group_num),
#                                       nn.ReLU(inplace=True),
#                                       nn.Conv2d(self.hg_size, self.output_disp, kernel_size=3, padding=1, stride=1,
#                                                 bias=False, dilation=1, groups=group_num)
#                                       )
#         self.classif1 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1, groups=group_num),
#                                       nn.ReLU(inplace=True),
#                                       nn.Conv2d(self.hg_size, self.output_disp, kernel_size=3, padding=1, stride=1,
#                                                 bias=False, dilation=1, groups=group_num))
#         self.classif2 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1, groups=group_num),
#                                       nn.ReLU(inplace=True),
#                                       nn.Conv2d(self.hg_size, self.output_disp, kernel_size=3, padding=1, stride=1,
#                                                 bias=False, dilation=1, groups=group_num))
#         self.classif3 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1, groups=group_num),
#                                       nn.ReLU(inplace=True),
#                                       nn.Conv2d(self.hg_size, self.output_disp, kernel_size=3, padding=1, stride=1,
#                                                 bias=False, dilation=1, groups=group_num))

#     def forward(self, volume, weight=None, disp_true=False,):
#         cost0 = self.dres0(volume)
#         cost0 = self.dres1(cost0) + cost0

#         out1 = self.encoder_decoder1(cost0)  # [2, hg_size, 64, 128]
#         out2 = self.encoder_decoder2(out1)
#         out3 = self.encoder_decoder3(out2)

#         if self.training:
#             cost0 = torch.mul(cost0, weight)
#             out1 = torch.mul(out1, weight)
#             out2 = torch.mul(out2, weight)
#             out3 = torch.mul(out3, weight)
#             cost0 = self.classif0(cost0)
#             cost1 = self.classif1(out1)
#             cost2 = self.classif2(out2)
#             cost3 = self.classif3(out3)

#             costs = [cost0, cost1, cost2, cost3]
#             outputs = []
#             for cost in costs:
#                 if cost.shape[1] != self.maxdisp:
#                     cost = torch.unsqueeze(cost, 1)
#                     cost = F.interpolate(cost, [
#                         self.maxdisp, self.full_shape[0], self.full_shape[1]], mode='trilinear', align_corners=False)
#                     cost = torch.squeeze(cost, 1)
#                 else:
#                     cost = F.interpolate(
#                         cost, [self.full_shape[0], self.full_shape[1]], mode='bilinear')

#                 pred = F.softmax(cost, dim=1)
#                 pred = disparity_regression(pred, self.maxdisp)
#                 outputs.append(pred)

#             return outputs

#         else:
#             out3 = torch.mul(out3, weight)
#             cost3 = self.classif3(out3)

#             if cost3.shape[1] != self.maxdisp:
#                 cost3 = torch.unsqueeze(cost3, 1)
#                 cost3 = F.interpolate(cost3, [
#                     self.maxdisp, self.full_shape[0], self.full_shape[1]], mode='trilinear', align_corners=False)
#                 cost3 = torch.squeeze(cost3, 1)
#             else:
#                 cost3 = F.interpolate(
#                         cost3, [self.full_shape[0], self.full_shape[1]], mode='bilinear')
            
#             pred3 = F.softmax(cost3, dim=1)
#             pred3 = disparity_regression(pred3, self.maxdisp)

#             return (pred3)

            

class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, bias=False, groups=groups)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * 1, kernel_size=1, bias=False, groups=groups)
        self.bn3 = nn.BatchNorm2d(planes * 1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

if __name__ == '__main__':
    from teacher.modules import teacher_main
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

