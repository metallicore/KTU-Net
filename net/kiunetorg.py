import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class kiunet_org(nn.Module):

    def __init__(self):
        super(kiunet_org, self).__init__()
        self.start = nn.Conv3d(1, 1, 3, stride=2, padding=1)

        self.encoder1 = nn.Conv3d(1, 16, 3, stride=1,
                                  padding=1)  # First Layer GrayScale Image , change to input channels to 3 in case of RGB
        self.en1_bn = nn.GroupNorm(2, 16)
        self.encoder2 = nn.Conv3d(16, 32, 3, stride=1, padding=1)
        self.en2_bn = nn.GroupNorm(2, 32)
        self.encoder3 = nn.Conv3d(32, 64, 3, stride=1, padding=1)
        self.en3_bn = nn.GroupNorm(2, 64)

        self.decoder1 = nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.conv = nn.Conv3d(64, 32, kernel_size=1)
        self.de1_bn = nn.GroupNorm(2, 32)
        self.decoder2 = nn.Conv3d(32, 16, 3, stride=1, padding=1)
        self.de2_bn = nn.GroupNorm(2, 16)
        self.decoder3 = nn.Conv3d(16, 8, 3, stride=1, padding=1)
        self.de3_bn = nn.GroupNorm(2, 8)

        self.decoderf1 = nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.def1_bn = nn.GroupNorm(2, 32)
        self.decoderf2 = nn.Conv3d(32, 16, 3, stride=1, padding=1)
        self.def2_bn = nn.GroupNorm(2, 16)
        self.decoderf3 = nn.Conv3d(16, 8, 3, stride=1, padding=1)
        self.def3_bn = nn.GroupNorm(2, 8)

        self.encoderf1 = nn.Conv3d(1, 16, 3, stride=1, padding=1)
        # First Layer GrayScale Image , change to input channels to 3 in case of RGB
        self.enf1_bn = nn.GroupNorm(2, 16)
        self.encoderf2 = nn.Conv3d(16, 32, 3, stride=1, padding=1)
        self.enf2_bn = nn.GroupNorm(2, 32)
        self.encoderf3 = nn.Conv3d(32, 64, 3, stride=1, padding=1)
        self.enf3_bn = nn.GroupNorm(2, 64)

        self.intere1_1 = nn.Conv3d(16, 16, 3, stride=1, padding=1)
        self.inte1_1bn = nn.GroupNorm(2, 16)
        self.intere2_1 = nn.Conv3d(32, 32, 3, stride=1, padding=1)
        self.inte2_1bn = nn.GroupNorm(2, 32)
        self.intere3_1 = nn.Conv3d(64, 64, 3, stride=1, padding=1)
        self.inte3_1bn = nn.GroupNorm(2, 64)

        self.intere1_2 = nn.Conv3d(16, 16, 3, stride=1, padding=1)
        self.inte1_2bn = nn.GroupNorm(2, 16)
        self.intere2_2 = nn.Conv3d(32, 32, 3, stride=1, padding=1)
        self.inte2_2bn = nn.GroupNorm(2, 32)
        self.intere3_2 = nn.Conv3d(64, 64, 3, stride=1, padding=1)
        self.inte3_2bn = nn.GroupNorm(2, 64)

        self.interd1_1 = nn.Conv3d(32, 32, 3, stride=1, padding=1)
        self.intd1_1bn = nn.GroupNorm(2, 32)
        self.interd2_1 = nn.Conv3d(16, 16, 3, stride=1, padding=1)
        self.intd2_1bn = nn.GroupNorm(2, 16)
        self.interd3_1 = nn.Conv3d(64, 64, 3, stride=1, padding=1)
        self.intd3_1bn = nn.GroupNorm(2, 64)

        self.interd1_2 = nn.Conv3d(32, 32, 3, stride=1, padding=1)
        self.intd1_2bn = nn.GroupNorm(2, 32)
        self.interd2_2 = nn.Conv3d(16, 16, 3, stride=1, padding=1)
        self.intd2_2bn = nn.GroupNorm(2, 16)
        self.interd3_2 = nn.Conv3d(64, 64, 3, stride=1, padding=1)
        self.intd3_2bn = nn.GroupNorm(2, 64)

        self.final = nn.Conv3d(8, 2, 1, stride=1, padding=0)
        self.fin = nn.Conv3d(2, 2, 1, stride=1, padding=0)

        self.map4 = nn.Sequential(
            nn.Conv3d(32, 2, 1, 1),
            nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear'),
            nn.Sigmoid()
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(16, 2, 1, 1),
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear'),
            nn.Sigmoid()
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(8, 2, 1, 1),
            nn.Upsample(scale_factor=(4, 2, 2), mode='trilinear'),
            nn.Sigmoid()
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, 2, 1, 1),
            nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear'),
            nn.Sigmoid()
        )

        self.soft = nn.Softmax(dim=1)
        self.apply(self.my_weight)

    def my_weight(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
            nn.init.kaiming_normal_(module.weight.data, 0.25)
            nn.init.constant_(module.bias.data, 0)
            print(1)

    def forward(self, x):
        # print("reset x", x.shape)  # [4, 1, 32, 32, 32]
        outx = self.start(x)  # [4, 1, 16, 16, 16]
        # print("start x", outx.shape)
        out = F.rrelu(self.en1_bn(F.max_pool3d(self.encoder1(outx), 2, 2)))  # U-Net branch

        out1 = F.rrelu(self.enf1_bn(F.interpolate(self.encoderf1(outx), scale_factor=(0.5, 1, 1),
                                     mode='trilinear')))  # Ki-Net branch

        tmp = out  # [4, 16, 8, 8, 8]
        # out out1 first shape torch.Size([4, 16, 8, 8, 8]) torch.Size([4, 16, 8, 16, 16])
        # print("out out1 first shape", out.shape, out1.shape)
        out = torch.add(out, F.interpolate(F.rrelu(self.inte1_1bn(self.intere1_1(out1))), scale_factor=(1, 0.5, 0.5)
                                           , mode='trilinear'))  # CRFB
        out1 = torch.add(out1, F.interpolate(F.rrelu(self.inte1_2bn(self.intere1_2(tmp))), scale_factor=(1, 2, 2)
                                             , mode='trilinear'))  # CRFB

        # out out1 second shape torch.Size([4, 16, 8, 8, 8]) torch.Size([4, 16, 8, 16, 16])
        # print("out out1 second shape", out.shape, out1.shape)

        u1 = out  # skip conn
        o1 = out1  # skip conn

        out = F.rrelu(self.en2_bn(F.max_pool3d(self.encoder2(out), 2, 2)))
        out1 = F.rrelu(self.enf2_bn(F.interpolate(self.encoderf2(out1), scale_factor=(1, 1, 1), mode='trilinear')))
        tmp = out

        # out out1 third shape torch.Size([4, 32, 4, 4, 4]) torch.Size([4, 32, 8, 16, 16])
        # print("out out1 third shape", out.shape, out1.shape)

        out = torch.add(out, F.interpolate(F.rrelu(self.inte2_1bn(self.intere2_1(out1))), scale_factor=(0.5, 0.25, 0.25)
                                           , mode='trilinear'))
        out1 = torch.add(out1, F.interpolate(F.rrelu(self.inte2_2bn(self.intere2_2(tmp))), scale_factor=(2, 4, 4)
                                             , mode='trilinear'))

        # out out1 forth shape torch.Size([4, 32, 4, 4, 4]) torch.Size([4, 32, 8, 16, 16])
        # print("out out1 forth shape", out.shape, out1.shape)

        u2 = out
        o2 = out1
        # out = F.pad(out, [0, 0, 0, 0, 0, 1])
        # print(out.shape)
        out = F.rrelu(self.en3_bn(F.max_pool3d(self.encoder3(out), 2, 2)))
        out1 = F.rrelu(self.enf3_bn(F.interpolate(self.encoderf3(out1), scale_factor=(2, 2, 2), mode='trilinear')))

        # out out1 fifth shape torch.Size([4, 64, 2, 2, 2]) torch.Size([4, 64, 16, 32, 32])
        # print("out out1 fifth shape", out.shape, out1.shape)

        tmp = out
        out = torch.add(out,
                        F.interpolate(F.rrelu(self.inte3_1bn(self.intere3_1(out1))), scale_factor=(0.125, 0.0625, 0.0625)
                                      , mode='trilinear'))
        out1 = torch.add(out1, F.interpolate(F.rrelu(self.inte3_2bn(self.intere3_2(tmp))), scale_factor=(8, 16, 16)
                                             , mode='trilinear'))

        # out out1 sixth shape torch.Size([4, 64, 2, 2, 2]) torch.Size([4, 64, 16, 32, 32])
        # print("out out1 sixth shape", out.shape, out1.shape)
        # End of encoder block

        # Start Decoder

        out = F.rrelu(self.de1_bn(F.interpolate(self.decoder1(out), scale_factor=(2, 2, 2), mode='trilinear')))  # U-NET
        out1 = F.rrelu(self.def1_bn(F.max_pool3d(self.decoderf1(out1), 2, 2)))  # Ki-NET
        tmp = out

        # out out1 one shape torch.Size([4, 32, 4, 4, 4]) torch.Size([4, 32, 8, 16, 16])
        # print("out out1 one shape", out.shape, out1.shape)

        out = torch.add(out, F.interpolate(F.rrelu(self.intd1_1bn(self.interd1_1(out1))), scale_factor=(0.5, 0.25, 0.25)
                                           , mode='trilinear'))
        out1 = torch.add(out1, F.interpolate(F.rrelu(self.intd1_2bn(self.interd1_2(tmp))), scale_factor=(2, 4, 4)
                                             , mode='trilinear'))

        # out out1 two shape torch.Size([4, 32, 4, 4, 4]) torch.Size([4, 32, 8, 16, 16])
        # print("out out1 two shape", out.shape, out1.shape)

        output1 = self.map4(out)

        # print("output1", output1.shape)

        out = torch.add(out, u2)  # skip conn
        out1 = torch.add(out1, o2)  # skip conn

        out = F.rrelu(self.de2_bn(F.interpolate(self.decoder2(out), scale_factor=(2, 2, 2), mode='trilinear')))
        out1 = F.rrelu(self.def2_bn(F.max_pool3d(self.decoderf2(out1), 1, 1)))

        # out out1 three shape torch.Size([4, 16, 8, 8, 8]) torch.Size([4, 16, 8, 16, 16])
        # print("out out1 three shape", out.shape, out1.shape)

        tmp = out
        out = torch.add(out, F.interpolate(F.rrelu(self.intd2_1bn(self.interd2_1(out1))), scale_factor=(1, 0.5, 0.5)
                                           , mode='trilinear'))
        out1 = torch.add(out1, F.interpolate(F.rrelu(self.intd2_2bn(self.interd2_2(tmp))), scale_factor=(1, 2, 2)
                                             , mode='trilinear'))
        output2 = self.map3(out)
        # print("output2", output2.shape)

        # out out1 four shape torch.Size([4, 16, 8, 8, 8]) torch.Size([4, 16, 8, 16, 16])
        # print("out out1 four shape", out.shape, out1.shape)

        out = torch.add(out, u1)
        out1 = torch.add(out1, o1)

        out = F.rrelu(self.de3_bn(F.interpolate(self.decoder3(out), scale_factor=(1, 2, 2), mode='trilinear')))
        out1 = F.rrelu(self.def3_bn(F.max_pool3d(self.decoderf3(out1), 1, 1)))

        # out out1 five shape torch.Size([4, 8, 8, 16, 16]) torch.Size([4, 8, 8, 16, 16])
        # print("out out1 five shape", out.shape, out1.shape)

        output3 = self.map2(out)
        # print("output3", output3.shape)

        out = torch.add(out, out1)  # fusion of both branches

        # print("out finish", out.shape)

        out = F.rrelu(self.final(out))  # 1*1 conv

        output4 = F.interpolate(self.fin(out), scale_factor=(4, 2, 2), mode='trilinear')
        # print(out.shape)
        # out = self.soft(out)
        print("output4", output4.shape)
        # if self.training is True:
        #     return output1, output2, output3, output4
        # else:
        return output4


