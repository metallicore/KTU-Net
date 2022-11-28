import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class kiunet_min(nn.Module):
    def __init__(self, training):
        super(kiunet_min, self).__init__()
        self.training = training
        self.encoder1 = nn.Conv3d(1, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2 = nn.Conv3d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3 = nn.Conv3d(64, 128, 3, stride=1, padding=1)
        self.encoder4 = nn.Conv3d(128, 256, 3, stride=1, padding=1)
        self.encoder5 = nn.Conv3d(256, 512, 3, stride=1, padding=1)

        self.kencoder1 = nn.Conv3d(1, 32, 3, stride=1, padding=1)
        self.kdecoder1 = nn.Conv3d(32, 2, 3, stride=1, padding=1)

        self.decoder1 = nn.Conv3d(512, 256, 3, stride=1, padding=1)  # b, 16, 5, 5
        self.decoder2 = nn.Conv3d(256, 128, 3, stride=1, padding=1)  # b, 8, 15, 1
        self.decoder3 = nn.Conv3d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 = nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv3d(32, 2, 3, stride=1, padding=1)

        self.map4 = nn.Sequential(
            nn.Conv3d(2, 2, 1, 1),
            nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear', align_corners=False),
            nn.Sigmoid()
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(64, 2, 1, 1),
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear', align_corners=False),
            nn.Sigmoid()
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(128, 2, 1, 1),
            nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear', align_corners=False),
            nn.Sigmoid()
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, 2, 1, 1),
            nn.Upsample(scale_factor=(16, 16, 16), mode='trilinear', align_corners=False),
            nn.Sigmoid()
        )

        self.soft = nn.Softmax(dim=1)

    def forward(self, x):

        out = F.relu(F.max_pool3d(self.encoder1(x), 2, 2))  # 4,32,16,16,16
        t1 = out
        # print("t1", t1.shape)
        out = F.relu(F.max_pool3d(self.encoder2(out), 2, 2))
        t2 = out
        out = F.relu(F.max_pool3d(self.encoder3(out), 2, 2))
        t3 = out
        out = F.relu(F.max_pool3d(self.encoder4(out), 2, 2))
        t4 = out
        out = F.relu(F.max_pool3d(self.encoder5(out), 2, 2))  # 4 512 1 1 1
        # print("out", out.shape)

        # t2 = out
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2, 2), mode='trilinear', align_corners=False))
        print("111", out.shape, t4.shape)
        # out = torch.add(F.pad(out, [0, 0, 0, 0, 0, 1]), t4)
        out = torch.add(out, t4)
        print("222", out.shape)
        output1 = self.map1(out)
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2, 2), mode='trilinear', align_corners=False))
        out = torch.add(out, t3)
        output2 = self.map2(out)
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2, 2), mode='trilinear', align_corners=False))
        out = torch.add(out, t2)
        output3 = self.map3(out)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2, 2), mode='trilinear', align_corners=False))
        out = torch.add(out, t1)
        print("first out", out.shape)
        out1 = F.relu(F.interpolate(self.kencoder1(x), scale_factor=(1, 2, 2), mode='trilinear', align_corners=False))
        print("out111", out1.shape)
        out1 = F.relu(
            F.interpolate(self.kdecoder1(out1), scale_factor=(1, 0.5, 0.5), mode='trilinear', align_corners=False))

        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2, 2), mode='trilinear', align_corners=False))
        print("second out out1", out.shape, out1.shape)
        out = torch.add(out, out1)
        output4 = self.map4(out)
        print("output4", output4.shape)
        # print(out.shape)

        # print(output1.shape,output2.shape,output3.shape,output4.shape)
        if self.training is True:
            return output1, output2, output3, output4
        else:
            return output4
