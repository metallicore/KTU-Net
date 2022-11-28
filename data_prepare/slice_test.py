import numpy as np
import SimpleITK as sitk
from glob import glob
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import os
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax

# def resize_image_itk(ct_image, newSize, resamplemethod):
#     resampler = sitk.ResampleImageFilter()
#     originSize = ct_image.GetSize()  # 原来的体素块尺寸
#     print(originSize)
#     originSpacing = ct_image.GetSpacing()
#     newSize = np.array(newSize, float)
#     factor = originSize / newSize
#     newSpacing = originSpacing * factor
#     newSize = newSize.astype(np.int)  # spacing肯定不能是整数
#     resampler.SetReferenceImage(ct_image)  # 需要重新采样的目标图像
#     resampler.SetSize(newSize.tolist())
#     resampler.SetOutputSpacing(newSpacing.tolist())
#     resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
#     resampler.SetInterpolator(resamplemethod)
#     itkimgResampled = resampler.Execute(ct_image)  # 得到重新采样后的图像
#     return itkimgResampled


# im_path = '../data/train_image/volume-0.npy'
# im = np.load(im_path)
# print(im.shape)
# img_path = '../LITS2017/CT/volume-27.nii'
# itkimage = sitk.ReadImage(img_path)
# print('1', itkimage.GetSize())
# itkimgResampled = resize_image_itk(itkimage, (160, 128, ), resamplemethod=sitk.sitkLinear) #这里要注意：mask用最近邻插值，CT图像用线性插值
# sitk.WriteImage(itkimgResampled, '../data/save_train_new/' + 'volume-0.nii')

# img_paths = glob('../data/train_image/*')
# a = list(map(lambda x: x.replace('volume', 'segmentation').replace('image', 'mask'), img_paths))
# print(a)

# a = [1]
# b = [2]
# c = [3]
#
# x = np.array([[a, b], [a, c], [b, c], [a, a]])
# print(x.shape)
# x = x[:, :, :, np.newaxis]
# print(x.shape)

# value = np.asarray([0]*2, dtype='float64')
# print(value)

# a = [4, 16, 16, 32, 32]
# a = torch.Tensor(a)
# print(a.shape)
# b = [4, 1, 16, 32, 32]
# b = torch.Tensor(b)
# print(b.shape)
# c = a+b
# print(c)
# print(c.shape)

# t4d = torch.empty(4, 256, 2, 4, 4)
# out = torch.empty(4, 256, 1, 2, 2)
# out = F.relu(F.interpolate(out,scale_factor=(2,2,2),mode ='trilinear', align_corners=False))
# print(out.size())
# out1 = F.pad(out, [0, 0, 0, 0, 0, 0])
# print(out1.size())
# # t = torch.empty(4, 256, 4, 8, 8)
# out2 = torch.add(out1, t4d)
# print(out2.size())

# x = torch.empty(4, 16, 8, 8, 8)
# encoder2 = nn.Conv3d(16, 32, 3, stride=1, padding=1)
# y = encoder2(x)
# print(y.size())
# z = F.max_pool3d(y, 2, 2)
# print(z.size())
# y = torch.empty(4, 16, 32)
# z = torch.empty(4, 16, 32)

# first = torch.add(x, y)
# second = torch.add(first, z)
#
# print(first.size())
# print(second.size())

# a = 5
# b = 6
# print(x.size())
# print(x.size()[:-1])
# new_x = x.size()[:-1] + (a, b)
# x = x.view(*new_x)
# print(x.size())
# t = x.permute(0, 2, 1, 3)
# print("t", t.size())
# t1 = t.transpose(-1, -2)
# print(t1.size())
#
# res = torch.matmul(t, t1)
# print("res", res.size())
# res = res / math.sqrt(6)
# res = nn.Softmax(dim=-1)(res)
# print("later res", res.size())
# res1 = torch.matmul(res, t)
# print("res1", res1.size())


# layers = [4, 4, 8, 2]
#
# print(len(layers))
# img_path = '../data/train_image/'
# saveimg_path = '../data/train_img/'
# img_names = os.listdir(img_path)
#
# for img_name in img_names:
#     print(img_name)
#     img = nib.load(img_path + img_name).get_fdata()  # 载入
#     img = np.array(img)
#     np.save(saveimg_path + str(img_name).split('.')[0] + '.npy', img)  # 保存

# width = 32
# features = [width * 2 ** i for i in range(4)]
# print(features)

# print(5**2)

# img_path = '../data/train_img/'
# image_file = glob(img_path + '*.npy')
# for i in range(len(image_file)):
#     img = np.load(image_file[i])
#     print(img.shape)

"""np.zeros"""


# x = torch.empty(64, 128, 128)
# print((np.zeros((1,) + x.shape).astype(np.float32)).shape)

# for i in range(0, 5):
#     print(i)

# ct_array_nor = torch.empty(200, 300, 300)
# seg_liver = np.zeros((1,) + ct_array_nor.shape).astype(np.float32)
# cnt_liver = np.zeros(ct_array_nor.shape).astype(np.float32)
# patch_size = [64, 128, 128]
# locations = [32, 64, 80]
# patch_stride = [16, 32, 32]
# image_shape = [128, 256, 256]
# for z in range(0, locations[0]):
#     zs = min(patch_stride[0] * z, image_shape[0] - patch_size[0])
#     print("我是zs", zs)
#     for x in range(0, locations[1]):
#         xs = min(patch_stride[1] * x, image_shape[1] - patch_size[1])
#         print("我是xs", xs)
#         for y in range(0, locations[2]):
#             ys = min(patch_stride[2] * y, image_shape[2] - patch_size[2])
#             print("我是ys", ys)
#
#             patch = ct_array_nor[zs:zs + patch_size[0], xs:xs + patch_size[1], ys:ys + patch_size[2]]
#             print("分割大小", patch.shape)
#             patch = np.expand_dims(np.expand_dims(patch, axis=0), axis=0).astype(np.float32)
#             print("升维", patch.shape)
#             seg_liver[:, zs:zs + patch_size[0], xs:xs + patch_size[1], ys:ys + patch_size[2]] \
#                 = seg_liver[:, zs:zs + patch_size[0], xs:xs + patch_size[1], ys:ys + patch_size[2]]
#             cnt_liver[zs:zs + patch_size[0], xs:xs + patch_size[1], ys:ys + patch_size[2]] \
#                 = cnt_liver[zs:zs + patch_size[0], xs:xs + patch_size[1], ys:ys + patch_size[2]] + 1
#             print(seg_liver.shape)
#     seg_liver = seg_liver / np.expand_dims(cnt_liver, axis=0)
#     print(seg_liver.shape)

# a = [163, 512, 512]
# spa = [0.75, 0.75, 3]
# zxy = [1.5, 0.8, 0.8]
# factor = spa / zxy
# print(factor)

# x = torch.randn(1, 4, 128, 128, 128)
# print(x.size())
# x = F.avg_pool3d(x, x.size()[2:])
# print(x.size())


# content = torch.load('../models/check/ckpt_best_10.pth')
# print(content.keys())   # keys()
# # 之后有其他需求比如要看 key 为 model 的内容有啥
# print(content['optimizer'])

# print(math.log(131))

# w = torch.empty(2, 3)
# v = torch.empty(2, 3)
# torch.nn.init.kaiming_uniform_(w)
# torch.nn.init.kaiming_normal_(v)
# print(w)
# print(v)
# F.elu()
# F.rrelu()
# F.selu()
# F.prelu()

# n_labels = 2
#
#
# unary = np.zeros((2, 3))
# seg = [[1., 0., 1.], [0., 1., 0.]]
# print(unary)
#
# unary[seg == 0] = 0.1
# unary[seg == 1] = 0.9
# U = np.stack((1 - unary, unary), axis=0)
# print(U)

# a = tuple([0.7, 0.7, 0.6])

# class ASPP_module(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ASPP_module, self).__init__()
#         # 定义空洞率，根据图示空洞率为1 6 12 18 ，说明：当空洞率为1时为普通卷积
#         dilations = [1, 3, 5]
#
#         self.Aspp1 = nn.Sequential(
#             nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1, 1), padding=0,
#                       dilation=dilations[0], bias=False),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU())
#
#         self.Aspp2 = nn.Sequential(
#             nn.Sequential(
#                 nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3),
#                           # padding与dilation相同原因：当原始卷积核为3x3时，使输入输出尺寸大小相同，计算见3中说明。
#                           padding=dilations[0], dilation=dilations[0], bias=False),
#                 nn.BatchNorm3d(out_channels),
#                 nn.ReLU()
#             ),
#             nn.Sequential(
#                 nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1, 1), padding=0,
#                           dilation=dilations[0], bias=False),
#                 nn.BatchNorm3d(out_channels),
#                 nn.ReLU()
#             )
#
#         )
#         self.Aspp3 = nn.Sequential(
#             nn.Sequential(
#                 nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3),
#                           # padding与dilation相同原因：当原始卷积核为3x3时，使输入输出尺寸大小相同，计算见3中说明。
#                           padding=dilations[0], dilation=dilations[0], bias=False),
#                 nn.BatchNorm3d(out_channels),
#                 nn.ReLU()
#             ),
#             nn.Sequential(
#                 nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3),
#                           # padding与dilation相同原因：当原始卷积核为3x3时，使输入输出尺寸大小相同，计算见3中说明。
#                           padding=dilations[1], dilation=dilations[1], bias=False),
#                 nn.BatchNorm3d(out_channels),
#                 nn.ReLU()
#             ),
#             nn.Sequential(
#                 nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1, 1), padding=0,
#                           dilation=dilations[0], bias=False),
#                 nn.BatchNorm3d(out_channels),
#                 nn.ReLU()
#             )
#         )
#
#         self.Aspp4 = nn.Sequential(
#             nn.Sequential(
#                 nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3),
#                           # padding与dilation相同原因：当原始卷积核为3x3时，使输入输出尺寸大小相同，计算见3中说明。
#                           padding=dilations[0], dilation=dilations[0], bias=False),
#                 nn.BatchNorm3d(out_channels),
#                 nn.ReLU()
#             ),
#             nn.Sequential(
#                 nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3),
#                           # padding与dilation相同原因：当原始卷积核为3x3时，使输入输出尺寸大小相同，计算见3中说明。
#                           padding=dilations[1], dilation=dilations[1], bias=False),
#                 nn.BatchNorm3d(out_channels),
#                 nn.ReLU()
#             ),
#             nn.Sequential(
#                 nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3),
#                           # padding与dilation相同原因：当原始卷积核为3x3时，使输入输出尺寸大小相同，计算见3中说明。
#                           padding=dilations[2], dilation=dilations[2], bias=False),
#                 nn.BatchNorm3d(out_channels),
#                 nn.ReLU()
#             ),
#             nn.Sequential(
#                 nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1, 1), padding=0,
#                           dilation=dilations[0], bias=False),
#                 nn.BatchNorm3d(out_channels),
#                 nn.ReLU()
#             )
#         )
#
#         # concat后通道为1280，用1x1卷积改变通道数
#         self.conv1 = nn.Conv3d(4*in_channels, 16, kernel_size=(1, 1, 1), bias=False)
#         self.bn1 = nn.BatchNorm3d(16)
#         # 初始化卷积核权重
#         # self._init_weight()
#
#     def forward(self, x):
#         x1 = self.Aspp1(x)
#         print("X1.shape", x1.size())
#         x2 = self.Aspp2(x)
#         print("X2.shape", x2.size())
#         x3 = self.Aspp3(x)
#         print("X3.shape", x3.size())
#         x4 = self.Aspp4(x)
#         print("X4.shape", x4.size())
#         # 利用双线性插值恢复x5的尺寸，再进行concat
#         # cat = torch.cat((x1, x2, x3), dim=1)
#         cat = torch.cat((x1, x2, x3, x4), dim=1)
#         print('cat.shape', cat.size())
#
#         # 此处的output，包含后面1x1卷积进行的通道数调整
#         output = self.conv1(cat)
#         output = self.bn1(output)
#         print('output.shape', output.size())
#         return output
#
#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
#                 print('qqq', n)
#                 m.weight.data.normal_(0, math.sqrt(2. / n))  # 初始化卷积核方式
#                 print('www', m)
#             elif isinstance(m, nn.BatchNorm3d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#
# if __name__ == '__main__':
#     # 测试输出尺寸使用
#     aspp = ASPP_module(16, 16)
#     input = torch.randn(1, 16, 128, 128, 128);
#     print('input_size:', input.size())
#     out = aspp(input)


angle = round(np.random.uniform(-10, 10), 2)
print(angle)
