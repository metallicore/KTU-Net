from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.nets import ViT


class UNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
            self,
            in_channels=1,
            out_channels=2,
            img_size=(16, 16, 16),
            feature_size: int = 16,
            hidden_size: int = 512,
            mlp_dim: int = 3072,
            num_heads: int = 8,
            pos_embed: str = "perceptron",
            norm_name: Union[Tuple, str] = "instance",
            conv_block: bool = False,
            res_block: bool = True,
            dropout_rate: float = 0.1,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
        Examples::
            # for single channel input 4-channel output with patch size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')
            # for 4-channel input 3-channel output with patch size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = 12
        self.patch_size = (16, 16, 16)
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        self.hidden_size = hidden_size
        upsample_stride = 1
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )

        self.encoder22 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )

        self.encoder1 = nn.Conv3d(1, 16, 3, stride=1,
                                  padding=1)  # First Layer GrayScale Image , change to input channels to 3 in case of RGB
        self.en1_bn = nn.BatchNorm3d(16)
        self.encoder2 = nn.Conv3d(16, 32, 3, stride=1, padding=1)
        self.en2_bn = nn.BatchNorm3d(32)
        self.encoder3 = nn.Conv3d(32, 64, 3, stride=1, padding=1)
        self.en3_bn = nn.BatchNorm3d(64)

        self.decoder1 = nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.conv = nn.Conv3d(64, 32, kernel_size=1)
        self.de1_bn = nn.BatchNorm3d(32)
        self.decoder2 = nn.Conv3d(32, 16, 3, stride=1, padding=1)
        self.de2_bn = nn.BatchNorm3d(16)
        self.decoder3 = nn.Conv3d(16, 8, 3, stride=1, padding=1)
        self.de3_bn = nn.BatchNorm3d(8)

        self.decoderf1 = nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.def1_bn = nn.BatchNorm3d(32)
        self.decoderf2 = nn.Conv3d(32, 16, 3, stride=1, padding=1)
        self.def2_bn = nn.BatchNorm3d(16)
        self.decoderf3 = nn.Conv3d(16, 8, 3, stride=1, padding=1)
        self.def3_bn = nn.BatchNorm3d(8)

        self.encoderf1 = nn.Conv3d(1, 16, 3, stride=1,
                                   padding=1)  # First Layer GrayScale Image , change to input channels to 3 in case of RGB
        self.enf1_bn = nn.BatchNorm3d(16)
        self.encoderf2 = nn.Conv3d(16, 32, 3, stride=1, padding=1)
        self.enf2_bn = nn.BatchNorm3d(32)
        self.encoderf3 = nn.Conv3d(32, 64, 3, stride=1, padding=1)
        self.enf3_bn = nn.BatchNorm3d(64)

        self.intere1_1 = nn.Conv3d(16, 16, 3, stride=1, padding=1)
        self.inte1_1bn = nn.BatchNorm3d(16)
        self.intere2_1 = nn.Conv3d(32, 32, 3, stride=1, padding=1)
        self.inte2_1bn = nn.BatchNorm3d(32)
        self.intere3_1 = nn.Conv3d(64, 64, 3, stride=1, padding=1)
        self.inte3_1bn = nn.BatchNorm3d(64)

        self.intere1_2 = nn.Conv3d(16, 16, 3, stride=1, padding=1)
        self.inte1_2bn = nn.BatchNorm3d(16)
        self.intere2_2 = nn.Conv3d(32, 32, 3, stride=1, padding=1)
        self.inte2_2bn = nn.BatchNorm3d(32)
        self.intere3_2 = nn.Conv3d(64, 64, 3, stride=1, padding=1)
        self.inte3_2bn = nn.BatchNorm3d(64)

        self.interd1_1 = nn.Conv3d(32, 32, 3, stride=1, padding=1)
        self.intd1_1bn = nn.BatchNorm3d(32)
        self.interd2_1 = nn.Conv3d(16, 16, 3, stride=1, padding=1)
        self.intd2_1bn = nn.BatchNorm3d(16)
        self.interd3_1 = nn.Conv3d(64, 64, 3, stride=1, padding=1)
        self.intd3_1bn = nn.BatchNorm3d(64)

        self.interd1_2 = nn.Conv3d(32, 32, 3, stride=1, padding=1)
        self.intd1_2bn = nn.BatchNorm3d(32)
        self.interd2_2 = nn.Conv3d(16, 16, 3, stride=1, padding=1)
        self.intd2_2bn = nn.BatchNorm3d(16)
        self.interd3_2 = nn.Conv3d(64, 64, 3, stride=1, padding=1)
        self.intd3_2bn = nn.BatchNorm3d(64)

        self.final = nn.Conv3d(8, 2, 1, stride=1, padding=0)
        self.fin = nn.Conv3d(2, 2, 1, stride=1, padding=0)

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(32, 2, 1, 1),
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear'),
            nn.Sigmoid()
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(16, 2, 1, 1),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear'),
            nn.Sigmoid()
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(8, 2, 1, 1),
            nn.Upsample(scale_factor=(2, 1, 1), mode='trilinear'),
            nn.Sigmoid()
        )

        self.soft = nn.Softmax(dim=1)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def load_from(self, weights):
        with torch.no_grad():
            res_weight = weights
            # copy weights from patch embedding
            for i in weights['state_dict']:
                print(i)
            self.vit.patch_embedding.position_embeddings.copy_(
                weights['state_dict']['module.transformer.patch_embedding.position_embeddings_3d'])
            self.vit.patch_embedding.cls_token.copy_(
                weights['state_dict']['module.transformer.patch_embedding.cls_token'])
            self.vit.patch_embedding.patch_embeddings[1].weight.copy_(
                weights['state_dict']['module.transformer.patch_embedding.patch_embeddings.1.weight'])
            self.vit.patch_embedding.patch_embeddings[1].bias.copy_(
                weights['state_dict']['module.transformer.patch_embedding.patch_embeddings.1.bias'])

            # copy weights from  encoding blocks (default: num of blocks: 12)
            for bname, block in self.vit.blocks.named_children():
                print(block)
                block.loadFrom(weights, n_block=bname)
            # last norm layer of transformer
            self.vit.norm.weight.copy_(weights['state_dict']['module.transformer.norm.weight'])
            self.vit.norm.bias.copy_(weights['state_dict']['module.transformer.norm.bias'])

    def forward(self, x_in):
        print(x_in.size())
        x, hidden_states_out = self.vit(x_in)
        # print("x", x.size())
        out = F.leaky_relu(self.en1_bn(F.max_pool3d(self.encoder1(x_in), 2, 2)))
        out1 = F.leaky_relu(self.enf1_bn(F.interpolate(self.encoderf1(x_in), scale_factor=(0.5, 1, 1),
                                                 mode='trilinear', align_corners=False)))  # Ki-Net branch
        tmp = out
        print("out1", out1.size())
        print("tmp", tmp.size())

        out = torch.add(out, F.interpolate(F.leaky_relu(self.inte1_1bn(self.intere1_1(out1))), scale_factor=(1, 0.5, 0.5)
                                           , mode='trilinear', align_corners=False))  # CRFB
        out1 = torch.add(out1, F.interpolate(F.leaky_relu(self.inte1_2bn(self.intere1_2(tmp))), scale_factor=(1, 2, 2)
                                             , mode='trilinear', align_corners=False))  # CRFB
        u1 = out
        o1 = out1

        out = F.leaky_relu(self.en2_bn(F.max_pool3d(self.encoder2(out), 2, 2)))
        out1 = F.leaky_relu(self.enf2_bn(F.interpolate(self.encoderf2(out1), scale_factor=(1, 1, 1)
                                                 , mode='trilinear', align_corners=False)))

        tmp = out
        out = torch.add(out, F.interpolate(F.leaky_relu(self.inte2_1bn(self.intere2_1(out1))), scale_factor=(0.5, 0.25, 0.25)
                                           , mode='trilinear', align_corners=False))
        out1 = torch.add(out1, F.interpolate(F.leaky_relu(self.inte2_2bn(self.intere2_2(tmp))), scale_factor=(2, 4, 4)
                                             , mode='trilinear', align_corners=False))
        u2 = out
        o2 = out1
        out = F.leaky_relu(self.en3_bn(F.max_pool3d(self.encoder3(out), 2, 2)))
        out1 = F.leaky_relu(self.enf3_bn(F.interpolate(self.encoderf3(out1), scale_factor=(2, 2, 2),
                                                 mode='trilinear', align_corners=False)))

        tmp = out

        out = torch.add(out, F.interpolate(F.leaky_relu(self.inte3_1bn(self.intere3_1(out1))),
                                           scale_factor=(0.125, 0.0625, 0.0625), mode='trilinear', align_corners=False))
        out1 = torch.add(out1, F.interpolate(F.leaky_relu(self.inte3_2bn(self.intere3_2(tmp))),
                                             scale_factor=(8, 16, 16), mode='trilinear', align_corners=False))

        # End of encoder block

        # Start Decoder
        x1 = hidden_states_out[9]
        enc = self.encoder22(self.proj_feat(x1, self.hidden_size, self.feat_size))
        out = F.interpolate(self.decoder1(out), scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)  # U-NET
        tmp = torch.multiply(out, enc)
        temp = F.leaky_relu(self.de1_bn(self.conv(tmp)))
        out1 = F.leaky_relu(self.def1_bn(F.max_pool3d(self.decoderf1(out1), 2, 2)))  # Ki-NET

        out = torch.add(temp, F.interpolate(F.leaky_relu(self.intd1_1bn(self.interd1_1(out1))),
                                            scale_factor=(0.5, 0.25, 0.25), mode='trilinear', align_corners=False))
        out1 = torch.add(out1, F.interpolate(F.leaky_relu(self.intd1_2bn(self.interd1_2(temp))),
                                             scale_factor=(2, 4, 4), mode='trilinear', align_corners=False))

        output1 = self.map3(out)

        out = torch.add(out, u2)  # skip conn
        out1 = torch.add(out1, o2)  # skip conn
        out = F.leaky_relu(self.de2_bn(F.interpolate(self.decoder2(out), scale_factor=(2, 2, 2),
                                               mode='trilinear', align_corners=False)))
        out1 = F.leaky_relu(self.def2_bn(F.max_pool3d(self.decoderf2(out1), 1, 1)))

        tmp = out
        out = torch.add(out, F.interpolate(F.leaky_relu(self.intd2_1bn(self.interd2_1(out1))), scale_factor=(1, 0.5, 0.5)
                                           , mode='trilinear', align_corners=False))
        out1 = torch.add(out1, F.interpolate(F.leaky_relu(self.intd2_2bn(self.interd2_2(tmp))), scale_factor=(1, 2, 2)
                                             , mode='trilinear', align_corners=False))
        output2 = self.map2(out)

        out = torch.add(out, u1)
        out1 = torch.add(out1, o1)

        out = F.leaky_relu(self.de3_bn(F.interpolate(self.decoder3(out), scale_factor=(1, 2, 2),
                                               mode='trilinear', align_corners=False)))
        out1 = F.leaky_relu(self.def3_bn(F.max_pool3d(self.decoderf3(out1), 1, 1)))

        output3 = self.map1(out)

        out = torch.add(out, out1)  # fusion of both branches
        out = F.leaky_relu(self.final(out))  # 1*1 conv

        output4 = F.interpolate(self.fin(out), scale_factor=(2, 1, 1), mode='trilinear', align_corners=False)

        if self.training is True:
            return output1, output2, output3, output4
        else:
            return output4

