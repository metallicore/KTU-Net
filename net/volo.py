from typing import Tuple, Union
import torch
import torch.nn as nn
from monai.networks.blocks.unetr_block import UnetrUpBlocka
from monai.networks.nets import ViT
from net.FFB import Unet, DeUp_Cat, DeBlock
from net.cbam import BiFusion_block


class UNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
            self,
            in_channels: int = 4,
            out_channels: int = 4,
            img_size=(128, 128, 128),
            feature_size: int = 16,
            hidden_size: int = 512,
            mlp_dim: int = 3072,
            num_heads: int = 8,
            pos_embed: str = "conv",
            norm_name: Union[Tuple, str] = "batch",
            conv_block: bool = False,
            res_block: bool = True,
            dropout_rate: float = 0.0,
    ) -> None:

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
        self.decoder5 = UnetrUpBlocka(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.Unet = Unet(in_channels=4, base_channels=16, num_classes=4)

        self.bif = BiFusion_block(ch_1=128, ch_2=128, r_2=16, ch_int=128, ch_out=128)

        self.DeUp4 = DeUp_Cat(in_channels=128, out_channels=64)
        self.DeBlock4 = DeBlock(in_channels=64)

        self.DeUp3 = DeUp_Cat(in_channels=64, out_channels=32)
        self.DeBlock3 = DeBlock(in_channels=32)

        self.DeUp2 = DeUp_Cat(in_channels=32, out_channels=16)
        self.DeBlock2 = DeBlock(in_channels=16)

        self.endconv = nn.Conv3d(16, 4, kernel_size=1)

        self.Softmax = nn.Softmax(dim=1)

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
        x1_1, x2_1, x3_1, output = self.Unet(x_in)
        # print('trans', trans.shape)
        x1, hidden_states_out = self.vit(x_in)
        # print("...t", x.shape)
        dec4 = self.proj_feat(x1, self.hidden_size, self.feat_size)
        # print("dec4", dec4.size())

        dec3 = self.decoder5(dec4)

        fuse = self.bif(output, dec3)

        # print("dec3", dec3.size())
        y4 = self.DeUp4(fuse, x3_1)  # (1, 64, 32, 32, 32)
        y4 = self.DeBlock4(y4)

        y3 = self.DeUp3(y4, x2_1)  # (1, 32, 64, 64, 64)
        y3 = self.DeBlock3(y3)

        y2 = self.DeUp2(y3, x1_1)  # (1, 16, 128, 128, 128)
        y2 = self.DeBlock2(y2)

        y = self.endconv(y2)
        y = self.Softmax(y)

        return y


if __name__ == '__main__':
    x = torch.randn(1, 4, 128, 128, 128)
    net = UNETR()
    y = net(x)
    print("params: ", sum(p.numel() for p in net.parameters()))
    print(y.shape)