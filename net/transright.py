import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(SingleDeconv3DBlock, self).__init__()
        self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)


class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(SingleConv3DBlock, self).__init__()
        self.block = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2))

    def forward(self, x):
        return self.block(x)


class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(Conv3DBlock, self).__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Deconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Deconv3DBlock, self).__init__()
        self.block = nn.Sequential(
            SingleDeconv3DBlock(in_planes, out_planes),
            SingleConv3DBlock(out_planes, out_planes, 3),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Attention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout):
        """
        hidden_size即embed_dim
        假设 hidden_size = 128, num_attention_heads = 8, dropout_prob = 0.2
        即隐层维度为128，注意力头设置为8个
        all_head_size = 128 即等于hidden_size, 一般自注意力输入输出前后维度不变
        """
        super(Attention, self).__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(embed_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(embed_dim, self.all_head_size)
        self.key = nn.Linear(embed_dim, self.all_head_size)
        self.value = nn.Linear(embed_dim, self.all_head_size)
        # 后期会将所有头的注意力拼接在一起然后乘上权重矩阵输出
        # out是为了后期准备的
        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

        self.vis = False

    """主要是把维度大小为 [batch_size * seq_length * embed_dim] 的 q,k,v 向量
       变换成 [batch_size * num_attention_heads * seq_length * attention_head_size]，便于后面做 Multi-Head Attention
       seq_length:字的个数
       把embed_dim分解为self.num_attention_heads * self.attention_head_size，
       然后再交换 seq_length 维度 和 num_attention_heads 维度
       因为attention是要对query中的每个字和key中的每个字做点积，即是在 seq_length 维度上
       query和key的点积是[seq_length * attention_head_size] * [attention_head_size * seq_length]=[seq_length * seq_length]
       """

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        """Take the dot product between "query" and "key" to get the raw attention scores.
           shape of attention_scores: batch_size * num_attention_heads * seq_length * seq_length"""
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # 做 Scaled，将方差统一到1，避免维度的影响
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        # shape of first context_layer: batch_size * num_attention_heads * seq_length * attention_head_size
        # shape of second context_layer: batch_size * seq_length * num_attention_heads * attention_head_size
        # context_layer 维度恢复到：batch_size * seq_length * hidden_size
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


# class Mlp(nn.Module):
#     def __init__(self, in_features=512, hidden_features=2048, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, in_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    # 之前用的768
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # Torch linears have a `b` by default.
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # return self.w_2(self.dropout(F.relu(self.w_1(x))))
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class Embeddings(nn.Module):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size, dropout):
        super(Embeddings, self).__init__()
        self.n_patches = int((cube_size[0] * cube_size[1] * cube_size[2]) / (patch_size * patch_size * patch_size))
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embeddings = nn.Conv3d(in_channels=input_dim, out_channels=embed_dim,
                                          kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embeddings(x)
        # print("ceshi", x.size())
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        # print("ceshi11", x.size())
        # print("ceshuu", self.position_embeddings.size())
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, cube_size, patch_size):
        super(Block, self).__init__()
        self.attention_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_dim = int((cube_size[0] * cube_size[1] * cube_size[2]) / (patch_size * patch_size * patch_size))
        self.mlp = PositionwiseFeedForward(embed_dim, 2048)
        # self.mlp = Mlp()
        self.attn = Attention(num_heads, embed_dim, dropout)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        h = x

        x = self.mlp_norm(x)
        x = self.mlp(x)

        x = x + h
        return x, weights


class Transformer(nn.Module):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size, num_heads, num_layers, dropout, extract_layers):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(input_dim, embed_dim, cube_size, patch_size, dropout)
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.extract_layers = extract_layers
        for _ in range(num_layers):
            layer = Block(embed_dim, num_heads, dropout, cube_size, patch_size)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        extract_layers = []
        hidden_states = self.embeddings(x)

        for depth, layer_block in enumerate(self.layer):
            hidden_states, _ = layer_block(hidden_states)
            if depth + 1 in self.extract_layers:
                extract_layers.append(hidden_states)

        return extract_layers


class UNeTR(nn.Module):
    def __init__(self):
        super(UNeTR, self).__init__()
        self.input_dim = 1
        self.output_dim = 2
        self.embed_dim = 512
        self.cube_size = [32, 32, 32]
        self.patch_size = 16
        self.num_heads = 4
        self.num_trans_layers = 12
        self.dropout = 0.1
        self.extract_layers = [3, 6, 9, 12]
        self.num_extract_layers = len(self.extract_layers)
        self.patch_dim = [int(cube / self.patch_size) for cube in self.cube_size]
        # Transformer Encoder
        self.transformer = Transformer(self.input_dim, self.embed_dim, self.cube_size, self.patch_size,
                                       self.num_heads, self.num_trans_layers, self.dropout, self.extract_layers)

        # Unet Decoder
        self.decoder0 = nn.Sequential(
            Conv3DBlock(self.input_dim, 16, 3),
            Conv3DBlock(16, 32, 3)
        )
        self.decoder3 = nn.Sequential(
            Deconv3DBlock(self.embed_dim, 256),
            Deconv3DBlock(256, 128),
            Deconv3DBlock(128, 64)
        )
        self.decoder6 = nn.Sequential(
            Deconv3DBlock(self.embed_dim, 256),
            Deconv3DBlock(256, 128),
        )
        self.decoder9 = nn.Sequential(
            Deconv3DBlock(self.embed_dim, 256),
        )
        self.decoder12_upsampler = SingleDeconv3DBlock(self.embed_dim, 256)
        self.decoder9_upsampler = nn.Sequential(
            Conv3DBlock(512, 256, 3),
            Conv3DBlock(256, 256, 3),
            Conv3DBlock(256, 256, 3),
            SingleDeconv3DBlock(256, 128)
        )
        self.decoder6_upsampler = nn.Sequential(
            Conv3DBlock(256, 128, 3),
            Conv3DBlock(128, 128, 3),
            SingleDeconv3DBlock(128, 64)
        )
        self.decoder3_upsampler = nn.Sequential(
            Conv3DBlock(128, 64, 3),
            Conv3DBlock(64, 64, 3),
            SingleDeconv3DBlock(64, 32)
        )
        self.decoder0_header = nn.Sequential(
            Conv3DBlock(64, 32, 3),
            Conv3DBlock(32, 32, 3),
            SingleConv3DBlock(32, self.output_dim, 1)
        )

    def forward(self, x):
        z = self.transformer(x)

        z0, z3, z6, z9, z12 = x, *z
        print("初始x", x.size())
        print("初始z0", z0.size())
        print("初始z3", z3.size())
        print("初始z6", z6.size())
        print("初始z9", z9.size())
        print("初始z12", z12.size())
        # print("初始*z", *z)
        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)

        print("第二z3", z3.size())
        print("第二z6", z6.size())
        print("第二z9", z9.size())
        print("第二z12", z12.size())
        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z6 = self.decoder6(z6)
        z3 = self.decoder3(z3)
        z0 = self.decoder0(z0)
        # [4, 32, 32, 32, 32] [4, 64, 16, 16, 16] [4, 128, 8, 8, 8] [4, 256, 4, 4, 4] [4, 256, 4, 4, 4]
        # print(z0.size(), z3.size(), z6.size(), z9.size(), z12.size())
        z12 = self.decoder12_upsampler(z12)
        print(z12.size())
        z9 = self.decoder9(z9)
        print("91", z9.size())
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        print("92", z9.size())
        z6 = self.decoder6(z6)
        print("61", z6.size())
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        print("62", z6.size())
        z3 = self.decoder3(z3)
        print("31", z3.size())
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        print("32", z3.size())
        z0 = self.decoder0(z0)
        print("01", z0.size())
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))
        print("output", output.size())
        return output
