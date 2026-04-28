# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2
from nnunetv2.networks.dinov2.models.vision_transformer import DinoVisionTransformer, vit_base, vit_large, vit_giant2, vit_small
from nnunetv2.networks.dinov2.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
from nnunetv2.networks.dinov2.models import vision_transformer as vits
import nnunetv2.networks.dinov2.utils as dinov2_utils
from urllib.parse import urlparse
import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
# from nnunetv2.MedSAM.segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
import argparse
import SimpleITK as sitk
from scipy import ndimage
logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

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

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        # print('in_channels', in_channels)
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        # print('hidden_states.size()', hidden_states.size())
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        # print('B', B)
        # print('n_patch', n_patch)
        # print('hidden', hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        # print('h', h)
        # print('w', w)
        x = hidden_states.permute(0, 2, 1)
        # print('x.shape', x.shape)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)  # Ensure the output channels match the first DecoderBlock
        # print('x.shape', x.shape)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None and i < self.config.n_skip:
                skip = features[i]
            else:
                skip = None
            # print('decoder_block', decoder_block)
            # print('x.shape', x.shape)
            x = decoder_block(x, skip=skip)
        # print('x.shape', x.shape)
        return x
def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    if urlparse(pretrained_weights).scheme:  # If it looks like an URL
        state_dict = torch.hub.load_state_dict_from_url(pretrained_weights, map_location="cpu")
    else:
        state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))
class UNet_Dinov2(nn.Module):
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(UNet_Dinov2, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        # DINOv2,Image Encoder
        # dinov2_kwargs = dict(img_size = 518,
        #                      patch_size = 14,
        #                      init_values = 1e-5,
        #                      ffn_layer = 'mlp',
        #                      block_chunks=0,
        #                      qkv_bias=True,
        #                      proj_bias=True,
        #                      ffn_bias=True,)
        # dinov2 = vits.__dict__["vit_large"](**dinov2_kwargs)
        # load_pretrained_weights(dinov2, '/home/litaozhao/nnUNet2/nnUNet/nnunetv2/networks/pretrained/dinov2_vitl14_pretrain.pth', "teacher")
        # dinov2.eval()
        # dinov2.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # self.encoder = DinoVisionTransformer()
        # self.encoder = vit_base()
        self.encoder = vit_large()
        # self.encoder = dinov2
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def get_pseudo_mask(self, tmp_supp_feat, query_feat_4, mask):
        tmp_supp_feat_4 = tmp_supp_feat * mask     # [1, 1024, 24, 24]
        q = query_feat_4                           # [1, 1024, 24, 24]
        s = tmp_supp_feat_4                        # [1, 1024, 24, 24]
        bsize, ch_sz, sp_sz, _ = q.size()[:]       # 1, 1024, 24, 24

        device = q.device

        tmp_query = q
        tmp_query = tmp_query.reshape(bsize, ch_sz, -1)    # [1, 1024, 576]
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

        tmp_supp = s
        tmp_supp = tmp_supp.reshape(bsize, ch_sz, -1)     # [1, 1024, 576]
        tmp_supp = tmp_supp.permute(0, 2, 1)             # [1, 576, 1024]
        # 确保tmp_supp和tmp_query都在同一个设备上
        tmp_supp = tmp_supp.to(device)
        tmp_query = tmp_query.to(device)
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)
        tmp_query_norm = tmp_query_norm.to(device)  # 确保tmp_query_norm也在同一

        cosine_eps = 1e-7
        similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)  # [1, 576, 576]
        # print('similarity.shape', similarity.shape)
        similarity = similarity.max(1)[0].reshape(bsize, sp_sz * sp_sz)  # [1, 576]
        # print('similarity.shape', similarity.shape)
        corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)   # [1, 1, 24, 24]
        # print('corr_query.shape', corr_query.shape)
        return corr_query

    def point_selection(self, mask_sim, topk=1):
        # Top-1 point selection
        w, h = mask_sim.shape
        topk_xy = mask_sim.flatten(0).topk(topk)[1]
        topk_x = (topk_xy // h).unsqueeze(0)
        topk_y = (topk_xy - topk_x * h)
        topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
        topk_label = np.array([1] * topk)
        topk_xy = topk_xy.cpu().numpy()

        # Top-last point selection
        last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
        last_x = (last_xy // h).unsqueeze(0)
        last_y = (last_xy - last_x * h)
        last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
        last_label = np.array([0] * topk)
        last_xy = last_xy.cpu().numpy()

        return topk_xy, topk_label, last_xy, last_label

    def forward(self, x):

        # x_temp encoder feature 保存为npy文件
        # skips = self.encoder.get_intermediate_layers(x, n=1, reshape=True)
        # x_feature = skips[-1]
        # x_feature = x_feature.cpu().numpy()
        # # np.save('/home/litaozhao/nnUNet2/nnUNet/nnunetv2/nnUNetFrame/DATASET/nnUNet_raw/Dataset605_private_Loc/template2/x_feature_all.npy', x_feature)
        # # np.save('/home/litaozhao/nnUNet2/nnUNet/nnunetv2/nnUNetFrame/DATASET/nnUNet_raw/Dataset602_PICAI_Loc/template2/x_feature_all.npy', x_feature)
        # np.save('/home/litaozhao/nnUNet2/nnUNet/nnunetv2/nnUNetFrame/DATASET/nnUNet_raw/Dataset602_PICAI_Loc/template3/x_feature_les.npy', x_feature)
        # # np.save('/home/litaozhao/nnUNet2/nnUNet/nnunetv2/nnUNetFrame/DATASET/nnUNet_raw/Dataset602_PICAI_Loc/template2/x_feature_bg.npy', x_feature)
        # x = x.float()
        # x_input = x[:,1,:,:].unsqueeze(1)
        # # print('x.shape', x.shape)   # [1,3,384,384]
        # if x.size()[1] == 1:
        #     x = x.repeat(1,3,1,1)
        """ x 提取encoder的特征 """
        skips = self.encoder.get_intermediate_layers(x, n=1, reshape=True)
        # for s in skips:
        #     print(f"UNet.forward.skips: {s.shape} {type(skips)}")
        # x = skips[-1].view(skips[-1].shape[0], 1024, -1).permute(0, 2, 1)
        x_feature = skips[-1]    # [1, 1024, 24, 24]
        #
        # """ x_temp 提取encoder的特征 """
        # x_temp_path = '/home/litaozhao/nnUNet2/nnUNet/nnunetv2/nnUNetFrame/DATASET/nnUNet_raw/Dataset605_private_Loc/template/x_feature_all.npy'
        x_temp_path = '/home/litaozhao/nnUNet2/nnUNet/nnunetv2/nnUNetFrame/DATASET/nnUNet_raw/Dataset602_PICAI_Loc/template3/x_feature_les.npy'
        x_temp_feature = np.load(x_temp_path)   # [1, 1024, 24, 24]
        x_temp_feature = torch.from_numpy(x_temp_feature)

        """temp mask 缩放至与x_temp同样大小"""
        # mask_path = '/home/litaozhao/nnUNet2/nnUNet/nnunetv2/nnUNetFrame/DATASET/nnUNet_raw/Dataset605_private_Loc/labelsTr/PCa_002.nii.gz'
        # mask_path = '/home/litaozhao/nnUNet2/nnUNet/nnunetv2/nnUNetFrame/DATASET/nnUNet_raw/Dataset602_PICAI_Loc/labelsTr/PCa_1000005.nii.gz'
        mask_path = '/home/litaozhao/nnUNet2/nnUNet/nnunetv2/nnUNetFrame/DATASET/nnUNet_raw/Dataset602_PICAI_Loc/labelsTr/PCa_1000019.nii.gz'
        mask_img = sitk.ReadImage(mask_path)
        mask_data = sitk.GetArrayFromImage(mask_img)[9, :, :]
        # mask_data = sitk.GetArrayFromImage(mask_img)
        mask_temp = torch.from_numpy(mask_data).unsqueeze(0).unsqueeze(0).float() # [1, 1, 154, 154]
        resize_size = x_feature.size(2)
        # mask_temp = F.interpolate(mask_temp, size=(resize_size, resize_size), mode='bilinear',
        #                          align_corners=True)  # [1, 1, 24, 24]
        mask_temp = F.interpolate(mask_temp, size=(resize_size, resize_size), mode='nearest')
        # print('mask_temp.shape', mask_temp.shape)
        """相似度图"""
        similarity_map = self.get_pseudo_mask(x_temp_feature, x_feature, mask_temp).float()  # [1, 1, 24, 24]

        """相似度图恢复到原始图像尺寸"""
        # resize_size_2 = x_input.size(2)
        similarity_map_resize = F.interpolate(similarity_map, size=(384, 384), mode='bilinear',
                                 align_corners=True)    # [1, 1, 384, 384]
        similarity_map_resize2 = 1 - similarity_map_resize
        similarity_map_resize_final = torch.stack((similarity_map_resize.squeeze(0), similarity_map_resize2.squeeze(0)), dim=1)
        # print('similarity_map_resize_final.shape', similarity_map_resize_final.shape)
        # similarity_map_resize = F.interpolate(similarity_map, size=(384, 384), mode='nearest')  # [1, 1, 384, 384]


        return similarity_map_resize_final



    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}