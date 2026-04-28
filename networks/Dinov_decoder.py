import numpy as np
import torch
from torch import nn
from typing import Union, List, Tuple, Type

from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp

class Dinov2Decoder(nn.Module):
    def __init__(self, encoder: 'DinoVisionTransformer', num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]], deep_supervision,
                 nonlin_first: bool = False, norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None, dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None, nonlin: Union[None, Type[nn.Module]] = None,
                 nonlin_kwargs: dict = None, conv_bias: bool = None):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)

        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        # Set default values for attributes if encoder does not provide them
        conv_bias = False if conv_bias is None else conv_bias
        norm_op = nn.BatchNorm2d if norm_op is None else norm_op
        dropout_op = nn.Dropout2d if dropout_op is None else dropout_op
        nonlin = nn.ReLU if nonlin is None else nonlin

        # Use the first norm_op_kwargs and dropout_op_kwargs if not provided
        norm_op_kwargs = {} if norm_op_kwargs is None else norm_op_kwargs
        dropout_op_kwargs = {} if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin_kwargs = {} if nonlin_kwargs is None else nonlin_kwargs

        transpconv_op = get_matching_convtransp(nn.Conv2d)

        # we start with the bottleneck and work our way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedConvBlocks(
                n_conv_per_stage[s-1], nn.Conv2d, 2 * input_features_skip, input_features_skip,
                3, 1, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
            ))

            # we always build the deep supervision outputs so that we can always load parameters
            seg_layers.append(nn.Conv2d(input_features_skip, num_classes, kernel_size=1, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            print('lres_input.shape', lres_input.shape)
            print('x.shape', x.shape)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            print('skips[-(s + 2)].shape', skips[-(s + 2)].shape)
            print('x.shape', x.shape)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]

        assert len(skip_sizes) == len(self.stages)

        output = np.int64(0)
        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output