import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet

import spconv
from spconv.modules import SparseModule

from torchplus.nn import Empty, GroupNorm, Sequential
from torchplus.tools import change_default_args

REGISTERED_RPN_CLASSES = {}

def register_rpn(cls, name=None):
    global REGISTERED_RPN_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_RPN_CLASSES, f"exist class: {REGISTERED_RPN_CLASSES}"
    REGISTERED_RPN_CLASSES[name] = cls
    return cls

def get_rpn_class(name):
    global REGISTERED_RPN_CLASSES
    assert name in REGISTERED_RPN_CLASSES, f"available class: {REGISTERED_RPN_CLASSES}"
    return REGISTERED_RPN_CLASSES[name]

@register_rpn
class RPN(nn.Module):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2),
                 num_filters=(128, 128, 256),
                 upsample_strides=(1, 2, 4),
                 num_upsample_filters=(256, 256, 256),
                 num_input_features=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 box_code_size=7,
                 num_direction_bins=2,
                 name='rpn'):
        """deprecated. exists for checkpoint backward compilability (SECOND v1.0)
        """
        super(RPN, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        assert len(layer_nums) == 3
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        upsample_strides = [
            np.round(u).astype(np.int64) for u in upsample_strides
        ]
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(
                layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(
                np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        block2_input_filters = num_filters[0]
        self.block1 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(
                num_input_features, num_filters[0], 3,
                stride=layer_strides[0]),
            BatchNorm2d(num_filters[0]),
            nn.ReLU(),
        )
        for i in range(layer_nums[0]):
            self.block1.add(
                Conv2d(num_filters[0], num_filters[0], 3, padding=1))
            self.block1.add(BatchNorm2d(num_filters[0]))
            self.block1.add(nn.ReLU())
        self.deconv1 = Sequential(
            ConvTranspose2d(
                num_filters[0],
                num_upsample_filters[0],
                upsample_strides[0],
                stride=upsample_strides[0]),
            BatchNorm2d(num_upsample_filters[0]),
            nn.ReLU(),
        )
        self.block2 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(
                block2_input_filters,
                num_filters[1],
                3,
                stride=layer_strides[1]),
            BatchNorm2d(num_filters[1]),
            nn.ReLU(),
        )
        for i in range(layer_nums[1]):
            self.block2.add(
                Conv2d(num_filters[1], num_filters[1], 3, padding=1))
            self.block2.add(BatchNorm2d(num_filters[1]))
            self.block2.add(nn.ReLU())
        self.deconv2 = Sequential(
            ConvTranspose2d(
                num_filters[1],
                num_upsample_filters[1],
                upsample_strides[1],
                stride=upsample_strides[1]),
            BatchNorm2d(num_upsample_filters[1]),
            nn.ReLU(),
        )
        self.block3 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(num_filters[1], num_filters[2], 3, stride=layer_strides[2]),
            BatchNorm2d(num_filters[2]),
            nn.ReLU(),
        )
        for i in range(layer_nums[2]):
            self.block3.add(
                Conv2d(num_filters[2], num_filters[2], 3, padding=1))
            self.block3.add(BatchNorm2d(num_filters[2]))
            self.block3.add(nn.ReLU())
        self.deconv3 = Sequential(
            ConvTranspose2d(
                num_filters[2],
                num_upsample_filters[2],
                upsample_strides[2],
                stride=upsample_strides[2]),
            BatchNorm2d(num_upsample_filters[2]),
            nn.ReLU(),
        )
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        self.conv_cls = nn.Conv2d(sum(num_upsample_filters), num_cls, 1)
        self.conv_box = nn.Conv2d(
            sum(num_upsample_filters), num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                sum(num_upsample_filters),
                num_anchor_per_loc * num_direction_bins, 1)

        if self._use_rc_net:
            self.conv_rc = nn.Conv2d(
                sum(num_upsample_filters), num_anchor_per_loc * box_code_size,
                1)

    def forward(self, x):
        # t = time.time()
        # torch.cuda.synchronize()

        x = self.block1(x)
        up1 = self.deconv1(x)
        x = self.block2(x)
        up2 = self.deconv2(x)
        x = self.block3(x)
        up3 = self.deconv3(x)
        x = torch.cat([up1, up2, up3], dim=1)
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)

        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds
        if self._use_rc_net:
            rc_preds = self.conv_rc(x)
            rc_preds = rc_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["rc_preds"] = rc_preds
        # torch.cuda.synchronize()
        # print("rpn forward time", time.time() - t)

        return ret_dict

class RPNNoHeadBase(nn.Module):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2),
                 num_filters=(128, 128, 256),
                 upsample_strides=(1, 2, 4),
                 num_upsample_filters=(256, 256, 256),
                 num_input_features=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 box_code_size=7,
                 num_direction_bins=2,
                 name='rpn'):
        """upsample_strides support float: [0.25, 0.5, 1]
        if upsample_strides < 1, conv2d will be used instead of convtranspose2d.
        """
        super(RPNNoHeadBase, self).__init__()
        self._layer_strides = layer_strides
        self._num_filters = num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = upsample_strides
        self._num_upsample_filters = num_upsample_filters
        self._num_input_features = num_input_features
        self._use_norm = use_norm
        self._use_groupnorm = use_groupnorm
        self._num_groups = num_groups
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(num_upsample_filters) == len(upsample_strides)
        self._upsample_start_idx = len(layer_nums) - len(upsample_strides)
        must_equal_list = []
        for i in range(len(upsample_strides)):
            must_equal_list.append(upsample_strides[i] / np.prod(
                layer_strides[:i + self._upsample_start_idx + 1]))
        for val in must_equal_list:
            assert val == must_equal_list[0]

        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        in_filters = [num_input_features, *num_filters[:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                num_filters[i],
                layer_num,
                stride=layer_strides[i])
            blocks.append(block)
            if i - self._upsample_start_idx >= 0:
                stride = upsample_strides[i - self._upsample_start_idx]
                if stride >= 1:
                    stride = np.round(stride).astype(np.int64)
                    deblock = nn.Sequential(
                        ConvTranspose2d(
                            num_out_filters,
                            num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride),
                        BatchNorm2d(
                            num_upsample_filters[i -
                                                 self._upsample_start_idx]),
                        nn.ReLU(),
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    deblock = nn.Sequential(
                        Conv2d(
                            num_out_filters,
                            num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride),
                        BatchNorm2d(
                            num_upsample_filters[i -
                                                 self._upsample_start_idx]),
                        nn.ReLU(),
                    )
                deblocks.append(deblock)
        self._num_out_filters = num_out_filters
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        raise NotImplementedError

    def forward(self, x):
        # blocks_total_time = 0
        # deblocks_total_time = 0

        ups = []
        stage_outputs = []
        for i in range(len(self.blocks)):
            # blocks_before = time.time()
            # print(f"Before block {i} shape:", x.shape)
            x = self.blocks[i](x)
            # blocks_after = time.time()
            # blocks_total_time += (blocks_after - blocks_before)
            # print(f"Block {i} shape:", x.shape)
            stage_outputs.append(x)
            if i - self._upsample_start_idx >= 0:
                # deblocks_before = time.time()
                res = self.deblocks[i - self._upsample_start_idx](x)
                # deblocks_after = time.time()
                # deblocks_total_time += (deblocks_after - deblocks_before)
                # print(f"Deblock {i} shape:", res.shape)
                ups.append(res)

        # print(">>>>>>Blocks total ms: %.2f" % (blocks_total_time * 1000),
        #       "Deblocks total ms: %.2f" % (deblocks_total_time * 1000))

        if len(ups) > 0:
            x = torch.cat(ups, dim=1)
        res = {}
        for i, up in enumerate(ups):
            res[f"up{i}"] = up
        for i, out in enumerate(stage_outputs):
            res[f"stage{i}"] = out
        res["out"] = x
        return res


class RPNBase(RPNNoHeadBase):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2),
                 num_filters=(128, 128, 256),
                 upsample_strides=(1, 2, 4),
                 num_upsample_filters=(256, 256, 256),
                 num_input_features=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 box_code_size=7,
                 num_direction_bins=2,
                 name='rpn'):
        """upsample_strides support float: [0.25, 0.5, 1]
        if upsample_strides < 1, conv2d will be used instead of convtranspose2d.
        """
        super(RPNBase, self).__init__(
            use_norm=use_norm,
            num_class=num_class,
            layer_nums=layer_nums,
            layer_strides=layer_strides,
            num_filters=num_filters,
            upsample_strides=upsample_strides,
            num_upsample_filters=num_upsample_filters,
            num_input_features=num_input_features,
            num_anchor_per_loc=num_anchor_per_loc,
            encode_background_as_zeros=encode_background_as_zeros,
            use_direction_classifier=use_direction_classifier,
            use_groupnorm=use_groupnorm,
            num_groups=num_groups,
            box_code_size=box_code_size,
            num_direction_bins=num_direction_bins,
            name=name)
        self._num_anchor_per_loc = num_anchor_per_loc
        self._num_direction_bins = num_direction_bins
        self._num_class = num_class
        self._use_direction_classifier = use_direction_classifier
        self._box_code_size = box_code_size

        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        if len(num_upsample_filters) == 0:
            final_num_filters = self._num_out_filters
        else:
            final_num_filters = sum(num_upsample_filters)
        self.conv_cls = nn.Conv2d(final_num_filters, num_cls, 1)
        self.conv_box = nn.Conv2d(final_num_filters,
                                  num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                final_num_filters, num_anchor_per_loc * num_direction_bins, 1)

    def forward(self, x):
        res = super().forward(x)
        x = res["out"]
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        C, H, W = box_preds.shape[1:]
        box_preds = box_preds.view(-1, self._num_anchor_per_loc,
                                   self._box_code_size, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()
        cls_preds = cls_preds.view(-1, self._num_anchor_per_loc,
                                   self._num_class, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()
        # box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        # cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()

        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.view(
                -1, self._num_anchor_per_loc, self._num_direction_bins, H,
                W).permute(0, 1, 3, 4, 2).contiguous()
            # dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds
        return ret_dict


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

@register_rpn
class ResNetRPN(RPNBase):
    def __init__(self, *args, **kw):
        self.inplanes = -1
        super(ResNetRPN, self).__init__(*args, **kw)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # if zero_init_residual:
        for m in self.modules():
            if isinstance(m, resnet.Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, resnet.BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        if self.inplanes == -1:
            self.inplanes = self._num_input_features
        block = resnet.BasicBlock
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers), self.inplanes

@register_rpn
class RPNV2(RPNBase):
    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        if self._use_norm:
            if self._use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=self._num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        block = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(inplanes, planes, 3, stride=stride),
            BatchNorm2d(planes),
            nn.ReLU(),
        )
        for j in range(num_blocks):
            block.add(Conv2d(planes, planes, 3, padding=1))
            block.add(BatchNorm2d(planes))
            block.add(nn.ReLU())

        return block, planes

@register_rpn
class RPNV2Conv2x2(RPNBase):
    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        if self._use_norm:
            if self._use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=self._num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        block = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(inplanes, planes, 3, stride=stride),
            BatchNorm2d(planes),
            nn.ReLU(),
        )
        for j in range(num_blocks):
            block.add(Conv2d(planes, planes, 3, padding=1))
            block.add(BatchNorm2d(planes))
            block.add(nn.ReLU())

        return block, planes

@register_rpn
class RPNNoHead(RPNNoHeadBase):
    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        if self._use_norm:
            if self._use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=self._num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        block = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(inplanes, planes, 3, stride=stride),
            BatchNorm2d(planes),
            nn.ReLU(),
        )
        for j in range(num_blocks):
            block.add(Conv2d(planes, planes, 3, padding=1))
            block.add(BatchNorm2d(planes))
            block.add(nn.ReLU())

        return block, planes

############################ Sparse implementation

class RPNNoHeadBaseSparse(nn.Module):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2),
                 num_filters=(128, 128, 256),
                 upsample_strides=(1, 2, 4),
                 num_upsample_filters=(256, 256, 256),
                 num_input_features=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 box_code_size=7,
                 num_direction_bins=2,
                 name='rpn'):
        """upsample_strides support float: [0.25, 0.5, 1]
        if upsample_strides < 1, conv2d will be used instead of convtranspose2d.
        """
        super(RPNNoHeadBaseSparse, self).__init__()
        self._layer_strides = layer_strides
        self._num_filters = num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = upsample_strides
        self._num_upsample_filters = num_upsample_filters
        self._num_input_features = num_input_features
        self._use_norm = use_norm
        self._use_groupnorm = use_groupnorm
        self._num_groups = num_groups
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(num_upsample_filters) == len(upsample_strides)
        self._upsample_start_idx = len(layer_nums) - len(upsample_strides)
        must_equal_list = []
        for i in range(len(upsample_strides)):
            must_equal_list.append(upsample_strides[i] / np.prod(
                layer_strides[:i + self._upsample_start_idx + 1]))
        for val in must_equal_list:
            assert val == must_equal_list[0]

        in_filters = [num_input_features, *num_filters[:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                num_filters[i],
                layer_num,
                i,
                stride=layer_strides[i])
            blocks.append(block)
            if i - self._upsample_start_idx >= 0:
                deblock = self._make_deblock(num_out_filters, i)
                deblocks.append(deblock)
        self._num_out_filters = num_out_filters
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        raise NotImplementedError


    def _make_deblock(self, num_out_filters, idx):
        stride = self._upsample_strides[idx - self._upsample_start_idx]
        if self._use_norm:
            if self._use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=self._num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(spconv.SparseConv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                spconv.SparseConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(spconv.SparseConv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                spconv.SparseConvTranspose2d)
        if stride >= 1:
            stride = np.round(stride).astype(np.int64)
            print("DEBLOCK CONV_TRANSPOSE STRIDE:", stride)
            deblock = spconv.SparseSequential(
                # PrintLayer(stride),
                ConvTranspose2d(
                    num_out_filters,
                    self._num_upsample_filters[idx - self._upsample_start_idx],
                    stride,
                    stride=stride),
                # PrintLayer(stride),
                BatchNorm2d(
                    self._num_upsample_filters[idx -
                                            self._upsample_start_idx]),
                nn.ReLU(),
            )
        else:
            stride = np.round(1 / stride).astype(np.int64)
            print("DEBLOCK CONV STRIDE:", stride)
            deblock = spconv.SparseSequential(
                # PrintLayer(stride),
                Conv2d(
                    num_out_filters,
                    self._num_upsample_filters[idx - self._upsample_start_idx],
                    stride, 
                    stride=stride),
                # PrintLayer(stride),
                BatchNorm2d(
                    self._num_upsample_filters[idx -
                                            self._upsample_start_idx]),
                nn.ReLU(),
            )
        return deblock


    def forward(self, x):
        ups = []
        stage_outputs = []

        x = spconv.SparseConvTensor.from_sparse(x)
        # print(" Before sparsity: %.4f" % x.sparity, "shape:", x.dense().shape)
        # after_from_dense = time.time()
        
        # blocks_total_time = 0
        # deblocks_total_time = 0
        # plot_pseudo_img(x, "Raw pseudoimage")
        for i in range(len(self.blocks)):
            # print("Block:")
            # print(self.blocks[i])
            # blocks_before = time.time()
            # print(f"Block {i} before x type:", type(x))
            x = self.blocks[i](x)

            # plot_pseudo_img(x, f"After block {i}")
            # blocks_after = time.time()
            # blocks_total_time += (blocks_after - blocks_before)
            # print(f"Block {i} sparsity: %.4f" % x.sparity, "shape:", x.dense().shape)
            stage_outputs.append(x)
            if i - self._upsample_start_idx >= 0:
                # deblocks_before = time.time()
                res = self.deblocks[i - self._upsample_start_idx](x)

                # plot_pseudo_img(res, f"After deblock {i}")
                # deblocks_after = time.time()
                # deblocks_total_time += (deblocks_after - deblocks_before)
                # print(f"Deblock {i} shape:", res.dense().shape)
                # print(f"Deblock {i} sparsity: %.4f" % res.sparity, "shape:", res.dense().shape)
                ups.append(res)
        # print(">>>>>>from_dense() time ms: %.2f" % ((after_from_dense - before_from_dense) * 1000), 
        # print("Blocks total ms: %.2f" % (blocks_total_time * 1000),
        #       "Deblocks total ms: %.2f" % (deblocks_total_time * 1000))
        if len(ups) > 0:
            lst = [e.dense() for e in ups]
            # print("DENSE SHAPES:", [e.shape for e in lst])
            x = torch.cat(lst, dim=1)
            # print("DENSE SHAPE:", x.shape)
            # x_sp = cat_sparse_dim1(ups).dense() #torch.cat(ups, dim=1)
            # print("SPARSE SHAPE:", x_sp.shape)
        res = {}
        for i, up in enumerate(ups):
            res[f"up{i}"] = up
        for i, out in enumerate(stage_outputs):
            res[f"stage{i}"] = out
        res["out"] = x
        return res

# def cat_sparse_dim1(lst):
#     out = spconv.SparseConvTensor(None, None, lst[0].spatial_shape, lst[0].batch_size)
#     dim1_lengths = [0] + [t.spatial_shape[0] for t in lst]
#     print("t.spatial_shape:", [t.spatial_shape for t in lst])
#     print("t.features.shape:",[t.features.shape for t in lst])
#     running_lengths = np.cumsum(dim1_lengths)
#     # Features require no scaling as they are values
#     out.features = torch.cat([t.features for t in lst], dim=0)
#     # Indices require scaling based on index
#     out.indices = torch.cat([t.indices + torch.tensor([0, running_lengths[idx], 0]).to(t.indices.device) for idx, t in enumerate(lst)])
#     out.spatial_shape = torch.Size([running_lengths[-1], out.spatial_shape[1]])
#     return out

def plot_pseudo_img(t, title):
    if isinstance(t, spconv.SparseConvTensor):
        t = t.dense().detach()
    elif isinstance(t, torch.Tensor):
        t = t.detach()
        if t.is_sparse:
            t = t.to_dense()
            t = t.permute(0, 3, 1, 2)
    t = t.cpu().numpy()
    print("Image shape:", t.shape)
    ts = [t[v] for v in range(t.shape[0])]
    imgs = [np.any(t, 0) for t in ts]

    import matplotlib.pyplot as plt
    for idx, img in enumerate(imgs):
        plt.subplot(2, 1, idx + 1)
        flt_img = img.flatten()
        nonzeros = flt_img.sum()
        plt.title(title + " Nonzeros: " + str(nonzeros) + "/" + str(flt_img.shape[0]) + " = %: " + str(nonzeros / flt_img.shape[0]))
        plt.imshow(img)
    plt.show()

class RPNBaseSparse(RPNNoHeadBaseSparse):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2),
                 num_filters=(128, 128, 256),
                 upsample_strides=(1, 2, 4),
                 num_upsample_filters=(256, 256, 256),
                 num_input_features=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 box_code_size=7,
                 num_direction_bins=2,
                 name='rpn'):
        """upsample_strides support float: [0.25, 0.5, 1]
        if upsample_strides < 1, conv2d will be used instead of convtranspose2d.
        """
        super(RPNBaseSparse, self).__init__(
            use_norm=use_norm,
            num_class=num_class,
            layer_nums=layer_nums,
            layer_strides=layer_strides,
            num_filters=num_filters,
            upsample_strides=upsample_strides,
            num_upsample_filters=num_upsample_filters,
            num_input_features=num_input_features,
            num_anchor_per_loc=num_anchor_per_loc,
            encode_background_as_zeros=encode_background_as_zeros,
            use_direction_classifier=use_direction_classifier,
            use_groupnorm=use_groupnorm,
            num_groups=num_groups,
            box_code_size=box_code_size,
            num_direction_bins=num_direction_bins,
            name=name)
        self._num_anchor_per_loc = num_anchor_per_loc
        self._num_direction_bins = num_direction_bins
        self._num_class = num_class
        self._use_direction_classifier = use_direction_classifier
        self._box_code_size = box_code_size

        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        if len(num_upsample_filters) == 0:
            final_num_filters = self._num_out_filters
        else:
            final_num_filters = sum(num_upsample_filters)
        self.conv_cls = nn.Conv2d(final_num_filters, num_cls, 1)
        self.conv_box = nn.Conv2d(final_num_filters,
                                  num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                final_num_filters, num_anchor_per_loc * num_direction_bins, 1)

    def forward(self, x):
        res = super().forward(x)
        x = res["out"]
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        C, H, W = box_preds.shape[1:]
        box_preds = box_preds.view(-1, self._num_anchor_per_loc,
                                   self._box_code_size, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()
        cls_preds = cls_preds.view(-1, self._num_anchor_per_loc,
                                   self._num_class, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()
        # box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        # cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()

        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.view(
                -1, self._num_anchor_per_loc, self._num_direction_bins, H,
                W).permute(0, 1, 3, 4, 2).contiguous()
            # dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds
        return ret_dict

class PrintLayer(SparseModule):
    def __init__(self, msg):
        super(PrintLayer, self).__init__()
        self.msg = msg
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(self.msg, "Shape:", x.dense().shape, "Sparsity:", x.sparity)
        return x

class SparseZeroPad2d(SparseModule):
    def __init__(self, pad):
        super(SparseModule, self).__init__()
        self.pad = pad

    def forward(self, x):
        w, h = x.spatial_shape
        x.spatial_shape = torch.Size([w + 2 * self.pad, h + 2 * self.pad])
        x.indices[:, 1] += self.pad
        x.indices[:, 2] += self.pad
        return x

@register_rpn
class RPNV2SemiSparse(RPNBaseSparse):
    def _make_layer(self, inplanes, planes, num_blocks, idx, stride=1):
        if self._use_norm:
            if self._use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=self._num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            SparseConv2d = change_default_args(bias=False)(spconv.SparseConv2d)
            DenseConv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                spconv.SparseConvTranspose2d)
        else:
            BatchNorm2d = Empty
            SparseConv2d = change_default_args(bias=True)(spconv.SparseConv2d)
            DenseConv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                spconv.SparseConvTranspose2d)
        print("STRIDE:", stride)

        if idx == 0:
            block = spconv.SparseSequential(
                SparseZeroPad2d(1),
                SparseConv2d(inplanes, planes, 3, stride=stride),
                BatchNorm2d(planes),
                nn.ReLU(),
            )
            for j in range(num_blocks):
                block.add(SparseConv2d(planes, planes, 3, padding=1))
                block.add(BatchNorm2d(planes))
                block.add(nn.ReLU())
        else:
            block = Sequential(
                nn.ZeroPad2d(1),
                DenseConv2d(inplanes, planes, 3, stride=stride),
                BatchNorm2d(planes),
                nn.ReLU(),
            )
            for j in range(num_blocks):
                block.add(DenseConv2d(planes, planes, 3, padding=1))
                block.add(BatchNorm2d(planes))
                block.add(nn.ReLU())

        return block, planes

    def _make_deblock(self, num_out_filters, idx):
        print("CUSTOM MAKE DEBLOCK")
        stride = self._upsample_strides[idx - self._upsample_start_idx]
        if self._use_norm:
            if self._use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=self._num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            SparseConvTranspose2d = change_default_args(bias=False)(
                spconv.SparseConvTranspose2d)
            DenseConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            SparseConvTranspose2d = change_default_args(bias=True)(
                spconv.SparseConvTranspose2d)
            DenseConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)
        stride = np.round(stride).astype(np.int64)
        if (idx == 0):
            deblock = spconv.SparseSequential(
                SparseConvTranspose2d(
                    num_out_filters,
                    self._num_upsample_filters[idx - self._upsample_start_idx],
                    stride,
                    stride=stride),
                BatchNorm2d(
                    self._num_upsample_filters[idx -
                                            self._upsample_start_idx]),
                nn.ReLU(),
                spconv.ToDense()
            )
        else:
            stride = np.round(stride).astype(np.int64)
            deblock = Sequential(
                DenseConvTranspose2d(
                    num_out_filters,
                    self._num_upsample_filters[idx - self._upsample_start_idx],
                    stride,
                    stride=stride),
                BatchNorm2d(
                    self._num_upsample_filters[idx -
                                            self._upsample_start_idx]),
                nn.ReLU(),
            )
        return deblock

@register_rpn
class RPNV2Sparse(RPNBaseSparse):
    def _make_layer(self, inplanes, planes, num_blocks, idx, stride=1):
        if self._use_norm:
            if self._use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=self._num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(spconv.SparseConv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                spconv.SparseConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(spconv.SparseConv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                spconv.SparseConvTranspose2d)
        print("STRIDE:", stride)
        block = spconv.SparseSequential(
            SparseZeroPad2d(1),
            Conv2d(inplanes, planes, 3, stride=stride),
            BatchNorm2d(planes),
            nn.ReLU(),
        )
        for j in range(num_blocks):
            block.add(Conv2d(planes, planes, 3, padding=1))
            block.add(BatchNorm2d(planes))
            block.add(nn.ReLU())

        return block, planes

class SparseScale2d(SparseModule):
    def __init__(self, scale):
        super(SparseModule, self).__init__()
        self.scale = scale

    def forward(self, x):
        w, h = x.spatial_shape
        x.spatial_shape = torch.Size([w * self.scale, h * self.scale])
        x.indices[:, 1:] = x.indices[:, 1:] * self.scale
        return x

@register_rpn
class RPNV2Pyramid(RPNBaseSparse):
    def _make_layer(self, inplanes, planes, num_blocks, idx, stride=1):
        print("NUM BLOCKS:", num_blocks)
        if self._use_norm:
            if self._use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=self._num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(spconv.SparseConv2d)
            SubMConv2d = change_default_args(bias=False)(spconv.SubMConv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                spconv.SparseConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(spconv.SparseConv2d)
            SubMConv2d = change_default_args(bias=True)(spconv.SubMConv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                spconv.SparseConvTranspose2d)

        block = spconv.SparseSequential(
            SparseZeroPad2d(1),         
            # PrintLayer(0),   
            Conv2d(inplanes, planes, 3, stride=stride),
            BatchNorm2d(planes),
            nn.ReLU(),
            # PrintLayer(1),
        )
        for j in range(num_blocks):
            block.add(SubMConv2d(planes, planes, 3, padding=1))
            block.add(BatchNorm2d(planes)),
            block.add(nn.ReLU())
            # block.add(PrintLayer(2 + j))

        return block, planes

@register_rpn
class RPNV2Pyramid2x2(RPNBaseSparse):
    def _make_layer(self, inplanes, planes, num_blocks, idx, stride=1):
        print("NUM BLOCKS:", num_blocks, "STRIDE:", stride)
        if self._use_norm:
            if self._use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=self._num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(spconv.SparseConv2d)
            SubMConv2d = change_default_args(bias=False)(spconv.SubMConv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                spconv.SparseConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(spconv.SparseConv2d)
            SubMConv2d = change_default_args(bias=True)(spconv.SubMConv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                spconv.SparseConvTranspose2d)

        block = spconv.SparseSequential(    
            # PrintLayer(0),   
            Conv2d(inplanes, planes, 2, stride=stride),
            BatchNorm2d(planes),
            nn.ReLU(),
            # PrintLayer(1),
        )
        for j in range(num_blocks):
            block.add(SubMConv2d(planes, planes, 3, padding=1))
            block.add(BatchNorm2d(planes)),
            block.add(nn.ReLU())
            # block.add(PrintLayer(2 + j))

        return block, planes
