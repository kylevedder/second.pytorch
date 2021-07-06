import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet

import spconv
from spconv.modules import SparseModule

from second.pytorch.models.rpn import register_rpn

import MinkowskiEngine as ME

from torchplus.nn import Empty, GroupNorm, Sequential
from torchplus.tools import change_default_args

LAST_SPARSE_IDX = 1

class ToDenseMink(SparseModule):
    def __init__(self, batch_size):
        super(ToDenseMink, self).__init__()
        self.min_coordinate = torch.IntTensor([0, 0])
        self.batch_size = batch_size
    
    def forward(self, x):
        shape = torch.Size([self.batch_size, 128, 248, 216])
        return x.dense(shape=shape, min_coordinate=self.min_coordinate)[0]

class RPNNoHeadBaseMESparse(nn.Module):
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
        super(RPNNoHeadBaseMESparse, self).__init__()
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
        # print("Blocks:")
        # for b in blocks:
        #     print(b)
        # print("Deblocks:")
        # for d in deblocks:
        #     print(d)

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
            BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(ME.MinkowskiBatchNorm)
            Conv2d = change_default_args(bias=False, dimension=2)(ME.MinkowskiConvolution)
            ConvTranspose2d = change_default_args(bias=False, dimension=2)(
                ME.MinkowskiConvolutionTranspose)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True, dimension=2)(ME.MinkowskiConvolution)
            ConvTranspose2d = change_default_args(bias=True, dimension=2)(
                ME.MinkowskiConvolutionTranspose)
        ReLU = ME.MinkowskiReLU()

        if stride >= 1:
            stride = np.round(stride).astype(np.int64)
            print("DEBLOCK CONV_TRANSPOSE STRIDE:", stride)
            deblock = Sequential(
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
                ReLU,
            )
        else:
            stride = np.round(1 / stride).astype(np.int64)
            print("DEBLOCK CONV STRIDE:", stride)
            deblock = Sequential(
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
                ReLU,
            )
        return deblock


    def forward(self, x):
        x, batch_size = x
        ups = []
        stage_outputs = []
        # print("Input shape:", x.shape)
        # plot_pseudo_img(x, "Coo")
        vals = x._values()
        # print("Vals:", vals.shape)
        idxs = x._indices().permute(1, 0).contiguous().int()
        # print("Idxs:", idxs.shape)
        
        x = ME.SparseTensor(vals, idxs) 
        to_dense = ToDenseMink(batch_size)
        #spconv.SparseConvTensor.from_sparse(x)
        # print(" Before sparsity: %.4f" % x.sparity, "shape:", x.dense().shape)
        # after_from_dense = time.time()
        
        # blocks_total_time = 0
        # deblocks_total_time = 0
        # plot_pseudo_img(x, "before")
        for i in range(len(self.blocks)):
            # print("Block:")
            # print(self.blocks[i])
            # blocks_before = time.time()
            # if i == (LAST_SPARSE_IDX + 1):
            # #     # print(f"Made x dense at step {i}")
            #     x = x.dense()
            # print(f"Block {i} before x type:", type(x))
            # print(self.blocks[i])
            x = self.blocks[i](x)
            # plot_pseudo_img(x, f"block{i}")
            # plot_pseudo_img(x, f"After block {i}")
            # blocks_after = time.time()
            # blocks_total_time += (blocks_after - blocks_before)
            # print(f"Block {i} sparsity: %.4f" % x.sparity, "shape:", x.dense().shape)
            stage_outputs.append(x)
            if i - self._upsample_start_idx >= 0:
                # deblocks_before = time.time()
                res = self.deblocks[i - self._upsample_start_idx](x)
                # plot_pseudo_img(res, f"deblock{i}")
                # plot_pseudo_img(res, f"After deblock {i}")
                # deblocks_after = time.time()
                # deblocks_total_time += (deblocks_after - deblocks_before)
                # print(f"Deblock {i} shape:", res.dense().shape)
                # print(f"Deblock {i} sparsity: %.4f" % res.sparity, "shape:", res.dense().shape)
                ups.append(to_dense(res))
        # print(">>>>>>from_dense() time ms: %.2f" % ((after_from_dense - before_from_dense) * 1000), 
        # print("Blocks total ms: %.2f" % (blocks_total_time * 1000),
        #       "Deblocks total ms: %.2f" % (deblocks_total_time * 1000))
        if len(ups) > 0:
            #lst = [e.dense() for e in ups]
            #x = torch.cat(lst, dim=1)
            # for idx, e in enumerate(ups):
            #     print(idx)
            #     tensor, min_coordinate, tensor_stride = e
            #     print(tensor.shape)
            #     print(min_coordinate.shape)
            #     print(tensor_stride.shape)
            # print([e.shape for e in ups])
            x = torch.cat(ups, dim=1)
            # print("Cat'd shape:", x.shape)
            #[2, 384, 248, 216]
            # [2, 384, 248, 216]
            # print("DENSE SHAPES:", [e.shape for e in lst])
            
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
    elif isinstance(t, ME.SparseTensor):
        t = t.dense()[0].detach()
    elif isinstance(t, torch.Tensor):
        t = t.detach()
        if t.is_sparse:
            t = t.to_dense()
            t = t.permute(0, 3, 1, 2)
    t = t.cpu().numpy()
    print("Image shape:", t.shape)
    ts = [t[v] for v in range(t.shape[0])]
    imgs = [np.any(t, 0) for t in ts]

    print(t.shape)
    import matplotlib.pyplot as plt
    import matplotlib
    # matplotlib.use('pgf')
    matplotlib.rcParams.update({# Use mathtext, not LaTeX
                            'text.usetex': False,
                            # Use the Computer modern font
                            'font.family': 'serif',
                            'font.serif': ['cmr10'],
                            'font.size' : 6,
                            'mathtext.fontset': 'cm',
                            # Use ASCII minus
                            'axes.unicode_minus': False,
                            })
    img = imgs[0]
    flt_img = img.flatten()
    nonzeros = flt_img.sum()
    print(str(nonzeros) + "/" + str(flt_img.shape[0]))
    print(img.shape)
    dpi = 100
    figsize = (img.shape[0] / dpi, img.shape[1] / dpi)
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(img, cmap='gray', interpolation='none')
    plt.clim(0,1)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    plt.axis('off')
    plt.show()
    # plt.savefig(f"{title}.pdf", bbox_inches='tight', pad_inches=0)
    # for idx, img in enumerate(imgs):
    #     plt.subplot(t.shape[0], 1, idx + 1)
    #     flt_img = img.flatten()
    #     nonzeros = flt_img.sum()
    #     plt.title(title + " Nonzeros: " + str(nonzeros) + "/" + str(flt_img.shape[0]) + " = %: " + str(nonzeros / flt_img.shape[0]))
    #     plt.imshow(img)
    # plt.show()

class RPNBaseMESparse(RPNNoHeadBaseMESparse):
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
        super(RPNBaseMESparse, self).__init__(
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

@register_rpn
class RPNV2MESemiSparse(RPNBaseMESparse):
    def _make_layer(self, inplanes, planes, num_blocks, idx, stride=1):
        if self._use_norm:
            if self._use_groupnorm:
                SparseBatchNorm2d = change_default_args(
                    num_groups=self._num_groups, eps=1e-3)(GroupNorm)
                DenseBatchNorm2d = change_default_args(
                    num_groups=self._num_groups, eps=1e-3)(GroupNorm)
            else:
                SparseBatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
                DenseBatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            SparseConv2d = change_default_args(bias=False)(spconv.SparseConv2d)
            DenseConv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                spconv.SparseConvTranspose2d)
        else:
            SparseBatchNorm2d = Empty
            DenseBatchNorm2d = Empty
            SparseConv2d = change_default_args(bias=True)(spconv.SparseConv2d)
            DenseConv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                spconv.SparseConvTranspose2d)
        print("STRIDE:", stride)

        if idx <= LAST_SPARSE_IDX:
            block = spconv.SparseSequential(
                SparseZeroPad2d(1),
                SparseConv2d(inplanes, planes, 3, stride=stride),
                SparseBatchNorm2d(planes),
                nn.ReLU(),
            )
            for j in range(num_blocks):
                block.add(SparseConv2d(planes, planes, 3, padding=1))
                block.add(SparseBatchNorm2d(planes))
                block.add(nn.ReLU())
        else:
            block = Sequential(
                nn.ZeroPad2d(1),
                DenseConv2d(inplanes, planes, 3, stride=stride),
                DenseBatchNorm2d(planes),
                nn.ReLU(),
            )
            for j in range(num_blocks):
                block.add(DenseConv2d(planes, planes, 3, padding=1))
                block.add(DenseBatchNorm2d(planes))
                block.add(nn.ReLU())

        return block, planes

    def _make_deblock(self, num_out_filters, idx):
        print("CUSTOM MAKE DEBLOCK")
        stride = self._upsample_strides[idx - self._upsample_start_idx]
        if self._use_norm:
            if self._use_groupnorm:
                SparseBatchNorm2d = change_default_args(
                    num_groups=self._num_groups, eps=1e-3)(GroupNorm)
                DenseBatchNorm2d = change_default_args(
                    num_groups=self._num_groups, eps=1e-3)(GroupNorm)
            else:
                SparseBatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(ME.MinkowskiBatchNorm)
                DenseBatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            SparseConvTranspose2d = change_default_args(bias=False, dimension=2)(
                ME.MinkowskiConvolutionTranspose)
            DenseConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            SparseBatchNorm2d = Empty
            DenseBatchNorm2d = Empty
            SparseConvTranspose2d = change_default_args(bias=True, dimension=2)(
                ME.MinkowskiConvolutionTranspose)
            DenseConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)
        ReLU = ME.MinkowskiReLU()
        stride = np.round(stride).astype(np.int64)
        if (idx <= LAST_SPARSE_IDX):
            deblock = Sequential(
                SparseConvTranspose2d(
                    num_out_filters,
                    self._num_upsample_filters[idx - self._upsample_start_idx],
                    stride,
                    stride=stride),
                SparseBatchNorm2d(
                    self._num_upsample_filters[idx -
                                            self._upsample_start_idx]),
                ReLU,
                ME.ToDense()
            )
        else:
            stride = np.round(stride).astype(np.int64)
            deblock = Sequential(
                DenseConvTranspose2d(
                    num_out_filters,
                    self._num_upsample_filters[idx - self._upsample_start_idx],
                    stride,
                    stride=stride),
                DenseBatchNorm2d(
                    self._num_upsample_filters[idx -
                                            self._upsample_start_idx]),
                ReLU,
            )
        return deblock

@register_rpn
class RPNV2MESparse(RPNBaseMESparse):
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
class RPNV2MEPyramid(RPNBaseMESparse):
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
class RPNV2MEPyramid2x2(RPNBaseMESparse):
    def _make_layer(self, inplanes, planes, num_blocks, idx, stride=1):
        print("NUM BLOCKS:", num_blocks, "STRIDE:", stride)
        if self._use_norm:
            BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(ME.MinkowskiBatchNorm)
            Conv2d = change_default_args(bias=False, dimension=2)(ME.MinkowskiConvolution)
            SubMConv2d = change_default_args(bias=False, dimension=2)(ME.MinkowskiConvolution)
            ConvTranspose2d = change_default_args(bias=False, dimension=2)(
                ME.MinkowskiConvolutionTranspose)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True, dimension=2)(ME.MinkowskiConvolution)
            SubMConv2d = change_default_args(bias=True, dimension=2)(ME.MinkowskiConvolution)
            ConvTranspose2d = change_default_args(bias=True, dimension=2)(
                ME.MinkowskiConvolutionTranspose)
        ReLU = ME.MinkowskiReLU()

        block = Sequential(    
            # PrintLayer(0),   
            Conv2d(inplanes, planes, 2, stride=stride),
            BatchNorm2d(planes),
            ReLU,
            # PrintLayer(1),
        )
        for j in range(num_blocks):
            block.add(SubMConv2d(planes, planes, 3))
            block.add(BatchNorm2d(planes)),
            block.add(ReLU)
            # block.add(PrintLayer(2 + j))

        return block, planes

@register_rpn
class RPNV2MEPyramid2x2Wide(RPNBaseMESparse):
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

        block = spconv.SparseSequential()

        wide_conv_schedule = [None, 7, 9]
        filter_radius = wide_conv_schedule[idx]

        if filter_radius is not None:
            print("Filter radius:", filter_radius)
            block.add(SubMConv2d(inplanes, inplanes, filter_radius))
            # Substitute for later convs.
            num_blocks -= 1


        block.add(Conv2d(inplanes, planes, 2, stride=stride))
        block.add(BatchNorm2d(planes))
        block.add(nn.ReLU())

        for j in range(num_blocks):
            block.add(SubMConv2d(planes, planes, 3, padding=1))
            block.add(BatchNorm2d(planes)),
            block.add(nn.ReLU())
            # block.add(PrintLayer(2 + j))

        return block, planes
