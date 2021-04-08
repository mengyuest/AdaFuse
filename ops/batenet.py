import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
from torch.nn.init import normal_, constant_

from ops.utils import count_conv2d_flops

__all__ = ['BateNet', 'batenet18', 'batenet34', 'batenet50', 'batenet101', 'batenet152']

model_urls = {
    'batenet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'batenet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'batenet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'batenet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'batenet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def list_sum(obj):
    if isinstance(obj, list):
        if len(obj)==0:
            return 0
        else:
            return sum(list_sum(x) for x in obj)
    else:
        return obj

def shift(x, n_segment, fold_div=3, inplace=False):
    nt, c, h, w = x.size()
    n_batch = nt // n_segment
    x = x.view(n_batch, n_segment, c, h, w)

    fold = c // fold_div
    if inplace:
        # Due to some out of order error when performing parallel computing.
        # May need to write a CUDA kernel.
        raise NotImplementedError
        # out = InplaceShift.apply(x, fold)
    else:
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

    return out.view(nt, c, h, w)

class PolicyBlock(nn.Module):
    def __init__(self, in_planes, out_planes, norm_layer, shared, args):
        super(PolicyBlock, self).__init__()
        self.args = args
        self.norm_layer = norm_layer
        self.shared = shared
        in_factor = 1
        out_factor = 2
        if self.args.gate_history:
            in_factor = 2
            if not self.args.gate_no_skipping:
                out_factor = 3
        self.action_dim = out_factor

        in_dim = in_planes * in_factor
        out_dim = out_planes * out_factor // self.args.granularity
        out_dim = out_dim * (args.gate_channel_ends - args.gate_channel_starts) // 64
        self.num_channels = out_dim // self.action_dim

        keyword = "%d_%d" % (in_dim, out_dim)
        if self.args.relative_hidden_size > 0:
            hidden_dim = int(self.args.relative_hidden_size * out_planes // self.args.granularity)
        elif self.args.hidden_quota > 0:
            hidden_dim = self.args.hidden_quota // (out_planes // self.args.granularity)
        else:
            hidden_dim = self.args.gate_hidden_dim

        if self.args.gate_conv_embed_type != "None":
            if self.args.gate_conv_embed_type == "conv3x3":
                self.gate_conv_embed = conv3x3(in_planes, in_planes, stride=1, groups=1, dilation=1)
            elif self.args.gate_conv_embed_type == "conv3x3dw":
                self.gate_conv_embed = conv3x3(in_planes, in_planes, stride=1, groups=in_planes, dilation=1)
            elif self.args.gate_conv_embed_type == "conv1x1":
                self.gate_conv_embed = conv1x1(in_planes, in_planes, stride=1, groups=1, dilation=1)
            elif self.args.gate_conv_embed_type == "conv1x1dw":
                self.gate_conv_embed = conv1x1(in_planes, in_planes, stride=1, groups=in_planes, dilation=1)
            self.gate_conv_embed_bn = nn.BatchNorm2d(in_planes)
            self.gate_conv_embed_relu = nn.ReLU(inplace=True)

        if self.args.single_linear:
            self.gate_fc0 = nn.Linear(in_dim, out_dim)
            if self.args.gate_bn_between_fcs:
                self.gate_bn = nn.BatchNorm1d(out_dim)
            if self.args.gate_relu_between_fcs:
                self.gate_relu = nn.ReLU(inplace=True)

            normal_(self.gate_fc0.weight, 0, 0.001)
            constant_(self.gate_fc0.bias, 0)
        elif self.args.triple_linear:
            self.gate_fc0 = nn.Linear(in_planes * in_factor, hidden_dim)
            if self.args.gate_bn_between_fcs:
                self.gate_bn = nn.BatchNorm1d(hidden_dim)
            if self.args.gate_relu_between_fcs:
                self.gate_relu = nn.ReLU(inplace=True)

            self.gate_fc1 = nn.Linear(hidden_dim, hidden_dim)

            if self.args.gate_bn_between_fcs:
                self.gate_bn1 = nn.BatchNorm1d(hidden_dim)
            if self.args.gate_relu_between_fcs:
                self.gate_relu1 = nn.ReLU(inplace=True)
            self.gate_fc2 = nn.Linear(hidden_dim, out_dim)

            normal_(self.gate_fc0.weight, 0, 0.001)
            constant_(self.gate_fc0.bias, 0)
            normal_(self.gate_fc1.weight, 0, 0.001)
            constant_(self.gate_fc1.bias, 0)
            normal_(self.gate_fc2.weight, 0, 0.001)
            constant_(self.gate_fc2.bias, 0)

        else:
            if self.args.shared_policy_net:
                self.gate_fc0 = self.shared[0][keyword]
            else:
                self.gate_fc0 = nn.Linear(in_dim, hidden_dim)
            if self.args.gate_bn_between_fcs:
                self.gate_bn = nn.BatchNorm1d(hidden_dim)
            if self.args.gate_relu_between_fcs:
                self.gate_relu = nn.ReLU(inplace=True)
            if self.args.shared_policy_net:
                self.gate_fc1 = self.shared[1][keyword]
            else:
                self.gate_fc1 = nn.Linear(hidden_dim, out_dim)

            if not self.args.shared_policy_net:
                normal_(self.gate_fc0.weight, 0, 0.001)
                constant_(self.gate_fc0.bias, 0)
                normal_(self.gate_fc1.weight, 0, 0.001)
                constant_(self.gate_fc1.bias, 0)



    def forward(self, x, **kwargs):
        # data preparation
        if self.args.gate_conv_embed_type != "None":
            x_input = self.gate_conv_embed(x)
            x_input = self.gate_conv_embed_bn(x_input)
            x_input = self.gate_conv_embed_relu(x_input)
        else:
            x_input = x

        if self.args.gate_reduce_type=="avg":
            x_c = nn.AdaptiveAvgPool2d((1, 1))(x_input)
        elif self.args.gate_reduce_type=="max":
            x_c = nn.AdaptiveMaxPool2d((1, 1))(x_input)
        x_c = torch.flatten(x_c, 1)
        _nt, _c = x_c.shape
        _t = self.args.num_segments
        _n = _nt // _t

        # history
        if self.args.gate_history:
            x_c_reshape = x_c.view(_n, _t, _c)
            h_vec = torch.zeros_like(x_c_reshape)
            h_vec[:, 1:] = x_c_reshape[:, :-1]
            h_vec = h_vec.view(_nt, _c)
            x_c = torch.cat([h_vec, x_c], dim=-1)

        # fully-connected embedding
        if self.args.single_linear:
            x_c = self.gate_fc0(x_c)
            if self.args.gate_bn_between_fcs:
                x_c = x_c.unsqueeze(-1)
                x_c = self.gate_bn(x_c)
                x_c = x_c.squeeze(-1)
            if self.args.gate_relu_between_fcs:
                x_c = self.gate_relu(x_c)

        elif self.args.triple_linear:
            x_c = self.gate_fc0(x_c)
            if self.args.gate_bn_between_fcs:
                x_c = x_c.unsqueeze(-1)
                x_c = self.gate_bn(x_c)
                x_c = x_c.squeeze(-1)
            if self.args.gate_relu_between_fcs:
                x_c = self.gate_relu(x_c)
            x_c = self.gate_fc1(x_c)

            if self.args.gate_bn_between_fcs:
                x_c = x_c.unsqueeze(-1)
                x_c = self.gate_bn1(x_c)
                x_c = x_c.squeeze(-1)
            if self.args.gate_relu_between_fcs:
                x_c = self.gate_relu1(x_c)
            x_c = self.gate_fc2(x_c)

        else:
            x_c = self.gate_fc0(x_c)
            if self.args.gate_bn_between_fcs:
                x_c = x_c.unsqueeze(-1)
                x_c = self.gate_bn(x_c)
                x_c = x_c.squeeze(-1)
            if self.args.gate_relu_between_fcs:
                x_c = self.gate_relu(x_c)
            x_c = self.gate_fc1(x_c)

        # gating operations
        x_c2d = x_c.view(x.shape[0], self.num_channels // self.args.granularity, self.action_dim)
        x_c2d = torch.log(F.softmax(x_c2d, dim=2).clamp(min=1e-8))
        mask = F.gumbel_softmax(logits=x_c2d, tau=kwargs["tau"], hard=not self.args.gate_gumbel_use_soft)

        if self.args.granularity>1:
            mask = mask.repeat(1, self.args.granularity, 1)
        if self.args.gate_channel_starts>0 or self.args.gate_channel_ends<64:
            full_channels = mask.shape[1] // (self.args.gate_channel_ends-self.args.gate_channel_starts) * 64
            channel_starts = full_channels // 64 * self.args.gate_channel_starts
            channel_ends = full_channels // 64 * self.args.gate_channel_ends
            outer_mask = torch.zeros(mask.shape[0], full_channels, mask.shape[2]).to(mask.device)
            outer_mask[:, :, -1] = 1.
            outer_mask[:, channel_starts:channel_ends] = mask

            return outer_mask
        else:
            return mask  # TODO: BT*C*ACT_DIM


def handcraft_policy_for_masks(x, out, num_channels, use_current, args):
    factor = 3 if args.gate_history else 2

    if use_current:
        mask = torch.zeros(x.shape[0], num_channels, factor, device=x.device)
        mask[:, :, -1] = 1.

    elif args.gate_all_zero_policy:
        mask = torch.zeros(x.shape[0], num_channels, factor, device=x.device)
    elif args.gate_all_one_policy:
        mask = torch.ones(x.shape[0], num_channels, factor, device=x.device)
    elif args.gate_only_current_policy:
        mask = torch.zeros(x.shape[0], num_channels, factor, device=x.device)
        mask[:, :, -1] = 1.
    elif args.gate_random_soft_policy:
        mask = torch.rand(x.shape[0], num_channels, factor, device=x.device)
    elif args.gate_random_hard_policy:
        tmp_value = torch.rand(x.shape[0], num_channels, device=x.device)
        mask = torch.zeros(x.shape[0], num_channels, factor, device=x.device)

        if len(args.gate_stoc_ratio) > 0:
            _ratio = args.gate_stoc_ratio
        else:
            _ratio = [0.333, 0.333, 0.334] if args.gate_history else [0.5, 0.5]
        mask[:, :, 0][torch.where(tmp_value < _ratio[0])] = 1
        if args.gate_history:
            mask[:, :, 1][torch.where((tmp_value < _ratio[1] + _ratio[0]) & (tmp_value > _ratio[0]))] = 1
            mask[:, :, 2][torch.where(tmp_value > _ratio[1] + _ratio[0])] = 1

    elif args.gate_threshold:
        stat = torch.norm(out, dim=[2, 3], p=1) / out.shape[2] / out.shape[3]
        mask = torch.ones_like(stat).float()
        if args.absolute_threshold is not None:
            mask[torch.where(stat < args.absolute_threshold)] = 0
        else:
            if args.relative_max_threshold is not None:
                mask[torch.where(
                    stat < torch.max(stat, dim=1)[0].unsqueeze(-1) * args.relative_max_threshold)] = 0
            else:
                mask = torch.zeros_like(stat)
                c_ids = torch.topk(stat, k=int(mask.shape[1] * args.relative_keep_threshold), dim=1)[1]  # TODO B*K
                b_ids = torch.tensor([iii for iii in range(mask.shape[0])]).to(mask.device).unsqueeze(-1).expand(c_ids.shape)  # TODO B*K
                mask[b_ids.detach().flatten(), c_ids.detach().flatten()] = 1

        mask = torch.stack([1 - mask, mask], dim=-1)

    return mask


def get_hmap(out, args, **kwargs):
    out_reshaped = out.view((-1, args.num_segments) + out.shape[1:])

    if args.gate_history:
        h_map_reshaped = torch.zeros_like(out_reshaped)
        h_map_reshaped[:, 1:] = out_reshaped[:, :-1]
    else:
        return None

    if args.gate_history_detach:
        h_map_reshaped = h_map_reshaped.detach()

    h_map_updated = h_map_reshaped.view((-1,) + out_reshaped.shape[2:])
    return h_map_updated


def fuse_out_with_mask(out, mask, h_map, args):
    out = out * mask[:, :, -1].unsqueeze(-1).unsqueeze(-1)
    if args.gate_history:
        out = out + h_map * mask[:, :, -2].unsqueeze(-1).unsqueeze(-1)
    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample0=None, downsample1=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, shared=None, shall_enable=None, args=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample0 = downsample0
        self.downsample1 = downsample1
        self.stride = stride

        self.args = args
        self.shall_enable = shall_enable
        self.num_channels = planes
        self.adaptive_policy = not any([self.args.gate_all_zero_policy,
                                        self.args.gate_all_one_policy,
                                        self.args.gate_only_current_policy,
                                        self.args.gate_random_soft_policy,
                                        self.args.gate_random_hard_policy,
                                        self.args.gate_threshold])
        if self.shall_enable==False and self.adaptive_policy:
            self.adaptive_policy=False
            self.use_current=True
        else:
            self.use_current=False

        if self.adaptive_policy:
            self.policy_net = PolicyBlock(in_planes=inplanes, out_planes=planes, norm_layer=norm_layer, shared=shared, args=args)

            if self.args.dense_in_block:
                self.policy_net2 = PolicyBlock(in_planes=planes, out_planes=planes, norm_layer=norm_layer, shared=shared, args=args)

        if self.args.gate_history_conv_type in ['ghost', 'ghostbnrelu']:
            self.gate_hist_conv = conv3x3(planes, planes, groups=planes)
            if self.args.gate_history_conv_type == 'ghostbnrelu':
                self.gate_hist_bnrelu = nn.Sequential(norm_layer(planes), nn.ReLU(inplace=True))

    def count_flops(self, input_data_shape, **kwargs):
        conv1_flops, conv1_out_shape = count_conv2d_flops(input_data_shape, self.conv1)
        conv2_flops, conv2_out_shape = count_conv2d_flops(conv1_out_shape, self.conv2)
        if self.downsample0 is not None:
            downsample0_flops, _ = count_conv2d_flops(input_data_shape, self.downsample0)
        else:
            downsample0_flops = 0

        return [conv1_flops, conv2_flops, downsample0_flops, 0], conv2_out_shape

    def forward(self, x, **kwargs):
        identity = x

        # shift operations
        if self.args.shift:
            x = shift(x, self.args.num_segments, fold_div=self.args.shift_div, inplace=False)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # gate functions
        h_map_updated = get_hmap(out, self.args, **kwargs)
        if self.adaptive_policy:
            mask = self.policy_net(x, **kwargs)
        else:
            mask = handcraft_policy_for_masks(x, out, self.num_channels, self.use_current, self.args)
        out = fuse_out_with_mask(out, mask, h_map_updated, self.args)

        x2 = out
        out = self.conv2(out)
        out = self.bn2(out)

        # gate functions
        if self.args.dense_in_block:
            h_map_updated2 = get_hmap(out, self.args, **kwargs)
            if self.adaptive_policy:
                mask2 = self.policy_net2(x2, **kwargs)
            else:
                mask2 = handcraft_policy_for_masks(x2, out, self.num_channels, self.use_current, self.args)
            out = fuse_out_with_mask(out, mask2, h_map_updated2, self.args)
            mask2 = mask2.view((-1, self.args.num_segments) + mask2.shape[1:])
        else:
            mask2 = None

        if self.downsample0 is not None:
            y = self.downsample0(x)
            identity = self.downsample1(y)
        out += identity
        out = self.relu(out)
        return out, mask.view((-1, self.args.num_segments) + mask.shape[1:]), mask2




class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample0=None, downsample1=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, shared=None, shall_enable=None, args=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample0 = downsample0
        self.downsample1 = downsample1
        self.stride = stride

        self.args = args
        self.shall_enable = shall_enable
        self.num_channels = width
        self.adaptive_policy = not any([self.args.gate_all_zero_policy,
                                        self.args.gate_all_one_policy,
                                        self.args.gate_only_current_policy,
                                        self.args.gate_random_soft_policy,
                                        self.args.gate_random_hard_policy,
                                        self.args.gate_threshold])
        if self.shall_enable==False and self.adaptive_policy:
            self.adaptive_policy=False
            self.use_current=True
        else:
            self.use_current=False

        if self.adaptive_policy:
            self.policy_net = PolicyBlock(in_planes=inplanes, out_planes=width, norm_layer=norm_layer, shared=shared, args=args)
            if self.args.dense_in_block:
                self.policy_net2 = PolicyBlock(in_planes=width, out_planes=width, norm_layer=norm_layer, shared=shared, args=args)

        if self.args.gate_history_conv_type in ['ghost', 'ghostbnrelu']:
            self.gate_hist_conv = conv3x3(width, width, groups=width)
            if self.args.gate_history_conv_type == 'ghostbnrelu':
                self.gate_hist_bnrelu = nn.Sequential(norm_layer(width), nn.ReLU(inplace=True))

    def count_flops(self, input_data_shape, **kwargs):
        conv1_flops, conv1_out_shape = count_conv2d_flops(input_data_shape, self.conv1)
        conv2_flops, conv2_out_shape = count_conv2d_flops(conv1_out_shape, self.conv2)
        conv3_flops, conv3_out_shape = count_conv2d_flops(conv2_out_shape, self.conv3)
        if self.downsample0 is not None:
            downsample0_flops, _ = count_conv2d_flops(input_data_shape, self.downsample0)
        else:
            downsample0_flops = 0

        return [conv1_flops, conv2_flops, conv3_flops, downsample0_flops, 0], conv3_out_shape

    def forward(self, x, **kwargs):
        identity = x

        # shift operations
        if self.args.shift:
            x = shift(x, self.args.num_segments, fold_div=self.args.shift_div, inplace=False)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # gate functions
        h_map_updated = get_hmap(out, self.args, **kwargs)
        if self.args.gate_history_conv_type in ['ghost', 'ghostbnrelu']:
            h_map_updated = self.gate_hist_conv(h_map_updated)
            if self.args.gate_history_conv_type == 'ghostbnrelu':
                h_map_updated = self.gate_hist_bnrelu(h_map_updated)

        if self.adaptive_policy:
            mask = self.policy_net(x, **kwargs)
        else:
            mask = handcraft_policy_for_masks(x, out, self.num_channels, self.use_current, self.args)

        out = fuse_out_with_mask(out, mask, h_map_updated, self.args)

        x2 = out
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # gate functions
        if self.args.dense_in_block:
            h_map_updated2 = get_hmap(out, self.args, **kwargs)
            if self.adaptive_policy:
                mask2 = self.policy_net2(x2, **kwargs)
            else:
                mask2 = handcraft_policy_for_masks(x2, out, self.num_channels, self.use_current, self.args)
            out = fuse_out_with_mask(out, mask2, h_map_updated2, self.args)
            mask2 = mask2.view((-1, self.args.num_segments) + mask2.shape[1:])
        else:
            mask2 = None

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample0 is not None:
            y = self.downsample0(x)
            identity = self.downsample1(y)

        out += identity
        out = self.relu(out)

        return out, mask.view((-1, self.args.num_segments) + mask.shape[1:]), mask2


class BateNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, args=None):
        super(BateNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.args = args

        if self.args.shared_policy_net:
            self.gate_fc0s = nn.ModuleDict()
            self.gate_fc1s = nn.ModuleDict()
        else:
            self.gate_fc0s = None
            self.gate_fc1s = None

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64 * 1, layers[0], layer_offset=0)
        self.layer2 = self._make_layer(block, 64 * 2, layers[1],
                                       stride=2, dilate=replace_stride_with_dilation[0], layer_offset=layers[0])
        self.layer3 = self._make_layer(block, 64 * 4, layers[2],
                                       stride=2, dilate=replace_stride_with_dilation[1], layer_offset=layers[0] + layers[1])
        self.layer4 = self._make_layer(block, 64 * 8, layers[3],
                                       stride=2, dilate=replace_stride_with_dilation[2], layer_offset=layers[0] + layers[1] + layers[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if self.args.policy_attention:
            self.attention_fc0 = nn.Linear(16*3, 16)
            self.attention_relu = nn.ReLU(inplace=True)
            self.attention_bn = nn.BatchNorm1d(16)
            self.attention_fc1 = nn.Linear(16, 1)
            normal_(self.attention_fc0.weight, 0, 0.001)
            constant_(self.attention_fc0.bias, 0)
            normal_(self.attention_fc1.weight, 0, 0.001)
            constant_(self.attention_fc1.bias, 0)
            nn.init.constant_(self.attention_bn.weight, 1)
            nn.init.constant_(self.attention_bn.bias, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        zero_init_residual = args.zero_init_residual
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    # for bn in m.bn2s:
                    #     nn.init.constant_(bn.weight, 0)
                    nn.init.constant_(m.bn2.weight, 0)

    def update_shared_net(self, in_planes, out_planes):
        in_factor = 1
        out_factor = 2
        if self.args.gate_history:
            in_factor = 2
            if not self.args.gate_no_skipping:
                out_factor = 3
        if self.args.relative_hidden_size > 0:
            hidden_dim = int(self.args.relative_hidden_size * out_planes // self.args.granularity)
        elif self.args.hidden_quota > 0:
            hidden_dim = self.args.hidden_quota // (out_planes // self.args.granularity)
        else:
            hidden_dim = self.args.gate_hidden_dim
        in_dim = in_planes * in_factor
        out_dim = out_planes * out_factor // self.args.granularity
        out_dim = out_dim // 64 * (self.args.gate_channel_ends - self.args.gate_channel_starts)
        keyword = "%d_%d" % (in_dim, out_dim)
        if keyword not in self.gate_fc0s:
            self.gate_fc0s[keyword] = nn.Linear(in_dim, hidden_dim)
            self.gate_fc1s[keyword] = nn.Linear(hidden_dim, out_dim)
            normal_(self.gate_fc0s[keyword].weight, 0, 0.001)
            constant_(self.gate_fc0s[keyword].bias, 0)
            normal_(self.gate_fc1s[keyword].weight, 0, 0.001)
            constant_(self.gate_fc1s[keyword].bias, 0)

    def _make_layer(self, block, planes_list_0, blocks, stride=1, dilate=False, layer_offset=-1):
        norm_layer = self._norm_layer
        downsample0 = None
        downsample1 = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes_list_0 * block.expansion:
            downsample0 = conv1x1(self.inplanes, planes_list_0 * block.expansion, stride)
            downsample1 = norm_layer(planes_list_0 * block.expansion)

        _d={1:0, 2:1, 4:2, 8:3}
        layer_idx = _d[planes_list_0//64]

        if len(self.args.enabled_layers) > 0:
            enable_policy = layer_offset in self.args.enabled_layers
            print("stage-%d layer-%d (abs: %d) enabled:%s"%(layer_idx, 0, layer_offset, enable_policy))
        elif len(self.args.enabled_stages) > 0:
            enable_policy = layer_idx in self.args.enabled_stages
        else:
            enable_policy = (layer_idx >= self.args.enable_from and layer_idx < self.args.disable_from)

        if self.args.shared_policy_net and enable_policy:
            self.update_shared_net(self.inplanes, planes_list_0)

        layers = nn.ModuleList()
        layers.append(block(self.inplanes, planes_list_0, stride, downsample0, downsample1, self.groups,
                            self.base_width, previous_dilation, norm_layer, shared=(self.gate_fc0s, self.gate_fc1s), shall_enable=enable_policy, args=self.args))
        self.inplanes = planes_list_0 * block.expansion
        for k in range(1, blocks):

            if len(self.args.enabled_layers) > 0:
                enable_policy = layer_offset + k in self.args.enabled_layers
                print("stage-%d layer-%d (abs: %d) enabled:%s" % (layer_idx, k, layer_offset + k, enable_policy))

            if self.args.shared_policy_net and enable_policy:
                self.update_shared_net(self.inplanes, planes_list_0)

            layers.append(block(self.inplanes, planes_list_0, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, shared=(self.gate_fc0s, self.gate_fc1s), shall_enable=enable_policy, args=self.args))
        return layers

    def count_flops(self, input_data_shape, **kwargs):
        flops_list = []
        _B, _T, _C, _H, _W = input_data_shape
        input2d_shape = _B*_T, _C, _H, _W

        flops_conv1, data_shape = count_conv2d_flops(input2d_shape, self.conv1)
        data_shape = data_shape[0], data_shape[1], data_shape[2]//2, data_shape[3]//2 #TODO pooling
        for li, layers in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            for bi, block in enumerate(layers):
                flops, data_shape = block.count_flops(data_shape, **kwargs)
                flops_list.append(flops)
        return flops_list

    def forward(self, input_data, **kwargs):
        # TODO x.shape (nt, c, h, w)
        if "tau" not in kwargs:
            kwargs["tau"] = 1
            kwargs["inline_test"] = True

        mask_stack_list = []  # TODO list for t-dimension
        for _, layers in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            for _, block in enumerate(layers):
                mask_stack_list.append(None)
                if self.args.dense_in_block:
                    mask_stack_list.append(None)

        x = self.conv1(input_data)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        idx = 0
        for li, layers in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            for bi, block in enumerate(layers):
                x, mask, mask2 = block(x, **kwargs)
                mask_stack_list[idx] = mask
                idx += 1
                if self.args.dense_in_block:
                    mask_stack_list[idx] = mask2
                    idx += 1

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)

        if self.args.policy_attention:
            mask_stat_list=[]
            for mask_stack in mask_stack_list:
                mask_stat = torch.stack([torch.mean(mask_stack[:, :, :, act_i], dim=-1) for act_i in range(3)], dim=-1)
                mask_stat_list.append(mask_stat)
            mask_stat_tensor = torch.stack(mask_stat_list, dim=-2)
            att_input = mask_stat_tensor.view(x.shape[0], len(mask_stack_list) * 3)
            att_mid = self.attention_fc0(att_input)
            att_mid = self.attention_bn(att_mid)
            att_mid = self.attention_fc1(att_mid)
            attention = torch.softmax(att_mid.view(-1, self.args.num_segments), dim=-1).clamp(min=1e-8)

            return out, mask_stack_list, attention
        else:
            return out, mask_stack_list



def _batenet(arch, block, layers, pretrained, progress, **kwargs):
    model = BateNet(block, layers, **kwargs)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls[arch],
                                                   progress=progress)
        # TODO okay now let's load ResNet to DResNet
        model_dict = model.state_dict()
        kvs_to_add = []
        old_to_new_pairs = []
        keys_to_delete = []
        for k in pretrained_dict:
            # TODO layer4.0.downsample.X.weight -> layer4.0.downsampleX.weight
            if "downsample.0" in k:
                old_to_new_pairs.append((k, k.replace("downsample.0", "downsample0")))
            elif "downsample.1" in k:
                old_to_new_pairs.append((k, k.replace("downsample.1", "downsample1")))

        for del_key in keys_to_delete:
            del pretrained_dict[del_key]

        for new_k, new_v in kvs_to_add:
            pretrained_dict[new_k] = new_v

        for old_key, new_key in old_to_new_pairs:
            pretrained_dict[new_key] = pretrained_dict.pop(old_key)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def batenet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _batenet('batenet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                    **kwargs)


def batenet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _batenet('batenet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                    **kwargs)


def batenet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _batenet('batenet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                    **kwargs)


def batenet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _batenet('batenet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                    **kwargs)


def batenet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _batenet('batenet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                    **kwargs)
