import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
import ops.cg_utils as G


# from ops.utils import count_conv2d_flops

def count_conv2d_flops(input_data_shape, conv):
    n, c_in, h_in, w_in = input_data_shape
    h_out = (h_in + 2 * conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) // conv.stride[0] + 1
    w_out = (w_in + 2 * conv.padding[1] - conv.dilation[1] * (conv.kernel_size[1] - 1) - 1) // conv.stride[1] + 1
    c_out = conv.out_channels
    bias = 1 if conv.bias is not None else 0
    flops = n * c_out * h_out * w_out * (c_in // conv.groups * conv.kernel_size[0] * conv.kernel_size[1] + bias)
    return flops, (n, c_out, h_out, w_out)


__all__ = ['CGNet', 'cgnet18', 'cgnet50',
           # 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           # 'wide_resnet50_2', 'wide_resnet101_2'
           ]


model_urls = {
    'cgnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'cgnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    # 'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    # 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    # 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    # 'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    # 'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    # 'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    # 'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    # 'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, args=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        # self.conv1 = conv3x3(inplanes, planes, stride)
        self.args = args
        self.conv1 = G.CGConv2dNew(inplanes, planes, kernel_size=3,
                                   stride=stride, padding=1, bias=False,
                                   p=args.partitions, th=args.ginit, alpha=args.alpha,
                                   use_group=args.use_group, shuffle=args.shuffle, sparse_bp=args.sparse_bp)

        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if args.dense_in_block:
            self.conv2 = G.CGConv2dNew(planes, planes, kernel_size=3,
                          stride=1, padding=1, bias=False,
                          p=args.partitions, th=args.ginit, alpha=args.alpha,
                          use_group=args.use_group, shuffle=args.shuffle, sparse_bp=args.sparse_bp)
        else:
            self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # # shift operations
        # if self.args.shift:
        #     x = shift(x, self.args.num_segments, fold_div=self.args.shift_div, inplace=False)

        out, n_all_pos = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.args.dense_in_block:
            out, n_all_pos2 = self.conv2(out)
        else:
            n_all_pos2 = None
            out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out, n_all_pos, n_all_pos2

    def count_flops(self, input_data_shape, **kwargs):
        conv1_flops, conv1_out_shape = count_conv2d_flops(input_data_shape, self.conv1)
        conv2_flops, conv2_out_shape = count_conv2d_flops(conv1_out_shape, self.conv2)
        if self.downsample is not None:
            downsample_flops, _ = count_conv2d_flops(input_data_shape, self.downsample[0])
        else:
            downsample_flops = 0

        gate_history_conv_flops = 0
        return [conv1_flops, conv2_flops, downsample_flops, gate_history_conv_flops], conv2_out_shape


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, args=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        # self.conv1 = conv1x1(inplanes, width)
        self.args = args
        self.conv1 = G.CGConv2dNew(inplanes, width, kernel_size=1,
                                   stride=1, padding=0, bias=False,
                                   p=args.partitions, th=args.ginit, alpha=args.alpha,
                                   use_group=args.use_group, shuffle=args.shuffle, sparse_bp=args.sparse_bp)

        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # shift operations
        # if self.args.shift:
        #     x = shift(x, self.args.num_segments, fold_div=self.args.shift_div, inplace=False)

        # out = self.conv1(x)
        out, n_all_pos = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out, n_all_pos, None

    def count_flops(self, input_data_shape, **kwargs):
        conv1_flops, conv1_out_shape = count_conv2d_flops(input_data_shape, self.conv1)
        conv2_flops, conv2_out_shape = count_conv2d_flops(conv1_out_shape, self.conv2)
        if self.downsample is not None:
            downsample_flops, _ = count_conv2d_flops(input_data_shape, self.downsample[0])
        else:
            downsample_flops = 0

        gate_history_conv_flops = 0
        return [conv1_flops, conv2_flops, downsample_flops, gate_history_conv_flops], conv2_out_shape


# def shift(x, n_segment, fold_div=3, inplace=False):
#     nt, c, h, w = x.size()
#     print(nt,c,h,w)
#     print(n_segment)
#     print()
#     n_batch = nt // n_segment
#     x = x.view(n_batch, n_segment, c, h, w)
#
#     fold = c // fold_div
#     if inplace:
#         # Due to some out of order error when performing parallel computing.
#         # May need to write a CUDA kernel.
#         raise NotImplementedError
#         # out = InplaceShift.apply(x, fold)
#     else:
#         out = torch.zeros_like(x)
#         out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
#         out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
#         out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
#
#     return out.view(nt, c, h, w)

class CGNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, **kwargs):
        super(CGNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.args = kwargs["args"]

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
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.affine == True:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, args=self.args))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, args=self.args))

        return nn.Sequential(*layers)


    def forward(self, input_data, **kwargs):
        _B, _T, _C, _H, _W = input_data.shape
        out_list = []
        mask_stack_list = []  # TODO list for t-dimension

        for _, layers in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            for _, block in enumerate(layers):
                mask_stack_list.append([])
                if self.args.dense_in_block:
                    mask_stack_list.append([])

        for t in range(_T):
            x = self.conv1(input_data[:, t])
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            idx = 0
            for li, layers in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
                for bi, block in enumerate(layers):
                    if self.args.dense_in_block:
                        x, n_all_pos, n_all_pos2 = block(x, **kwargs)
                        mask_stack_list[idx].append(n_all_pos)
                        mask_stack_list[idx+1].append(n_all_pos2)
                        idx += 2
                    else:
                        x, n_all_pos, n_all_pos2 = block(x, **kwargs)
                        mask_stack_list[idx].append(n_all_pos)
                        idx += 1

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            out = self.fc(x)
            out_list.append(out)
        # print(mask_stack_list)
        return torch.stack(out_list, dim=1), mask_stack_list




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
        # print(list_sum(flops_list)+flops_conv1+512*200)
        # print(flops_list)
        return flops_list

def _cgnet(arch, block, layers, pretrained, progress, **kwargs):
    model = CGNet(block, layers, **kwargs)
    if pretrained:
        # state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        # model.load_state_dict(state_dict)
        pretrained_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model_dict = model.state_dict()

        # different sets
        kvs_to_add = []
        old_to_new_pairs = []
        keys_to_delete = []

        # conditions

        # operations
        for del_key in keys_to_delete:
            del pretrained_dict[del_key]
        for new_k, new_v in kvs_to_add:
            pretrained_dict[new_k] = new_v
        for old_key, new_key in old_to_new_pairs:
            pretrained_dict[new_key] = pretrained_dict.pop(old_key)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def cgnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _cgnet('cgnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def cgnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _cgnet('cgnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)