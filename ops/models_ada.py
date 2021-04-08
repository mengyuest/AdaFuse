from torch import nn
from ops.transforms import *
from torch.nn.init import normal_, constant_


def init_hidden(batch_size, cell_size):
    init_cell = torch.Tensor(batch_size, cell_size).zero_()
    if torch.cuda.is_available():
        init_cell = init_cell.cuda()
    return init_cell


class TSN_Ada(nn.Module):
    def __init__(self, args):
        super(TSN_Ada, self).__init__()
        self.reshape = True
        self.dropout = args.dropout
        self.pretrain = args.pretrain

        self.is_shift = args.shift
        self.shift_div = args.shift_div
        self.shift_place = args.shift_place
        self.fc_lr5 = False
        self.temporal_pool = False

        self.args = args
        self.rescale_to=args.rescale_to

        self.base_model_name = args.arch

        self._prepare_base_model(self.base_model_name)
        self._prepare_fc(self.args.num_class)

        self._enable_pbn = not args.no_partialbn
        if not args.no_partialbn:
            self.partialBN(True)

    def _prep_a_net(self, model_name, shall_pretrain):
        if "BNInception" in self.args.arch:
            from bn_archs.bn_inception import bninception
            model = bninception(args=self.args)
        else:
            model = getattr(torchvision.models, model_name)(shall_pretrain)
        model.last_layer_name = 'fc'
        return model

    def _make_a_shift(self, base_model):
        print('Adding temporal shift...')
        from ops.temporal_shift import make_temporal_shift
        make_temporal_shift(self.base_model, self.args.num_segments,
                            n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

    def _prepare_base_model(self, base_model):
        self.base_model = self._prep_a_net(base_model, self.pretrain == 'imagenet')

        self.input_size = 224
        if "BNInception" in self.args.arch:
            self.input_mean = self.base_model.mean
            self.input_std = self.base_model.std
        else:
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

        if self.is_shift:
            self._make_a_shift(base_model)

    def _prepare_fc(self, num_class):
        def make_a_linear(input_dim, output_dim):
            linear_model = nn.Linear(input_dim, output_dim)
            normal_(linear_model.weight, 0, 0.001)
            constant_(linear_model.bias, 0)
            return linear_model

        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
        self.new_fc = make_a_linear(feature_dim, num_class)


    def forward(self, *argv, **kwargs):
        input_data = kwargs["input"][0]
        _b, _tc, _h, _w = input_data.shape
        _t, _c = _tc // 3, 3
        input_2d = input_data.view(_b * _t, _c, _h, _w)
        feat = self.base_model(input_2d)
        base_out = self.new_fc(feat).view(_b, _t, -1)
        output = base_out.mean(dim=1).squeeze(1)  # self.consensus(base_out)
        return output, None, None, None

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        if flip:
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        else:
            print('#' * 20, 'NO FLIP!!!')
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN_Ada, self).train(mode)
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            count = 0
            bn_scale = len(self.args.num_filters_list)
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m,nn.BatchNorm3d):
                    count += 1
                    if count >= (2*bn_scale if self._enable_pbn else bn_scale):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        if mode and len(self.args.frozen_layers) > 0 and self.args.freeze_corr_bn:
            for layer_idx in self.args.frozen_layers:
                for km in self.base_model.named_modules():
                    k, m = km
                    if layer_idx == 0:
                        if "bn1" in k and "layer" not in k and (
                                isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d)):
                            m.eval()
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False
                    else:
                        if "layer%d" % (layer_idx) in k and (
                                isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d)):
                            m.eval()
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))

            elif isinstance(m, torch.nn.LSTMCell):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                normal_weight.append(ps[1])
                normal_bias.append(ps[2])
                normal_bias.append(ps[3])

            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_ops"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},
        ]