import numpy as np
import torch
import torch.nn.functional as F
from ops.net_flops_table import get_gflops_params

def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_multi_hot(test_y, classes, assumes_starts_zero=True):
    bs = test_y.shape[0]
    label_cnt = 0

    if not assumes_starts_zero:
        for label_val in torch.unique(test_y):
            if label_val >= 0:
                test_y[test_y == label_val] = label_cnt
                label_cnt += 1

    gt = torch.zeros(bs, classes + 1)
    for i in range(test_y.shape[1]):
        gt[torch.LongTensor(range(bs)), test_y[:, i]] = 1

    return gt[:, :classes]

def cal_map(output, old_test_y):
    batch_size = output.size(0)
    num_classes = output.size(1)
    ap = torch.zeros(num_classes)
    test_y = old_test_y.clone()

    gt = get_multi_hot(test_y, num_classes, False)

    probs = F.softmax(output, dim=1)

    rg = torch.range(1, batch_size).float()
    for k in range(num_classes):
        scores = probs[:, k]
        targets = gt[:, k]
        _, sortind = torch.sort(scores, 0, True)
        truth = targets[sortind]
        tp = truth.float().cumsum(0)
        precision = tp.div(rg)
        ap[k] = precision[truth.byte()].sum() / max(float(truth.sum()), 1)
    return ap.mean()*100, ap*100


class Recorder:
    def __init__(self, larger_is_better=True):
        self.history=[]
        self.larger_is_better=larger_is_better
        self.best_at=None
        self.best_val=None

    def is_better_than(self, x, y):
        if self.larger_is_better:
            return x>y
        else:
            return x<y

    def update(self, val):
        self.history.append(val)
        if len(self.history)==1 or self.is_better_than(val, self.best_val):
            self.best_val = val
            self.best_at = len(self.history)-1

    def is_current_best(self):
        return self.best_at == len(self.history)-1

    def at(self, idx):
        return self.history[idx]

def adjust_learning_rate(optimizer, epoch, length, iteration, lr_type, lr_steps, args):
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    elif lr_type == 'linear':
        factor = min(1.0, (epoch * length + iteration + 1)/(args.warmup_epochs * length))
        lr = args.lr * factor
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        if lr_type != 'linear':
            param_group['weight_decay'] = decay * param_group['decay_mult']


def count_conv2d_flops(input_data_shape, conv):
    n, c_in, h_in, w_in = input_data_shape
    h_out = (h_in + 2 * conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) // conv.stride[0] + 1
    w_out = (w_in + 2 * conv.padding[1] - conv.dilation[1] * (conv.kernel_size[1] - 1) - 1) // conv.stride[1] + 1
    c_out = conv.out_channels
    bias = 1 if conv.bias is not None else 0
    flops = n * c_out * h_out * w_out * (c_in // conv.groups * conv.kernel_size[0] * conv.kernel_size[1] + bias)
    return flops, (n, c_out, h_out, w_out)


def product(tuple1):
    """Calculates the product of a tuple"""
    prod = 1
    for x in tuple1:
        prod = prod * x
    return prod

def count_bn_flops(input_data_shape):
    flops = product(input_data_shape) * 2
    output_data_shape = input_data_shape
    return flops, output_data_shape


def count_relu_flops(input_data_shape):
    flops = product(input_data_shape) * 1
    output_data_shape = input_data_shape
    return flops, output_data_shape


def count_fc_flops(input_data_shape, fc):
    output_data_shape = input_data_shape[:-1] + (fc.out_features, )
    flops = product(output_data_shape) * (fc.in_features + 1)
    return flops, output_data_shape

def init_gflops_table(model, args):
    if "cgnet" in args.arch:
        base_model_gflops = 1.8188 if "net18" in args.arch else 4.28
        params = get_gflops_params(args.arch, args.reso_list[0], args.num_class, -1, args=args)[1]
    else:
        base_model_gflops, params = get_gflops_params(args.arch, args.reso_list[0], args.num_class, -1, args=args)

    if args.ada_reso_skip:
        gflops_list = model.base_model.count_flops((1, 1, 3, args.reso_list[0], args.reso_list[0]))
        if "AdaBNInc" in args.arch:
            gflops_list, g_meta = gflops_list
        else:
            g_meta = None
        print("Network@%d (%.4f GFLOPS, %.4f M params) has %d blocks" % (args.reso_list[0], base_model_gflops, params, len(gflops_list)))
        for i, block in enumerate(gflops_list):
            print("block", i, ",".join(["%.4f GFLOPS" % (x / 1e9) for x in block]))
        return base_model_gflops, gflops_list, g_meta
    else:
        print("Network@%d (%.4f GFLOPS, %.4f M params)" % (args.reso_list[0], base_model_gflops, params))
        return base_model_gflops, None, None

def compute_gflops_by_mask(mask_tensor_list, base_model_gflops, gflops_list, g_meta, args):
    upperbound_gflops = base_model_gflops
    real_gflops = base_model_gflops

    if "bate" in args.arch:
        for m_i, mask in enumerate(mask_tensor_list):
            #compute precise GFLOPS
            upsave = torch.zeros_like(mask[:, :, :, 0]) # B*T*C*K->B*T*C
            for t in range(mask.shape[1]-1):
                if args.gate_history:
                    upsave[:, t] = (1 - mask[:, t, :, -1]) * (1 - mask[:, t + 1, :, -2])
                else:
                    upsave[:, t] = 1 - mask[:, t, :, -1] # since no reusing, as long as not keeping, save from upstream conv
            upsave[:, -1] = 1 - mask[:, t, :, -1]
            upsave = torch.mean(upsave)

            if args.gate_no_skipping: # downstream conv gflops' saving is from skippings
                downsave = upsave * 0
            else:
                downsave = torch.mean(mask[:, :, :, 0])

            conv_offset = 0
            real_count = 1.
            if args.dense_in_block:
                layer_i = m_i // 2  # because we have twice masks as the #(blocks)
                if m_i % 2 == 1:  # means we come to the second mask in the block
                    if "net50" in args.arch or "net101" in args.arch:  # because we have 3 convs in BottleNeck
                        conv_offset = 1
                    else:  # because we can't compute flops saving among blocks (due to residual op), so we skip this (as this is the case only in BasicBlock)
                        real_count = 0
            else:
                layer_i = m_i
            up_flops = gflops_list[layer_i][0 + conv_offset] / 1e9
            down_flops = gflops_list[layer_i][1 + conv_offset] * real_count / 1e9
            embed_conv_flops = gflops_list[layer_i][-1] * real_count / 1e9

            upperbound_gflops = upperbound_gflops - downsave * (down_flops - embed_conv_flops) # in worst case, we only compute saving from downstream conv
            real_gflops = real_gflops - upsave * up_flops - downsave * (down_flops - embed_conv_flops)
    elif "AdaBNInc" in args.arch:
        for m_i, mask in enumerate(mask_tensor_list):
            upsave = torch.zeros_like(mask[:, :, :, 0])  # B*T*C*K->B*T*C
            for t in range(mask.shape[1]-1):
                if args.gate_history:
                    upsave[:, t] = (1 - mask[:, t, :, -1]) * (1 - mask[:, t + 1, :, -2])
                else:
                    upsave[:, t] = 1 - mask[:, t, :, -1] # since no reusing, as long as not keeping, save from upstream conv
            upsave[:, -1] = 1 - mask[:, t, :, -1]
            upsave = torch.mean(upsave, dim=[0,1]) # -> C

            if len(gflops_list[m_i]) == 7:
                _a,_b,_c,_d = g_meta[m_i]
                upsaves = [torch.mean(upsave[:_a]),
                           torch.mean(upsave[_a:_a + _b]),
                           torch.mean(upsave[_a + _b:_a + _b + _c]),
                           torch.mean(upsave[_a + _b + _c:])]
                out_corr_list = [0, 2, 5, 6]  # to the id of last convs in each partition
                if m_i < len(gflops_list)-1 and len(gflops_list[m_i+1]) == 5:
                    next_in_corr_list = [0, 2]
                else:
                    next_in_corr_list = [0, 1, 3, 6]
            elif len(gflops_list[m_i]) == 5:
                _a, _b, _c= g_meta[m_i]
                upsaves = [torch.mean(upsave[:_a]),
                            torch.mean(upsave[_a:_a+_b]),
                            torch.mean(upsave[_a+_b:])]
                out_corr_list = [1, 4]  # to the id of last convs in each partition
                next_in_corr_list = [0, 1, 3, 6]
            up_flops_save = sum([upsaves[f_i] * gflops_list[m_i][out_corr_list[f_i]] for f_i in range(len(out_corr_list))]) / 1e9
            if args.gate_no_skipping: # downstream conv gflops' saving is from skippings
                downsave = upsaves[0] * 0
            else:
                downsave = torch.mean(mask[:, :, :, 0])
            down_flops_save = up_flops_save * 0
            if m_i < len(mask_tensor_list)-1:
                # to the id of first convs in each partition in the next layer
                down_flops_save = downsave * sum([gflops_list[m_i+1][next_in_corr_list[f_i]] for f_i in range(len(next_in_corr_list))]) / 1e9
            upperbound_gflops = upperbound_gflops - down_flops_save
            real_gflops = real_gflops - up_flops_save - down_flops_save
    else:
        # s0 for sparsity savings
        # s1 for history
        s0 = [1 - 1.0 * torch.sum(mask[:, :, 1]) / torch.sum(mask[:, :, 0]) for mask in mask_tensor_list]

        if args.dense_in_block:
            savings0 = sum([s0[i*2] * gflops_list[i][0] * (1 - 1.0 / args.partitions) for i in range(len(gflops_list))])
            savings1 = sum([s0[i*2+1] * gflops_list[i][1] * (1 - 1.0 / args.partitions) for i in range(len(gflops_list))])
            savings = savings0 + savings1
        else:
            savings = sum([s0[i] * gflops_list[i][0] * (1 - 1.0 / args.partitions) for i in range(len(gflops_list))])
        real_gflops = base_model_gflops - savings / 1e9
        upperbound_gflops = real_gflops

    return upperbound_gflops, real_gflops