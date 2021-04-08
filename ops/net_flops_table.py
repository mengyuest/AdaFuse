import sys
sys.path.insert(0, "../")

import torch
import torchvision
from torch import nn
from thop import profile
import ops.batenet
import ops.cgnet
import ops.cg_utils


feat_dim_dict = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    'batenet18': 512,
    'batenet34': 512,
    'batenet50': 2048,
    'batenet101': 2048,
    'cgnet18': 512,
    'cgnet50': 2048,
    "BNInception": 1024,
    "AdaBNInc": 1024,
    }

def get_gflops_params(model_name, resolution, num_classes, seg_len=-1, pretrained=True,args=None):
    last_layer = "fc"
    if "resnet" in model_name:
        model = getattr(torchvision.models, model_name)(pretrained)
    elif "BNInception" in model_name:
        from bn_archs.bn_inception import bninception
        model = bninception(args=args)
    elif "AdaBNInc" in model_name:
        from bn_archs.bn_inception_ada import bninception_ada
        model = bninception_ada(args=args)
    elif "batenet" in model_name:
        model = getattr(ops.batenet, model_name)(pretrained=False, args=args)
    elif "cgnet" in model_name:
        model = getattr(ops.cgnet, model_name)(pretrained=False, args=args)
    else:
        exit("I don't know what is %s" % model_name)

    setattr(model, last_layer, nn.Linear(feat_dim_dict[model_name], num_classes))

    if seg_len == -1:
        dummy_data = torch.randn(1, 3, resolution, resolution)
        if "batenet" in model_name:
            dummy_data = torch.randn(1 * args.num_segments, 3, resolution, resolution)
        elif "AdaBNInc" in model_name:
            dummy_data = torch.randn(1 * args.num_segments, 3, resolution, resolution)
        elif "cgnet" in model_name:
            dummy_data = torch.randn(1, args.num_segments, 3, resolution, resolution)
    else:
        dummy_data = torch.randn(1, 3, seg_len, resolution, resolution)

    flops, params = profile(model, inputs=(dummy_data,))

    if args.shared_policy_net:
        args.shared_policy_net = False
        if "AdaBNInc" in model_name:
            from bn_archs.bn_inception_ada import bninception_ada
            model = bninception_ada(args=args)
        else:
            model = getattr(ops.batenet, model_name)(pretrained=False, args=args)
        setattr(model, last_layer, nn.Linear(feat_dim_dict[model_name], num_classes))
        flops, _ = profile(model, inputs=(dummy_data,))
        args.shared_policy_net = True
    flops = flops / dummy_data.shape[0]

    return flops / 1e9, params / 1e6


if __name__ == "__main__":
    do = 1