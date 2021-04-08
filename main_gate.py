import warnings

warnings.filterwarnings("ignore")

import os
import sys
import time
import multiprocessing
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
from ops.dataset import TSNDataSet
from ops.models_gate import TSN_Gate
from ops.models_ada import TSN_Ada
from ops.transforms import *
from ops import dataset_config
from ops.utils import AverageMeter, accuracy, cal_map, Recorder, \
    init_gflops_table, compute_gflops_by_mask, adjust_learning_rate
from opts import parser
from ops.my_logger import Logger
import numpy as np
import common
from os.path import join as ospj
from shutil import copyfile


def main():
    args = parser.parse_args()
    common.set_manual_data_path(args.data_path, args.exps_path)
    test_mode = (args.test_from != "")

    set_random_seed(args.random_seed, args)

    args.num_class, args.train_list, args.val_list, args.root_path, prefix = \
        dataset_config.return_dataset(args.dataset, args.data_path)

    if args.gpus is not None:
        print("Use GPU: {} for training".format(args.gpus))

    logger = Logger()
    sys.stdout = logger

    if args.ada_reso_skip:
        model = TSN_Gate(args=args)
    else:
        model = TSN_Ada(args=args)

    base_model_gflops, gflops_list, g_meta = init_gflops_table(model, args)
    policies = model.get_optim_policies()
    optimizer = torch.optim.SGD(policies, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    if test_mode or args.base_pretrained_from != "":
        the_model_path = args.base_pretrained_from
        if test_mode:
            if "pth.tar" not in args.test_from:
                the_model_path = ospj(args.test_from, "models", "ckpt.best.pth.tar")
            else:
                the_model_path = args.test_from
        the_model_path = common.EXPS_PATH + "/" + the_model_path
        sd = torch.load(the_model_path)['state_dict']
        model_dict = model.state_dict()
        model_dict.update(sd)
        model.load_state_dict(model_dict)

    cudnn.benchmark = True

    train_loader, val_loader = get_data_loaders(model, prefix, args)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    exp_full_path = setup_log_directory(args.exp_header, test_mode, args, logger)
    if not test_mode:
        with open(os.path.join(exp_full_path, 'args.txt'), 'w') as f:
            f.write(str(args))

    map_record, mmap_record, prec_record, prec5_record = get_recorders(4)
    best_train_usage_str = None
    best_val_usage_str = None

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        if not args.skip_training and not test_mode:
            set_random_seed(args.train_random_seed + epoch, args)
            adjust_learning_rate(optimizer, epoch, -1, -1, args.lr_type, args.lr_steps, args)
            train_usage_str = train(train_loader, model, criterion, optimizer, epoch, base_model_gflops, gflops_list,
                                    g_meta, args)
        else:
            train_usage_str = "(Eval mode)"
        torch.cuda.empty_cache()

        # evaluation
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            set_random_seed(args.random_seed, args)
            mAP, mmAP, prec1, prec5, val_usage_str = \
                validate(val_loader, model, criterion, epoch, base_model_gflops, gflops_list, g_meta, exp_full_path,
                         args)

            # remember best prec@1 and save checkpoint
            map_record.update(mAP)
            mmap_record.update(mmAP)
            prec_record.update(prec1)
            prec5_record.update(prec5)

            if prec_record.is_current_best():
                best_train_usage_str = train_usage_str if not args.skip_training else "(Eval Mode)"
                best_val_usage_str = val_usage_str

            print('Best Prec@1: %.3f (epoch=%d) w. Prec@5: %.3f' % (
                prec_record.best_val, prec_record.best_at,
                prec5_record.at(prec_record.best_at)))

            if test_mode or args.skip_training:  # only runs for one epoch
                break
            else:
                saved_things = {'state_dict': model.state_dict()}
                save_checkpoint(saved_things, prec_record.is_current_best(), False, exp_full_path, "ckpt.best")
                save_checkpoint(saved_things, True, False, exp_full_path, "ckpt.latest")

                if epoch in args.backup_epoch_list:
                    save_checkpoint(None, False, True, exp_full_path, str(epoch))
                torch.cuda.empty_cache()

    # after fininshing all the epochs
    if test_mode:
        if args.skip_log == False:
            os.rename(logger._log_path, ospj(logger._log_dir_name, logger._log_file_name[:-4] +
                                             "_mm_%.2f_a_%.2f_f.txt" % (mmap_record.best_val, prec_record.best_val)))
    else:
        if args.ada_reso_skip:
            print("Best train usage:%s\nBest val usage:%s" % (best_train_usage_str, best_val_usage_str))


def build_dataflow(dataset, is_train, batch_size, workers, not_pin_memory):
    workers = min(workers, multiprocessing.cpu_count())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train,
                                              num_workers=workers, pin_memory=not not_pin_memory, sampler=None,
                                              drop_last=is_train)
    return data_loader


def get_data_loaders(model, prefix, args):
    train_transform_flip = torchvision.transforms.Compose([
        model.module.get_augmentation(flip=True),
        Stack(roll=("BNInc" in args.arch)),
        ToTorchFormatTensor(div=("BNInc" not in args.arch)),
        GroupNormalize(model.module.input_mean, model.module.input_std),
    ])

    train_transform_nofl = torchvision.transforms.Compose([
        model.module.get_augmentation(flip=False),
        Stack(roll=("BNInc" in args.arch)),
        ToTorchFormatTensor(div=("BNInc" not in args.arch)),
        GroupNormalize(model.module.input_mean, model.module.input_std),
    ])

    val_transform = torchvision.transforms.Compose([
        GroupScale(int(model.module.scale_size)),
        GroupCenterCrop(model.module.crop_size),
        Stack(roll=("BNInc" in args.arch)),
        ToTorchFormatTensor(div=("BNInc" not in args.arch)),
        GroupNormalize(model.module.input_mean, model.module.input_std),
    ])

    train_dataset = TSNDataSet(args.root_path, args.train_list,
                               num_segments=args.num_segments,
                               image_tmpl=prefix,
                               transform=(train_transform_flip, train_transform_nofl),
                               dense_sample=args.dense_sample,
                               dataset=args.dataset,
                               filelist_suffix=args.filelist_suffix,
                               folder_suffix=args.folder_suffix,
                               save_meta=args.save_meta,
                               always_flip=args.always_flip,
                               conditional_flip=args.conditional_flip,
                               adaptive_flip=args.adaptive_flip)

    val_dataset = TSNDataSet(args.root_path, args.val_list,
                             num_segments=args.num_segments,
                             image_tmpl=prefix,
                             random_shift=False,
                             transform=(val_transform, val_transform),
                             dense_sample=args.dense_sample,
                             dataset=args.dataset,
                             filelist_suffix=args.filelist_suffix,
                             folder_suffix=args.folder_suffix,
                             save_meta=args.save_meta)

    train_loader = build_dataflow(train_dataset, True, args.batch_size, args.workers, args.not_pin_memory)
    val_loader = build_dataflow(val_dataset, False, args.batch_size, args.workers, args.not_pin_memory)

    return train_loader, val_loader


def train(train_loader, model, criterion, optimizer, epoch, base_model_gflops, gflops_list, g_meta, args):
    batch_time, data_time, top1, top5 = get_average_meters(4)
    losses_dict = {}
    if args.ada_reso_skip:

        if "batenet" in args.arch or "AdaBNInc" in args.arch:
            mask_stack_list_list = [0 for _ in gflops_list]
        else:
            mask_stack_list_list = [[] for _ in gflops_list]
        upb_batch_gflops_list = []
        real_batch_gflops_list = []

    tau = args.init_tau

    # switch to train mode
    model.module.partialBN(not args.no_partialbn)
    model.train()

    end = time.time()
    print("#%s# lr:%.6f\ttau:%.4f" % (args.exp_header, optimizer.param_groups[-1]['lr'] * 0.1, tau))

    for i, input_tuple in enumerate(train_loader):
        data_time.update(time.time() - end)
        if args.warmup_epochs > 0:
            adjust_learning_rate(optimizer, epoch, len(train_loader), i, "linear", None, args)

        # input and target
        batchsize = input_tuple[0].size(0)

        input_var_list = [torch.autograd.Variable(input_item).cuda(non_blocking=True) for input_item in
                          input_tuple[:-1]]
        target = input_tuple[-1].cuda(non_blocking=True)

        target_var = torch.autograd.Variable(target)

        # model forward function & measure losses and accuracy
        output, mask_stack_list, _, _ = \
            model(input=input_var_list, tau=tau, is_training=True, curr_step=epoch * len(train_loader) + i)

        if args.ada_reso_skip:
            upb_gflops_tensor, real_gflops_tensor = compute_gflops_by_mask(mask_stack_list, base_model_gflops,
                                                                           gflops_list, g_meta, args)
            loss_dict = compute_losses(criterion, output, target_var, mask_stack_list,
                                       upb_gflops_tensor, real_gflops_tensor, epoch, model,
                                       base_model_gflops, args)
            upb_batch_gflops_list.append(upb_gflops_tensor.detach())
            real_batch_gflops_list.append(real_gflops_tensor.detach())
        else:
            loss_dict = {"loss": criterion(output, target_var[:, 0])}
        prec1, prec5 = accuracy(output.data, target[:, 0], topk=(1, 5))

        # record losses and accuracy
        if len(losses_dict) == 0:
            losses_dict = {loss_name: get_average_meters(1)[0] for loss_name in loss_dict}
        for loss_name in loss_dict:
            losses_dict[loss_name].update(loss_dict[loss_name].item(), batchsize)
        top1.update(prec1.item(), batchsize)
        top5.update(prec5.item(), batchsize)

        # compute gradient and do SGD step
        loss_dict["loss"].backward()

        if args.clip_gradient is not None:
            clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()
        optimizer.zero_grad()

        # gather masks
        if args.ada_reso_skip:
            for layer_i, mask_stack in enumerate(mask_stack_list):
                if "batenet" in args.arch:
                    mask_stack_list_list[layer_i] += torch.sum(mask_stack.detach(), dim=0)
                elif "AdaBNInc" in args.arch:
                    mask_stack_list_list[layer_i] += torch.sum(mask_stack.detach(), dim=0)
                else:  # TODO CGNet
                    mask_stack_list_list[layer_i].append(mask_stack.detach())  # TODO removed cpu()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # logging
        if i % args.print_freq == 0:
            print_output = ('Epoch:[{0:02d}][{1:03d}/{2:03d}] lr {3:.6f} '
                            'Time {batch_time.val:.3f}({batch_time.avg:.3f}) '
                            '{data_time.val:.3f} ({data_time.avg:.3f})\t'
                            'Loss{loss.val:.4f}({loss.avg:.4f}) '
                            'Prec@1 {top1.val:.3f}({top1.avg:.3f}) '
                            'Prec@5 {top5.val:.3f}({top5.avg:.3f})\t'.format(
                epoch, i, len(train_loader), optimizer.param_groups[-1]['lr'] * 0.1, batch_time=batch_time,
                data_time=data_time, loss=losses_dict["loss"], top1=top1, top5=top5))  # TODO

            for loss_name in losses_dict:
                if loss_name == "loss" or "mask" in loss_name:
                    continue
                print_output += ' {header:s} ({loss.avg:.3f})'. \
                    format(header=loss_name[0], loss=losses_dict[loss_name])
            print(print_output)
    if args.ada_reso_skip:
        if "cgnet" in args.arch:
            for layer_i in range(len(mask_stack_list_list)):
                mask_stack_list_list[layer_i] = torch.cat(mask_stack_list_list[layer_i], dim=0)
        upb_batch_gflops = torch.mean(torch.stack(upb_batch_gflops_list))
        real_batch_gflops = torch.mean(torch.stack(real_batch_gflops_list))

    if args.ada_reso_skip:
        usage_str = get_policy_usage_str(upb_batch_gflops, real_batch_gflops)
        print(usage_str)
    else:
        usage_str = "Base Model"
    return usage_str


def validate(val_loader, model, criterion, epoch, base_model_gflops, gflops_list, g_meta, exp_full_path, args):
    batch_time, top1, top5 = get_average_meters(3)
    all_results = []
    all_targets = []

    tau = args.init_tau

    if args.ada_reso_skip:
        if "batenet" in args.arch or "AdaBNInc" in args.arch:
            mask_stack_list_list = [0 for _ in gflops_list]
        else:
            mask_stack_list_list = [[] for _ in gflops_list]
        upb_batch_gflops_list = []
        real_batch_gflops_list = []

    losses_dict = {}

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, input_tuple in enumerate(val_loader):
            # input and target
            batchsize = input_tuple[0].size(0)

            input_data = input_tuple[0].cuda(non_blocking=True)

            target = input_tuple[-1].cuda(non_blocking=True)

            # model forward function
            output, mask_stack_list, _, gate_meta = \
                model(input=[input_data], tau=tau, is_training=False, curr_step=0)

            # measure losses, accuracy and predictions
            if args.ada_reso_skip:
                upb_gflops_tensor, real_gflops_tensor = compute_gflops_by_mask(mask_stack_list, base_model_gflops,
                                                                               gflops_list, g_meta, args)
                loss_dict = compute_losses(criterion, output, target, mask_stack_list,
                                           upb_gflops_tensor, real_gflops_tensor, epoch, model,
                                           base_model_gflops, args)
                upb_batch_gflops_list.append(upb_gflops_tensor)
                real_batch_gflops_list.append(real_gflops_tensor)
            else:
                loss_dict = {"loss": criterion(output, target[:, 0])}

            prec1, prec5 = accuracy(output.data, target[:, 0], topk=(1, 5))

            all_results.append(output)
            all_targets.append(target)

            # record loss and accuracy
            if len(losses_dict) == 0:
                losses_dict = {loss_name: get_average_meters(1)[0] for loss_name in loss_dict}
            for loss_name in loss_dict:
                losses_dict[loss_name].update(loss_dict[loss_name].item(), batchsize)
            top1.update(prec1.item(), batchsize)
            top5.update(prec5.item(), batchsize)

            if args.ada_reso_skip:
                # gather masks
                for layer_i, mask_stack in enumerate(mask_stack_list):
                    if "batenet" in args.arch:
                        mask_stack_list_list[layer_i] += torch.sum(mask_stack.detach(), dim=0)  # TODO remvoed .cpu()
                    elif "AdaBNInc" in args.arch:
                        mask_stack_list_list[layer_i] += torch.sum(mask_stack.detach(), dim=0)  # TODO remvoed .cpu()
                    else:  # TODO CGNet
                        mask_stack_list_list[layer_i].append(mask_stack.detach())  # TODO remvoed .cpu()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print_output = ('Test: [{0:03d}/{1:03d}] '
                                'Time {batch_time.val:.3f}({batch_time.avg:.3f})\t'
                                'Loss{loss.val:.4f}({loss.avg:.4f})'
                                'Prec@1 {top1.val:.3f}({top1.avg:.3f}) '
                                'Prec@5 {top5.val:.3f}({top5.avg:.3f})\t'.
                                format(i, len(val_loader), batch_time=batch_time,
                                       loss=losses_dict["loss"], top1=top1, top5=top5))

                for loss_name in losses_dict:
                    if loss_name == "loss" or "mask" in loss_name:
                        continue
                    print_output += ' {header:s} {loss.val:.3f}({loss.avg:.3f})'. \
                        format(header=loss_name[0], loss=losses_dict[loss_name])
                print(print_output)
    if args.ada_reso_skip:
        if "cgnet" in args.arch:
            for layer_i in range(len(mask_stack_list_list)):
                mask_stack_list_list[layer_i] = torch.cat(mask_stack_list_list[layer_i], dim=0)
        upb_batch_gflops = torch.mean(torch.stack(upb_batch_gflops_list))
        real_batch_gflops = torch.mean(torch.stack(real_batch_gflops_list))

    mAP, _ = cal_map(torch.cat(all_results, 0).cpu(),
                     torch.cat(all_targets, 0)[:, 0:1].cpu())  # single-label mAP
    mmAP, _ = cal_map(torch.cat(all_results, 0).cpu(), torch.cat(all_targets, 0).cpu())  # multi-label mAP

    print('Testing: mAP {mAP:.3f} mmAP {mmAP:.3f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(mAP=mAP, mmAP=mmAP, top1=top1, top5=top5, loss=losses_dict["loss"]))
    if args.ada_reso_skip:
        usage_str = get_policy_usage_str(upb_batch_gflops, real_batch_gflops)
        print(usage_str)
    else:
        usage_str = "Base Model"

    return mAP, mmAP, top1.avg, top5.avg, usage_str


def set_random_seed(the_seed, args):
    np.random.seed(the_seed)
    torch.manual_seed(the_seed)


def compute_losses(criterion, prediction, target, mask_stack_list, upb_gflops_tensor, real_gflops_tensor, epoch_i,
                   model,
                   base_model_gflops, args):
    loss_dict = {}
    if args.gflops_loss_type == "real":
        gflops_tensor = real_gflops_tensor
    else:
        gflops_tensor = upb_gflops_tensor

    # accuracy loss
    acc_loss = criterion(prediction, target[:, 0])
    loss_dict["acc_loss"] = acc_loss
    loss_dict["eff_loss"] = acc_loss * 0

    # gflops loss
    gflops_loss = acc_loss * 0
    if args.gate_gflops_loss_weight > 0 and epoch_i > args.eff_loss_after:
        if args.gflops_loss_norm == 1:
            gflops_loss = torch.abs(gflops_tensor - args.gate_gflops_bias) * args.gate_gflops_loss_weight
        elif args.gflops_loss_norm == 2:
            gflops_loss = ((
                                   gflops_tensor / base_model_gflops - args.gate_gflops_threshold) ** 2) * args.gate_gflops_loss_weight
        loss_dict["gflops_loss"] = gflops_loss
        loss_dict["eff_loss"] += gflops_loss

    # threshold loss for cgnet
    thres_loss = acc_loss * 0
    if "cgnet" in args.arch:
        for name, param in model.named_parameters():
            if 'threshold' in name:
                thres_loss += args.threshold_loss_weight * torch.sum((param - args.gtarget) ** 2)
        loss_dict["thres_loss"] = thres_loss
        loss_dict["eff_loss"] += thres_loss
    loss = acc_loss + gflops_loss + thres_loss
    loss_dict["loss"] = loss

    return loss_dict


def get_policy_usage_str(upb_gflops, real_gflops):
    return "Equivalent GFLOPS: upb: %.4f   real: %.4f" % (upb_gflops.item(), real_gflops.item())


def get_recorders(number):
    return [Recorder() for _ in range(number)]


def get_average_meters(number):
    return [AverageMeter() for _ in range(number)]


def save_checkpoint(state, is_best, shall_backup, exp_full_path, decorator):
    if is_best:
        torch.save(state, '%s/models/%s.pth.tar' % (exp_full_path, decorator))
    if shall_backup:
        copyfile("%s/models/ckpt.best.pth.tar" % exp_full_path,
                 "%s/models/oldbest.%s.pth.tar" % (exp_full_path, decorator))


def setup_log_directory(exp_header, test_mode, args, logger):
    exp_full_name = "g%s_%s" % (logger._timestr, exp_header)
    if test_mode:
        exp_full_path = ospj(common.EXPS_PATH, args.test_from)
    else:
        exp_full_path = ospj(common.EXPS_PATH, exp_full_name)

        os.makedirs(exp_full_path)
        os.makedirs(ospj(exp_full_path, "models"))
    if args.skip_log == False:
        logger.create_log(exp_full_path, test_mode, args.num_segments, args.batch_size, args.top_k)
    return exp_full_path


if __name__ == '__main__':
    t0 = time.time()
    main()
    print("Finished in %.4f seconds\n" % (time.time() - t0))
