import os
import time
import numpy as np

import torch
from torch import sigmoid
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
from torch.autograd import Variable

import sys
sys.path.append("../")

# Local imports
import utils.utils as utils
from utils.utils import AverageMeter

def train(args, train_loader, model, optimizer, epoch, curr_lr, win_feats5, win_fusion, viz, global_step, accumulation_steps, loss_fn=None, scaler=None, use_fp16=False, teacher=None, alpha=0.7, temperature=3.0):
    if loss_fn is None:
        loss_fn = WeightedMultiLabelSigmoidLoss

    batch_time = AverageMeter()
    data_time = AverageMeter()
    feats5_losses = AverageMeter()
    fusion_losses = AverageMeter()
    distill_losses = AverageMeter()
    total_losses = AverageMeter()

    # switch to eval mode to make BN unchanged.
    model.train()
    optimizer.zero_grad()

    end = time.time()
    for i, (img, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # Input for Image CNN.
        img_var = utils.check_gpu(0, img) # BS X 3 X H X W
        target_var = utils.check_gpu(0, target) # BS X H X W X NUM_CLASSES

        bs = img.size()[0] * accumulation_steps

        # Forward pass in FP16 if enabled
        with torch.amp.autocast('cuda', enabled=use_fp16):
            score_feats5, fused_feats = model(img_var) # BS X NUM_CLASSES X 472 X 472

        # Hard loss computed in FP32 for numerical stability
        feats5_loss = loss_fn(score_feats5.float(), target_var)
        fused_feats_loss = loss_fn(fused_feats.float(), target_var)
        hard_loss = feats5_loss + fused_feats_loss

        # Knowledge distillation loss
        if teacher is not None:
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=use_fp16):
                    teacher_feats5, teacher_fused = teacher(img_var)
                # Safety guard for spatial mismatch
                if teacher_feats5.shape != score_feats5.shape:
                    teacher_feats5 = F.interpolate(teacher_feats5.float(), score_feats5.shape[2:], mode='bilinear', align_corners=False)
                    teacher_fused = F.interpolate(teacher_fused.float(), fused_feats.shape[2:], mode='bilinear', align_corners=False)
                soft_target_feats5 = torch.sigmoid(teacher_feats5.float() / temperature)
                soft_target_fused = torch.sigmoid(teacher_fused.float() / temperature)

            # Binary KL divergence via BCE: equivalent to KL(teacher || student) + const
            distill_feats5 = F.binary_cross_entropy_with_logits(
                score_feats5.float() / temperature, soft_target_feats5)
            distill_fused = F.binary_cross_entropy_with_logits(
                fused_feats.float() / temperature, soft_target_fused)
            distill_loss = (distill_feats5 + distill_fused) * (temperature ** 2)

            loss = alpha * hard_loss + (1 - alpha) * distill_loss
        else:
            distill_loss = None
            loss = hard_loss

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        del img_var
        del target_var
        del score_feats5
        if teacher is not None:
            del teacher_feats5, teacher_fused, soft_target_feats5, soft_target_fused

        # increase batch size by factor of accumulation steps (Gradient accumulation) for training with limited memory
        if (i+1) % accumulation_steps == 0:
            feats5_losses.update(feats5_loss.data, bs)
            fusion_losses.update(fused_feats_loss.data, bs)
            if distill_loss is not None:
                distill_losses.update(distill_loss.data, bs)
            total_losses.update(loss.data, bs)

            if viz is not None:
                trn_feats5_loss = feats5_loss.clone().cpu().data.numpy()
                trn_fusion_loss = fused_feats_loss.clone().cpu().data.numpy()
                viz.line(win=win_feats5, name='train_feats5', update='append', X=np.array([global_step]), Y=np.array([trn_feats5_loss]))
                viz.line(win=win_fusion, name='train_fusion', update='append', X=np.array([global_step]), Y=np.array([trn_fusion_loss]))

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if ((i+1) % args.print_freq == 0):
                print("\n\n")
                distill_str = ''
                if teacher is not None:
                    distill_str = 'Distill Loss {distill_loss.val:.11f} ({distill_loss.avg:.11f})\t'.format(
                        distill_loss=distill_losses)
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Total Loss {total_loss.val:.11f} ({total_loss.avg:.11f})\t'
                      '{distill_str}\n'
                      'lr {learning_rate:.10f}\t'
                      .format(epoch, int((i+1)/accumulation_steps), int(len(train_loader)/accumulation_steps), batch_time=batch_time,
                       data_time=data_time, total_loss=total_losses, distill_str=distill_str,
                       learning_rate=curr_lr))

    del feats5_loss
    del fused_feats_loss
    del feats5_losses
    del fusion_losses
    del distill_losses
    del total_losses
    # torch.cuda.empty_cache()
    return global_step

def validate(args, val_loader, model, epoch, win_feats5, win_fusion, viz, global_step, loss_fn=None, use_fp16=False):
    if loss_fn is None:
        loss_fn = WeightedMultiLabelSigmoidLoss

    batch_time = AverageMeter()
    data_time = AverageMeter()
    feats5_losses = AverageMeter()
    fusion_losses = AverageMeter()
    total_losses = AverageMeter()

    # switch to eval mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (img, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # Input for Image CNN.
            img_var = utils.check_gpu(0, img) # BS X 3 X H X W
            target_var = utils.check_gpu(0, target) # BS X H X W X NUM_CLASSES

            bs = img.size()[0]

            with torch.amp.autocast('cuda', enabled=use_fp16):
                score_feats5, fused_feats = model(img_var) # BS X NUM_CLASSES X 472 X 472

            # Loss in FP32 for numerical stability
            feats5_loss = loss_fn(score_feats5.float(), target_var)
            fused_feats_loss = loss_fn(fused_feats.float(), target_var)
            loss = feats5_loss + fused_feats_loss

            feats5_losses.update(feats5_loss.data, bs)
            fusion_losses.update(fused_feats_loss.data, bs)
            total_losses.update(loss.data, bs)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            del img_var
            del target_var
            del score_feats5
            del fused_feats_loss
            del feats5_loss

            if (i % args.print_freq == 0):
                print("\n\n")
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Total Loss {total_loss.val:.4f} ({total_loss.avg:.4f})\n'
                      .format(epoch, i, len(val_loader), batch_time=batch_time,
                       data_time=data_time, total_loss=total_losses))

    if viz is not None:
        viz.line(win=win_feats5, name='val_feats5', update='append', X=np.array([global_step]), Y=np.array([feats5_losses.avg.cpu()]))
        viz.line(win=win_fusion, name='val_fusion', update='append', X=np.array([global_step]), Y=np.array([fusion_losses.avg.cpu()]))

    return fusion_losses.avg

def WeightedMultiLabelSigmoidLoss(model_output, target):
    """
    model_output: BS X NUM_CLASSES X H X W
    target: BS X H X W X NUM_CLASSES
    """
    # Calculate weight. (edge pixel and non-edge pixel)
    weight_sum = utils.check_gpu(0, target.sum(dim=1).sum(dim=1).sum(dim=1).float().data) # BS
    edge_weight = utils.check_gpu(0, weight_sum.data / float(target.size()[1]*target.size()[2]))
    non_edge_weight = utils.check_gpu(0, (target.size()[1]*target.size()[2]-weight_sum.data) / float(target.size()[1]*target.size()[2]))
    one_sigmoid_out = sigmoid(model_output)
    zero_sigmoid_out = 1 - one_sigmoid_out
    target = target.transpose(1,3).transpose(2,3).float() # BS X NUM_CLASSES X H X W
    loss = -non_edge_weight.unsqueeze(1).unsqueeze(2).unsqueeze(3)*target*torch.log(one_sigmoid_out.clamp(min=1e-10)) - \
            edge_weight.unsqueeze(1).unsqueeze(2).unsqueeze(3)*(1-target)*torch.log(zero_sigmoid_out.clamp(min=1e-10))

    return loss.mean(dim=0).sum()

def WeightedMultiLabelFocalLoss(model_output, target, gamma=2.0):
    """
    Hybrid focal loss: per-sample adaptive weights + focal modulation.
    model_output: BS X NUM_CLASSES X H X W
    target: BS X H X W X NUM_CLASSES
    gamma: focusing parameter — down-weights easy examples
    """
    # Per-sample adaptive weights (from original loss)
    weight_sum = utils.check_gpu(0, target.sum(dim=1).sum(dim=1).sum(dim=1).float().data) # BS
    edge_weight = utils.check_gpu(0, weight_sum.data / float(target.size()[1]*target.size()[2]))
    non_edge_weight = utils.check_gpu(0, (target.size()[1]*target.size()[2]-weight_sum.data) / float(target.size()[1]*target.size()[2]))

    one_sigmoid_out = sigmoid(model_output)
    zero_sigmoid_out = 1 - one_sigmoid_out
    target = target.transpose(1,3).transpose(2,3).float() # BS X NUM_CLASSES X H X W

    # Focal modulating factors
    pos_focal_weight = zero_sigmoid_out ** gamma   # hard positives get higher weight
    neg_focal_weight = one_sigmoid_out ** gamma     # hard negatives get higher weight

    loss = -non_edge_weight.unsqueeze(1).unsqueeze(2).unsqueeze(3) * pos_focal_weight * target * torch.log(one_sigmoid_out.clamp(min=1e-10)) - \
           edge_weight.unsqueeze(1).unsqueeze(2).unsqueeze(3) * neg_focal_weight * (1 - target) * torch.log(zero_sigmoid_out.clamp(min=1e-10))

    return loss.mean(dim=0).sum()
