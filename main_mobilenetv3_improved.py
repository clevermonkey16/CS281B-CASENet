import os
import argparse
import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable

# Local imports
import utils.utils as utils

# For model
from modules.CASENet import CASENet_mobilenetv3

# For training and validation
import train_val.model_play_improved as model_play

# For visualization
import visdom

# For settings
import config

args = config.get_args()

# Additional args for improvements (parsed separately to avoid modifying shared config.py)
improvement_parser = argparse.ArgumentParser(parents=[], add_help=False)
improvement_parser.add_argument('--focal-loss', action='store_true', help='Use focal loss instead of weighted BCE')
improvement_parser.add_argument('--gamma', default=2.0, type=float, help='Focal loss gamma (default: 2.0)')
improvement_parser.add_argument('--alpha', default=0.75, type=float, help='Focal loss alpha for positive class (default: 0.75)')
improvement_parser.add_argument('--augmentation', action='store_true', help='Use enhanced data augmentation (ColorJitter, GaussianBlur, GaussianNoise, RandomRotation)')
improvement_parser.add_argument('--random-erasing', action='store_true', help='Use RandomErasing augmentation (can combine with --augmentation)')
improvement_parser.add_argument('--fp16', action='store_true', help='Use FP16 mixed precision training')
imp_args, _ = improvement_parser.parse_known_args()

# Build experiment suffix from active flags
suffix = ""
if imp_args.focal_loss:
    suffix += "_focal"
if imp_args.augmentation:
    suffix += "_aug"
if imp_args.random_erasing:
    suffix += "_erase"
if imp_args.fp16:
    suffix += "_fp16"

viz = visdom.Visdom(env=f'CASENet-MobileNetV3{suffix}')

def main():
    global args, imp_args, suffix
    print("config:{0}".format(args))
    print("improvements: focal_loss={0} gamma={1} alpha={2} augmentation={3} random_erasing={4} fp16={5}".format(
        imp_args.focal_loss, imp_args.gamma, imp_args.alpha, imp_args.augmentation, imp_args.random_erasing, imp_args.fp16))

    checkpoint_dir = args.checkpoint_folder + suffix if suffix else args.checkpoint_folder + "_improved"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Select loss function
    if imp_args.focal_loss:
        loss_fn = partial(model_play.WeightedMultiLabelFocalLoss, gamma=imp_args.gamma, alpha=imp_args.alpha)
        print("Using Focal Loss (gamma={0}, alpha={1})".format(imp_args.gamma, imp_args.alpha))
    else:
        loss_fn = model_play.WeightedMultiLabelSigmoidLoss
        print("Using Weighted Multi-Label Sigmoid Loss (baseline)")

    # FP16 mixed precision
    scaler = torch.cuda.amp.GradScaler() if imp_args.fp16 else None
    if imp_args.fp16:
        print("Using FP16 mixed precision training")

    global_step = 0
    min_val_loss = 999999999

    title = 'train|val loss '
    init = np.nan
    win_feats5 = viz.line(
        X=np.column_stack((np.array([init]), np.array([init]))),
        Y=np.column_stack((np.array([init]), np.array([init]))),
        opts={'title': title, 'xlabel': 'Iter', 'ylabel': 'Loss', 'legend': ['train_feats5', 'val_feats5']},
    )

    win_fusion = viz.line(
        X=np.column_stack((np.array([init]), np.array([init]))),
        Y=np.column_stack((np.array([init]), np.array([init]))),
        opts={'title': title, 'xlabel': 'Iter', 'ylabel': 'Loss', 'legend': ['train_fusion', 'val_fusion']},
    )

    # Select dataloader based on augmentation flags
    if imp_args.augmentation or imp_args.random_erasing:
        import prep_dataset.prep_cityscapes_dataset_augmented as prep_cityscapes_dataset
        train_loader, val_loader = prep_cityscapes_dataset.get_dataloader(
            args, augmentation=imp_args.augmentation, random_erasing=imp_args.random_erasing)
    else:
        import prep_dataset.prep_cityscapes_dataset_mobilenetv3 as prep_cityscapes_dataset
        train_loader, val_loader = prep_cityscapes_dataset.get_dataloader(args)
    model = CASENet_mobilenetv3(pretrained=True, num_classes=args.cls_num)

    if args.multigpu:
        model = torch.nn.DataParallel(model.cuda())
    else:
        model = model.cuda()

    policies = get_model_policy(model) # Set the lr_mult=10 of new layer
    optimizer = torch.optim.SGD(policies, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    cudnn.benchmark = True

    if args.pretrained_model:
        utils.load_pretrained_model(model, args.pretrained_model)

    if args.resume_model:
        checkpoint = torch.load(args.resume_model)
        args.start_epoch = checkpoint['epoch']+1
        min_val_loss = checkpoint['min_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint
        torch.cuda.empty_cache()

    for epoch in range(args.start_epoch, args.epochs):
        torch.cuda.empty_cache()
        curr_lr = utils.adjust_learning_rate(args.lr, args, optimizer, global_step, args.lr_steps)

        global_step = model_play.train(args, train_loader, model, optimizer, epoch, curr_lr,
                                 win_feats5, win_fusion, viz, global_step, args.acc_steps,
                                 loss_fn=loss_fn, scaler=scaler, use_fp16=imp_args.fp16)
        torch.cuda.empty_cache()

        curr_loss = model_play.validate(args, val_loader, model, epoch, win_feats5, win_fusion, viz, global_step,
                                        loss_fn=loss_fn, use_fp16=imp_args.fp16)
        torch.cuda.empty_cache()

        # Always store current model to avoid process crashed by accident.
        utils.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'min_loss': min_val_loss,
        }, epoch, folder=checkpoint_dir, filename=f"curr_checkpoint{suffix}.pth.tar")

        if curr_loss < min_val_loss:
            min_val_loss = curr_loss
            utils.save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'min_loss': min_val_loss,
            }, epoch, folder=checkpoint_dir, filename=f"min_loss_checkpoint{suffix}.pth.tar")
            print("Min loss is {0}, in {1} epoch.".format(min_val_loss, epoch))

def get_model_policy(model):
    score_feats_conv_weight = []
    score_feats_conv_bias = []
    score_param_ids = set()
    for m in model.named_modules():
        if m[0] != '' and m[0] != 'module':
            if ('score' in m[0] or 'fusion' in m[0]) and isinstance(m[1], torch.nn.Conv2d):
                ps = list(m[1].parameters())
                score_feats_conv_weight.append(ps[0])
                score_param_ids.add(id(ps[0]))
                if len(ps) == 2:
                    score_feats_conv_bias.append(ps[1])
                    score_param_ids.add(id(ps[1]))
                print("Totally new layer:{0}".format(m[0]))

    other_pts = [p for p in model.parameters() if id(p) not in score_param_ids]

    return [
            {'params': score_feats_conv_weight, 'lr_mult': 10, 'name': 'score_conv_weight'},
            {'params': score_feats_conv_bias, 'lr_mult': 20, 'name': 'score_conv_bias'},
            {'params': filter(lambda p: p.requires_grad, other_pts), 'lr_mult': 1, 'name': 'other'},
    ]

if __name__ == '__main__':
    main()
