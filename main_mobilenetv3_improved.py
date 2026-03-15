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

# For settings
import config

args = config.get_args()

# Additional args for improvements (parsed separately to avoid modifying shared config.py)
improvement_parser = argparse.ArgumentParser(parents=[], add_help=False)
improvement_parser.add_argument('--focal-loss', action='store_true', help='Use focal loss instead of weighted BCE')
improvement_parser.add_argument('--gamma', default=2.0, type=float, help='Focal loss gamma (default: 2.0)')
improvement_parser.add_argument('--augmentation', action='store_true', help='Use enhanced data augmentation (ColorJitter, GaussianBlur, GaussianNoise, RandomRotation)')
improvement_parser.add_argument('--random-erasing', action='store_true', help='Use RandomErasing augmentation (can combine with --augmentation)')
improvement_parser.add_argument('--fp16', action='store_true', help='Use FP16 mixed precision training')
improvement_parser.add_argument('--visdom', action='store_true', help='Enable Visdom visualization')
improvement_parser.add_argument('--quantize', type=str, default=None, metavar='CHECKPOINT',
                                help='Post-training quantization: provide path to trained checkpoint. Produces FP32, FP16, and INT8 versions.')
improvement_parser.add_argument('--distillation', action='store_true',
                                help='Enable knowledge distillation from ResNet-101 teacher')
improvement_parser.add_argument('--teacher-path', type=str, default='pretrained_models/model_casenet.pth.tar',
                                help='Path to teacher model checkpoint (raw state dict)')
improvement_parser.add_argument('--alpha', type=float, default=0.7,
                                help='Hard loss weight (1-alpha for distillation loss, default: 0.7)')
improvement_parser.add_argument('--temperature', type=float, default=3.0,
                                help='Distillation temperature (default: 3.0)')
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
if imp_args.distillation:
    suffix += "_distill"

def quantize_model(checkpoint_path, num_classes):
    """Post-training quantization: saves FP32, FP16, and INT8 versions of a trained model."""
    import torch.quantization

    output_dir = os.path.dirname(checkpoint_path) or '.'
    base_name = os.path.splitext(os.path.splitext(os.path.basename(checkpoint_path))[0])[0]  # strip .pth.tar

    # Load model on CPU for quantization
    model = CASENet_mobilenetv3(pretrained=False, num_classes=num_classes)
    utils.load_pretrained_model(model, checkpoint_path)
    model.eval()
    model.cpu()

    # 1. FP32 baseline
    fp32_path = os.path.join(output_dir, f"{base_name}_fp32.pth")
    torch.save(model.state_dict(), fp32_path)

    # 2. FP16
    fp16_model = CASENet_mobilenetv3(pretrained=False, num_classes=num_classes)
    fp16_model.load_state_dict(model.state_dict())
    fp16_model.half()
    fp16_path = os.path.join(output_dir, f"{base_name}_fp16.pth")
    torch.save(fp16_model.state_dict(), fp16_path)
    del fp16_model

    # 3. INT8 dynamic quantization
    int8_model = torch.quantization.quantize_dynamic(
        model, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
    )
    int8_path = os.path.join(output_dir, f"{base_name}_int8.pth")
    torch.save(int8_model.state_dict(), int8_path)
    del int8_model

    # Print results table
    fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
    fp16_size = os.path.getsize(fp16_path) / (1024 * 1024)
    int8_size = os.path.getsize(int8_path) / (1024 * 1024)

    print("\nPost-Training Quantization Results:")
    print("-" * 50)
    print(f"{'Format':<10} {'Size (MB)':<15} {'Reduction'}")
    print("-" * 50)
    print(f"{'FP32':<10} {fp32_size:<15.2f} {'1.0x (baseline)'}")
    print(f"{'FP16':<10} {fp16_size:<15.2f} {fp32_size/fp16_size:.1f}x")
    print(f"{'INT8':<10} {int8_size:<15.2f} {fp32_size/int8_size:.1f}x")
    print("-" * 50)
    print(f"Saved to: {output_dir}/")


def main():
    global args, imp_args, suffix
    print("config:{0}".format(args))
    print("improvements: focal_loss={0} gamma={1} augmentation={2} random_erasing={3} fp16={4} visdom={5} distillation={6}".format(
        imp_args.focal_loss, imp_args.gamma, imp_args.augmentation, imp_args.random_erasing, imp_args.fp16, imp_args.visdom, imp_args.distillation))

    # Post-training quantization mode — skip training entirely
    if imp_args.quantize:
        quantize_model(imp_args.quantize, args.cls_num)
        return

    if imp_args.visdom:
        import visdom
        viz = visdom.Visdom(env=f'CASENet-MobileNetV3{suffix}')
    else:
        viz = None

    checkpoint_dir = args.checkpoint_folder + suffix if suffix else args.checkpoint_folder + "_improved"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Select loss function
    if imp_args.focal_loss:
        loss_fn = partial(model_play.WeightedMultiLabelFocalLoss, gamma=imp_args.gamma)
        print("Using Focal Loss (gamma={0}, per-sample adaptive weights)".format(imp_args.gamma))
    else:
        loss_fn = model_play.WeightedMultiLabelSigmoidLoss
        print("Using Weighted Multi-Label Sigmoid Loss (baseline)")

    # FP16 mixed precision
    scaler = torch.amp.GradScaler('cuda') if imp_args.fp16 else None
    if imp_args.fp16:
        print("Using FP16 mixed precision training")

    global_step = 0
    min_val_loss = 999999999

    win_feats5 = None
    win_fusion = None
    if imp_args.visdom:
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

    # Load and freeze teacher for distillation
    teacher = None
    if imp_args.distillation:
        from modules.CASENet import CASENet_resnet101
        teacher = CASENet_resnet101(pretrained=False, num_classes=args.cls_num)
        utils.load_official_pretrained_model(teacher, imp_args.teacher_path)
        teacher = teacher.cuda()
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        if imp_args.fp16:
            teacher = teacher.half()
        if args.multigpu:
            teacher = torch.nn.DataParallel(teacher)
        print("Teacher loaded from {0} (frozen, {1}), alpha={2}, T={3}".format(
            imp_args.teacher_path, "FP16" if imp_args.fp16 else "FP32",
            imp_args.alpha, imp_args.temperature))

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
                                 loss_fn=loss_fn, scaler=scaler, use_fp16=imp_args.fp16,
                                 teacher=teacher, alpha=imp_args.alpha, temperature=imp_args.temperature)
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
