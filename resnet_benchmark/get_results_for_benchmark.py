import os
import sys
import argparse
import time
import csv
import gc

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import cv2
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import zipfile
import shutil
import h5py
from imageio import imwrite

import torch
from torch import sigmoid
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from modules.CASENet import CASENet_resnet101
from prep_dataset.prep_cityscapes_dataset import RGB2BGR, ToTorchFormatTensor

import utils.utils as utils

NUM_CLS = 19


def run_inference(
    model_path,
    output_dir,
    image_list='',
    image_dir='',
    image_file='',
    shard_idx=0,
    num_shards=1,
    num_threads=None,
):
    """
    Run CASENet inference and write per-class PNGs to output_dir.

    Call with either image_list (path to file listing image paths) or image_file
    (single image). image_dir is prepended to paths when provided.
    """
    # Optionally limit intra-op CPU threads (helps on CPU-heavy runs)
    if num_threads is not None:
        try:
            num_threads = int(num_threads)
            if num_threads > 0:
                torch.set_num_threads(num_threads)
        except Exception:
            pass

    # load input path
    if image_list and os.path.exists(image_list):
        with open(image_list) as f:
            ori_test_lst = [x.strip().split()[0] for x in f.readlines()]
            if image_dir:
                test_lst = [
                    image_dir + x if os.path.isabs(x)
                    else os.path.join(image_dir, x)
                    for x in ori_test_lst]
            else:
                test_lst = ori_test_lst
        if num_shards and num_shards > 1:
            shard_idx = int(shard_idx)
            num_shards = int(num_shards)
            if shard_idx < 0 or shard_idx >= num_shards:
                raise ValueError('shard_idx must be in [0, num_shards)')
            test_lst = test_lst[shard_idx::num_shards]
    else:
        # Handle single-image input
        if image_dir:
            if os.path.isabs(image_file):
                path = image_dir + image_file
            else:
                path = os.path.join(image_dir, image_file)
        else:
            path = image_file
        if os.path.exists(path):
            ori_test_lst = [image_file]
            test_lst = [path]
        else:
            raise IOError('nothing to be tested!')

    # load net
    model = CASENet_resnet101(pretrained=False, num_classes=NUM_CLS)
    model.eval()
    utils.load_pretrained_model(model, model_path)

    os.makedirs(output_dir, exist_ok=True)
    for cls_idx in range(NUM_CLS):
        dir_path = os.path.join(output_dir, str(cls_idx))
        os.makedirs(dir_path, exist_ok=True)

    normalize = transforms.Normalize(mean=[104.008, 116.669, 122.675], std=[1, 1, 1])
    img_transform = transforms.Compose([
        RGB2BGR(roll=True),
        ToTorchFormatTensor(div=False),
        normalize,
    ])

    inference_times = []

    for idx_img in range(len(test_lst)):
        img = Image.open(test_lst[idx_img]).convert('RGB')
        processed_img = img_transform(img).unsqueeze(0)
        height = processed_img.size()[2]
        width = processed_img.size()[3]
        processed_img_var = utils.check_gpu(None, processed_img)

        t0 = time.perf_counter()
        with torch.no_grad():
            score_feats5, score_fuse_feats = model(processed_img_var)
        score_output = sigmoid(score_fuse_feats.transpose(1, 3).transpose(1, 2)).squeeze(0)[:height, :width, :]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        img_base_name_noext = os.path.splitext(os.path.basename(test_lst[idx_img]))[0]
        inference_times.append((img_base_name_noext, elapsed))

        for cls_idx in range(NUM_CLS):
            im_arr = (score_output[:, :, cls_idx].data.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            imwrite(os.path.join(output_dir, str(cls_idx), img_base_name_noext + '.png'), im_arr)
        print('processed: {} ({:.3f}s)'.format(test_lst[idx_img], elapsed))
        del score_feats5, score_fuse_feats, score_output, processed_img_var, processed_img
        gc.collect()

    times_path = os.path.join(output_dir, 'inference_times.csv')
    with open(times_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'inference_time_sec'])
        writer.writerows(inference_times)
    print('Inference times written to {}'.format(times_path))
    print('Done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('-m', '--model', type=str,
                        help="path to the pytorch(.pth) containing the trained weights")
    parser.add_argument('-l', '--image_list', type=str, default='',
                        help="list of image files to be tested")
    parser.add_argument('-f', '--image_file', type=str, default='',
                        help="a single image file to be tested")
    parser.add_argument('-d', '--image_dir', type=str, default='',
                        help="root folder of the image files in the list or the single image file")
    parser.add_argument('-o', '--output_dir', type=str, default='.',
                        help="folder to store the test results")
    parser.add_argument('--shard_idx', type=int, default=0,
                        help="shard index for parallel runs (default: 0)")
    parser.add_argument('--num_shards', type=int, default=1,
                        help="total number of shards for parallel runs (default: 1)")
    parser.add_argument('--num_threads', type=int, default=None,
                        help="limit torch CPU threads (default: leave at PyTorch default)")
    args = parser.parse_args(sys.argv[1:])

    run_inference(
        model_path=args.model,
        output_dir=args.output_dir,
        image_list=args.image_list,
        image_dir=args.image_dir,
        image_file=args.image_file,
        shard_idx=args.shard_idx,
        num_shards=args.num_shards,
        num_threads=args.num_threads,
    )
