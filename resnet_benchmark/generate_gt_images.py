"""
Generate ground truth edge label images for a single validation image.

Outputs 19 per-class binary PNGs (black = no edge, white = edge) matching
the format of predictions in output/val_pred/.

Usage:
    python resnet_benchmark/generate_gt_images.py \
        -i frankfurt_000000_000294_leftImg8bit \
        -l cityscapes-preprocess/data_proc/val.txt \
        -o output/val_gt
"""

import os
import sys
import argparse

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import h5py
from imageio import imwrite

NUM_CLS = 19
LABEL_FILE = os.path.join(_project_root, 'val_label_binary_np.h5')


def load_gt(h5_f, label_path):
    """Load ground truth edge maps from HDF5, returning (H, W, 19) uint8 array."""
    gt_key = 'data/' + label_path.replace('/', '_').replace('bin', 'npy')
    gt_raw = h5_f[gt_key][:]  # (H, W, 19)
    gt = gt_raw[:, :, ::-1].copy()
    return gt


def main():
    parser = argparse.ArgumentParser(description='Generate ground truth edge label images')
    parser.add_argument('-i', '--image_name', type=str, required=True,
                        help="image base name without extension (e.g. frankfurt_000000_000294_leftImg8bit)")
    parser.add_argument('-l', '--image_list', type=str, required=True,
                        help="val.txt file with '<image_path> <label_path>' per line")
    parser.add_argument('-o', '--output_dir', type=str, default='output/val_gt',
                        help="output directory (default: output/val_gt)")
    args = parser.parse_args()

    # Parse val.txt to find the label path for the requested image
    with open(args.image_list) as f:
        lines = [x.strip().split() for x in f.readlines()]

    label_path = None
    for img_path, lbl_path in lines:
        base = os.path.splitext(os.path.basename(img_path))[0]
        if base == args.image_name:
            label_path = lbl_path
            break

    if label_path is None:
        print('Error: image "{}" not found in {}'.format(args.image_name, args.image_list))
        sys.exit(1)

    # Create output directories
    for cls_idx in range(NUM_CLS):
        os.makedirs(os.path.join(args.output_dir, str(cls_idx)), exist_ok=True)

    # Load ground truth and save per-class PNGs
    h5_f = h5py.File(LABEL_FILE, 'r')
    gt = load_gt(h5_f, label_path)  # (H, W, 19) uint8 with 0/1 values
    h5_f.close()

    for cls_idx in range(NUM_CLS):
        im_arr = (gt[:, :, cls_idx] * 255).astype(np.uint8)
        out_path = os.path.join(args.output_dir, str(cls_idx), args.image_name + '.png')
        imwrite(out_path, im_arr)

    print('Ground truth images saved to {}'.format(args.output_dir))


if __name__ == '__main__':
    main()
