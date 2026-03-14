"""
Evaluate CASENet predictions against ground truth using the paper's protocol:
  - Per-class ODS (Maximum F-measure at Optimal Dataset Scale)
  - Per-class AP (Average Precision)

This script reads the per-class PNG predictions produced by get_results_for_benchmark.py
and compares them against ground truth from the HDF5 label file.

Usage:
    python evaluate.py -p output_dir -l cityscapes-preprocess/data_proc/val.txt
The prediction PNGs are expected at: <pred_dir>/<cls_idx>/<image_name>.png
(the layout produced by get_results_for_benchmark.py).
"""

import os
import sys
import argparse
import csv

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import cv2
import h5py
import torch

CITYSCAPES_CLASS_NAMES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

NUM_CLS = 19
LABEL_FILE = os.path.join(_project_root, 'val_label_binary_np.h5')


def load_gt(h5_f, label_path):
    """Load ground truth edge maps from HDF5, returning (H, W, 19) uint8 array."""
    gt_key = 'data/' + label_path.replace('/', '_').replace('bin', 'npy')
    gt_raw = h5_f[gt_key][:]  # (H, W, 19)
    # Channel reversal: class k uses channel num_cls-1-k
    gt = gt_raw[:, :, ::-1].copy()
    return gt


def load_pred(pred_dir, image_name, num_cls):
    """Load per-class prediction PNGs, returning (H, W, num_cls) float32 array in [0, 1]."""
    channels = []
    for k in range(num_cls):
        path = os.path.join(pred_dir, str(k), image_name + '.png')
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise FileNotFoundError('Prediction not found: {}'.format(path))
        channels.append(im.astype(np.float32) / 255.0)
    return np.stack(channels, axis=2)


def main():
    parser = argparse.ArgumentParser(description='Evaluate CASENet predictions (ODS & AP)')
    parser.add_argument('-p', '--pred_dir', type=str, required=True,
                        help="directory containing per-class prediction PNGs (output of get_results_for_benchmark.py)")
    parser.add_argument('-l', '--image_list', type=str, required=True,
                        help="val.txt file with '<image_path> <label_path>' per line")
    parser.add_argument('-o', '--output_dir', type=str, default='',
                        help="directory to write evaluation_metrics.csv (default: pred_dir)")
    parser.add_argument('--half_res', action='store_true', default=True,
                        help="downsample to half resolution for evaluation (default: True, per paper Sec 4.2)")
    parser.add_argument('--no_half_res', action='store_false', dest='half_res',
                        help="evaluate at full resolution")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: {}'.format(
        torch.cuda.get_device_name(0) if device.type == 'cuda' else device))

    output_dir = args.output_dir if args.output_dir else args.pred_dir

    # Parse val.txt
    with open(args.image_list) as f:
        lines = [x.strip().split() for x in f.readlines()]
    image_paths = [x[0] for x in lines]
    label_paths = [x[1] for x in lines]

    # Open ground truth HDF5
    h5_f = h5py.File(LABEL_FILE, 'r')

    # Threshold grid for ODS sweep (PyTorch on device for multi-threaded CPU or GPU)
    num_thresh = 99
    thresholds = np.linspace(0.01, 0.99, num_thresh)
    thresholds_t = torch.from_numpy(thresholds).float().to(device)
    total_tp = torch.zeros((NUM_CLS, num_thresh), dtype=torch.float64, device=device)
    total_fp = torch.zeros((NUM_CLS, num_thresh), dtype=torch.float64, device=device)
    total_fn = torch.zeros((NUM_CLS, num_thresh), dtype=torch.float64, device=device)

    for idx in range(len(image_paths)):
        # Derive image base name (matches what get_results_for_benchmark.py saves)
        image_name = os.path.splitext(os.path.basename(image_paths[idx]))[0]

        # Load prediction and ground truth
        pred = load_pred(args.pred_dir, image_name, NUM_CLS)  # (H, W, 19) float32
        gt = load_gt(h5_f, label_paths[idx])                   # (H, W, 19) uint8

        height, width = pred.shape[0], pred.shape[1]

        # Downsample to half resolution (per paper Section 4.2 for Cityscapes)
        if args.half_res:
            eval_h, eval_w = height // 2, width // 2
            pred = cv2.resize(pred, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
            gt = cv2.resize(gt, (eval_w, eval_h), interpolation=cv2.INTER_NEAREST)

        pred_t = torch.from_numpy(pred).float().to(device)
        gt_t = torch.from_numpy(gt.astype(np.float32)).to(device)
        for k in range(NUM_CLS):
            pred_k = pred_t[:, :, k:k + 1]                    # (H, W, 1)
            gt_k = gt_t[:, :, k]                              # (H, W)
            pred_bin = (pred_k >= thresholds_t.view(1, 1, -1)).float()  # (H, W, num_thresh)
            gt_exp = gt_k.unsqueeze(-1)                       # (H, W, 1)
            total_tp[k] += (pred_bin * gt_exp).sum(dim=(0, 1))
            total_fp[k] += (pred_bin * (1 - gt_exp)).sum(dim=(0, 1))
            total_fn[k] += ((1 - pred_bin) * gt_exp).sum(dim=(0, 1))

        print('evaluated: [{}/{}] {}'.format(idx + 1, len(image_paths), image_name))

    h5_f.close()

    total_tp = total_tp.cpu().numpy()
    total_fp = total_fp.cpu().numpy()
    total_fn = total_fn.cpu().numpy()

    # Compute metrics
    precision = total_tp / (total_tp + total_fp + 1e-10)
    recall = total_tp / (total_tp + total_fn + 1e-10)
    f_measure = 2 * precision * recall / (precision + recall + 1e-10)

    # ODS: best threshold per class
    ods_per_class = np.max(f_measure, axis=1)
    ods_thresh_per_class = thresholds[np.argmax(f_measure, axis=1)]

    # AP: area under P-R curve per class
    ap_per_class = np.zeros(NUM_CLS)
    for k in range(NUM_CLS):
        sorted_idx = np.argsort(recall[k])
        ap_per_class[k] = np.trapezoid(precision[k][sorted_idx], recall[k][sorted_idx])

    mean_ods = np.mean(ods_per_class)
    mean_ap = np.mean(ap_per_class)

    # Print summary table
    print('\n{:<20s} {:>10s} {:>10s} {:>12s}'.format('Class', 'ODS (MF%)', 'AP%', 'ODS Thresh'))
    print('-' * 54)
    for k in range(NUM_CLS):
        print('{:<20s} {:>10.2f} {:>10.2f} {:>12.2f}'.format(
            CITYSCAPES_CLASS_NAMES[k],
            ods_per_class[k] * 100,
            ap_per_class[k] * 100,
            ods_thresh_per_class[k]))
    print('-' * 54)
    print('{:<20s} {:>10.2f} {:>10.2f}'.format('Mean', mean_ods * 100, mean_ap * 100))

    # Save to CSV
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    eval_path = os.path.join(output_dir, 'evaluation_metrics.csv')
    with open(eval_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'ods_mf', 'ods_threshold', 'ap'])
        for k in range(NUM_CLS):
            writer.writerow([CITYSCAPES_CLASS_NAMES[k],
                             '{:.6f}'.format(ods_per_class[k]),
                             '{:.2f}'.format(ods_thresh_per_class[k]),
                             '{:.6f}'.format(ap_per_class[k])])
        writer.writerow(['mean',
                         '{:.6f}'.format(mean_ods),
                         '',
                         '{:.6f}'.format(mean_ap)])
    print('\nEvaluation metrics written to {}'.format(eval_path))


if __name__ == '__main__':
    main()
