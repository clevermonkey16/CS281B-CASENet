"""
Run CASENet inference on all Cityscapes val images and then evaluate.

Invokes get_results_for_benchmark.py once over val.txt, then evaluate.py once.
"""

import argparse
import os
import subprocess
import sys


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    default_data_root = os.path.join(project_root, "cityscapes-preprocess", "data_proc")
    default_val_list = "val.txt"
    default_pred_dir = os.path.join(project_root, "output", "val_pred")

    parser = argparse.ArgumentParser(
        description="Batch run inference and evaluation on Cityscapes val set"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Path to the CASENet .pth weights",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=default_data_root,
        help="Root folder for Cityscapes data_proc (default: cityscapes-preprocess/data_proc)",
    )
    parser.add_argument(
        "--val_list",
        type=str,
        default=default_val_list,
        help="Name of val list file under data_root (default: val.txt)",
    )
    parser.add_argument(
        "-o",
        "--pred_dir",
        type=str,
        default=default_pred_dir,
        help="Directory for prediction outputs (default: output/val_pred)",
    )
    parser.add_argument(
        "--eval_output_dir",
        type=str,
        default="",
        help="Directory for evaluation CSV (default: same as pred_dir)",
    )
    args = parser.parse_args()

    val_list_path = os.path.join(args.data_root, args.val_list)
    if not os.path.isfile(val_list_path):
        print("Error: val list not found: {}".format(val_list_path), file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.model):
        print("Error: model not found: {}".format(args.model), file=sys.stderr)
        sys.exit(1)

    eval_output_dir = args.eval_output_dir if args.eval_output_dir else args.pred_dir

    inference_script = os.path.join(script_dir, "get_results_for_benchmark.py")
    cmd_inference = [
        sys.executable,
        inference_script,
        "-m", args.model,
        "-l", val_list_path,
        "-d", args.data_root,
        "-o", args.pred_dir,
    ]
    print("Running inference on full val set...")
    ret = subprocess.run(cmd_inference, cwd=project_root)
    if ret.returncode != 0:
        print("Inference failed with exit code {}".format(ret.returncode), file=sys.stderr)
        sys.exit(ret.returncode)

    eval_script = os.path.join(script_dir, "evaluate.py")
    cmd_eval = [
        sys.executable,
        eval_script,
        "-p", args.pred_dir,
        "-l", val_list_path,
        "-o", eval_output_dir,
    ]
    print("Running evaluation...")
    ret = subprocess.run(cmd_eval, cwd=project_root)
    if ret.returncode != 0:
        print("Evaluation failed with exit code {}".format(ret.returncode), file=sys.stderr)
        sys.exit(ret.returncode)
    print("Done. Metrics in {}".format(eval_output_dir))


if __name__ == "__main__":
    main()
