import h5py
import numpy as np

with h5py.File("val_label_binary_np.h5", "r") as f:
    keys = list(f["data"].keys())
    print(f"Datasets ({len(keys)}): {keys}\n")

    for key in keys:
        arr = f["data"][key][:]
        arr_sum = np.sum(arr)
        arr_min, arr_max = np.min(arr), np.max(arr)
        min_idx = np.unravel_index(np.argmin(arr), arr.shape)
        max_idx = np.unravel_index(np.argmax(arr), arr.shape)

        print(f"[{key}]")
        print(f"  shape: {arr.shape}")
        print(f"  sum: {arr_sum}")
        print(f"  min: {arr_min} at index {min_idx}")
        print(f"  max: {arr_max} at index {max_idx}")
        print()