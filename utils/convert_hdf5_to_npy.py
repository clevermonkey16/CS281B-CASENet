"""
Convert HDF5 label files to individual .npy files for multiprocess-safe DataLoader access.

Usage:
    python utils/convert_hdf5_to_npy.py train_label_binary_np.h5 train_label_npy/
    python utils/convert_hdf5_to_npy.py val_label_binary_np.h5 val_label_npy/
"""
import h5py
import numpy as np
import os
import sys

def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_hdf5_to_npy.py <input.h5> <output_dir>")
        sys.exit(1)

    h5_path = sys.argv[1]
    out_dir = sys.argv[2]

    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(h5_path, 'r') as f:
        keys = list(f['data'].keys())
        print(f"Converting {len(keys)} samples from {h5_path} to {out_dir}/")
        for i, key in enumerate(keys):
            arr = f['data'][key][:]
            np.save(os.path.join(out_dir, key), arr)
            if (i + 1) % 100 == 0 or (i + 1) == len(keys):
                print(f"  {i+1}/{len(keys)}")

    print("Done.")

if __name__ == '__main__':
    main()
