import numpy as np
from PIL import Image
import os
import sys
import zipfile
import shutil
import h5py
from tqdm import tqdm

import torch

def _bitfield_torch(label_data, h, w, cls_num=19):
    """Original per-pixel torch logic; returns array (for testing)."""
    label_list = list(label_data)
    all_bit_tensor_list = []
    for n in label_list:
        bitfield = np.asarray([int(digit) for digit in bin(n)[2:]])
        bit_tensor = torch.from_numpy(bitfield)
        actual_len = bit_tensor.size()[0]
        padded_bit_tensor = torch.cat((torch.zeros(cls_num - actual_len).byte(), bit_tensor.byte()), dim=0)
        all_bit_tensor_list.append(padded_bit_tensor)
    all_bit_tensor_list = torch.stack(all_bit_tensor_list).view(h, w, cls_num)
    return all_bit_tensor_list.numpy()


def _bitfield_vectorized(label_data, h, w, cls_num=19):
    """Vectorized version; same bit layout (MSB at index 0). Returns array."""
    bit_indices = np.arange(cls_num - 1, -1, -1, dtype=np.uint32)
    bits = (np.reshape(label_data, (-1, 1)).astype(np.uint32) >> bit_indices) & 1
    return bits.reshape(h, w, cls_num).astype(np.uint8)


def test_bitfield_equivalence(cls_num=19, h=32, w=64, n_trials=100):
    """Assert that vectorized and torch implementations produce identical output."""
    np.random.seed(42)
    for _ in range(n_trials):
        label_data = np.random.randint(0, 2**cls_num, size=(h * w,), dtype=np.uint32)
        torch_out = _bitfield_torch(label_data, h, w, cls_num=cls_num)
        vec_out = _bitfield_vectorized(label_data, h, w, cls_num=cls_num)
        assert np.array_equal(torch_out, vec_out), "Vectorized output != torch output"
    print("OK: vectorized and torch bitfield outputs match (test_bitfield_equivalence)")


def convert_num_to_bitfield(label_data, h, w, npz_name, root_folder, h5_file, cls_num=19):
    all_bits = _bitfield_vectorized(label_data, h, w, cls_num=cls_num)
    h5_file.create_dataset('data/' + npz_name.replace('/', '_'), data=all_bits)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_bitfield_equivalence()
        sys.exit(0)
    torch.set_num_threads(8)
    f = open("cityscapes-preprocess/data_proc/train.txt", 'r')
    lines = f.readlines()
    root_folder = "cityscapes-preprocess/data_proc/"

    h5_file = h5py.File("train_label_binary_np.h5", 'w')
    for ori_line in tqdm(lines):
        line = ori_line.split()
        bin_name = line[1]
        img_name = line[0]
        
        label_path = os.path.join(root_folder, bin_name.lstrip('/')) 
        img_path = os.path.join(root_folder, img_name.lstrip('/'))

        print(root_folder)

        img = Image.open(img_path).convert('RGB')
        w, h = img.size # Notice: not h, w! This is very important! Otherwise, the label is wrong for each pixel.

        label_data = np.fromfile(label_path, dtype=np.uint32)
        npz_name = bin_name.replace("bin", "npy")
        convert_num_to_bitfield(label_data, h, w, npz_name, root_folder, h5_file)
