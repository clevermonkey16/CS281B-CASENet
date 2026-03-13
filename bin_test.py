import numpy as np
p = "cityscapes-preprocess/data_proc/gtFine/val/frankfurt/frankfurt_000000_008206_gtFine_edge.bin"
d = np.fromfile(p, dtype=np.uint32)
print(d.size, d.max(), d.sum())