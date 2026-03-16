# CASENet: PyTorch Implementation

This is a PyTorch implementation of [CASENet: Deep Category-Aware Semantic Edge Detection](https://arxiv.org/abs/1705.09759), adapted for the Cityscapes dataset.

Forked from [anirudh-chakravarthy/CASENet](https://github.com/anirudh-chakravarthy/CASENet), which provides the ResNet-101 backbone, Caffe weight conversion, and baseline training pipeline.

## New contributions

This fork adds a **MobileNetV3-Large backbone** and several training improvements:

### New files
| File | Description |
|------|-------------|
| `modules/CASENet.py` (CASENet_MobileNetV3 class) | CASENet with MobileNetV3-Large backbone |
| `main_mobilenetv3.py` | Training script for MobileNetV3 baseline |
| `main_mobilenetv3_improved.py` | Training script with focal loss, augmentation, FP16, distillation, quantization |
| `train_val/model_play_improved.py` | Training/validation loop with focal loss, FP16 mixed precision, knowledge distillation |
| `prep_dataset/prep_cityscapes_dataset_mobilenetv3.py` | MobileNetV3 data preprocessing (ImageNet normalization) |
| `prep_dataset/prep_cityscapes_dataset_augmented.py` | Augmentation pipeline (ColorJitter, GaussianBlur, GaussianNoise, RandomRotation, RandomErasing) |
| `mobilenetv3_benchmark/` | Evaluation, benchmarking, and visualization scripts for MobileNetV3 |
| `utils/convert_hdf5_to_npy.py` | HDF5 to per-image .npy conversion for multiprocess DataLoader |

### Modified files
| File | Changes |
|------|---------|
| `config.py` | Added `--backbone` argument |
| `main.py` | Added `--visdom` toggle |
| `dataloader/cityscapes_data.py` | Loads labels from .npy files instead of HDF5 |

## Input pre-processing
The author's preprocessing repository was used. Instructions for use can be found in the cityscapes-preprocess directory. This is used to generate binary files for each label in the dataset.

For data loading into hdf5 file, an hdf5 file containing these binary files needs to be generated. For this conversion, run:
```
python utils/convert_bin_to_hdf5.py
```
after updating the directory paths (use absolute paths).

Then convert to per-image .npy files for multiprocess-safe loading:
```
python utils/convert_hdf5_to_npy.py
```

## Model
The original model uses the ResNet-101 variant of CASENet.

The model configuration (.prototxt) can be found: [here.](https://github.com/Chrisding/seal/blob/master/exper/sbd/config/deploy.prototxt)

The download links for pretrained weights for CASENet can be found: [here.](https://github.com/Chrisding/seal#usage)

### Converting Caffe weights to PyTorch
To use the pretrained caffemodel in PyTorch, use [extract-caffe-params](https://github.com/nilboy/extract-caffe-params) to save each layer's weights in numpy format. The code along with instructions for usage can be found in the utils folder.

To load these numpy weights for each layer into a PyTorch model, run:
```
python modules/CASENet.py
```
after updating the directory path to the parent directory containing the numpy files (use absolute paths).

**NOTE**: The Pytorch pre-trained weights can be downloaded from: [Google Drive.](https://drive.google.com/open?id=1zxshISZtq0_S6zFB37F-FhE9wT1ZBrGK)

## Training

### ResNet-101
```
python main.py --pretrained-model pretrained_models/model_casenet.pth.tar
```

### MobileNetV3 (baseline)
```
python main_mobilenetv3.py
```

### MobileNetV3 (with improvements)
```
python main_mobilenetv3_improved.py [OPTIONS]

Options:
    --focal-loss            use focal loss
    --gamma                 focal loss gamma (default: 2.0)
    --augmentation          enhanced data augmentation
    --random-erasing        RandomErasing augmentation
    --fp16                  FP16 mixed precision
    --distillation          knowledge distillation from ResNet-101 teacher
    --teacher-path          path to teacher checkpoint
    --alpha                 hard loss weight (default: 0.7)
    --temperature           distillation temperature (default: 3.0)
    --quantize CHECKPOINT   post-training quantization
    --visdom                enable Visdom
```

### Common options
```
    --checkpoint-folder     path to checkpoint dir (default: ./checkpoint)
    --multigpu              use multiple GPUs
    -j, --workers           number of data loading workers (default: 16)
    --epochs                number of total epochs to run (default: 150)
    --start-epoch           manual epoch number (useful on restarts)
    --cls-num               number of classes (default: 19 for Cityscapes)
    --lr-steps              iterations to decay learning rate by 10
    --acc-steps             gradient accumulation steps (default: 1)
    -b, --batch-size        mini-batch size (default: 1)
    --lr                    learning rate (default: 1e-7)
    --momentum              momentum (default: 0.9)
    --weight-decay          weight decay (default: 5e-4)
    -p, --print-freq        print frequency (default: 1)
    --resume-model          path to latest checkpoint
    --pretrained-model      path to pretrained checkpoint
```

## Visualization
For visualizing feature maps, ground truths and predictions, run from the project root:
```
python resnet_benchmark/visualize_multilabel.py -m pretrained_models/model_casenet.pth.tar -f leftImg8bit/val/lindau/lindau_000045_000019_leftImg8bit.png -d cityscapes-preprocess/data_proc/ -o output/
```

For MobileNetV3:
```
python mobilenetv3_benchmark/visualize_multilabel.py -m checkpoint/min_loss_checkpoint.pth.tar -f leftImg8bit/val/lindau/lindau_000045_000019_leftImg8bit.png -d cityscapes-preprocess/data_proc/ -o output/
```

## Testing
For testing a pretrained model on new images:
```
python resnet_benchmark/get_results_for_benchmark.py -m pretrained_models/model_casenet.pth.tar -f img1.png -d images/ -o output/
```

To run full validation inference and evaluation:
```
python resnet_benchmark/batch_eval_val.py -m pretrained_models/model_casenet.pth.tar
python mobilenetv3_benchmark/batch_eval_val.py -m checkpoint/min_loss_checkpoint.pth.tar
```

## Acknowledgements
1. Data processing: <https://github.com/Chrisding/cityscapes-preprocess>
2. Caffe to numpy conversion: <https://github.com/nilboy/extract-caffe-params>
3. ResNet-101 CASENet reference implementation: <https://github.com/lijiaman/CASENet>
4. Original Caffe implementation: <http://www.merl.com/research/license#CASENet>
5. Forked from: <https://github.com/anirudh-chakravarthy/CASENet>
