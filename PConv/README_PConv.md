# PointConv-Ras
This folder contains the code for the following work incorporated with our PointRas. 
**PointConv: Deep Convolutional Networks on 3D Point Clouds.** CVPR 2019  
Wenxuan Wu, Zhongang Qi, Li Fuxin.

## Installation
The code is based on [PointNet](https://github.com/charlesq34/pointnet)， and [PointNet++](https://github.com/charlesq34/pointnet2). Please install [TensorFlow](https://www.tensorflow.org/install/), and follow the instruction in [PointNet++](https://github.com/charlesq34/pointnet2) to compile the customized TF operators.  
The code has been tested with Python 2.7.16, TensorFlow 1.13.1, CUDA 10.1 and cuDNN 7.5.0 on Ubuntu 18.04. 
You can also construct the virtual environment via running the following terminal command: 
```
conda env create -f environment.yaml
conda activate pconv_ras
```

## Usage

### ScanetNet DataSet Segmentation

Download the ScanNetv2 dataset from [here](http://www.scan-net.org/), and see `scannet/README` for details of preprocessing.

To train a model to segment Scannet Scenes:

```
CUDA_VISIBLE_DEVICES=0 python train_ras.py --model pointconv_rend --log_dir $SPECIFY_YOUR_LOG_DIR --batch_size 8
```

After training, to evaluate the segmentation IoU accuracies:

```
CUDA_VISIBLE_DEVICES=0 python evaluate_scannet_uncertain.py --model pointconv_rend_uncertain --batch_size 8 --model_path $SPECIFY_YOUR_LOG_DIR/best_model_epoch_*.ckpt --ply_path DataSet/ScanNetv2/scans
```

Modify the model_path to your .ckpt file path. 
We provide a [pre-trained model here](https://mega.nz/folder/HtpV2YIL#ANa1865qUa5uxSI6zBy8Yg). To reproduce our reported results: 
```
CUDA_VISIBLE_DEVICES=0 python evaluate_scannet_uncertain.py --model pointconv_rend_uncertain --batch_size 8 --model_path pointconv_scannet_rend_2020_04_16_14_32_31/best_model_epoch_465.ckpt --ply_path DataSet/ScanNetv2/scans
```

## License
This repository is released under MIT License (see LICENSE file for details).
