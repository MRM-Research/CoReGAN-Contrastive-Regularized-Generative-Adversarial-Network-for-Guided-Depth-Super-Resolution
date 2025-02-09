# CoReGAN: Contrastive Regularized GAN for Guided Depth Map Super Resolution

## Overview
This repository contains the official implementation of CoReGAN (Contrastive Regularized Generative Adversarial Network), a novel approach for guided depth map super-resolution. The model leverages contrastive learning to regularize features extracted from dual encoders and a decoder, achieving state-of-the-art results in depth map super-resolution tasks.

## Paper
**Title**: CoReGAN: Contrastive Regularized Generative Adversarial Network for Guided Depth Map Super Resolution

**Abstract**: Consumer-grade depth sensors provide low-resolution depth maps; however, a high-resolution RGB camera is usually mounted on the same device and acquires a high-resolution image of the same scene. While deep learning and guided filtering methods gave decent results, recent works have highlighted the superiority of using RGB images for Depth Super Resolution. This paper proposes CoReGAN, a generative data fusion model that employs contrastive learning to regularize the extracted features of 2 independent encoders and 1 decoder for Guided Depth Super Resolution, demonstrating state-of-the-art results.

**Paper Link**: [ACM Digital Library](https://dl.acm.org/doi/10.1145/3639856.3639897)

## Dataset
The model is trained and evaluated on the NYUv2 Dataset, which contains RGB-D images of indoor scenes. You can access the dataset [here](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html).

## Evaluation Metrics
The model's performance is evaluated using the following metrics:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

## Arguments
Below are the available command-line arguments for training and testing the model:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--hr_dir` | str | `/content/drive/MyDrive/nyuv2-dataset/train/images` | Directory path for high-resolution training images |
| `--tar_dir` | str | `/content/drive/MyDrive/nyuv2-dataset/train/depths` | Directory path for target depth maps (training) |
| `--hr_val_dir` | str | `/content/drive/MyDrive/nyuv2-dataset/val/images` | Directory path for high-resolution validation images |
| `--tar_val_dir` | str | `/content/drive/MyDrive/nyuv2-dataset/val/depths` | Directory path for target depth maps (validation) |
| `--hr_test_dir` | str | `/content/drive/MyDrive/nyuv2-dataset/test/images` | Directory path for high-resolution test images |
| `--tar_test_dir` | str | `/content/drive/MyDrive/nyuv2-dataset/test/depths` | Directory path for target depth maps (testing) |
| `--batch_size` | int | 8 | Number of samples per training batch |
| `--epochs` | int | 250 | Number of training epochs |
| `--device` | str | 'cuda' | Computing device ('cuda' for GPU, 'cpu' for CPU) |
| `--encoder` | str | 'resnet34' | Encoder architecture to use |
| `--encoder_weights` | str | 'imagenet' | Pre-trained weights for the encoder |
| `--lr` | float | 1e-3 | Learning rate for optimization |
| `--beta` | float | 1 | Weight parameter for loss function |
| `--loss_weight` | float | 2000 | Weight factor for the loss computation |
| `--gan_type` | str | 'lsgan' | Type of GAN loss to use |

## Citation
If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{10.1145/3639856.3639897,
author = {Kasliwal, Aditya and Gakhar, Ishaan and Kamani, Aryan Bhavin},
title = {CoReGAN: Contrastive Regularized Generative Adversarial Network for Guided Depth Map Super Resolution},
year = {2024},
isbn = {9798400716492},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3639856.3639897},
doi = {10.1145/3639856.3639897},
booktitle = {Proceedings of the Third International Conference on AI-ML Systems},
articleno = {41},
numpages = {5},
location = {Bangalore, India},
series = {AIMLSystems '23}
}
