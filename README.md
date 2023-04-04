# TSNnet
This is the implementation of the paper: "Teacher, Student, and the Naive : A Cascade Knowledge Distillation Training", ICCV2023.

## Introduction
Stereo matching has been studied for years, there are many effective methods[5, 7, 14, 47] with high computational complexity/memory occupation. On the other hand, there are only a few studies[2, 4, 38, 45] focused on stereo matching with limited computing resources. In this paper, we aim to bridge the performance gap
between theoretical approaches and practical applications by proposing an efficient cascaded knowledge distilla-
tion scheme for training lightweight networks (TSNnet). Three lightweight networks with different sizes(FLOPs, Pa
rameters) are proposed: Teacher(36.2G, 2.21M), Student(16.4G, 0.84M) and the Na ̈ıve(7.16G, 0.63M). The Student act as an intermediate to digest and bypass the knowledge from Teacher to the Na ̈ıve. Our novelties include two folds. First, we proposed the first 2D convolution only lightweight neural network for stereo matching, the network is efficient in both inference and consumed memory. Second, we propose the knowledge distillation scheme
to improve the inferior performance of lightweight network. We are able to improve the Na ̈ıve by 0.4px on
SceneFlow(EPE)[31] and 0.6% on KITTI2015(D1)[11]. Extensive experiments were conducted on two commonly used benchmarks, Sceneflow [31] and KITTI [11]. All the experiments are conducted with Nvidia V100 and PyTorch.
Our method has archived 0.67px on Sceneflow(EPE), 0.7px on KITTI2012(EPE), and 2.70% on KITTI2015(D1). And
only 17ms for a single-time inference.


![image](https://github.com/pan0793/TSNnet/blob/main/img/workflow.png)

# How to use

## Requirements
* Python>=3.9
* Pytorch>=1.10

## Enviroment
conda create -n TSNnet python=3.9
conda activate TSNnet
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
pip install opencv-python
pip install albumentations