# TSNnet
This is the implementation of the ICCV2023 submission-No.11899: "Teacher, Student, and the Na ̈ıve: A Cascade Knowledge Distillation Training".

## Introduction
Stereo matching has been studied for years, there are many effective methods[5, 7, 14, 47] with high computational complexity/memory occupation. On the other hand, there are only a few studies[2, 4, 38, 45] focused on stereo matching with limited computing resources. In this paper, we aim to bridge the performance gap between theoretical approaches and practical applications by proposing an efficient cascaded knowledge distillation scheme for training lightweight networks (TSNnet). Three lightweight networks with different sizes(FLOPs, Parameters) are proposed: Teacher(36.2G, 2.21M), Student(16.4G, 0.84M) and the Na ̈ıve(7.16G, 0.63M). The Student act as an intermediate to digest and bypass the knowledge from Teacher to the Na ̈ıve. Our novelties include two folds. First, we proposed the first 2D convolution only lightweight neural network for stereo matching, the network is efficient in both inference and consumed memory. Second, we propose the knowledge distillation scheme to improve the inferior performance of lightweight network. We are able to improve the Na ̈ıve by 0.4px on SceneFlow(EPE)[31] and 0.6% on KITTI2015(D1)[11]. Extensive experiments were conducted on two commonly used benchmarks, Sceneflow [31] and KITTI [11]. All the experiments are conducted with Nvidia V100 and PyTorch. Our method has archived 0.67px on Sceneflow(EPE), 0.7px on KITTI2012(EPE), and 2.70% on KITTI2015(D1). And only 17ms for a single-time inference.


![image](https://github.com/pan0793/TSNnet/blob/main/img/workflow.png)


## Results
### Results on KITTI 2015 leaderboard [<font size =1>link</font>](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
<!-- [Leaderboard Link](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) -->

| Method | D1-bg (All) | D1-fg (All) | D1-all (All) | Runtime (ms) |
|:-:|:-:|:-:|:-:|:-:|
| [TSNnet_Teacher](https://www.cvlibs.net/datasets/kitti/eval_scene_flow_detail.php?benchmark=stereo&result=39290e76173f581a8ca318bb1e9a12e16b8f3ca5) |2.24 % | 4.99 % | 2.70 % | 108 |
| [TSNnet_Student](https://www.cvlibs.net/datasets/kitti/eval_scene_flow_detail.php?benchmark=stereo&result=525e1ae0f0f15a64e0bf06b8fd194f0783ec9416) | 2.35 %|6.70 %|3.07 %| 24 |
| [TSNnet_Naive](https://www.cvlibs.net/datasets/kitti/eval_scene_flow_detail.php?benchmark=stereo&result=77e2ffe05f35444bc1d61761468c49529f5fe99f) |2.64 %|6.47 %|3.28 %| 17 |

### Examples of our results
![image](https://github.com/pan0793/TSNnet/tree/main/img/comparison.png)

<!-- ![image](https://github.com/pan0793/TSNnet/tree/main/img/qualitative/imgL6.png) -->
<!-- ![image](https://github.com/pan0793/TSNnet/tree/main/img/qualitative/col6.png) -->



# How to use

### Requirements
* Python>=3.9
* Pytorch>=1.10

### Prepare the Enviroment 
```
conda create -n TSNnet python=3.9
conda activate TSNnet
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
pip install opencv-python
pip install albumentations
```
### Prepare the Data
Download [Sceneflow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo), [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

## Train
Use the following command to train the TSNnet on Sceneflow

First, train TSNnet-Teacher network on Sceneflow solely,
```
python train.py
```
or to Train on kitti
```
python train.py --finetune kitti
```
Secondly, train the TSNnet-student or TSNnet-naive with Knowledge Distillation
```
python train_knowledge.py
```
or to Knowledge Distillation on kitti:

```
python train_knowledge.py --finetune kitti
```

## Test

```
python eval.py
```


