# XJH-Net
这个是我的本科毕设代码，数据集和预训练权重暂不开源。
该工作暂定投稿于CVPR2024
  
## 环境

* 建议使用python3.8
```
conda create --name XJHNet python=3.8.18
```
* 我自己用的torch版本
```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```



## 运行前的准备
预训练权重存放于`./weight`

训练数据集包括事件体素的npy文件、低光帧图像和真实标签。

这两部分后续会陆续开源。

## 训练
运行`train.py`
```bash
python train.py --event_path VOXEL_PATH --frame_path FRAME_PATH --gt_path GT_PATH --weight_save_path WEIGHT_PATH --epochs 50 --batch_size 4
```
其中`VOXEL_PATH`为体素的文件夹，`FRAME_PATH `为低光帧图像的文件夹，`GT_PATH `为真实标签的文件夹，`WEIGHT_PATH `为输出训练权重地址。

**示例**：
```bash
python train.py --event_path dataset\xjh\voxel --frame_path dataset\xjh\frame --gt_path dataset\xjh\gt --weight_save_path weight\good.pth --epochs 50 --batch_size 4
```
这里默认了训练集xjh已经存放于`./dataset`

**Tips**：
1、在训练时，`./log`中会生成图像，可以查看训练过程中loss的变化过程。（每次新训练会刷新）
2、在`train.py`里面有一个`xjtrain`函数，可以调整具体训练哪一个模块。

## 预测

运行`predict.py	`
```bash
python predict.py --event_path VOXEL_PATH --frame_path FRAME_PATH  --output_path OUTPUT_PATH --weight_path WEIGHT_PATH
```
其中`VOXEL_PATH`为体素的文件夹，`FRAME_PATH `为低光帧图像的文件夹，`OUTPUT_PATH `为输出文件夹，`WEIGHT_PATH `为预训练权重地址。

**示例**：
```bash
python predict.py --event_path dataset\xjh\voxel --frame_path dataset\xjh\frame --output_path output\xjh--weight_path weight\good.pth
```
