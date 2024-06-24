# XJH-Net
华中科技大学2024高级机器学习课程报告配套代码

  
## 环境

* 建议使用python3.8
```
conda create --name XJHNet python=3.8.18
```
* 参考的torch版本
```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```
* 其他相关库安装
```
pip install -r requirements.txt
```



## 运行前的准备

预训练权重存放于`./weight`

训练数据集包括事件体素的npy文件、低光帧图像和真实标签，存放于`./dataset`。

预训练权重和提供测试的部分数据集可以在[这里](https://pan.baidu.com/s/17W8xrqw4a0v275uT5tOC0A?pwd=kd93 )下载：
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
python predict.py --event_path VOXEL_PATH --frame_path FRAME_PATH  --output_path OUTPUT_PATH --weight_path WEIGHT_PATH --flag OUTPUT_FLAG
```
其中`VOXEL_PATH`为体素的文件夹，`FRAME_PATH `为低光帧图像的文件夹，`OUTPUT_PATH `为输出文件夹，`WEIGHT_PATH `为预训练权重地址，`OUTPUT_FLAG`为想要输出的目标图像，从0~5分别表示“正常光、分解R、分解I、去噪R、图像热图、事件热图”，默认输出为正常光的结果。

**示例**：
```bash
python predict.py --event_path dataset\xjh\voxel --frame_path dataset\xjh\frame --output_path output\xjh--weight_path weight\good.pth
```
