import datetime
import logging
import numpy as np
import torch
import os
import argparse

from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from f_dataset.dataset import ImageEventDataset
from model.Xnet import Xnet


"""
参考代码：
python predict.py --event_path data_test/event --frame_path data_test/frame --output_path data_test/predict --weight_path weight/xjh.pth --resize_flag True --target_size 540 720
"""

# 读入形参，这个if是用来缩放节省空间的
if True:
    parser = argparse.ArgumentParser("xjh_net/predict.py")
    parser.add_argument('--event_path', type=str, help='存放npy格式事件的文件夹地址')
    parser.add_argument('--frame_path', type=str, help='存放低光RGB帧图像的文件夹地址')
    parser.add_argument('--output_path', type=str, help='输出预测图像的文件夹地址')
    parser.add_argument('--weight_path', type=str, help='预训练权重所在地址')
    parser.add_argument('--flag', type=int, default=0, help='正常光、分解R、分解I、去噪R、图像热图、事件热图')

    args = parser.parse_args()

    event_path = args.event_path
    frame_path = args.frame_path
    output_path = args.output_path
    weight_path = args.weight_path
    flag = args.flag

if not os.path.exists(output_path):  # 检查输出文件夹是否存在
    os.makedirs(output_path)  # 如果不存在，创建文件夹

# 初始化模型
model = Xnet()
model.load_state_dict(torch.load(weight_path))

# 确保模型转移到正确的设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('当前设备：', device)
model = model.to(device)

dataset = ImageEventDataset(image_dir=frame_path, event_dir=event_path, mode='predict')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

count = 0
# 开始预测
for image, event_signal in tqdm(dataloader, desc="Processing images", total=len(dataloader)):
    image, event_signal = image.to(device), event_signal.to(device)
    # 如果有GPU可用，将模型和数据移到GPU上
    image = image.to(device)
    event_signal = event_signal.to(device)
    img_in, output_R_low, output_I_low, output_I_delta, output_R_denoise, output_S, img_map, event_map = model.predict(image, event_signal)

    # 将结果移回CPU并转换为numpy数组 (如果需要)
    if flag == 0:
        prediction = output_S.cpu().numpy()
        # 将 ndarray 转换为 PIL 图像对象
        prediction = (prediction * 255).astype(np.uint8)
        predict_img = Image.fromarray(prediction.squeeze().transpose(1, 2, 0))  # 去掉第一个维度，转换通道顺序
        target_filename = f"{count:06}.png"  # 生成目标文件名，如000001.png、000002.png，从0开始
        target_path = os.path.join(output_path, target_filename)
        # 保存图像为 PNG 文件
        predict_img.save(target_path)
        predict_img.close()
    elif flag == 1:
        prediction = output_R_low.cpu().numpy()
        # 将 ndarray 转换为 PIL 图像对象
        prediction = (prediction * 255).astype(np.uint8)
        predict_img = Image.fromarray(prediction.squeeze().transpose(1, 2, 0))  # 去掉第一个维度，转换通道顺序
        target_filename = f"{count:06}.png"  # 生成目标文件名，如000001.png、000002.png，从0开始
        target_path = os.path.join(output_path, target_filename)
        # 保存图像为 PNG 文件
        predict_img.save(target_path)
        predict_img.close()
    elif flag == 2:
        prediction = output_I_low.cpu().numpy()
        # 将 ndarray 转换为 PIL 图像对象
        prediction = (prediction * 255).astype(np.uint8)
        predict_img = Image.fromarray(prediction.squeeze().transpose(1, 2, 0))  # 去掉第一个维度，转换通道顺序
        target_filename = f"{count:06}.png"  # 生成目标文件名，如000001.png、000002.png，从0开始
        target_path = os.path.join(output_path, target_filename)
        # 保存图像为 PNG 文件
        predict_img.save(target_path)
        predict_img.close()
    elif flag == 3:
        prediction = output_R_denoise.cpu().numpy()
        # 将 ndarray 转换为 PIL 图像对象
        prediction = (prediction * 255).astype(np.uint8)
        predict_img = Image.fromarray(prediction.squeeze().transpose(1, 2, 0))  # 去掉第一个维度，转换通道顺序
        target_filename = f"{count:06}.png"  # 生成目标文件名，如000001.png、000002.png，从0开始
        target_path = os.path.join(output_path, target_filename)
        # 保存图像为 PNG 文件
        predict_img.save(target_path)
        predict_img.close()
    elif flag == 4:
        map = img_map.cpu().numpy()
        map = map.squeeze()
        plt.imshow(map, cmap='hot', interpolation='nearest')
        plt.colorbar()
        target_filename = f"{count:06}attention.png"  # 生成目标文件名，如000001.png、000002.png，从0开始
        target_path = os.path.join(output_path, target_filename)
        plt.savefig(target_path)
        plt.close()
    elif flag == 5:
        map = event_map.cpu().numpy()
        map = map.squeeze()
        plt.imshow(map, cmap='hot', interpolation='nearest')
        plt.colorbar()
        target_filename = f"{count:06}attention.png"  # 生成目标文件名，如000001.png、000002.png，从0开始
        target_path = os.path.join(output_path, target_filename)
        plt.savefig(target_path)
        plt.close()

    count += 1

print(f'所有图片已保存至{output_path}')

# 将运行相关信息写入日志
logging.basicConfig(filename='log/predict.log', level=logging.INFO, format='%(message)s')
current_time = datetime.datetime.now()
current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
logging.info("%s\n事件路径：%s\n低光图像路径：%s\n所使用的预训练权重：%s\n图像总数：%s\n\n"
             "********************************\n", current_time_str, event_path, frame_path, weight_path,
             len(dataset))

