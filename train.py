import datetime
import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import logging
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from f_dataset.dataset import ImageEventDataset
from model.Xnet import Xnet
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# CUDA_VISIBLE_DEVICES=0
def plot_losses(phase_losses):
    save_path = os.path.join('log/loss_plot.png')
    if os.path.exists(save_path):       # 删掉原来的
        os.remove(save_path)
    plt.figure()
    for phase, losses in phase_losses.items():
        plt.plot(losses, label=phase)
    plt.title('Losses for Different Phases')
    plt.xlabel('imgs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_epoch_losses(phase_losses):
    save_path = os.path.join('log', 'epoch_loss.png')
    if os.path.exists(save_path):  # 删掉原来的
        os.remove(save_path)

    plt.figure()
    for phase, losses in phase_losses.items():
        plt.plot(losses, label=phase)
        # # 在每个点旁添加文本显示具体值
        # for i, loss in enumerate(losses):
        #     plt.text(i, loss, f'{loss:.5f}', fontsize=8)

    plt.title('Average Losses for Different Phases')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def xjtrain(model, data_loader, num_epochs, weigh_save_path):
    setting = 30        # 早停，不过记得考虑phases数量
    phase_losses = {"Decom": [], "Denoise": [], "Relight": [], "together": []}  # 记录每个阶段的损失
    epoch_losses_history = {"Decom": [], "Denoise": [], "Relight": [], "together": []}  # 记录每个 epoch 的总损失
    recent_losses = []
    flag_break = False
    # 循环遍历训练数据
    for epoch in range(num_epochs):
        # 动态调整 train_phase
        if epoch <= 0.9 * num_epochs:
            phases = ["Decom", "Denoise", "Relight"]
        else:
            phases = ["together"]

        for phase in phases:
            model.train_phase = phase
            # 使用 tqdm 包装数据加载器以显示进度条
            data_loader_iter = tqdm(data_loader, desc=f"Epoch {epoch + 1}, Phase {phase}\t", leave=True)
            total_loss = 0
            num = 0
            for image, extra_data, label in data_loader_iter:
                # 调用训练方法
                loss = model.my_train(image, extra_data, label, phase)
                phase_losses[phase].append(loss)  # 记录损失
                total_loss = total_loss + loss
                num = num + 1
                average_loss = total_loss / num
                # 更新进度条显示的损失信息
                data_loader_iter.set_postfix(loss=loss, avg_loss=average_loss)

            epoch_losses_history[phase].append(average_loss)  # 记录损失
            recent_losses.append(average_loss)
            # 如果记录的平均损失数量超过一定数量，移除最旧的一个
            if len(recent_losses) > setting:
                recent_losses.pop(0)
            if len(recent_losses) == setting and all(recent_losses[0] <= recent_losses[i] for i in range(setting)):
                flag_break = True
        if 0 < epoch <= 0.9 * num_epochs:
            for phase in phases:
                list = epoch_losses_history[phase]
                print(f"模块 {phase} 与上次相比，平均损失变化为： {list[-1]-list[-2]}")

        # 保存模型权重
        if (epoch + 1) % 1 == 0:  # 每1个epoch保存一次
            torch.save(model.state_dict(), weigh_save_path)
            torch.save(model.DecomNet.state_dict(), 'weight/DecomNet_weights.pth')
            torch.save(model.DenoiseNet.state_dict(), 'weight/DenoiseNet_weights.pth')
            torch.save(model.RelightNet.state_dict(), 'weight/RelightNet_weights.pth')
        plot_losses(phase_losses)
        plot_epoch_losses(epoch_losses_history)
        if flag_break:
            break
    return average_loss, epoch + 1

if True:
    parser = argparse.ArgumentParser("xjh_net/train.py")
    parser.add_argument('--event_path', type=str, help='存放npy格式事件的文件夹地址')
    parser.add_argument('--frame_path', type=str, help='存放低光RGB帧图像的文件夹地址')
    parser.add_argument('--gt_path', type=str, help='存放gt图像的文件夹地址')
    parser.add_argument('--weight_save_path', type=str, default='weight/unnamed.pth', help='网络权重地址')
    parser.add_argument('--epochs', type=int, default=10, help='epoch数量')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size大小')
    parser.add_argument('--pretrained_weight', type=str, default=None, help='如果在原有基础上继续训练，则给出地址')

    args = parser.parse_args()
    event_path = args.event_path
    frame_path = args.frame_path
    gt_path = args.gt_path
    weigh_save_path = args.weight_save_path

    epochs = args.epochs
    batch_size = args.batch_size
    pretrained_weight = args.pretrained_weight


dataset = ImageEventDataset(image_dir=frame_path, event_dir=event_path, gt_dir=gt_path, mode='train')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print('当前设备：\t', os.environ['CUDA_VISIBLE_DEVICES'])
print('初始权重：\t', pretrained_weight)
print('图像总数：\t', len(dataset))
print('设定epoch总数：\t', epochs)
print('batch_size：\t', batch_size)
print('权重保存至：\t', weigh_save_path)

model = Xnet().cuda()
# 如果是在已经训练过一点的基础上继续训练
if pretrained_weight:
    checkpoint = torch.load(pretrained_weight)
    model.load_state_dict(checkpoint)

# train_model(model, dataloader, epochs, weigh_save_path)
final_loss, epoch_num = xjtrain(model, dataloader, epochs, weigh_save_path)


# 将运行相关信息写入日志
logging.basicConfig(filename='log/train.log', level=logging.INFO, format='%(message)s')
current_time = datetime.datetime.now()
current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
loss_path = 'log/last_loss_plot.png'
logging.info("%s\n事件路径：%s\n低光图像路径：%s\ngt路径：%s\n初始权重：%s\n权重保存于：%s\n图像总数：%s\nepochs：%s\n"
             "batch_size：%s\nloss图路径：%s\nfinal_loss：%f\n\n********************************\n", current_time_str, event_path,
             frame_path, gt_path, pretrained_weight, weigh_save_path, len(dataset), epoch_num, batch_size, loss_path, final_loss)

