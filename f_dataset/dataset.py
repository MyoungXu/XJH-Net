import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class ImageEventDataset(Dataset):
    def __init__(self, image_dir, event_dir, gt_dir=None, mode='train', crop_size=(256, 256)):
        """
        Args:
            image_dir (string): 图像文件的路径。
            event_dir (string): 事件信号文件的路径。
            gt_dir (string, optional): 标签gt文件的路径。如果mode为predict则不需要。
            transform (callable, optional): 一个可选的转换函数，可以用来转换样本。
            mode (string, optional): train表训练，predict表预测
        """
        self.image_dir = image_dir
        self.event_dir = event_dir
        self.gt_dir = gt_dir

        self.crop_size = crop_size
        self.filenames = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        self.mode = mode
        assert mode == 'train' or mode == 'predict', "mode只能为train或者predict！"

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # 获取frame和事件的地址
        img_name = os.path.join(self.image_dir, self.filenames[idx])  # frame图像路径
        event_name = os.path.join(self.event_dir, self.filenames[idx].split('.')[0] + '.npy')  # 事件信号npy文件的路径
        assert os.path.isfile(event_name), "与frame中图像同名的事件信号npy文件不存在！请检查数据集或者输入的event_dir！"

        # 使用PIL库打开图像
        frame_img = Image.open(img_name).convert('RGB')
        frame = transforms.ToTensor()(frame_img)
        frame_img.close()

        # 读取事件信号数据
        event_signal = np.load(event_name)

        event_signal = torch.tensor(event_signal, dtype=torch.float32)  # 将事件信号转换为PyTorch张量

        if self.mode == 'train':
            # 获取gt的地址
            gt_name = os.path.join(self.gt_dir, self.filenames[idx])  # gt图像路径
            assert os.path.isfile(gt_name), "与frame中图像同名的gt图像不存在！请检查数据集或者输入的gt_dir！"
            # 使用PIL库打开图像
            gt_img = Image.open(gt_name).convert('RGB')
            # 如果定义了transforms，进行图像转换
            gt = transforms.ToTensor()(gt_img)
            gt_img.close()
            assert gt.size(1) % 4 == 0 and gt.size(2) % 4 == 0, "输入图像的长和宽必须都是4的倍数"

            h, w = frame.size(1), frame.size(2)
            th, tw = self.crop_size
            i = torch.randint(0, h - th + 1, (1,)).item()
            j = torch.randint(0, w - tw + 1, (1,)).item()

            frame = frame[:, i:i + th, j:j + tw]
            event_signal = event_signal[:, i:i + th, j:j + tw]
            gt = gt[:, i:i + th, j:j + tw]

            return frame, event_signal, gt

        elif self.mode == 'predict':
            frame = frame[:, :240, :320]
            event_signal = event_signal[:, :240, :320]
            return frame, event_signal
