import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import ms_ssim


def MS_SSIMLoss(out_image, gt_image):
    return 1 - ms_ssim(out_image, gt_image, data_range=1, size_average=True)

def compute_vgg_loss(enhanced_result, input_high):
    # 将输入数据和模型都放在同一个设备上
    device = enhanced_result.device
    instance_norm = nn.InstanceNorm2d(512, affine=False).to(device)
    # 加载 VGG16 模型，但只使用特征提取部分，去除全连接层
    vgg = models.vgg16(pretrained=True).features.to(device)
    vgg.eval()
    for param in vgg.parameters():
        param.requires_grad = False
    img_fea = vgg(enhanced_result)
    target_fea = vgg(input_high)
    loss = torch.mean((instance_norm(img_fea) - instance_norm(target_fea)) ** 2)
    return loss


# 定义SE注意力机制的类
class se_block(nn.Module):
    # 初始化, in_channel代表输入特征图的通道数, ratio代表第一个全连接下降通道的倍数
    def __init__(self, in_channel, ratio=4):
        # 继承父类初始化方法
        super(se_block, self).__init__()

        # 属性分配
        # 全局平均池化，输出的特征图的宽高=1
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # 第一个全连接层将特征图的通道数下降4倍
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        # relu激活
        self.relu = nn.ReLU()
        # 第二个全连接层恢复通道数
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)
        # sigmoid激活函数，将权值归一化到0-1
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):  # inputs 代表输入特征图

        # 获取输入特征图的shape
        b, c, h, w = inputs.shape
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        x = self.avg_pool(inputs)
        # 维度调整 [b,c,1,1]==>[b,c]
        x = x.view([b, c])

        # 第一个全连接下降通道 [b,c]==>[b,c//4]
        x = self.fc1(x)
        x = self.relu(x)
        # 第二个全连接上升通道 [b,c//4]==>[b,c]
        x = self.fc2(x)
        # 对通道权重归一化处理
        x = self.sigmoid(x)

        # 调整维度 [b,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])

        # 将输入特征图和通道权重相乘
        outputs = x * inputs
        return outputs

class ResidualModule0(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(ResidualModule0, self).__init__()
        self.Relu = nn.LeakyReLU()

        self.conv0 = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(channel, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv3 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv4 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)

        self.conv = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = x
        out0 = self.Relu(self.conv0(x))
        out1 = self.Relu(self.conv1(out0))
        out2 = self.Relu(self.conv2(out1))
        out3 = self.Relu(self.conv3(out2))
        out4 = self.Relu(self.conv4(out3))
        out = self.Relu(self.conv(residual))

        final_out = torch.cat((out, out4), dim=1)

        return final_out
class ResidualModule1(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(ResidualModule1, self).__init__()
        self.Relu = nn.LeakyReLU()

        self.conv0 = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(channel, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv3 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv4 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)

        self.conv = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = x
        out0 = self.Relu(self.conv0(x))
        out1 = self.Relu(self.conv1(out0))
        out2 = self.Relu(self.conv2(out1))
        out3 = self.Relu(self.conv3(out2))
        out4 = self.Relu(self.conv4(out3))
        out = self.Relu(self.conv(residual))

        final_out = torch.cat((out, out4), dim=1)

        return final_out
class ResidualModule2(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(ResidualModule2, self).__init__()
        self.Relu = nn.LeakyReLU()

        self.conv0 = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(channel, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv3 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv4 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)

        self.conv = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = x
        out0 = self.Relu(self.conv0(x))
        out1 = self.Relu(self.conv1(out0))
        out2 = self.Relu(self.conv2(out1))
        out3 = self.Relu(self.conv3(out2))
        out4 = self.Relu(self.conv4(out3))
        out = self.Relu(self.conv(residual))

        final_out = torch.cat((out, out4), dim=1)

        return final_out
class ResidualModule3(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(ResidualModule3, self).__init__()
        self.Relu = nn.LeakyReLU()

        self.conv0 = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(channel, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv3 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv4 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)

        self.conv = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = x
        out0 = self.Relu(self.conv0(x))
        out1 = self.Relu(self.conv1(out0))
        out2 = self.Relu(self.conv2(out1))
        out3 = self.Relu(self.conv3(out2))
        out4 = self.Relu(self.conv4(out3))
        out = self.Relu(self.conv(residual))

        final_out = torch.cat((out, out4), dim=1)

        return final_out


class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()

        self.activation = nn.LeakyReLU()

        self.conv0 = nn.Conv2d(4, channel, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.RM0 = ResidualModule0()
        self.RM1 = ResidualModule1()
        self.RM2 = ResidualModule2()
        self.RM3 = ResidualModule3()
        self.conv1 = nn.Conv2d(channel * 2, channel, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(channel * 2, channel, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv3 = nn.Conv2d(channel * 2, channel, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv4 = nn.Conv2d(channel * 2, channel, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv5 = nn.Conv2d(channel, 4, kernel_size=1, stride=1, padding=0)

    def forward(self, input_im):
        input_max = torch.max(input_im, dim=1, keepdim=True)[0]
        input_img = torch.cat((input_max, input_im), dim=1)

        out0 = self.activation(self.conv0(input_img))
        out1 = self.RM0(out0)
        out2 = self.activation(self.conv1(out1))
        out3 = self.RM1(out2)
        out4 = self.activation(self.conv2(out3))
        out5 = self.RM2(out4)
        out6 = self.activation(self.conv3(out5))
        out7 = self.RM3(out6)
        out8 = self.activation(self.conv4(out7))
        out9 = self.activation(self.conv5(out8))

        R = torch.sigmoid(out9[:, 0:3, :, :])
        L = torch.sigmoid(out9[:, 3:4, :, :])

        return R, L


class DenoiseNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DenoiseNet, self).__init__()
        self.Relu = nn.LeakyReLU()
        self.Denoise_conv0_1 = nn.Conv2d(4, channel, kernel_size, padding=2, padding_mode='replicate', dilation=2)
        self.Denoise_conv0_2 = nn.Conv2d(channel, channel, kernel_size, padding=2, padding_mode='replicate',
                                         dilation=2)  # 96*96
        self.conv0 = nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv3 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv4 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.Denoise_subsampling0 = nn.Conv2d(channel*2, channel, kernel_size=2, stride=2, padding=0)  # 48*48
        self.conv5 = nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv7 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv8 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv9 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.Denoise_subsampling1 = nn.Conv2d(channel*2, channel, kernel_size=2, stride=2, padding=0)  # 24*24
        self.conv10 = nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0)
        self.conv11 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv12 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv13 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv14 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.Denoise_subsampling2 = nn.Conv2d(channel*2, channel, kernel_size=2, stride=2, padding=0)  # 12*12
        self.conv15 = nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0)
        self.conv16 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv17 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv18 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv19 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.Denoise_deconv0 = nn.ConvTranspose2d(channel*2, channel, kernel_size=2, stride=2, padding=0,
                                                  output_padding=0)  # 24*24
        self.conv20 = nn.Conv2d(channel * 2, channel * 2, kernel_size=1, stride=1, padding=0)
        self.conv21 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv22 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv23 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv24 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.Denoise_deconv1 = nn.ConvTranspose2d(channel*2, channel, kernel_size=2, stride=2, padding=0,
                                                  output_padding=0)  # 48*48
        self.conv25 = nn.Conv2d(channel * 2, channel * 2, kernel_size=1, stride=1, padding=0)
        self.conv26 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv27 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv28 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv29 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.Denoise_deconv2 = nn.ConvTranspose2d(channel*2, channel, kernel_size=2, stride=2, padding=0,
                                                  output_padding=0)  # 96*96

        self.Denoiseout0 = nn.Conv2d(channel * 2, channel, kernel_size, padding=1, padding_mode='replicate')
        self.Denoiseout1 = nn.Conv2d(channel, 3, kernel_size=1, stride=1)

    def forward(self, input_L, input_R):
        input_img = torch.cat((input_R, input_L), dim=1)
        out0 = self.Relu(self.Denoise_conv0_1(input_img))
        out1 = self.Relu(self.Denoise_conv0_2(out0))
        out2 = self.Relu(self.conv0(out1))
        out3 = self.Relu(self.conv1(out2))
        out4 = self.Relu(self.conv2(out3))
        out5 = self.Relu(self.conv3(out4))
        out6 = self.Relu(self.conv4(out5))
        down0 = self.Relu(self.Denoise_subsampling0(torch.cat((out1, out6), dim=1)))
        out7 = self.Relu(self.conv5(down0))
        out8 = self.Relu(self.conv6(out7))
        out9 = self.Relu(self.conv7(out8))
        out10 = self.Relu(self.conv8(out9))
        out11 = self.Relu(self.conv9(out10))
        down1 = self.Relu(self.Denoise_subsampling1(torch.cat((down0, out11), dim=1)))
        out12 = self.Relu(self.conv10(down1))
        out13 = self.Relu(self.conv11(out12))
        out14 = self.Relu(self.conv12(out13))
        out15 = self.Relu(self.conv13(out14))
        out16 = self.Relu(self.conv14(out15))
        down2 = self.Relu(self.Denoise_subsampling2(torch.cat((down1, out16), dim=1)))
        out17 = self.Relu(self.conv15(down2))
        out18 = self.Relu(self.conv16(out17))
        out19 = self.Relu(self.conv17(out18))
        out20 = self.Relu(self.conv18(out19))
        out21 = self.Relu(self.conv19(out20))
        up0 = self.Relu(self.Denoise_deconv0(torch.cat((down2, out21), dim=1)))
        out22 = self.Relu(self.conv20(torch.cat((up0, out16), dim=1)))
        out23 = self.Relu(self.conv21(out22))
        out24 = self.Relu(self.conv22(out23))
        out25 = self.Relu(self.conv23(out24))
        out26 = self.Relu(self.conv24(out25))
        up1 = self.Relu(self.Denoise_deconv1(torch.cat((up0, out26), dim=1)))
        out27 = self.Relu(self.conv25(torch.cat((up1, out11), dim=1)))
        out28 = self.Relu(self.conv26(out27))
        out29 = self.Relu(self.conv27(out28))
        out30 = self.Relu(self.conv28(out29))
        out31 = self.Relu(self.conv29(out30))
        up2 = self.Relu(self.Denoise_deconv2(torch.cat((up1, out31), dim=1)))
        out32 = self.Relu(self.Denoiseout0(torch.cat((out6, up2), dim=1)))
        out33 = self.Relu(self.Denoiseout1(out32))
        denoise_R = out33

        return denoise_R

class Conv5(nn.Module):
    def __init__(self, channel_in, channel, kernel_size=3):
        super(Conv5, self).__init__()
        self.Relu = nn.LeakyReLU()
        self.conv0 = nn.Conv2d(channel_in, channel * 2, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv3 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv4 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)

        # self.xjh = nn.Conv2d(channel_in, channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Relu(self.conv0(x))
        x = self.Relu(self.conv1(x))
        x = self.Relu(self.conv2(x))
        x = self.Relu(self.conv3(x))
        x = self.Relu(self.conv4(x))
        # x = self.Relu(self.xjh(x))
        return x


class MergeModule(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(MergeModule, self).__init__()
        self.Relu = nn.LeakyReLU()
        self.Sigmoid = nn.Sigmoid()
        # self.se = se_block()
        self.img_inCov0 = nn.Conv2d(4, channel, kernel_size, padding=2, padding_mode='replicate', dilation=2)
        self.img_inCov1 = nn.Conv2d(channel, channel, kernel_size, padding=2, padding_mode='replicate', dilation=2)
        self.event_inCov0 = nn.Conv2d(5, channel, kernel_size, padding=2, padding_mode='replicate', dilation=2)
        self.event_inCov1 = nn.Conv2d(channel, channel, kernel_size, padding=2, padding_mode='replicate', dilation=2)

        self.Conv5_img0 = Conv5(channel * 2, channel, kernel_size)
        self.Enhance_subsampling0 = nn.Conv2d(channel * 2, channel, kernel_size=2, stride=2, padding=0)
        self.Conv5_event0 = Conv5(channel, channel, kernel_size)
        self.Enhance_subsampling0_ = nn.Conv2d(channel, channel, kernel_size=2, stride=2, padding=0)

        self.Conv5_img1 = Conv5(channel * 2, channel, kernel_size)
        self.Enhance_subsampling1 = nn.Conv2d(channel * 2, channel, kernel_size=2, stride=2, padding=0)
        self.Conv5_event1 = Conv5(channel, channel, kernel_size)
        self.Enhance_subsampling1_ = nn.Conv2d(channel, channel, kernel_size=2, stride=2, padding=0)

        self.Conv5_img2 = Conv5(channel * 2, channel, kernel_size)
        self.Enhance_subsampling2 = nn.Conv2d(channel * 2, channel, kernel_size=2, stride=2, padding=0)
        self.Conv5_event2 = Conv5(channel, channel, kernel_size)
        self.Enhance_subsampling2_ = nn.Conv2d(channel, channel, kernel_size=2, stride=2, padding=0)

        self.Conv5_img3 = Conv5(channel * 2, channel, kernel_size)

        self.Enhance_deconv0 = nn.UpsamplingBilinear2d(scale_factor=2.0)
        self.down_channel0 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.Conv5_img4 = Conv5(channel * 2, channel, kernel_size)

        self.Enhance_deconv1 = nn.UpsamplingBilinear2d(scale_factor=2.0)
        self.down_channel1 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.Conv5_img5 = Conv5(channel * 2, channel, kernel_size)

        self.Enhance_deconv2 = nn.UpsamplingBilinear2d(scale_factor=2.0)
        self.down_channel2 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)

        self.Enhanceout0 = nn.Conv2d(channel * 2, channel, kernel_size, padding=1, padding_mode='replicate')
        self.Enhanceout1 = nn.Conv2d(channel * 4, channel, kernel_size, padding=1, padding_mode='replicate')
        self.Enhanceout2 = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
        self.Enhanceout3 = nn.Conv2d(channel, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, input_L, denoise_R, extra_data):
        input_img = torch.cat((input_L, denoise_R), dim=1)
        img_out0 = self.Relu(self.img_inCov0(input_img))
        img_out0 = self.Relu(self.img_inCov1(img_out0))
        event_out0 = self.Relu(self.event_inCov0(extra_data))
        event_out0 = self.Relu(self.event_inCov1(event_out0))

        img_out1 = self.Relu(self.Conv5_img0(torch.cat((img_out0, event_out0), dim=1)))
        down0 = self.Relu(self.Enhance_subsampling0(torch.cat((img_out1, img_out0), dim=1)))
        event_out1 = self.Relu(self.Conv5_event0(event_out0))
        event_out1 = self.Relu(self.Enhance_subsampling0_(event_out1))

        img_out2 = self.Relu(self.Conv5_img1(torch.cat((down0, event_out1), dim=1)))
        down1 = self.Relu(self.Enhance_subsampling1(torch.cat((img_out2, down0), dim=1)))
        event_out2 = self.Relu(self.Conv5_event1(event_out1))
        event_out2 = self.Relu(self.Enhance_subsampling1_(event_out2))

        img_out3 = self.Relu(self.Conv5_img2(torch.cat((down1, event_out2), dim=1)))
        down2 = self.Relu(self.Enhance_subsampling2(torch.cat((img_out3, down1), dim=1)))
        event_out3 = self.Relu(self.Conv5_event2(event_out2))
        event_out3 = self.Relu(self.Enhance_subsampling2_(event_out3))

        img_out4 = self.Relu(self.Conv5_img3(torch.cat((down2, event_out3), dim=1)))

        up0 = self.Relu(self.Enhance_deconv0(torch.cat((img_out4, down2), dim=1)))
        up0 = self.Relu(self.down_channel0(up0))
        img_out5 = self.Relu(self.Conv5_img4(torch.cat((up0, img_out3), dim=1)))

        up1 = self.Relu(self.Enhance_deconv1(torch.cat((img_out5, up0), dim=1)))
        up1 = self.Relu(self.down_channel1(up1))
        img_out6 = self.Relu(self.Conv5_img5(torch.cat((up1, img_out2), dim=1)))

        up2 = self.Relu(self.Enhance_deconv2(torch.cat((img_out6, up1), dim=1)))
        up2 = self.Relu(self.down_channel2(up2))

        img_out7 = self.Relu(self.Enhanceout0(torch.cat((up2, img_out1), dim=1)))

        up0_1 = F.interpolate(up0, size=(input_img.size()[2], input_img.size()[3]))
        up1_1 = F.interpolate(up1, size=(input_img.size()[2], input_img.size()[3]))
        up2_1 = F.interpolate(up2, size=(input_img.size()[2], input_img.size()[3]))

        output = self.Relu(self.Enhanceout1(torch.cat((up0_1, up1_1, up2_1, img_out7), dim=1)))
        output = self.Relu(self.Enhanceout2(output))
        output = self.Sigmoid(self.Enhanceout3(output))
        return output


class RelightNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(RelightNet, self).__init__()
        # self.EnhanceModule = EnhanceModule1()
        self.MergeModule = MergeModule()

    def forward(self, input_L, denoise_R, extra_data):

        output = self.MergeModule(input_L, denoise_R, extra_data)
        return output


class DoubleConvBlock(nn.Module):
    """double conv layers block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class DownBlock(nn.Module):
    """Downscale block: maxpool -> double conv block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
class BridgeDown(nn.Module):
    """Downscale bottleneck block: maxpool -> conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
class BridgeUP(nn.Module):
    """Downscale bottleneck block: conv -> transpose conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_up = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.conv_up(x)
class UpBlock(nn.Module):
    """Upscale block: double conv block -> transpose conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConvBlock(in_channels * 2, in_channels)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return torch.relu(self.up(x))
class OutputBlock(nn.Module):
    """Output block: double conv block -> output conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_conv = nn.Sequential(
            DoubleConvBlock(in_channels * 2, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        return self.out_conv(x)
class RestoreNet(nn.Module):
    def __init__(self):
        super(RestoreNet, self).__init__()
        self.n_channels = 3
        self.encoder_inc = DoubleConvBlock(self.n_channels, 24)
        self.encoder_down1 = DownBlock(24, 48)
        self.encoder_down2 = DownBlock(48, 96)
        self.encoder_down3 = DownBlock(96, 192)
        self.encoder_bridge_down = BridgeDown(192, 384)
        self.decoder_bridge_up = BridgeUP(384, 192)
        self.decoder_up1 = UpBlock(192, 96)
        self.decoder_up2 = UpBlock(96, 48)
        self.decoder_up3 = UpBlock(48, 24)
        self.decoder_out = OutputBlock(24, self.n_channels)

    def forward(self, x):
        x1 = self.encoder_inc(x)
        x2 = self.encoder_down1(x1)
        x3 = self.encoder_down2(x2)
        x4 = self.encoder_down3(x3)
        x5 = self.encoder_bridge_down(x4)
        x = self.decoder_bridge_up(x5)
        x = self.decoder_up1(x, x4)
        x = self.decoder_up2(x, x3)
        x = self.decoder_up3(x, x2)
        out = self.decoder_out(x, x1)
        return out

def angle(a, b):
    vector = torch.mul(a, b)
    up = torch.sum(vector)
    down = torch.sqrt(torch.sum(torch.square(a))) * torch.sqrt(torch.sum(torch.square(b)))
    theta = torch.acos(up / down)  # 弧度制
    return theta
def color_loss(out_image, gt_image):  # 颜色损失  希望增强前后图片的颜色一致性 (b,c,h,w)
    loss = torch.mean(angle(out_image[:, 0, :, :], gt_image[:, 0, :, :]) +
                      angle(out_image[:, 1, :, :], gt_image[:, 1, :, :]) +
                      angle(out_image[:, 2, :, :], gt_image[:, 2, :, :]))
    return loss

class Xnet(nn.Module):
    def __init__(self):
        super(Xnet, self).__init__()
        self.DecomNet = DecomNet()
        self.DenoiseNet = DenoiseNet()
        self.RelightNet = RelightNet()
        self.RestoreNet = RestoreNet()

        self.mse_loss = nn.MSELoss()

        self.train_op_Decom = torch.optim.Adam(self.DecomNet.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0001)
        self.train_op_Denoise = torch.optim.Adam(self.DenoiseNet.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0001)
        self.train_op_Relight = torch.optim.Adam(self.RelightNet.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0001)
        self.train_op_Restore = torch.optim.Adam(self.RestoreNet.parameters(), lr=1e-4, betas=(0.9, 0.999),weight_decay=0.0001)

        parameters = list(self.RelightNet.parameters()) + list(self.RestoreNet.parameters())
        self.train_op_Both = torch.optim.Adam(parameters, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0001)

        ckpt_dict = torch.load(r'weight/Decom/28000.tar')
        self.DecomNet.load_state_dict(ckpt_dict)
        ckpt_dict = torch.load(r'weight/Denoise/28000.tar')
        self.DenoiseNet.load_state_dict(ckpt_dict)

    def forward(self, image, extra_data, label):
        R_low, I_low = self.DecomNet(image)
        denoise_R = self.DenoiseNet(I_low, R_low)
        I_delta = self.RelightNet(I_low, denoise_R, extra_data)
        I_delta_3 = torch.cat((I_delta, I_delta, I_delta), dim=1)

        result = self.RestoreNet(denoise_R * I_delta_3)
        if self.train_phase == "Relight":
            self.Relight_loss = F.l1_loss(denoise_R * I_delta_3, label).cuda()
            # self.Relight_vgg = compute_vgg_loss(denoise_R * I_delta_3, label).cuda()
            # relight_mse = self.mse_loss(denoise_R * I_delta_3, label).cuda()
            # SSIMloss = MS_SSIMLoss(denoise_R * I_delta_3, label).cuda()
            self.loss_Relight = self.Relight_loss
        elif self.train_phase == "Restore":
            self.Restore_loss = F.l1_loss(result, label).cuda() + color_loss(result, label).cuda()
            # self.Restore_vgg = compute_vgg_loss(result, label).cuda()
            self.loss_Restore = self.Restore_loss
        elif self.train_phase == "Both":
            self.loss_Both = F.l1_loss(denoise_R * I_delta_3, label).cuda() + F.l1_loss(result, label).cuda() + color_loss(result, label).cuda()
        elif self.train_phase == "Decom":
            R_high, I_high = self.DecomNet(label)
            I_low_3 = torch.cat((I_low, I_low, I_low), dim=1)
            I_high_3 = torch.cat((I_high, I_high, I_high), dim=1)
            # # DecomNet_loss
            self.vgg_loss = compute_vgg_loss(R_low * I_low_3, image).cuda() + compute_vgg_loss(R_high * I_high_3, label).cuda()
            self.recon_loss_low = F.l1_loss(R_low * I_low_3, image).cuda()
            self.recon_loss_high = F.l1_loss(R_high * I_high_3, label).cuda()
            self.recon_loss_mutal_low = F.l1_loss(R_high * I_low_3, image).cuda()
            self.recon_loss_mutal_high = F.l1_loss(R_low * I_high_3, label).cuda()
            self.loss_Decom = self.recon_loss_low + \
                              self.recon_loss_high + \
                              0.1 * self.recon_loss_mutal_low + \
                              0.1 * self.recon_loss_mutal_high + \
                              0.1 * self.vgg_loss
        elif self.train_phase == "Denoise":
            R_high, I_high = self.DecomNet(label)
            self.denoise_loss = F.l1_loss(denoise_R, R_high).cuda()
            # self.denoise_vgg = compute_vgg_loss(denoise_R, R_high).cuda()
            self.loss_Denoise = self.denoise_loss

        # self.output_R_low = R_low.detach().cpu()
        # self.output_I_low = I_low_3.detach().cpu()
        # self.output_I_delta = I_delta_3.detach().cpu()
        # self.output_R_denoise = denoise_R.detach().cpu()
        # self.output_S = denoise_R.detach().cpu() * I_delta_3.detach().cpu()

    def predict(self, image, extra_data):
        R_low, I_low = self.DecomNet(image)
        denoise_R = self.DenoiseNet(I_low, R_low)
        I_delta = self.RelightNet(I_low, denoise_R, extra_data)
        I_delta_3 = torch.cat((I_delta, I_delta, I_delta), dim=1)
        I_low_3 = torch.cat((I_low, I_low, I_low), dim=1)
        result = self.RestoreNet(I_delta_3*denoise_R)
        self.output_R_low = R_low.detach().cpu()
        self.output_I_low = I_low_3.detach().cpu()
        self.output_I_delta = I_delta_3.detach().cpu()
        self.output_R_denoise = denoise_R.detach().cpu()
        self.output_S = denoise_R.detach().cpu() * I_delta_3.detach().cpu()
        self.result = result.detach().cpu()

        return image, self.output_R_low, self.output_I_low, self.output_I_delta, self.output_R_denoise, self.output_S, self.result

    def my_train(self, image, extra_data, label, train_phase):
        image = image.cuda()  # 将输入数据移动到与模型相同的设备上
        extra_data = extra_data.cuda()
        label = label.cuda()

        self.train_phase = train_phase
        self.forward(image, extra_data, label)
        if self.train_phase == "Decom":
            self.train_op_Decom.zero_grad()
            self.loss_Decom.backward()
            self.train_op_Decom.step()
            loss = self.loss_Decom.item()
        elif self.train_phase == 'Denoise':
            self.train_op_Denoise.zero_grad()
            self.loss_Denoise.backward()
            self.train_op_Denoise.step()
            loss = self.loss_Denoise.item()
        elif self.train_phase == "Relight":
            self.train_op_Relight.zero_grad()
            self.loss_Relight.backward()
            self.train_op_Relight.step()
            loss = self.loss_Relight.item()
        elif self.train_phase == "Restore":
            self.train_op_Restore.zero_grad()
            self.loss_Restore.backward()
            self.train_op_Restore.step()
            loss = self.loss_Restore.item()
        elif self.train_phase == "Both":
            self.train_op_Both.zero_grad()
            self.loss_Both.backward()
            self.train_op_Both.step()
            loss = self.loss_Both.item()

        else:
            print("不存在这种模式\n")
            loss = None

        return loss
