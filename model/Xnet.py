import os
from model.attention import eca_block
from model.attention import spatial_attention
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.restoration import *


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

class Res(nn.Module):
    def __init__(self, channel, kernel_size=3):
        super(Res, self).__init__()
        self.Relu = nn.LeakyReLU()

        self.conv0 = nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(channel * 2, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)

        # self.xjh = nn.Conv2d(channel_in, channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        x0 = self.Relu(self.conv0(x))
        x1 = self.Relu(self.conv1(x0))
        x2 = self.Relu(self.conv2(x1))
        output = x2 + x
        return output


class MergeModule(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(MergeModule, self).__init__()
        self.Relu = nn.LeakyReLU()
        self.Sigmoid = nn.Sigmoid()

        self.img_inCov0 = nn.Conv2d(4, channel, kernel_size, padding=2, padding_mode='replicate', dilation=2)
        self.img_inCov1 = nn.Conv2d(channel, int(channel/2), kernel_size, padding=2, padding_mode='replicate', dilation=2)
        self.event_inCov0 = nn.Conv2d(5, channel, kernel_size, padding=2, padding_mode='replicate', dilation=2)
        self.event_inCov1 = nn.Conv2d(channel, int(channel/2), kernel_size, padding=2, padding_mode='replicate', dilation=2)

        self.spatial_event = spatial_attention()
        self.spatial_img = spatial_attention()
        self.channel_attention = eca_block(channel)

        self.res0 = Res(channel)
        self.Enhance_subsampling0 = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1)
        self.res1 = Res(channel)
        self.Enhance_subsampling1 = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1)
        self.res2 = Res(channel)
        self.Enhance_subsampling2 = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1)
        self.res3 = Res(channel)
        self.Enhance_subsampling3 = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1)

        self.res4 = Res(channel)
        self.up0 = nn.UpsamplingBilinear2d(scale_factor=2.0)
        self.down_channel0 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)

        self.res5 = Res(channel)
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2.0)
        self.down_channel1 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)

        self.res6 = Res(channel)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2.0)
        self.down_channel2 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)

        self.res7 = Res(channel)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2.0)

        self.Enhanceout0 = nn.Conv2d(channel * 2, channel, kernel_size, padding=1, padding_mode='replicate')
        self.Enhanceout1 = nn.Conv2d(channel * 4, channel, kernel_size, padding=1, padding_mode='replicate')
        self.Enhanceout2 = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
        self.Enhanceout3 = nn.Conv2d(channel, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, denoise_R, input_L, extra_data):
        input_img = torch.cat((denoise_R, input_L), dim=1)
        img_out0 = self.Relu(self.img_inCov0(input_img))
        img_out0 = self.Relu(self.img_inCov1(img_out0))
        event_out0 = self.Relu(self.event_inCov0(extra_data))
        event_out0 = self.Relu(self.event_inCov1(event_out0))

        img_out1, img_map = self.spatial_event(img_out0)
        event_out1, event_map = self.spatial_img(event_out0)
        out0 = self.channel_attention(torch.cat((img_out1, event_out1), dim=1))

        out1 = self.Relu(self.res0(out0))
        out1 = self.Relu(self.Enhance_subsampling0(out1))
        out2 = self.Relu(self.res1(out1))
        out2 = self.Relu(self.Enhance_subsampling1(out2))
        out3 = self.Relu(self.res2(out2))
        out3 = self.Relu(self.Enhance_subsampling2(out3))
        out4 = self.Relu(self.res3(out3))
        out4 = self.Relu(self.Enhance_subsampling3(out4))

        out5 = self.Relu(self.res4(out4))
        out5 = self.up0(out5)
        out5 = self.Relu(self.down_channel0(torch.cat((out5, out3), dim=1)))

        out6 = self.Relu(self.res5(out5))
        out6 = self.up1(out6)
        out6 = self.Relu(self.down_channel1(torch.cat((out6, out2), dim=1)))

        out7 = self.Relu(self.res6(out6))
        out7 = self.up2(out7)
        out7 = self.Relu(self.down_channel2(torch.cat((out7, out1), dim=1)))

        out8 = self.Relu(self.res7(out7))
        out8 = self.up3(out8)
        out8 = self.Relu(self.Enhanceout0(torch.cat((out8, out0), dim=1)))

        up0_1 = F.interpolate(out5, size=(input_img.size()[2], input_img.size()[3]))
        up1_1 = F.interpolate(out6, size=(input_img.size()[2], input_img.size()[3]))
        up2_1 = F.interpolate(out7, size=(input_img.size()[2], input_img.size()[3]))

        output = self.Relu(self.Enhanceout1(torch.cat((up0_1, up1_1, up2_1, out8), dim=1)))
        output = self.Relu(self.Enhanceout2(output))
        output = self.Sigmoid(self.Enhanceout3(output))
        return output, img_map, event_map


class RelightNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(RelightNet, self).__init__()
        # self.EnhanceModule = EnhanceModule1()
        self.MergeModule = MergeModule()

    def forward(self, denoise_R, input_L, extra_data):

        output = self.MergeModule(denoise_R, input_L, extra_data)
        return output


class RestoreNet(nn.Module):
    def __init__(self):
        super(RestoreNet, self).__init__()

        def ConvBlock(in_channels, out_channels):
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            return block

        def UpConv(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # Encoder
        self.enc1 = ConvBlock(3, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder
        self.upconv4 = UpConv(1024, 512)
        self.dec4 = ConvBlock(1024, 512)
        self.upconv3 = UpConv(512, 256)
        self.dec3 = ConvBlock(512, 256)
        self.upconv2 = UpConv(256, 128)
        self.dec2 = ConvBlock(256, 128)
        self.upconv1 = UpConv(128, 64)
        self.dec1 = ConvBlock(128, 64)

        # Final Convolution: map back to the original number of channels
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc1p = self.pool(enc1)
        enc2 = self.enc2(enc1p)
        enc2p = self.pool(enc2)
        enc3 = self.enc3(enc2p)
        enc3p = self.pool(enc3)
        enc4 = self.enc4(enc3p)
        enc4p = self.pool(enc4)

        # Bottleneck
        bottleneck = self.bottleneck(enc4p)

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return self.final_conv(dec1)



class Xnet(nn.Module):
    def __init__(self):
        super(Xnet, self).__init__()
        self.DecomNet = DecomNet()
        self.DenoiseNet = DenoiseNet()
        self.RelightNet = RelightNet()
        # self.RestoreNet = RestoreNet()

        self.mse_loss = nn.MSELoss()

        self.train_op_Decom = torch.optim.Adam(self.DecomNet.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0001)
        self.train_op_Denoise = torch.optim.Adam(self.DenoiseNet.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0001)
        self.train_op_Relight = torch.optim.Adam(self.RelightNet.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0001)
        all_params = list(self.DecomNet.parameters()) + list(self.DenoiseNet.parameters()) + list(self.RelightNet.parameters())
        self.train_op_together = torch.optim.Adam(all_params, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0001)

        ckpt_dict = torch.load(r'weight/DecomNet_weights.pth')
        self.DecomNet.load_state_dict(ckpt_dict)
        ckpt_dict = torch.load(r'weight/DenoiseNet_weights.pth')
        self.DenoiseNet.load_state_dict(ckpt_dict)
        ckpt_dict = torch.load(r'weight/RelightNet_weights.pth')
        self.RelightNet.load_state_dict(ckpt_dict)
        # ckpt_dict = torch.load(r'weight/RestoreNet_weights.pth')
        # self.RestoreNet.load_state_dict(ckpt_dict)

    def forward(self, image, extra_data, label):
        R_low, I_low = self.DecomNet(image)
        denoise_R = self.DenoiseNet(I_low, R_low)
        I_delta, _, _ = self.RelightNet(denoise_R, I_low, extra_data)
        I_delta_3 = torch.cat((I_delta, I_delta, I_delta), dim=1)

        # result = self.RestoreNet(denoise_R * I_delta_3)
        if self.train_phase == "Relight":
            self.Relight_loss = F.l1_loss(denoise_R * I_delta_3, label).cuda()
            # self.Relight_vgg = compute_vgg_loss(denoise_R * I_delta_3, label).cuda()
            # relight_mse = self.mse_loss(denoise_R * I_delta_3, label).cuda()
            # SSIMloss = MS_SSIMLoss(denoise_R * I_delta_3, label).cuda()
            self.loss_Relight = self.Relight_loss # + self.Relight_vgg * 0.1
        # elif self.train_phase == "Restore":
        #     self.Restore_loss = F.l1_loss(result, label).cuda()
            # self.Restore_loss = self.mse_loss(result, label).cuda()
            # self.Restore_loss = color_loss(result, label).cuda()
            # self.Restore_vgg = compute_vgg_loss(result, label).cuda()
            # self.loss_Restore = self.Restore_loss
        elif self.train_phase == "Decom":
            R_high, I_high = self.DecomNet(label)
            I_low_3 = torch.cat((I_low, I_low, I_low), dim=1)
            I_high_3 = torch.cat((I_high, I_high, I_high), dim=1)
            # # DecomNet_loss
            # self.vgg_loss = compute_vgg_loss(R_low * I_low_3, image).cuda() + compute_vgg_loss(R_high * I_high_3, label).cuda()
            self.recon_loss_low = F.l1_loss(R_low * I_low_3, image).cuda()
            self.recon_loss_high = F.l1_loss(R_high * I_high_3, label).cuda()
            self.recon_loss_mutal_low = F.l1_loss(R_high * I_low_3, image).cuda()
            self.recon_loss_mutal_high = F.l1_loss(R_low * I_high_3, label).cuda()
            self.loss_Decom = self.recon_loss_low + self.recon_loss_high + 0.1 * self.recon_loss_mutal_low + 0.1 * self.recon_loss_mutal_high # + 0.1 * self.vgg_loss
        elif self.train_phase == "Denoise":
            R_high, I_high = self.DecomNet(label)
            self.denoise_loss = F.l1_loss(denoise_R, R_high).cuda()
            # self.denoise_vgg = compute_vgg_loss(denoise_R, R_high).cuda()
            self.loss_Denoise = self.denoise_loss # + self.denoise_vgg * 0.1
        elif self.train_phase == "together":
            self.together_loss = F.l1_loss(denoise_R * I_delta_3, label).cuda()
            self.together_vgg = compute_vgg_loss(denoise_R * I_delta_3, label).cuda()
            self.loss_together = self.together_loss + self.together_vgg * 0.1

        # self.output_R_low = R_low.detach().cpu()
        # self.output_I_low = I_low_3.detach().cpu()
        # self.output_I_delta = I_delta_3.detach().cpu()
        # self.output_R_denoise = denoise_R.detach().cpu()
        # self.output_S = denoise_R.detach().cpu() * I_delta_3.detach().cpu()

    def predict(self, image, extra_data):
        R_low, I_low = self.DecomNet(image)
        denoise_R = self.DenoiseNet(I_low, R_low)
        I_delta, img_map, event_map = self.RelightNet(denoise_R, I_low, extra_data)
        I_delta_3 = torch.cat((I_delta, I_delta, I_delta), dim=1)
        I_low_3 = torch.cat((I_low, I_low, I_low), dim=1)
        # result = self.RestoreNet(I_delta_3*denoise_R)
        self.output_R_low = R_low.detach().cpu()
        self.output_I_low = I_low_3.detach().cpu()
        self.output_I_delta = I_delta_3.detach().cpu()
        self.output_R_denoise = denoise_R.detach().cpu()
        self.output_S = denoise_R.detach().cpu() * I_delta_3.detach().cpu()
        # self.result = result.detach().cpu()
        self.img_map = img_map.detach().cpu()
        self.event_map = event_map.detach().cpu()
        return image, self.output_R_low, self.output_I_low, self.output_I_delta, self.output_R_denoise, self.output_S, self.img_map, self.event_map

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
        elif self.train_phase == "together":
            self.train_op_together.zero_grad()
            self.loss_together.backward()
            self.train_op_together.step()
            loss = self.loss_together.item()
        # elif self.train_phase == "Restore":
        #     self.train_op_Restore.zero_grad()
        #     self.loss_Restore.backward()
        #     self.train_op_Restore.step()
        #     loss = self.loss_Restore.item()
        # elif self.train_phase == "Both":
        #     self.train_op_Both.zero_grad()
        #     self.loss_Both.backward()
        #     self.train_op_Both.step()
        #     loss = self.loss_Both.item()

        else:
            print("不存在这种模式\n")
            loss = None

        return loss
