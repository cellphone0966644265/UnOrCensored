
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

import sys
from pathlib import Path
project_root_for_import = Path(__file__).resolve().parents[2]
if str(project_root_for_import) not in sys.path:
    sys.path.append(str(project_root_for_import))

from scripts_AI import model_util
# *** SỬA LỖI: Import trực tiếp phiên bản spectral_norm cũ để khớp với file .pth ***
from torch.nn.utils import spectral_norm as SpectralNorm

#<editor-fold desc="Kiến trúc BiSeNet (Đã đúng)">
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))

class Spatial_path(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock1 = ConvBlock(in_channels=3, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=128)
        self.convblock3 = ConvBlock(in_channels=128, out_channels=256)
    def forward(self, input):
        x = self.convblock1(input)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x

class AttentionRefinementModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    def forward(self, input):
        x = self.avgpool(input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        x = self.sigmoid(x)
        x = torch.mul(input, x)
        return x

class FeatureFusionModule(torch.nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        x = self.avgpool(feature)
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x

class BiSeNet(torch.nn.Module):
    def __init__(self, num_classes, context_path='resnet18', train_flag=True):
        super().__init__()
        self.saptial_path = Spatial_path()
        self.sigmoid = nn.Sigmoid()
        self.context_path = model_util.resnet18(pretrained=train_flag)
        self.attention_refinement_module1 = AttentionRefinementModule(256, 256)
        self.attention_refinement_module2 = AttentionRefinementModule(512, 512)
        self.supervision1 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
        self.supervision2 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
        self.feature_fusion_module = FeatureFusionModule(num_classes, 1024)
        self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)
        self.init_weight()

    def init_weight(self):
        for name, m in self.named_modules():
            if 'context_path' not in name:
                if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-5; m.momentum = 0.1
                    nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def forward(self, input):
        sx = self.saptial_path(input)
        x = self.context_path.conv1(input); x = self.context_path.relu(self.context_path.bn1(x)); x = self.context_path.maxpool(x)
        feature1 = self.context_path.layer1(x); feature2 = self.context_path.layer2(feature1)
        feature3 = self.context_path.layer3(feature2); feature4 = self.context_path.layer4(feature3)
        tail = torch.mean(feature4, 3, keepdim=True); tail = torch.mean(tail, 2, keepdim=True)
        cx1, cx2 = feature3, feature4
        cx1 = self.attention_refinement_module1(cx1); cx2 = self.attention_refinement_module2(cx2)
        cx2 = torch.mul(cx2, tail)
        cx1 = F.interpolate(cx1, size=sx.size()[-2:], mode='bilinear', align_corners=True)
        cx2 = F.interpolate(cx2, size=sx.size()[-2:], mode='bilinear', align_corners=True)
        cx = torch.cat((cx1, cx2), dim=1)
        result = self.feature_fusion_module(sx, cx)
        result = F.interpolate(result, scale_factor=8, mode='bilinear', align_corners=True)
        result = self.conv(result)
        if self.training:
            cx1_sup = F.interpolate(self.supervision1(cx1), size=input.size()[-2:], mode='bilinear', align_corners=True)
            cx2_sup = F.interpolate(self.supervision2(cx2), size=input.size()[-2:], mode='bilinear', align_corners=True)
            return self.sigmoid(result), self.sigmoid(cx1_sup), self.sigmoid(cx2_sup)
        return self.sigmoid(result)
#</editor-fold>

#<editor-fold desc="Kiến trúc ResnetGenerator (Đã đúng)">
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect': conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate': conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero': p = 1
        else: raise NotImplementedError()
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout: conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == 'reflect': conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate': conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero': p = 1
        else: raise NotImplementedError()
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)
    def forward(self, x):
        return x + self.conv_block(x)

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, 7, padding=0, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf*mult, ngf*mult*2, 3, stride=2, padding=1, bias=use_bias), norm_layer(ngf*mult*2), nn.ReLU(True)]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type, norm_layer, use_dropout, use_bias)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf*mult, int(ngf*mult/2), 3, stride=2, padding=1, output_padding=1, bias=use_bias), norm_layer(int(ngf*mult/2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, 7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)
    def forward(self, input):
        return self.model(input)
#</editor-fold>

#<editor-fold desc="Kiến trúc BVDNet (Sửa lỗi)">
class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1):
        super().__init__()
        self.convup = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.ReflectionPad2d(padding),
                                  SpectralNorm(nn.Conv2d(in_channel, out_channel, kernel_size)), nn.LeakyReLU(0.2))
    def forward(self, input): return self.convup(input)

class Encoder2d(nn.Module):
    def __init__(self, input_nc, ngf=64, n_downsampling=3, activation=nn.LeakyReLU(0.2)):
        super(Encoder2d, self).__init__()
        model = [nn.ReflectionPad2d(3), SpectralNorm(nn.Conv2d(input_nc, ngf, 7, padding=0)), activation]
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.ReflectionPad2d(1), SpectralNorm(nn.Conv2d(ngf*mult, ngf*mult*2, 3, stride=2, padding=0)), activation]
        self.model = nn.Sequential(*model)
    def forward(self, input): return self.model(input)

class Encoder3d(nn.Module):
    def __init__(self, input_nc, ngf=64, n_downsampling=3, activation=nn.LeakyReLU(0.2)):
        super(Encoder3d, self).__init__()
        model = [SpectralNorm(nn.Conv3d(input_nc, ngf, 3, padding=1)), activation]
        for i in range(n_downsampling):
            mult = 2**i
            model += [SpectralNorm(nn.Conv3d(ngf*mult, ngf*mult*2, 3, stride=2, padding=1)), activation]
        self.model = nn.Sequential(*model)
    def forward(self, input): return self.model(input)

class BVDNet(nn.Module):
    def __init__(self, N=2, n_downsampling=3, n_blocks=4, input_nc=3, output_nc=3, activation=nn.LeakyReLU(0.2)):
        super(BVDNet, self).__init__()
        self.N = N
        self.encoder3d = Encoder3d(input_nc, 64, n_downsampling, activation)
        self.encoder2d = Encoder2d(input_nc, 64, n_downsampling, activation)
        ngf = 64; mult = 2**n_downsampling
        self.blocks = nn.Sequential(*[model_util.ResnetBlockSpectralNorm(ngf*mult, 'reflect', activation) for _ in range(n_blocks)])
        decoder = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            decoder += [UpBlock(ngf*mult, int(ngf*mult/2))]
        decoder += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, 7, padding=0), nn.Tanh()]
        self.decoder = nn.Sequential(*decoder)

    def forward(self, stream, previous):
        this_shortcut = stream[:,:,self.N]
        stream = self.encoder3d(stream)
        stream = stream.view(stream.size(0), stream.size(1), stream.size(3), stream.size(4))
        previous = self.encoder2d(previous)
        x = stream + previous
        x = self.blocks(x)
        x = self.decoder(x)
        x = x + this_shortcut
        return torch.tanh(x) # Dùng Tanh thay vì Limiter để nhất quán
#</editor-fold>
