
import functools
from math import exp
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo
import numpy as np

from torch.nn.utils import spectral_norm as SpectralNorm

#<editor-fold desc="Các hàm và lớp phụ trợ">
def todevice(net,gpu_id):
    if gpu_id != '-1':
        net.cuda()
    return net

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list): nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters(): param.requires_grad = requires_grad

def get_norm_layer(norm_type='instance', mod='2d'):
    if norm_type == 'batch': norm_layer = functools.partial(nn.BatchNorm2d if mod == '2d' else nn.BatchNorm3d, affine=True)
    elif norm_type == 'instance': norm_layer = functools.partial(nn.InstanceNorm2d if mod == '2d' else nn.InstanceNorm3d, affine=False, track_running_stats=True)
    else: norm_layer = None
    return norm_layer

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal': init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier': init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming': init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal': init.orthogonal_(m.weight.data, gain=gain)
            if hasattr(m, 'bias') and m.bias is not None: init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain); init.constant_(m.bias.data, 0.0)
    net.apply(init_func)

class ResnetBlockSpectralNorm(nn.Module):
    def __init__(self, dim, padding_type, activation=nn.LeakyReLU(0.2), use_dropout=False):
        super(ResnetBlockSpectralNorm, self).__init__()
        conv_block = []
        p = 0
        if padding_type == 'reflect': conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate': conv_block += [nn.ReplicationPad2d(1)]
        else: p = 1
        conv_block += [SpectralNorm(nn.Conv2d(dim, dim, 3, padding=p)), activation]
        if use_dropout: conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == 'reflect': conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate': conv_block += [nn.ReplicationPad2d(1)]
        else: p = 1
        conv_block += [SpectralNorm(nn.Conv2d(dim, dim, 3, padding=p))]
        self.conv_block = nn.Sequential(*conv_block)
    def forward(self, x): return x + self.conv_block(x)

model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'}
def conv3x3(in_planes, out_planes, stride=1): return nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)
def conv1x1(in_planes, out_planes, stride=1): return nn.Conv2d(in_planes, out_planes, 1, stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride); self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes); self.bn2 = norm_layer(planes)
        self.downsample = downsample; self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        if self.downsample is not None: identity = self.downsample(x)
        out += identity; out = self.relu(out)
        return out

class ResNet(nn.Module):
    # *** SỬA LỖI: Loại bỏ tầng self.avgpool và self.fc không cần thiết ***
    def __init__(self, block, layers, norm_layer=nn.BatchNorm2d):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes*block.expansion, stride), norm_layer(planes*block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample, norm_layer=norm_layer)]
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks): layers.append(block(self.inplanes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return x # Trả về output của layer cuối, không qua avgpool và fc

def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        try:
            # Tải với strict=False để bỏ qua các key fc.weight, fc.bias không khớp
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
        except Exception as e:
            print(f"Không thể tải pre-trained weights cho ResNet18. Lỗi: {e}. Tiếp tục với trọng số ngẫu nhiên.")
    return model

#<Phần còn lại của file giữ nguyên>
class GANLoss(nn.Module):
    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss() if gan_mode == 'lsgan' else nn.BCEWithLogitsLoss()
    def get_target_tensor(self, prediction, target_is_real):
        return self.real_label.expand_as(prediction) if target_is_real else self.fake_label.expand_as(prediction)
    def __call__(self, prediction, target_is_real):
        return self.loss(prediction, self.get_target_tensor(prediction, target_is_real))

class HingeLossD(nn.Module):
    def forward(self, dis_fake, dis_real): return torch.mean(F.relu(1. - dis_real)) + torch.mean(F.relu(1. + dis_fake))

class HingeLossG(nn.Module):
    def forward(self, dis_fake): return -torch.mean(dis_fake)

class VGGLoss(nn.Module):
    def __init__(self, gpu_id='-1'):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().to(torch.device(f"cuda:{gpu_id}" if gpu_id != '-1' else "cpu"))
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = sum(self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach()) for i in range(len(x_vgg)))
        return loss

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        try: # Dùng API mới nếu có
            vgg_pretrained_features = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        except: # Dùng API cũ
            vgg_pretrained_features = models.vgg19(pretrained=True).features
            
        self.slice1 = nn.Sequential(*[vgg_pretrained_features[x] for x in range(2)])
        self.slice2 = nn.Sequential(*[vgg_pretrained_features[x] for x in range(2, 7)])
        self.slice3 = nn.Sequential(*[vgg_pretrained_features[x] for x in range(7, 12)])
        self.slice4 = nn.Sequential(*[vgg_pretrained_features[x] for x in range(12, 21)])
        self.slice5 = nn.Sequential(*[vgg_pretrained_features[x] for x in range(21, 30)])
        if not requires_grad:
            for param in self.parameters(): param.requires_grad = False
    def forward(self, X):
        h1 = self.slice1(X)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        return [h1, h2, h3, h4, h5]

# ... các lớp còn lại không thay đổi ...
#</editor-fold>
