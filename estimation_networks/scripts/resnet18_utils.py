'''
Refers to and be adapted from
https://github.com/julianstastny/VAE-ResNet18-PyTorch
'''

import torch
from torch import nn, optim
import torch.nn.functional as F
from pprint import pprint

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  
        max_out, _ = torch.max(x, dim=1, keepdim=True)  
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        attention_map = self.sigmoid(out)
        return x * attention_map

class ResizeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockEnc(nn.Module):
    def __init__(self, in_planes, stride=1, attention=False, att_kernel_size=7):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if attention:
            self.satt = SpatialAttention(kernel_size=att_kernel_size)
        else:
            self.satt = None

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        fea = self.bn2(self.conv2(out))
        if self.satt is not None:
            fea = self.satt(fea)
        out = fea + self.shortcut(x)
        out = torch.relu(out)
        return out

class BasicBlockDec(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet18Enc(nn.Module):
    def __init__(self, img_size=512, latent_dim=64, in_chs=3, downsize=4, in_planes=32, num_Blocks=[2, 2, 2, 2]):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        
        # input layer, downsample 2
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_chs, in_planes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True)
        )
        
        # middle layer
        self.in_planes = in_planes
        self.layer1 = self._make_layer(BasicBlockEnc, 64,  num_Blocks[0], stride=2)#, attention=True, kernel_size=7) # 256 x 256
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2, attention=True, kernel_size=7) # 128 x 128
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2, attention=True, kernel_size=7) #  64 x 64
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2, attention=True, kernel_size=7) #  32 x 32
        
        # output layer
        last_size = img_size//2**5 # if img_size==512, last_size=16
        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(last_size//downsize, last_size//downsize)), # 4x4
            nn.Flatten(),
            nn.Linear(512*last_size**2//downsize**2, latent_dim)
        )

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride, attention=False, kernel_size=7):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride, attention, kernel_size)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output_layer(x)
        return x

class ResNet18Dec(nn.Module):
    def __init__(self, last_size=4, resize_factor=4, latent_dim=64, out_chs=3, in_planes=512, num_Blocks=[2, 2, 2, 2]):
        super().__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(latent_dim, in_planes*last_size**2),
            nn.Unflatten(1, (in_planes, last_size, last_size)),
            nn.Upsample(scale_factor=resize_factor)
        )
        
        self.in_planes = in_planes
        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64,  num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 32,  num_Blocks[0], stride=2) 
        
        self.output_layer = nn.Sequential(
            ResizeConv2d(32, out_chs, kernel_size=3, scale_factor=2),
            nn.Sigmoid() # converge faster with sigmoid()
        ) 

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.input_layer(z)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.output_layer(x)
        return x

class ResNetImgAE18(nn.Module):

    def __init__(self, img_size=512, in_chs=3, latent_dim=64, ret_emed=False):
        super().__init__()
        self.ret_embed = ret_emed
        self.encoder = ResNet18Enc(
            img_size=img_size, latent_dim=latent_dim, in_chs=in_chs, downsize=4, in_planes=32, num_Blocks=[2, 2, 2, 2]
        )
        self.decoder = ResNet18Dec( 
            last_size=4, resize_factor=4, latent_dim=latent_dim, out_chs=in_chs, in_planes=512, num_Blocks=[2, 2, 2, 2]
        )

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        if self.ret_embed:
            return x, z
        else:
            return x


import torch
class SimpleCNNWithAttention(nn.Module):
    def __init__(self):
        super(SimpleCNNWithAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.sa = SpatialAttention(kernel_size=7) 
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.sa(x)  
        x = self.relu(self.conv2(x))
        return x


if __name__ == '__main__':
    input_tensor = torch.randn(16, 3, 128, 128)  
    model = SimpleCNNWithAttention()
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # (16, 128, 32, 32)
    
    