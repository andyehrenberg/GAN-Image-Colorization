import torch.nn.functional as F
import torch
import torch.nn as nn
from spectral_norm import SpectralNorm

class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, normalize=None, kernel_size=4, stride=2, padding=1, bias=True, dropout=0, activation_fn=nn.LeakyReLU(0.2)):
        super(ConvBlock, self).__init__()
        conv = nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        torch.nn.init.xavier_uniform_(conv.weight)
        model = [conv]
        if normalize == "batch":
            model.append(nn.BatchNorm2d(out_size))
        elif normalize == "spectral":
            model = [SpectralNorm(conv)]
            
        model.append(activation_fn)
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        x = self.model(x)
        return x

class ConvTransposeBlock(nn.Module):
    def __init__(self, in_size, out_size, normalize="batch", kernel_size=4, stride=2, padding=1, bias=True, dropout=0, activation_fn=nn.ReLU()):
        super(ConvTransposeBlock, self).__init__()
        deconv = nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        torch.nn.init.xavier_uniform_(deconv.weight)
        model = [deconv]
        if normalize == "batch":
            model.append(nn.BatchNorm2d(out_size))

        model.append(activation_fn)       
        self.model = nn.Sequential(*model)
        
    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)  
        return x

class Generator(nn.Module):
    def __init__(self, normalization_type=None):
        super(Generator, self).__init__()
        self.norm = normalization_type
        
        self.conv1 = ConvBlock(1, 64, normalize=self.norm, kernel_size=4, stride=1, padding=0, bias=True, dropout=0.0)
        self.conv2 = ConvBlock(64, 128, normalize=self.norm, kernel_size=4, stride=2, padding=1, bias=True, dropout=0.0)
        self.conv3 = ConvBlock(128, 256, normalize=self.norm, kernel_size=4, stride=2, padding=1, bias=True, dropout=0.0)
        self.conv4 = ConvBlock(256, 512, normalize=self.norm, kernel_size=4, stride=2, padding=1, bias=True, dropout=0.0)
        self.conv5 = ConvBlock(512, 512, normalize=self.norm, kernel_size=4, stride=2, padding=1, bias=True, dropout=0.0)
        
        self.deconv1 = ConvTransposeBlock(512, 512, normalize=self.norm, kernel_size=4, stride=2, padding=1, bias=True, dropout=0.5)
        self.deconv2 = ConvTransposeBlock(1024, 256, normalize=self.norm, kernel_size=4, stride=2, padding=1, bias=True, dropout=0.5)
        self.deconv3 = ConvTransposeBlock(512, 128, normalize=self.norm, kernel_size=4, stride=2, padding=1, bias=True, dropout=0.0)
        self.deconv4 = ConvTransposeBlock(256, 64, normalize=self.norm, kernel_size=4, stride=2, padding=1, bias=True, dropout=0.0)
        self.conv6 = ConvBlock(128, 2, normalize=None, kernel_size=1, stride=1, padding=0, bias=True, dropout=0.0, activation_fn=nn.Tanh())

        
    def forward(self, x):
        x = F.interpolate(x, size=(35, 35), mode='bilinear')
        down1 = self.conv1(x)
        down2 = self.conv2(down1)
        down3 = self.conv3(down2)
        down4 = self.conv4(down3)
        down5 = self.conv5(down4)
        up1 = self.deconv1(down5,down4)
        up2 = self.deconv2(up1,down3)
        up3 = self.deconv3(up2,down2)
        up4 = self.deconv4(up3,down1)    
        x = self.conv6(up4) 
        return x

class Discriminator(nn.Module):
    def __init__(self, normalization_type):
        super(Discriminator, self).__init__()
        self.norm = normalization_type
        
        self.conv1 = ConvBlock(3, 64, normalize=None, kernel_size=4, stride=1, padding=0, bias=True, dropout=0.0)
        self.conv2 = ConvBlock(64, 128, normalize=self.norm, kernel_size=4, stride=2, padding=1, bias=True, dropout=0.0)
        self.conv3 = ConvBlock(128, 256, normalize=self.norm, kernel_size=4, stride=2, padding=1, bias=True, dropout=0.0)
        self.conv4 = ConvBlock(256, 512, normalize=self.norm, kernel_size=4, stride=2, padding=1, bias=True, dropout=0.0)
        self.conv5 = ConvBlock(512, 1, normalize=None, kernel_size=4, stride=1, padding=0, bias=True, dropout=0.0, activation_fn=nn.Tanh())
        
    def forward(self, x):
        x = F.interpolate(x, size=(35, 35), mode='bilinear') 
        down1 = self.conv1(x)
        down2 = self.conv2(down1)
        down3 = self.conv3(down2)
        down4 = self.conv4(down3)
        x = self.conv5(down4)
        x = x.view(x.size()[0], -1)
        return x