import torch
import torch.nn as nn
from torchvision import models

class MakeLoss(nn.Module):
    def __init__(self, opt,device='cuda'):
        super(MakeLoss, self).__init__()
        self.opt = opt
        self.pix_loss=PixelLoss(p=2)
        self.vgg_loss=VGGLoss(device)
        self.device=device

    def forward(self,img1,img2):
        loss=0
        loss=self.pix_loss(img1,img2)+loss
        #if self.opt["optimizer"]=="Eye" or self.opt["optimizer"]=="Face":
        #    loss=self.vgg_loss(img1,img2)+loss
        return loss




class PixelLoss(nn.Module):

    def __init__(self, p=1.0):
        super(PixelLoss, self).__init__()
        self.p = p

    def forward(self, canvas, gt, ignore_color=False):
        if ignore_color:
            canvas = torch.mean(canvas, dim=1)
            #gt = torch.mean(gt, dim=1)
        loss = torch.mean(torch.abs(canvas-gt)**self.p)
        #loss = torch.pow(torch.mean(self.norm(canvas,gt,self.p)),1/self.p)

        return loss

    def norm(self,img1,img2,p):
        img=torch.pow(torch.abs(torch.abs(img1)-torch.abs(img2)),p)
        return img

class VGGLoss(nn.Module):
    def __init__(self,device):
        super(VGGLoss, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.contentLayers = nn.Sequential(*list(vgg.features)[:31]).eval().to(device)
        for param in self.contentLayers.parameters():
            param.requires_grad = False

    def forward(self, img1, img2):
        img1=img1.unsqueeze(0)
        img2=img2.unsqueeze(0)
        MSELoss = nn.MSELoss()
        content_loss = MSELoss(self.contentLayers(img1), self.contentLayers(img2))
        return content_loss