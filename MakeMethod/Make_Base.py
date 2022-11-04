from pathlib import Path
from loss import MakeLoss
from optimizer import MakeOptimizer
import torch
import torchvision

class Base:
    def __init__(self,opt,name,device='cuda'):
        self.opt=opt
        self.name=Path(name)
        self.name.mkdir(exist_ok=True) 
        self.loss=MakeLoss(opt,device)
        self.optimizer=MakeOptimizer(opt)
        self.device=device

    def preprocess(self):
        return None

    def postprocess(self):
        return None

    def saveimg(self,name):
        all=torch.stack((self.in_save,self.out_save,self.makeup_save))
        all=torch.stack((self.in_save,self.out_save,self.makeup_save,torch.abs(self.makeup_save-self.out_save)))
        torchvision.utils.save_image(all,str(self.name/(name+".png")))
        torchvision.utils.save_image(self.makeup_save,str(self.name/("make_"+name+".png")))
    
    def backward(self):
        loss=self.loss(self.makeup_loss,self.out_loss)
        loss.backward()
        print(self.epoch,loss)

    def train(self):
        return None,None

    def test(self):
        return None,None
        
    def test_stroke(self):
        return None,None


    def make(self,img_dict,out_img,mode,params):
        self.mode=mode
        self.break_num=10
        
        self.in_img=img_dict["original"]
        self.mask=img_dict["mask"]
        self.landmark=img_dict["landmark"]
        
        if mode=="train":
            self.out_img=out_img
            return self.train()
        if mode=="test":
            self.out_img=out_img
            with torch.no_grad():
                return self.test(params)
        if mode=="test_stroke":
            self.mode="test"
            self.out_img=out_img
            with torch.no_grad():
                return self.test_stroke(params)

        return None,None