from pathlib import Path
from .Make_Base import Base
import utils.utils as utils
import cv2
import numpy as np
import torch

class Foundation(Base):
    def __init__(self,opt,name,device):
        super().__init__(opt,name,device)
        self.name=Path(name)/"Foundation"
        self.name.mkdir(exist_ok=True) 

    def preprocess(self):
        mask=self.get_facemask(self.in_img)
        self.mean=np.mean(self.in_img[np.nonzero(mask)])
        self.std=np.std(self.in_img[np.nonzero(mask)])
        in_img= 128+60*((self.in_img)-self.mean)/self.std
        in_img=in_img.astype(np.float32)
        
        self.in_img=(torch.tensor(cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB),dtype=torch.float32)/255).permute(2,0,1).to(self.device)
        self.mask=(torch.tensor(mask,dtype=torch.float32).permute(2,0,1)).to(self.device)
        self.in_face=self.mask*self.in_img

        if self.mode=="train":
            out_img= 128+60*((self.out_img)-self.mean)/self.std
            out_img=out_img.astype(np.float32)
            self.out_img=(torch.tensor(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB),dtype=torch.float32)/255).permute(2,0,1).to(self.device)
            self.out_face=self.mask*self.out_img
            self.out_loss=self.out_face
            self.in_save,self.out_save=(self.in_face,self.out_face)

    
    def postprocess(self,img):
        res_img=self.in_img*(1-self.mask)+img
        res_img=(255*res_img).permute(1,2,0).to('cpu').detach().numpy().copy()
        res_img=cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)
        res_img=self.std*(res_img-128)/60+self.mean
        res_img=res_img.clip(0,255)

        return res_img

    def get_facemask(self,img):
        h,w,c=img.shape
        parsing=self.mask
        mask=((parsing==1)*1 + (parsing==10)*1).astype(np.uint8)
        mask=cv2.resize(mask,dsize=(w,h))
        mask=np.array([mask,mask,mask]).transpose(1,2,0)
        return mask

    def make_process(self,params_dict,LABmean,in_face_lab):
        RGB_c=params_dict["RGB"]
        alpha_c=params_dict["alpha"]
        
        alpha=alpha_c.to(self.device)           
        RGB=RGB_c.to(self.device)
        RGB=RGB.unsqueeze(1)
        RGB=RGB.unsqueeze(1)
        RGB=utils.dclamp(RGB,min=0.001,max=1.0)
        LAB=utils.RGB2LAB(RGB)
        addLAB=LAB-LABmean
        addimg_LAB=in_face_lab+addLAB
        addimg_LAB[0]=utils.dclamp(addimg_LAB[0],min=0.001,max=100)
        addimg_LAB[1]=utils.dclamp(addimg_LAB[1],min=-127,max=127)
        addimg_LAB[2]=utils.dclamp(addimg_LAB[2],min=-127,max=127)
        addimg=utils.LAB2RGB(addimg_LAB)

        img=self.in_face+alpha*addimg*self.mask

        return img

    def backward(self):
        loss=self.loss(self.makeup_loss*self.mask,self.out_loss*self.mask)
        loss.backward()
        print(self.epoch,loss)

    def train(self):

        self.preprocess()
        
        in_face_lab=utils.RGB2LAB(self.in_face)
        LABmean=torch.stack((torch.mean(in_face_lab[0][self.mask[0]==1]),torch.mean(in_face_lab[1][self.mask[0]==1]),torch.mean(in_face_lab[2][self.mask[0]==1])))
        LABmean=LABmean.unsqueeze(1)
        LABmean=LABmean.unsqueeze(1)


        RGB_c=self.opt["Params"]["RGB"]
        alpha_c=self.opt["Params"]["alpha"]
        RGB_c=torch.tensor(RGB_c,requires_grad = True)
        alpha_c=torch.tensor(alpha_c,requires_grad = True)
        params_name=["RGB","alpha"]
        for epoch in range(self.opt["epoch"]):
            self.epoch=epoch
            params_list=[RGB_c,alpha_c]
            params_dict=dict(zip(params_name,params_list))
            optimizer,params_dict=self.optimizer.make(params_dict,epoch)

            img=self.make_process(params_dict,LABmean,in_face_lab)
            self.makeup_loss=img
            self.makeup_save=img

            optimizer.zero_grad()
            self.backward()
            optimizer.step()
            
            self.saveimg(str(epoch))


        res_img=self.postprocess(img)
        
        params_list=[RGB_c.tolist(),alpha_c.tolist()]
        params_dict=dict(zip(params_name,params_list))

        return res_img,params_dict

    def test(self,params):
        self.preprocess()

        in_face_lab=utils.RGB2LAB(self.in_face)
        LABmean=torch.stack((torch.mean(in_face_lab[0][self.mask[0]==1]),torch.mean(in_face_lab[1][self.mask[0]==1]),torch.mean(in_face_lab[2][self.mask[0]==1])))
        LABmean=LABmean.unsqueeze(1)
        LABmean=LABmean.unsqueeze(1)

        RGB_c=params["RGB"]
        alpha_c=params["alpha"]
        RGB_c=torch.tensor(RGB_c)
        alpha_c=torch.tensor(alpha_c)
        params_name=["RGB","alpha"]
        params_list=[RGB_c,alpha_c]
        params_dict=dict(zip(params_name,params_list))

        img=self.make_process(params_dict,LABmean,in_face_lab)
        res_img=self.postprocess(img)

        return res_img,params

    def test_stroke(self,params):
        in_img=self.in_img.copy()
        res_img,_=self.test(params)
        res_img=res_img.astype(np.uint8)
        h,w,c=res_img.shape
        imgs=[]
        for j in range(self.break_num):
            div=int(h*(j+1)/self.break_num)
            res_j=cv2.hconcat([res_img[:,:div,:],in_img[:,div:,:]])
            imgs.append(res_j)
        return res_img,imgs
