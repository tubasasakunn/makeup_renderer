from cgitb import strong
from pathlib import Path
from .Make_Base import Base
import utils.utils as utils
import cv2
import numpy as np
import torch
from utils.trimEye import TrimEye
from utils.DRenderer import Renderer
from torch.nn import functional as F
import torchvision

params_name=["start_x","start_y","end_x","end_y","middle_x","middle_y","thickness","RGB","alpha"]

class Eye(Base):
    def __init__(self,opt,name,device):
        super().__init__(opt,name,device)
        self.name=Path(name)/"Eye"
        self.name.mkdir(exist_ok=True)

    def preprocess(self):
        if self.mode=='test':
            self.out_img=self.in_img
        self.r_eye=TrimEye(self.in_img,self.out_img,self.landmark)
        self.l_eye=TrimEye(self.in_img,self.out_img,self.landmark,False)
        in_r_eye,out_r_eye=self.r_eye.get_eye()
        in_l_eye,out_l_eye=self.l_eye.get_eye()
        self.in_r_img = (torch.tensor(cv2.cvtColor(in_r_eye, cv2.COLOR_BGR2RGB),dtype=torch.float32)/255).permute(2,0,1).to(self.device)
        self.out_r_img = (torch.tensor(cv2.cvtColor(out_r_eye, cv2.COLOR_BGR2RGB),dtype=torch.float32)/255).permute(2,0,1).to(self.device)
        self.in_l_img = (torch.tensor(cv2.cvtColor(in_l_eye, cv2.COLOR_BGR2RGB),dtype=torch.float32)/255).permute(2,0,1).to(self.device)
        self.out_l_img = (torch.tensor(cv2.cvtColor(out_l_eye, cv2.COLOR_BGR2RGB),dtype=torch.float32)/255).permute(2,0,1).to(self.device)

    def postprocess(self,img_r,img_l):
        img_r=(255*img_r).permute(1,2,0).to('cpu').detach().numpy().copy()
        img_r=cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR)
        img_l=(255*img_l).permute(1,2,0).to('cpu').detach().numpy().copy()
        img_l=cv2.cvtColor(img_l, cv2.COLOR_RGB2BGR)

        img=self.r_eye.set_eye(img_r)
        self.l_eye.change_img(img)
        img=self.l_eye.set_eye(img_l)
        return img

    def saveimg(self,name):
        stroke_3alpha=torch.stack((self.stroke_alpha,self.stroke_alpha,self.stroke_alpha))
        all=torch.stack((self.in_save,self.out_save,self.makeup_save,stroke_3alpha))
        torchvision.utils.save_image(all,str(self.name/(name+".png")))
        name=Path(name)
        torchvision.utils.save_image(self.makeup_save,str(self.name/name.parent/("make"+name.name+".png")))
        torchvision.utils.save_image(stroke_3alpha,str(self.name/name.parent/("stroke"+name.name+".png")))

    def make_process(self,in_img,params_dict,LABmean,in_img_lab,num,left=False,break_time=1):
        start_x=params_dict["start_x"][num]
        start_y=params_dict["start_y"][num]
        end_x=params_dict["end_x"][num]
        end_y=params_dict["end_y"][num]
        middle_x=params_dict["middle_x"][num]
        middle_y=params_dict["middle_y"][num]
        thickness=params_dict["thickness"][num]
        RGB_c=params_dict["RGB"][num]
        alpha_c=params_dict["alpha"][num]


        c,w,h=in_img.shape
        canvas=torch.zeros([w,h,3])
        self.renderer=Renderer(canvas,self.renderMode,device=self.device,break_time=break_time)

        params=torch.cat((start_x,start_y,end_x,end_y,middle_x,middle_y,thickness)).to(self.device)
        params=utils.dclamp(params,min=0.0001,max=1)
        G_mask=self.renderer.draw_stroke(params)
        alpha=alpha_c.to(self.device)
        stroke=G_mask[0][0]
        stroke_alpha=stroke*alpha

        if left:
            mirror=torch.fliplr(stroke_alpha)
            stroke_alpha=mirror

        self.stroke_alpha=stroke_alpha

        #化粧過程        
        RGB=RGB_c.to(self.device)
        RGB=RGB.unsqueeze(1)
        RGB=RGB.unsqueeze(1)
        RGB=utils.dclamp(RGB,min=0.001,max=1.0)
        LAB=utils.RGB2LAB(RGB)

        addLAB=LAB-LABmean
        addimg_LAB=in_img_lab+addLAB
        addimg=utils.LAB2RGB(addimg_LAB)
        img=in_img*(1-stroke_alpha)+stroke_alpha*addimg

        return img

    def backward(self,img_r,img_l):

        if self.useMirror:
            c,w,h=img_r.shape
            self.makeup_save=torch.cat((img_r,F.interpolate(img_l.unsqueeze(0),(w,h),mode='bilinear')[0]),dim=1).detach()
            self.in_save=torch.cat((self.in_r_img,F.interpolate(self.in_l_img.unsqueeze(0),(w,h),mode='bilinear')[0]),dim=1).detach()
            self.out_save=torch.cat((self.out_r_img,F.interpolate(self.out_l_img.unsqueeze(0),(w,h),mode='bilinear')[0]),dim=1).detach()
            
            loss_r=self.loss(img_r,self.out_r_img)
            loss_l=self.loss(img_l,self.out_l_img)
            loss=loss_l+loss_r
        else:
            if self.right:
                self.makeup_save=img_r.detach()
                self.in_save=self.in_r_img.detach()
                self.out_save=self.out_r_img.detach()
                loss=self.loss(img_r,self.out_r_img)
            else:
                self.makeup_save=img_l.detach()
                self.in_save=self.in_l_img.detach()
                self.out_save=self.out_l_img.detach()
                loss=self.loss(img_l,self.out_l_img)

        print(self.epoch,loss)
        loss.backward()

    def list2torch(self,params):       
        start_x=params["start_x"]
        start_y=params["start_y"]
        end_x=params["end_x"]
        end_y=params["end_y"]
        middle_x=params["middle_x"]
        middle_y=params["middle_y"]
        thickness=params["thickness"]
        RGB_c=params["RGB"]
        alpha_c=params["alpha"]

        start_x=torch.tensor(start_x,requires_grad = True)
        start_y=torch.tensor(start_y,requires_grad = True)
        end_x=torch.tensor(end_x,requires_grad = True)
        end_y=torch.tensor(end_y,requires_grad = True)
        middle_x=torch.tensor(middle_x,requires_grad = True)
        middle_y=torch.tensor(middle_y,requires_grad = True)
        thickness=torch.tensor(thickness,requires_grad = True)
        RGB_c=torch.tensor(RGB_c,requires_grad = True)
        alpha_c=torch.tensor(alpha_c,requires_grad = True)

        params_list=[start_x,start_y,end_x,end_y,middle_x,middle_y,thickness,RGB_c,alpha_c]
        params_dict=dict(zip(params_name,params_list))
        return params_dict

    def train(self):
        self.preprocess()

        in_l_img_lab=utils.RGB2LAB(self.in_l_img)
        LABmean_l=torch.mean(in_l_img_lab,dim=[1,2])
        LABmean_l=LABmean_l.unsqueeze(1)
        LABmean_l=LABmean_l.unsqueeze(1)

        in_r_img_lab=utils.RGB2LAB(self.in_r_img)
        LABmean_r=torch.mean(in_r_img_lab,dim=[1,2])
        LABmean_r=LABmean_r.unsqueeze(1)
        LABmean_r=LABmean_r.unsqueeze(1)

        res_params_dict={}
        img_r=self.in_r_img
        img_l=self.in_l_img
        for key,params in self.opt["Params"].items():
            key_path=Path(key)
            (self.name/key_path).mkdir(exist_ok=True)
            
            params_dict_i={}
            self.useMirror=params["useMirror"]
            self.useMulti=params["useMulti"]
            self.renderMode=params["renderMode"]
            if not self.useMirror:
                self.right=params["right"]
                params_dict_i["right"]=self.right

            params_dict_i["useMirror"]=self.useMirror
            params_dict_i["useMulti"]=self.useMulti
            params_dict_i["renderMode"]=self.renderMode

            params_dict=self.list2torch(params)

            if self.useMulti:
                for epoch in range(self.opt["epoch"]):
                    self.epoch=epoch
                    img_r_i=img_r.detach()
                    img_l_i=img_l.detach()
                    optimizer,params_dict=self.optimizer.make(params_dict,epoch,key,None)

                    for i in range(len(params_dict["start_x"])):
                        if self.useMirror:
                            img_r_i=self.make_process(img_r_i,params_dict,LABmean_r,in_r_img_lab,i,False)
                            img_l_i=self.make_process(img_l_i,params_dict,LABmean_l,in_l_img_lab,i,True)
                        else:
                            if self.right:
                                img_r_i=self.make_process(img_r_i,params_dict,LABmean_r,in_r_img_lab,i,False)
                            else:
                                img_l_i=self.make_process(img_l_i,params_dict,LABmean_l,in_l_img_lab,i,True)

                    optimizer.zero_grad()
                    self.backward(img_r_i,img_l_i)
                    optimizer.step()
                    self.saveimg(str(key_path/str(epoch)))
                img_l=img_l_i
                img_r=img_r_i
                
            else:
                 for i in range(len(params_dict["start_x"])):

                    for epoch in range(self.opt["epoch"]):
                        self.epoch=epoch
                        img_r_i=img_r.detach()
                        img_l_i=img_l.detach()
                        optimizer,params_dict=self.optimizer.make(params_dict,epoch,key,i)
                        if self.useMirror:
                            img_r_i=self.make_process(img_r_i,params_dict,LABmean_r,in_r_img_lab,i,False)
                            img_l_i=self.make_process(img_l_i,params_dict,LABmean_l,in_l_img_lab,i,True)
                        else:
                            if self.right:
                                img_r_i=self.make_process(img_r_i,params_dict,LABmean_r,in_r_img_lab,i,False)

                            else:
                                img_l_i=self.make_process(img_l_i,params_dict,LABmean_l,in_l_img_lab,i,True)

                        optimizer.zero_grad()
                        self.backward(img_r_i,img_l_i)
                        optimizer.step()
                        self.saveimg(str(key_path/(str(i)+"_"+str(epoch))))
                    img_l=img_l_i
                    img_r=img_r_i

            cv2.imwrite(str(self.name/(str(i)+key+'.png')),self.postprocess(img_r,img_l))
            
            for name in params_name:
                params_dict_i[name]=params_dict[name].tolist()
            res_params_dict[key]=params_dict_i
            
        res_img=self.postprocess(img_r,img_l)
        return res_img,res_params_dict
                        
    def test(self,in_params):
        self.preprocess()

        in_l_img_lab=utils.RGB2LAB(self.in_l_img)
        LABmean_l=torch.mean(in_l_img_lab,dim=[1,2])
        LABmean_l=LABmean_l.unsqueeze(1)
        LABmean_l=LABmean_l.unsqueeze(1)

        in_r_img_lab=utils.RGB2LAB(self.in_r_img)
        LABmean_r=torch.mean(in_r_img_lab,dim=[1,2])
        LABmean_r=LABmean_r.unsqueeze(1)
        LABmean_r=LABmean_r.unsqueeze(1)


        img_r=self.in_r_img
        img_l=self.in_l_img
        for key,params in in_params.items():
            self.useMirror=params["useMirror"]
            self.useMulti=params["useMulti"]
            self.renderMode=params["renderMode"]
            if not self.useMirror:
                self.right=params["right"]

 
            params_dict=self.list2torch(params)




            for i in range(len(params_dict["start_x"])):
                if self.useMirror:
                    img_r=self.make_process(img_r,params_dict,LABmean_r,in_r_img_lab,i,False)
                    img_l=self.make_process(img_l,params_dict,LABmean_l,in_l_img_lab,i,True)
                else:
                    if self.right:
                        img_r=self.make_process(img_r,params_dict,LABmean_r,in_r_img_lab,i,False)
                    else:
                        img_l=self.make_process(img_l,params_dict,LABmean_l,in_l_img_lab,i,True)

        res_img=self.postprocess(img_r,img_l)
        return res_img,in_params

    def test_stroke(self,in_params):
        self.preprocess()

        in_l_img_lab=utils.RGB2LAB(self.in_l_img)
        LABmean_l=torch.mean(in_l_img_lab,dim=[1,2])
        LABmean_l=LABmean_l.unsqueeze(1)
        LABmean_l=LABmean_l.unsqueeze(1)

        in_r_img_lab=utils.RGB2LAB(self.in_r_img)
        LABmean_r=torch.mean(in_r_img_lab,dim=[1,2])
        LABmean_r=LABmean_r.unsqueeze(1)
        LABmean_r=LABmean_r.unsqueeze(1)


        img_r=self.in_r_img
        img_l=self.in_l_img
        imgs=[]
        for key,params in in_params.items():
            self.useMirror=params["useMirror"]
            self.useMulti=params["useMulti"]
            self.renderMode=params["renderMode"]
            if not self.useMirror:
                self.right=params["right"]

 
            params_dict=self.list2torch(params)




            
            for i in range(len(params_dict["start_x"])):
                img_l_tmp=img_l
                img_r_tmp=img_r
                for j in range(self.break_num):
                    if self.useMirror:
                        img_r_tmp=self.make_process(img_r,params_dict,LABmean_r,in_r_img_lab,i,False,(j+1)/self.break_num)
                        img_l_tmp=self.make_process(img_l,params_dict,LABmean_l,in_l_img_lab,i,True,(j+1)/self.break_num)
                    else:
                        if self.right:
                            img_r_tmp=self.make_process(img_r,params_dict,LABmean_r,in_r_img_lab,i,False,(j+1)/self.break_num)
                        else:
                            img_l_tmp=self.make_process(img_l,params_dict,LABmean_l,in_l_img_lab,i,True,(j+1)/self.break_num)
                    res_img=self.postprocess(img_r_tmp,img_l_tmp)
                    imgs.append(res_img)
                img_l=img_l_tmp
                img_r=img_r_tmp

        res_img=self.postprocess(img_r,img_l)
        return res_img,imgs