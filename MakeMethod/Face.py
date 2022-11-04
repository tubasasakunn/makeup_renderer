from pathlib import Path
import utils.utils as utils
import cv2
import numpy as np
import torch
from utils.trimFace import TrimFace
from utils.DRenderer import Renderer
from torch.nn import functional as F
from .Make_Base import Base
import torchvision

params_name=["start_x","start_y","end_x","end_y","middle_x","middle_y","thickness","RGB","alpha"]

class Face(Base):
    def __init__(self,opt,name,device):
        super().__init__(opt,name,device)
        self.name=Path(name)/"Face"
        self.name.mkdir(exist_ok=True)

    def preprocess(self):
        if self.mode=='test':
            self.out_img=self.in_img
        self.face=TrimFace(self.in_img,self.out_img,self.mask,self.landmark)
        in_face,out_face=self.face.get_face()
        self.in_face = (torch.tensor(cv2.cvtColor(in_face, cv2.COLOR_BGR2RGB),dtype=torch.float32)/255).permute(2,0,1).to(self.device)
        self.out_face = (torch.tensor(cv2.cvtColor(out_face, cv2.COLOR_BGR2RGB),dtype=torch.float32)/255).permute(2,0,1).to(self.device)

        self.out_loss=self.out_face
        self.in_save,self.out_save=(self.in_face,self.out_face)

    def postprocess(self,img):
        img=(255*img).permute(1,2,0).to('cpu').detach().numpy().copy()
        img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img=self.face.set_face(img)

        return img  

    def saveimg(self,name):
        stroke_3alpha=torch.stack((self.stroke_alpha,self.stroke_alpha,self.stroke_alpha))
        all=torch.stack((self.in_save,self.out_save,self.makeup_save,stroke_3alpha))
        torchvision.utils.save_image(all,str(self.name/(name+".png")))
        name=Path(name)
        torchvision.utils.save_image(self.makeup_save,str(self.name/name.parent/("make"+name.name+".png")))
        torchvision.utils.save_image(stroke_3alpha,str(self.name/name.parent/("stroke"+name.name+".png")))

    def make_process(self,in_img,params_dict,LABmean,in_img_lab,num,mirror=False,break_time=1):
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

        if mirror:
            mirror=torch.fliplr(stroke_alpha)
            stroke_alpha=stroke_alpha+mirror
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

        in_img_lab=utils.RGB2LAB(self.in_face)
        LABmean=torch.mean(in_img_lab,dim=[1,2])
        LABmean=LABmean.unsqueeze(1)
        LABmean=LABmean.unsqueeze(1)

        res_params_dict={}
        img_face=self.in_face
        for key,params in self.opt["Params"].items():
            key_path=Path(key)
            (self.name/key_path).mkdir(exist_ok=True)
            params_dict_i={}
            self.useMirror=params["useMirror"]
            self.useMulti=params["useMulti"]
            self.renderMode=params["renderMode"]

            params_dict_i["useMirror"]=self.useMirror
            params_dict_i["useMulti"]=self.useMulti
            params_dict_i["renderMode"]=self.renderMode

            params_dict=self.list2torch(params)

            if self.useMulti:
                for epoch in range(self.opt["epoch"]):
                    self.epoch=epoch
                    img_face_i=img_face.detach()
                    optimizer,params_dict=self.optimizer.make(params_dict,epoch,key,None)

                    for i in range(len(params_dict["start_x"])):
                        img_face_i=self.make_process(img_face_i,params_dict,LABmean,in_img_lab,i,self.useMirror)


                    self.makeup_loss=img_face_i
                    self.makeup_save=img_face_i
                    optimizer.zero_grad()
                    self.backward()
                    optimizer.step()
                    self.saveimg(str(key_path/str(epoch)))
                img_face=img_face_i
            else:
                 for i in range(len(params_dict["start_x"])):

                    for epoch in range(self.opt["epoch"]):
                        self.epoch=epoch
                        img_face_i=img_face.detach()
                        optimizer,params_dict=self.optimizer.make(params_dict,epoch,key,i)
                        img_face_i=self.make_process(img_face_i,params_dict,LABmean,in_img_lab,i,self.useMirror)

                        self.makeup_loss=img_face_i
                        self.makeup_save=img_face_i
                        optimizer.zero_grad()
                        self.backward()
                        optimizer.step()
                        self.saveimg(str(key_path/(str(i)+"_"+str(epoch))))
                    img_face=img_face_i

            cv2.imwrite(str(self.name/(key+str(i)+'.png')),self.postprocess(img_face))
            
            for name in params_name:
                params_dict_i[name]=params_dict[name].tolist()
            res_params_dict[key]=params_dict_i
            
        res_img=self.postprocess(img_face)
        return res_img,res_params_dict
                        
    def test(self,in_params):
        self.preprocess()

        in_img_lab=utils.RGB2LAB(self.in_face)
        LABmean=torch.mean(in_img_lab,dim=[1,2])
        LABmean=LABmean.unsqueeze(1)
        LABmean=LABmean.unsqueeze(1)


        img_face=self.in_face
        for key,params in in_params.items():
            self.useMirror=params["useMirror"]
            self.useMulti=params["useMulti"]
            self.renderMode=params["renderMode"]

 
            params_dict=self.list2torch(params)


            for i in range(len(params_dict["start_x"])):
                img_face=self.make_process(img_face,params_dict,LABmean,in_img_lab,i,self.useMirror)
                    
        res_img=self.postprocess(img_face)
        return res_img,in_params

    def test_stroke(self,in_params):
        self.preprocess()

        in_img_lab=utils.RGB2LAB(self.in_face)
        LABmean=torch.mean(in_img_lab,dim=[1,2])
        LABmean=LABmean.unsqueeze(1)
        LABmean=LABmean.unsqueeze(1)


        img_face=self.in_face
        imgs=[]
        for key,params in in_params.items():
            self.useMirror=params["useMirror"]
            self.useMulti=params["useMulti"]
            self.renderMode=params["renderMode"]

 
            params_dict=self.list2torch(params)


            for i in range(len(params_dict["start_x"])):

                for j in range(self.break_num):
                    img_face_tmp=self.make_process(img_face,params_dict,LABmean,in_img_lab,i,self.useMirror,(j+1)/self.break_num)
                    res_img=self.postprocess(img_face_tmp)
                    imgs.append(res_img)
                img_face=img_face_tmp
                    
        res_img=self.postprocess(img_face)
        return res_img,imgs