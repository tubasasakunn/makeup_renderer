from pathlib import Path
from MakeMethod.Foundation import Foundation
from MakeMethod.Lipstick import Lipstick
from MakeMethod.Eye import Eye
from MakeMethod.Face import Face
import utils.utils as utils
import cv2

class Makeup:
    def __init__(self,in_img,out_img,config,name='res'):
        self.in_img=in_img
        self.out_img=out_img
        self.img=in_img
        self.name=Path(name)
        self.name.mkdir(exist_ok=True) 
        self.output_path=self.name/"Makeup"
        self.output_path.mkdir(exist_ok=True)
        self.device=config.device

    def foundation(self,img_dict,out_img=None,mode="train",params=None,opt=None):
        make=Foundation(opt,self.name,self.device)
        img,params=make.make(img_dict,out_img,mode,params)
        return img,params

    def lipstick(self,img_dict,out_img=None,mode="train",params=None,opt=None):
        make=Lipstick(opt,self.name,self.device)
        img,params=make.make(img_dict,out_img,mode,params)
        return img,params

    def eye(self,img_dict,out_img=None,mode="train",params=None,opt=None):
        make=Eye(opt,self.name,self.device)
        img,params=make.make(img_dict,out_img,mode,params)
        return img,params

    def face(self,img_dict,out_img=None,mode="train",params=None,opt=None):
        make=Face(opt,self.name,self.device)
        img,params=make.make(img_dict,out_img,mode,params)
        return img,params




    def name2work(self,name):
        if name=="Foundation":
            return self.foundation
        if name=="Lipstick":
            return self.lipstick
        if name=="Eye":
            return self.eye
        if name=="Face":
            return self.face

    def train(self,opt):
        self.img=self.in_img
        params_dict={}
        img_dict={'original':self.img}
        img_dict["landmark"]=utils.get_landmark(self.img)
        img_dict["mask"]=utils.get_parsing(self.img)
        for i,make_name in enumerate(opt.make_list):
            print(make_name)
            work = self.name2work(make_name)
            img_dict["original"],params=work(img_dict,self.out_img,'train',opt=opt.make[make_name])
            cv2.imwrite(str(self.output_path/(str(i)+'.png')),img_dict["original"])
            params_dict[make_name]=params
        return params_dict,img_dict["original"]

    def test(self,in_img,params,mode='test_stroke'):
        self.img=in_img
        img_dict={'original':self.img}
        img_dict["landmark"]=utils.get_landmark(self.img)
        img_dict["mask"]=utils.get_parsing(self.img)
        imgs=[]
        for make_name,params_i in params.items():
            print(make_name)
            work = self.name2work(make_name)
            img_dict["original"],add_imgs=work(img_dict,None,mode=mode,params=params_i)
            imgs.extend(add_imgs)
        return imgs

'''
    def test(self,in_img,params,mode='test'):
        self.img=in_img
        img_dict={'original':self.img}
        img_dict["landmark"]=utils.get_landmark(self.img)
        img_dict["mask"]=utils.get_parsing(self.img)
        for make_name,params_i in params.items():
            print(make_name)
            work = self.name2work(make_name)
            img_dict["original"],_=work(img_dict,None,mode=mode,params=params_i)
        return img_dict["original"]
'''
