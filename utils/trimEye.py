from numpy import angle, e
from torch import optim
import os
from tqdm import tqdm
import torchvision.utils
from pathlib import Path
import utils.utils as utils
import cv2
import copy
import numpy as np

rate=1

class TrimEye():
    def __init__(self,in_img,out_img,landmark,right=True):

        #回転でボケるため
        in_img=cv2.resize(in_img,dsize=None,fx=rate,fy=rate)
        out_img=cv2.resize(out_img,dsize=None,fx=rate,fy=rate)
        
        self.ori_img=in_img
        self.trimming_eye(in_img,out_img,landmark,right)
        self.right=right

    def trimming_eye(self,in_img,out_img,landmark,right=True):
        self.ori_h, self.ori_w = in_img.shape[:2]
        #in_img=np.pad(in_img,((self.ori_h,self.ori_h),(self.ori_w,self.ori_w),(0,0)))
        #out_img=np.pad(out_img,((self.ori_h,self.ori_h),(self.ori_w,self.ori_w),(0,0)))
        h, w = in_img.shape[:2]
        if right:
            EYE_COD=utils.RIGHT_EYE
        else:
            EYE_COD=utils.LEFT_EYE

        size=np.array((w,h))
        x_axis=(landmark[EYE_COD[0]]-landmark[EYE_COD[15]])[:2]*size
        y_axis=(landmark[EYE_COD[11]]-landmark[EYE_COD[4]])[:2]*size
        center=np.array([landmark[x] for x in EYE_COD]).mean(axis=0)[:2]*size
            
        x_norm=np.linalg.norm(x_axis, ord=2)*1.5
        y_norm=np.linalg.norm(y_axis, ord=2)*1.5
        angle=np.degrees(np.arctan(x_axis[1]/x_axis[0]))

        M = cv2.getRotationMatrix2D(center, angle, 1)
        in_img=cv2.warpAffine(in_img, M, (w, h))
        out_img=cv2.warpAffine(out_img, M, (w, h))

        in_eye=in_img[int(center[1]-y_norm):int(center[1]+y_norm),int(center[0]-x_norm):int(center[0]+x_norm),:]
        out_eye=out_img[int(center[1]-y_norm):int(center[1]+y_norm),int(center[0]-x_norm):int(center[0]+x_norm),:]

        self.in_img = in_img
        self.out_img = out_img
        self.in_eye = in_eye
        self.out_eye = out_eye
        self.angle=angle
        self.center=center
        self.x_norm=x_norm
        self.y_norm=y_norm
        self.landmark=landmark

    def get_eye(self):
        return self.in_eye,self.out_eye

    def change_img(self,img):
        img=cv2.resize(img,dsize=None,fx=rate,fy=rate)
        self.ori_img=img
        

    def set_eye(self,in_eye):
        in_img=self.in_img.copy()
        in_img[int(self.center[1]-self.y_norm):int(self.center[1]+self.y_norm),int(self.center[0]-self.x_norm):int(self.center[0]+self.x_norm),:]=in_eye

        point=(int(self.center[0]),int(self.center[1]))
        #point = ((int(self.center[0])),(int(self.center[1])))
        mask=np.zeros(self.in_img.shape,dtype=np.uint8)
        mask[int(self.center[1]-self.y_norm):int(self.center[1]+self.y_norm),int(self.center[0]-self.x_norm):int(self.center[0]+self.x_norm),:]=255

        #mask[int(self.center[1]-self.y_norm):int(self.center[1]+self.y_norm),int(self.center[0]-self.x_norm):int(self.center[0]+self.x_norm)]=255
        #in_img=cv2.seamlessClone(in_img.astype(np.uint8),self.in_img.astype(np.uint8),mask,point,cv2.NORMAL_CLONE)


        in_img = cv2.seamlessClone(in_img.astype(np.uint8),self.in_img.astype(np.uint8), mask, point, cv2.NORMAL_CLONE).astype(np.float32)

        '''        
        face_h,face_w,_=in_eye.shape
        alpha_x=np.arange(face_w)-face_w/2
        alpha_y=np.arange(face_h)-face_h/2
        alpha=np.array(np.meshgrid(alpha_x,alpha_y))
        alpha=np.max(alpha,axis=0)
        #alpha=np.sqrt(alpha[0]**2+alpha[1]**2)
        my=(face_w-2)//4
        mx=(face_h-2)//4
        r=min(alpha[my,(face_w-2)//2],alpha[(face_h-2)//2,mx])
        alpha=alpha/r-1
        alpha[alpha<0]=0
        r=min(alpha[0,(face_w-2)//2],alpha[(face_h-2)//2,0])
        alpha[alpha>r]=r
        alpha=alpha/alpha.max()
        alpha=np.dstack((alpha,alpha,alpha))
        mask[int(self.center[1]-self.y_norm):int(self.center[1]+self.y_norm),int(self.center[0]-self.x_norm):int(self.center[0]+self.x_norm)]=255
        #point=(int(self.center[0]),int(self.center[1]))
        #cv2.imwrite("res/reerer.png",self.in_img)
        #cv2.imwrite("res/reerer1.png",in_img)
        #in_img=np.clip(in_img,0,255)
        #self.in_img=np.clip(self.in_img,0,255)
        #in_img=cv2.seamlessClone(self.in_img,in_img,mask,point,cv2.NORMAL_CLONE)
        #in_img=np.clip(in_img,0,255)
        #cv2.imwrite("res/reerer2.png",in_img)
        #cv2.imwrite("res/reerer3.png",in_img*mask)
        #in_img=self.in_img*mask+in_img*(1-mask)
        '''


        M = cv2.getRotationMatrix2D(self.center, -1*self.angle, 1)
        h, w = in_img.shape[:2]
        img=cv2.warpAffine(in_img, M, (w, h))

        margin=np.sqrt(self.y_norm**2+self.x_norm**2+0.000001)
        original=self.ori_img.copy()
        original[int(self.center[1]-margin):int(self.center[1]+margin),int(self.center[0]-margin):int(self.center[0]+margin),:] = img[int(self.center[1]-margin):int(self.center[1]+margin),int(self.center[0]-margin):int(self.center[0]+margin),:]

        #original=cv2.seamlessClone(self.ori_img,original,mask,point,cv2.NORMAL_CLONE)

        img=cv2.resize(original,dsize=None,fx=1/rate,fy=1/rate)
        return img


