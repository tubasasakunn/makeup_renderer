import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import structural_similarity as sk_cpt_ssim
import dlib
from imutils import face_utils
import os
import glob
import random

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils

from lib.faceparsing import test as parsingtest
from torch.autograd import Function
import mediapipe as mp

#mediapipeの準備
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=True,min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

faceparsing_path='model/faceparsing.pth'
predictor_path = 'model/dlib.dat'

class test(Function):
    @staticmethod
    def forward(ctx,params):
        ctx.save_for_backward(params)
        print("test in")
        print("in",params.max(),params.min())
        return params
    @staticmethod
    def backward(ctx,loss):
        epoch = ctx.saved_tensors
        print("test out",epoch[0].min())
        print("loss",loss.max(),loss.min())
        return loss
M_RENDERING_SAMPLES_PER_EPOCH = 50000

#微分可能なclamp
class DifferentiableClamp(Function):
    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def dclamp(input, min, max):
    return DifferentiableClamp.apply(input, min, max)

#色空間の変換
#参考　https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
#RGB∈[0,1],XYZ∈[0,1]
#L∈[0,100],a∈[-127,127],b∈[-127,127]
def RGB2XYZ(RGBimg):
    
    RGBimg=dclamp(RGBimg,0,1)
    MAT=torch.Tensor([[0.412453,0.357580,0.180423],[0.212671,0.715160,0.072169],[0.019334,0.119193,0.950227]])
    X=MAT[0,0]*RGBimg[0]+MAT[0,1]*RGBimg[1]+MAT[0,2]*RGBimg[2]
    Y=MAT[1,0]*RGBimg[0]+MAT[1,1]*RGBimg[1]+MAT[1,2]*RGBimg[2]
    Z=MAT[2,0]*RGBimg[0]+MAT[2,1]*RGBimg[1]+MAT[2,2]*RGBimg[2]
    XYZimg=torch.stack((X,Y,Z))
    return XYZimg

def XYZ2RGB(XYZimg):
    XYZimg=dclamp(XYZimg,0,1)
    MAT=torch.Tensor([[3.240479,-1.53715,-0.498535],[-0.969256,1.875991,0.041556],[0.055648,-0.204043,1.057311]])
    R=MAT[0,0]*XYZimg[0]+MAT[0,1]*XYZimg[1]+MAT[0,2]*XYZimg[2]
    G=MAT[1,0]*XYZimg[0]+MAT[1,1]*XYZimg[1]+MAT[1,2]*XYZimg[2]
    B=MAT[2,0]*XYZimg[0]+MAT[2,1]*XYZimg[1]+MAT[2,2]*XYZimg[2]
    RGBimg=torch.stack((R,G,B))
    return RGBimg

def LABfn(t):
    t_mask=(t>0.008856)*1
    ft=t_mask*torch.pow(t,1/3)+(1-t_mask)*(7.787*t+16/116)

    return ft

def invLABfn(t):
    t_mask=(t>0.2069)*1 #0.2069はしきい値での値
    ft=t_mask*torch.pow(t,3)+(1-t_mask)*((t-16/116)/7.787)
    
    return ft


#入出力は[0,1]
def XYZ2LAB(XYZimg):
    
    XYZimg=dclamp(XYZimg,0,1)
    X,Y,Z=XYZimg
    Xn=0.950456
    Zn=1.088754

    X=X/Xn
    Z=Z/Zn
    Y_mask=(Y>0.008856)*1
    L=Y_mask*(116*torch.pow(Y,1/3)-16)+(1-Y_mask)*903.3*Y
    a=500*(LABfn(X)-LABfn(Y))
    b=200*(LABfn(Y)-LABfn(Z))
    
    
    LABimg=torch.stack((L,a,b))
    return LABimg

def LAB2XYZ(LABimg):
    L,A,B=LABimg

    L=dclamp(L,min=0.001,max=100)
    A=dclamp(A,min=-127,max=127)
    B=dclamp(B,min=-127,max=127)

    Xn=0.950456
    Zn=1.088754
    Ln=7.9996 #Y=0.008856のときのL

    L_mask=(L>Ln)*1
    Y=L_mask*(torch.pow((L+16)/116,3))+(1-L_mask)*(L/903.3)
    X=invLABfn(A/500+LABfn(Y))*Xn
    Z=invLABfn(LABfn(Y)-B/200)*Zn


    XYZimg=torch.stack((X,Y,Z))
    return XYZimg

def RGB2LAB(RGBimg):
    XYZimg=RGB2XYZ(RGBimg)
    LABimg=XYZ2LAB(XYZimg)
    return LABimg
    
def LAB2RGB(LABimg):
    XYZimg=LAB2XYZ(LABimg)
    RGBimg=XYZ2RGB(XYZimg)
    return RGBimg

def get_parsing(img):

    img=img.astype(np.uint8)
    parsing=parsingtest.evaluate(save_pth=faceparsing_path,img=img)
    return parsing


#mediapipe用
RIGHT_EYE=[263,249,390,373,374,380,381,382,466,388,387,386,385,384,398,362]
LEFT_EYE=[33,7,163,144,145,153,154,155,246,161,160,159,158,157,173,133]
#最初(0)が外側，最後(15)が内側，下4，上11
FACE_OVAL=[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
#上0 下18 右9 左27
def face(results, annotated_image):
    label = ["x", "y", "z"]
    data = []
    if results.face_landmarks:
        # ランドマークを描画する
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.face_landmarks,
            connections=mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

        for landmark in results.face_landmarks.landmark:
            data.append([landmark.x, landmark.y, landmark.z])

    else:  # 検出されなかったら欠損値nanを登録する
        data.append([np.nan, np.nan, np.nan])

    return data

def get_landmark(img):
    image=img.astype(np.uint8)
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    annotated_image = image.copy()

    landmark = face(results, annotated_image)
    landmark=np.array(landmark)
    return landmark

'''
#dlib用
def get_landmark(img):
    img=img.astype(np.uint8)
    face_detector = dlib.get_frontal_face_detector()
    face_predictor = dlib.shape_predictor(predictor_path)
    img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(img_gry, 1)
    face=faces[0]
    landmark = face_predictor(img_gry, face)
    landmark = face_utils.shape_to_np(landmark)

    return landmark
'''

