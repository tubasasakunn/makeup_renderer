import argparse
from importlib import import_module
import cv2
from pathlib import Path
from setproctitle import setproctitle, getproctitle
import torch
from makeup import Makeup
import numpy as np




def video2imgs(video_path):
    cap = cv2.VideoCapture(video_path)
    imgs=[]
    while True:
        ret, frame = cap.read()
        if ret:
            imgs.append(frame)
        else:
            return imgs,cap.get(cv2.CAP_PROP_FPS)

def imgs2video(output_path,imgs,fps):
    h,w,c=imgs[0].shape
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(output_path,fourcc, fps, (w,h))
    for img in imgs:
        video.write(img)
    video.release()

def main(input_path,output_path,model_path,opt,mode=None):
    imgs,fps=video2imgs(input_path)
    makeup_imgs=[]
    connect_imgs=[]
    make_dict=torch.load(model_path)
    make=Makeup(imgs[0],imgs[0],opt,name='res')
    for i,img in enumerate(imgs):
        print(i)

        try:
            makeup_img=np.uint8(make.test(img,make_dict))
        except:
            print("できない")
            makeup_img=img


        makeup_imgs.append(makeup_img)
        connect_img=cv2.vconcat((img,makeup_img))
        connect_imgs.append(connect_img)
        
        if i%100==20:
            if mode==None:
                imgs2video(output_path,makeup_imgs,fps)
                imgs2video(output_path.replace('.mp4','con.mp4'),connect_imgs,fps)
            elif mode=="connect":
                imgs2video(output_path,connect_imgs,fps)
            else:
                imgs2video(output_path,makeup_imgs,fps)

    if mode==None:
        imgs2video(output_path,makeup_imgs,fps)
        imgs2video(output_path.replace('.mp4','con.mp4'),connect_imgs,fps)
    elif mode=="connect":
        imgs2video(output_path,connect_imgs,fps)
    else:
        imgs2video(output_path,makeup_imgs,fps)

if __name__ == '__main__':
    in_path="input.mp4"
    out_path="output.mp4"
    pt_path="model.pt"
    p=Path("config.py")
    lib=''
    for i in list(p.parts)[:-1]:
        lib=lib+i+'.'
    lib=lib+p.stem

    option = import_module(lib)
    opt = option.Options()
    setproctitle(opt.name)
    main(in_path,out_path,pt_path,opt)
