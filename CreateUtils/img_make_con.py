import argparse
from importlib import import_module
import cv2
from pathlib import Path
from setproctitle import setproctitle, getproctitle
import torch
from makeup import Makeup
import numpy as np






def main(input_path,output_path,opt,mode='stroke'):
    img=cv2.imread(input_path)
    make_dict={}
    for name in opt.make_list:
        make_dict[name]=opt.make[name]["Params"]
    make=Makeup(img,img,opt,name='res')
    if mode=='stroke':
        imgs=np.uint8(make.test(img,make_dict))
        h,w,c=imgs[0].shape
        fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
        video  = cv2.VideoWriter(output_path, fourcc, 20.0, (w,h))
        for img in imgs:
            video.write(img)
            video.write(img)
        video.release()
    else:
        makeup_img=np.uint8(make.test(img,make_dict))
        cv2.imwrite(output_path,makeup_img)
