from pathlib import Path
from .Foundation import Foundation
import utils.utils as utils
import cv2
import numpy as np
import torch

class Lipstick(Foundation):
    def __init__(self,opt,name,device):
        super().__init__(opt,name,device)
        self.name=Path(name)/"Lipstick"
        self.name.mkdir(exist_ok=True) 

    def get_facemask(self,img):
        h,w,c=img.shape
        parsing=self.mask
        mask=((parsing==12)*1+(parsing==13)*1).astype(np.uint8)
        mask=cv2.resize(mask,dsize=(w,h))
        mask=np.array([mask,mask,mask]).transpose(1,2,0)
        return mask