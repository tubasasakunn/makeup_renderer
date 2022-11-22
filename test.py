from re import I
from wsgiref.simple_server import WSGIRequestHandler
from cv2 import imread
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms
from pathlib import Path
from torchvision.io import read_image
from torch.utils.data import DataLoader
from train import get_model
import random
from makeup import Makeup
import cv2
import numpy as np
from make_dataset import PostProcessing
from torchvision.utils import save_image

class Options(object):
    make_list=["Foundation","Lipstick","Eye","Face"]
    name='test_ref'
    device='cuda'

def MakeParams(preset,make_name,right=True,renderMode='powder'):

    config={}
    if make_name in ["Foundation", "Lipstick"]:
        for i,key in enumerate(['RGB', 'alpha']):
            for j in range(len(preset)):
                noise=random.normalvariate(0,0.3)
                preset[i][j]=min(max(preset[i][j]+noise,0),1)
            config[key]=[preset[i]]

    if make_name in ["Eye","Face"]:
        for i,key in enumerate(['start_x', 'start_y', 'end_x', 'end_y', 'middle_x', 'middle_y', 'thickness', 'RGB', 'alpha']):
            config[key]=[]
            for j in range(len(preset)):
                if key=="RGB":
                    config[key].append([preset[j][-4],preset[j][-3],preset[j][-2]])
                elif key=="alpha":
                    config[key].append([preset[j][-1]])
                else:
                    config[key].append([preset[j][i]])


        config['useMirror']=True
        config['useMulti']=False
        config['renderMode']=renderMode
        if make_name=="Eye":
            config['right']=right
    params={}
    params[make_name]={"test":config}
    return params

def main(img,model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = "cpu"
    print("use device is %s"%(device))
    net=get_model(11,device)
    net.load_state_dict(torch.load(model_path))
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=torch.from_numpy(img.astype(np.float32)).clone()/255
    img=img.permute(2,0,1)
    img=img.unsqueeze(0).to(device)

    params=net(img)
    params=torch.clamp(params,0,1)
    print(params)
    return params[0].to("cpu").tolist()

if __name__ == '__main__':
    params=[]
    
    img=cv2.imread("unit_test/canmake4/B_out.png")
    img=cv2.imread("dataset/dataset_Face2/[[0.141468523957264], [0.8129193917250117], [0.6393019292692518], [0.8699712122133516], [0.6152468271187818], [0.8362301247366903], [0.2951296628551199], [0.0, 0.5507578991891822, 0.0], [0.911575536226868]].jpg")
    img=PostProcessing(img)
    model_paths=Path("model/Makeup_Model/MultiFace")
    model_paths=list(model_paths.iterdir())
    for model_path in model_paths:
        print(model_path)
        if "4" in str(model_path):
            params.append(main(img,"model_00700.pth"))
            #params.append([0.141,0.812,0.63,0.8696,0.615,0.8363,0.299,0.0, 0.552, 0.0,0.911])

        #params.append(main(img,model_path))
    make_dict=MakeParams(params,"Face")

    opt = Options()
    img=cv2.imread("unit_test/canmake4/A_in.png")
    make=Makeup(img,img,opt,name='res')
    makeup_img=np.uint8(make.test(img,make_dict))[-1]
    cv2.imwrite("aa.png",makeup_img)
