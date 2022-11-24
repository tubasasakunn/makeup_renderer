from re import I
from sqlite3 import paramstyle
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
import train
import model

class Options(object):
    make_list=["Foundation","Lipstick","Eye","Face"]
    name='test_ref'
    device='cuda'

def MakeParams(preset,make_name,right=True,renderMode='powder'):

    config={}
    '''
    if make_name in ["Foundation", "Lipstick"]:
        for i,key in enumerate(['RGB', 'alpha']):
            for j in range(len(preset)):
                noise=random.normalvariate(0,0.3)
                preset[i][j]=min(max(preset[i][j]+noise,0),1)
            config[key]=[preset[i]]
    '''
    

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
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,dsize=(256,256))
    img=torch.from_numpy(img.astype(np.float32)).clone()/255
    img=img.permute(2,0,1)
    img=img.unsqueeze(0).to(device)

    psgan_net=model.Generator().to(device)
    net=train.BaseModel().to(device)
    psgan_net.load_state_dict(torch.load("model/G.pth"))
    net.load_state_dict(torch.load(model_path))
    print(img.shape)
    latent=psgan_net(img)
    params=net(latent)
    print(params)
    params=torch.clamp(params,0,1)
    print(params)

    return params[0].to("cpu").tolist()

def str2list(moji):
    moji_list=moji.replace("]","").replace("[","").split(',')
    float_list=[float(i) for i in moji_list]
    return float_list

def test():
    params=[]
    
    img=cv2.imread("unit_test/canmake4/B_out.png")
    img=PostProcessing(img)
    model_paths=Path("model/Makeup_Model/MultiFace")
    model_paths=list(model_paths.iterdir())
    for model_path in model_paths:
        print(model_path)
        params.append(main(img,str(model_path)))

    make_dict=MakeParams(params,"Face")

    opt = Options()
    img=cv2.imread("unit_test/canmake4/A_in.png")
    make=Makeup(img,img,opt,name='res')
    makeup_img=np.uint8(make.test(img,make_dict))[-1]
    cv2.imwrite("aa.png",makeup_img)

def compire():
    name="dataset/dataset_Face0/[[0.36956802171659703], [0.5006924544087464], [0.36834583410701216], [0.3904635960359865], [0.24307970985796754], [0.34143806625343737], [0.1886615643802483], [0.0, 0.7321485666861673, 0.0], [0.9230607316649082]].jpg"
    img=cv2.imread(name)
    model_path="model/Makeup_Model/MultiFace/model_last.pth"
    params=main(img,model_path)
    seikai=str2list(Path(name).stem)


    opt = Options()
    img=cv2.imread("unit_test/canmake4/A_in.png")
    cv2.imwrite("0_or.png",img)

    print(seikai)
    make_dict=MakeParams([seikai],"Face")
    make=Makeup(img,img,opt,name='res')
    makeup_img=np.uint8(make.test(img,make_dict))[-1]
    cv2.imwrite("1_seikai.png",makeup_img)

    print(params)
    make_dict=MakeParams([params],"Face")
    make=Makeup(img,img,opt,name='res')
    makeup_img=np.uint8(make.test(img,make_dict))[-1]
    cv2.imwrite("2_predict.png",makeup_img)




if __name__ == '__main__':
    #compire()
    test()

