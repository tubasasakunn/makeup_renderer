from re import I
from sqlite3 import paramstyle
from wsgiref.simple_server import WSGIRequestHandler
from cv2 import imread, imwrite
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
import random
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
    


    img_paths=Path("test_img/virtual_makeup")
    img_paths=list(img_paths.iterdir())
    for path in img_paths:
        inimg=cv2.imread(str(path/"B_out.png"))
        outimg=cv2.imread(str(path/"A_in.png"))

        inimg_pro=PostProcessing(inimg)
        model_paths=Path("model/Makeup_Model/Face")
        model_paths=list(model_paths.iterdir())
        for model_path in model_paths:
            print(model_path)
            if model_path.suffix==".pth":
                params.append(main(inimg_pro,str(model_path)))

        make_dict=MakeParams(params,"Face")

        opt = Options()
        make=Makeup(outimg,outimg,opt,name='res')
        makeup_img=np.uint8(make.test(outimg,make_dict))[-1]
        cv2.imwrite("res/{}.png".format(path.name),np.hstack([inimg,outimg,makeup_img]))

def use_best_test():
    params=[]
    


    img_paths=Path("test_img/virtual_makeup")
    img_paths=list(img_paths.iterdir())
    for path in img_paths:
        inimg=cv2.imread(str(path/"B_out.png"))
        outimg=cv2.imread(str(path/"A_in.png"))

        inimg_pro=PostProcessing(inimg)
        model_paths=Path("model/Makeup_Model/Multi")
        model_paths=list(model_paths.iterdir())
        for model_path in model_paths:
            print(model_path)
            if model_path.is_file():
                continue
            if len(list(model_path.iterdir()))<3 :
                continue
            params.append(use_best(inimg_pro,str(model_path)))

        make_dict=MakeParams(params,"Face")

        opt = Options()
        make=Makeup(outimg,outimg,opt,name='res')
        makeup_img=np.uint8(make.test(outimg,make_dict))[-1]
        cv2.imwrite("res/{}.png".format(path.name),np.hstack([inimg,outimg,makeup_img]))

def compire():
    faces=list(Path("validation").iterdir())
    #faces=list(Path("dataset").iterdir())
    face=random.choice(faces)

    name=random.choice(list(face.iterdir()))
    img_in=cv2.imread(str(name))

    path=Path("model/Makeup_Model/Face")/(face.name+'.pth')
    print(face.name)
    params=main(img_in,path)
    
    seikai=str2list(Path(name).stem)

    opt = Options()
    img=cv2.imread("unit_test/canmake4/A_in.png")
    cv2.imwrite("res/0_or.png",img)

    print(seikai)
    make_dict=MakeParams([seikai],"Face")
    make=Makeup(img,img,opt,name='res')
    makeup_imgA=np.uint8(make.test(img,make_dict))[-1]
    cv2.imwrite("res/1_seikai.png",makeup_imgA)

    print(params)
    make_dict=MakeParams([params],"Face")
    make=Makeup(img,img,opt,name='res')
    makeup_imgB=np.uint8(make.test(img,make_dict))[-1]
    cv2.imwrite("res/2_predict.png",makeup_imgB)
    print("useFace",face.name)
    print("正解:",[round(j,2) for j in seikai ])
    print("推論",[round(j,2) for j in params ])
    name="res/"+"正解:"+str([round(j,2) for j in seikai ])+"推論"+str([round(j,2) for j in params ])+".png"
    img_in=cv2.resize(img_in,dsize=img.shape[:2])
    cv2.imwrite(name,np.hstack([img_in,img,makeup_imgA,makeup_imgB]))

def use_best_compire():
      
    faces=list(Path("validation").iterdir())
    #faces=list(Path("dataset").iterdir())
    face=random.choice(faces)

    name=random.choice(list(face.iterdir()))
    img_in=cv2.imread(str(name))

    path=Path("model/Makeup_Model/Multi")/face.name
    params=use_best(img_in,path)
    
    seikai=str2list(Path(name).stem)

    opt = Options()
    img=cv2.imread("unit_test/canmake4/A_in.png")
    cv2.imwrite("res/0_or.png",img)

    print(seikai)
    make_dict=MakeParams([seikai],"Face")
    make=Makeup(img,img,opt,name='res')
    makeup_imgA=np.uint8(make.test(img,make_dict))[-1]
    cv2.imwrite("res/1_seikai.png",makeup_imgA)

    print(params)
    make_dict=MakeParams([params],"Face")
    make=Makeup(img,img,opt,name='res')
    makeup_imgB=np.uint8(make.test(img,make_dict))[-1]
    cv2.imwrite("res/2_predict.png",makeup_imgB)
    print("useFace",face.name)
    print("正解:",[round(j,2) for j in seikai ])
    print("推論",[round(j,2) for j in params ])
    name="res/"+"正解:"+str([round(j,2) for j in seikai ])+"推論"+str([round(j,2) for j in params ])+".png"
    img_in=cv2.resize(img_in,dsize=img.shape[:2])
    cv2.imwrite(name,np.hstack([img_in,img,makeup_imgA,makeup_imgB]))

def use_best(img,path):
    model_paths=Path(path)
    model_paths=list(model_paths.iterdir())
    paramses=[]
    for model_path in model_paths:
        paramses.append(main(img,model_path))

    paramses=np.array(paramses)
    params=np.median(paramses,axis=0).tolist()

    return params




if __name__ == '__main__':
    #use_best_compire()
    #use_best_test()
    #for i in range(10):
    #    compire()
    test()

