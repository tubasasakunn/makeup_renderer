import enum
from unicodedata import name
from makeup import Makeup
from utils.trimFace import TrimFace
import random
import utils.utils as utils
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

face_path=Path("dataset/nonMakeupDataset")
presets=[]
presets.append([[0.25],[0.4],[0.45],[0.4],[0.35],[0.4],[0.2],[0.2,0.2,0.2],[0.91]])
presets.append([[0.1],[0.5],[0.3],[0.9],[0.2],[0.8],[0.3],[0.2,0.2,0.2],[0.91]])
presets.append([[0.3],[0.9],[0.6],[0.9],[0.5],[0.95],[0.3],[0.2,0.2,0.2],[0.91]])
presets.append([[0.6],[0.8],[0.8],[0.7],[0.7],[0.8],[0.3],[0.2,0.2,0.2],[0.91]])
presets.append([[0.8],[0.6],[0.8],[1-0.6],[0.9],[0.5],[0.3],[0.2,0.2,0.2],[0.91]])
presets.append([[0.5],[0.1],[0.5],[0.2],[0.5],[0.15],[0.2],[0.9,0.9,0.9],[0.91]])
presets.append([[0.38],[0.4],[0.4],[0.38],[0.4],[0.4],[0.2],[0.9,0.9,0.9],[0.91]])
presets.append([[0.7],[0.5],[0.75],[0.5],[0.725],[0.5],[0.2],[0.9,0.9,0.9],[0.91]])    
presets.append([[0.9],[0.5],[0.95],[0.5],[0.925],[0.5],[0.2],[0.9,0.9,0.9],[0.91]])     
presets.append([[0.1],[0.2],[0.1],[0.8],[0.1],[0.5],[0.2],[0.9,0.9,0.9],[0.91]])        
presets.append([[0.4],[0.5],[0.5],[0.5],[0.45],[0.5],[0.2],[0.9,0.9,0.9],[0.91]])
presets.append([[0.5],[0.5],[0.6],[0.5],[0.55],[0.5],[0.2],[0.9,0.9,0.9],[0.91]])
presets.append([[0.1],[0.5],[0.3],[0.9],[0.2],[0.8],[0.3],[0.2,0.2,0.2],[0.91]])
presets.append([[0.3],[0.9],[0.6],[0.9],[0.5],[0.95],[0.3],[0.2,0.2,0.2],[0.91]])

class Options(object):
    make_list=["Foundation","Lipstick","Eye","Face"]
    name='test_ref'
    device='cuda'

def PostProcessing(img):
    landmark=utils.get_landmark(img)
    mask=utils.get_parsing(img)
    face=TrimFace(img,img,mask,landmark)
    face_img,_=face.get_face()
    face_img=cv2.resize(face_img,dsize=(256,256))
    return face_img

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
            for j in range(len(preset[i])):
                noise=random.normalvariate(0,0.3)
                preset[i][j]=min(max(preset[i][j]+noise,0.0),1.0)
            config[key]=[preset[i]]

        config['useMirror']=False
        config['useMulti']=False
        config['renderMode']=renderMode
        if make_name=="Eye":
            config['right']=right
    params={}
    params[make_name]={"test":config}
    return params


def make_datasets(name,num,preset):
    name=Path(name)
    name.mkdir(exist_ok=True) 
    opt = Options()
    paths=list(face_path.iterdir())
    random.shuffle(paths)
    for i in tqdm(range(num)):
        try:
            make_dict=MakeParams(preset,"Face")
            img_path=paths[i%len(paths)]
            img=cv2.imread(str(img_path))
            make=Makeup(img,img,opt,name='res')
            makeup_img=np.uint8(make.test(img,make_dict))[-1]
            print(makeup_img.shape)
            makeup_img=PostProcessing(makeup_img)
            cv2.imwrite(str(name/(str(preset)+'.jpg')),makeup_img)
        except:
            print(paths[i%len(paths)])

if __name__ == '__main__':
    for i,preset in enumerate(presets):
        make_datasets("dataset/dataset_Face%d"%(i),1000,preset)