from importlib import import_module
import cv2
from pathlib import Path
import numpy as np
import sys
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

def path_sort(path_list):
    path_str=[]
    for path in path_list:
        path_str.append(str(path))
    path_str.sort()
    new_path_list=[]
    for path in path_str:
        new_path_list.append(Path(path))
    return new_path_list

def video2imgs(video_path):
    cap = cv2.VideoCapture(video_path)
    imgs=[]
    while True:
        ret, frame = cap.read()
        if ret:
            imgs.append(frame)
        else:
            return imgs,cap.get(cv2.CAP_PROP_FPS)

def pil2cv(imgPIL):
    imgCV_RGB = np.array(imgPIL, dtype = np.uint8)
    imgCV_BGR = np.array(imgPIL)[:, :, ::-1]
    return imgCV_BGR

def cv2pil(imgCV):
    imgCV_RGB = imgCV[:, :, ::-1]
    imgPIL = Image.fromarray(imgCV_RGB)
    return imgPIL

def imgs_text(imgs,names):
    res=[]
    for i in range(len(imgs)):
        img=imgs[i]
        name=names[i]
        img = Image.fromarray(img)   
        draw = ImageDraw.Draw(img)  
        font=ImageFont.truetype("/usr/share/fonts/truetype/fonts-japanese-mincho.ttf",200)
        draw.text((0, 0),name,(255,255,255),font=font)
        img = np.array(img) 
        res.append(img)
        


    return res

def imgs2video(output_path,imgs,fps):
    h,w,c=imgs[0].shape
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(output_path,fourcc, fps, (w,h))
    for img in imgs:
        video.write(img)
    video.release()

def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

def reshape(imgs,row=2):
    l=len(imgs)
    margin=imgs[0]*0
    new_imgs=[]
    k=0
    for i in range(-(-l//row)):
        row_imgs=[]
        for j in range(row):
            if k < l:
                row_imgs.append(imgs[k])
                k=k+1
            else:
                row_imgs.append(margin)
        new_imgs.append(row_imgs)

    return new_imgs

def main(input_paths,output_path):
    imgss=[]
    res_imgs=[]
    names=["入力画像","提案手法","PSGAN","CPM"]

    for input_path in input_paths:
        imgs,fps=video2imgs(input_path)
        imgss.append(imgs)

    for i in tqdm(range(len(imgss[0]))):
        imgs=[imgss[j][i] for j in range(len(imgss))]

        imgs=imgs_text(imgs,names)
        imgs=reshape(imgs,2)
        img=concat_tile(imgs)
        img=cv2.resize(img,dsize=None,fx=0.5,fy=0.5)
        res_imgs.append(img)

        if i%100==20:
            imgs2video(output_path,res_imgs,fps)

    imgs2video(output_path,res_imgs,fps)
    print("finish")

if __name__ == '__main__':
    args = sys.argv
    name=args[1]
    path=Path(name)
    output=str(path/"res.mov")
    file_list=list(path.iterdir())
    file_list=path_sort(file_list)
    file_list=[str(p) for p in file_list]
    main(file_list,output)