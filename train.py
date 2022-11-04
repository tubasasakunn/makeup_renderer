
import argparse
from importlib import import_module
import cv2
from makeup import Makeup
from pathlib import Path
from setproctitle import setproctitle, getproctitle
import torch


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config_path")
parser.add_argument("-i","--in_path", type=str, default='test_img/canmake2_in.png')
parser.add_argument("-o","--out_path", type=str, default='test_img/canmake2_out.png')
parser.add_argument("-t","--test_path", type=str, default='test_img/canmake2_in.png')
parser.add_argument("-r","--res_path", type=str, default='res')
parser.add_argument("-m","--mode", type=str, default='single')
args = parser.parse_args()

p=Path(args.config_path)
lib=''
for i in list(p.parts)[:-1]:
    lib=lib+i+'.'
lib=lib+p.stem

option = import_module(lib)
opt = option.Options()
in_path=args.in_path
out_path=args.out_path
test_path=args.test_path
res_path=args.res_path
mode=args.mode
setproctitle(opt.name)

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im.astype('int32'), (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation) for im in im_list]
    return cv2.hconcat(im_list_resize)


def single(in_path,out_path,test_path,res_path,opt):


    res_path=Path(res_path)
    res_path.mkdir(exist_ok=True) 

    in_img=cv2.imread(in_path)
    out_img=cv2.imread(out_path)
    test_img=cv2.imread(test_path)
    make=Makeup(in_img,out_img,opt,name=str(res_path/opt.name))

    dict,res_img=make.train(opt)
    print("test")
    img=make.test(test_img,dict)
    
    cv2.imwrite(str(res_path/opt.name/"test.png"),img)


def multi(in_path,res_path,opt):
    imgs_path=Path(in_path)
    res_path=Path(res_path)
    res_path.mkdir(exist_ok=True) 
    files=list(imgs_path.iterdir())
    print(files)
    mistakeA=[]
    mistakeB=[]
    for p in files:
        try:
        #if True:
            p_name=p.name
            in_img=cv2.imread(str(p/'A_in.png'))
            out_img=cv2.imread(str(p/'A_out.png'))
            trans_in_img=cv2.imread(str(p/'B_in.png'))
            trans_out_img=cv2.imread(str(p/'B_out.png'))

            makeup=Makeup(in_img,out_img,opt,name=str(res_path/p_name))
            params_dict,res_img=makeup.train(opt)
            torch.save(params_dict,str(res_path/p_name/'params.pt'))
            trans_res_img=makeup.test(trans_in_img,params_dict)
            cv2.imwrite(str(res_path/p_name/'in_img.png'),in_img)
            cv2.imwrite(str(res_path/p_name/'out_img.png'),out_img)
            cv2.imwrite(str(res_path/p_name/'res_img.png'),res_img)
            cv2.imwrite(str(res_path/p_name/'trans_in_img.png'),trans_in_img)
            cv2.imwrite(str(res_path/p_name/'trans_out_img.png'),trans_out_img)
            cv2.imwrite(str(res_path/p_name/'trans_res_img.png'),trans_res_img)
            all=hconcat_resize_min((in_img,out_img,res_img,trans_in_img,trans_out_img,trans_res_img))
            cv2.imwrite(str(res_path/p_name/'all_img.png'),all)
        except:
           mistakeA.append(p)    
    for p in files:
        try:
            p_name=p.name+'_inv'
            in_img=cv2.imread(str(p/'B_in.png'))
            out_img=cv2.imread(str(p/'B_out.png'))
            trans_in_img=cv2.imread(str(p/'A_in.png'))
            trans_out_img=cv2.imread(str(p/'A_out.png'))

            makeup=Makeup(in_img,out_img,opt,name=str(res_path/p_name))
            params_dict,res_img=makeup.train(opt)
            torch.save(params_dict,str(res_path/p_name/'params.pt'))
            trans_res_img=makeup.test(trans_in_img,params_dict)
            cv2.imwrite(str(res_path/p_name/'in_img.png'),in_img)
            cv2.imwrite(str(res_path/p_name/'out_img.png'),out_img)
            cv2.imwrite(str(res_path/p_name/'res_img.png'),res_img)
            cv2.imwrite(str(res_path/p_name/'trans_in_img.png'),trans_in_img)
            cv2.imwrite(str(res_path/p_name/'trans_out_img.png'),trans_out_img)
            cv2.imwrite(str(res_path/p_name/'trans_res_img.png'),trans_res_img)
            all=hconcat_resize_min((in_img,out_img,res_img,trans_in_img,trans_out_img,trans_res_img))
            cv2.imwrite(str(res_path/p_name/'all_img.png'),all)
        except:
           mistakeA.append(p)    



    print(mistakeA,mistakeB)


if __name__ == '__main__':    

    if mode=="single":
        single(in_path,out_path,test_path,res_path,opt)
    elif mode=="multi":
        multi(in_path,res_path,opt)
    else:
        print("正しくないmode")
