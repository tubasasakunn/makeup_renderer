import argparse
from CreateUtils import movie_make
from CreateUtils import img_make
from CreateUtils import img_make_con
from CreateUtils import connect
from CreateUtils import copy
from importlib import import_module
from pathlib import Path
from setproctitle import setproctitle, getproctitle

parser = argparse.ArgumentParser()
parser.add_argument("-i","--in_path", type=str, default='test_img/wakaiki.mp4')
parser.add_argument("-o","--out_path", type=str, default='test.mp4')
parser.add_argument("-s","--save_path", type=str, default='test.mp4')
parser.add_argument("-p","--pt_path", type=str, default='res_multi_normal/canmake4/params.pt')
parser.add_argument("-c", "--config_path",type=str, default='option/config.py')
parser.add_argument("-m", "--mode",type=str, default='MakeVideo')


if __name__ == '__main__':
    args = parser.parse_args()
    in_path=Path(args.in_path)
    out_path=Path(args.out_path)
    save_path=Path(args.save_path)
    pt_path=Path(args.pt_path)
    config_path=Path(args.config_path)
    mode=args.mode

    #opt作成
    lib=''
    for i in list(config_path.parts)[:-1]:
        lib=lib+i+'.'
    lib=lib+config_path.stem
    option = import_module(lib)
    opt = option.Options()
    setproctitle(opt.name)

    root=Path('.')
    for parent in list(save_path.parts)[:-1]:
        root=root/parent
        root.mkdir(exist_ok=True) 

    if not '.' in list(save_path.parts)[-1]:
        root=root/list(save_path.parts)[-1]
        root.mkdir(exist_ok=True) 

    if mode=="MakeVideo":
        movie_make.main(str(in_path),str(save_path),str(pt_path),opt)
    elif mode=="test":
        img_make.main(str(in_path),str(save_path),str(pt_path),opt,"test")
    elif mode=="test_stroke":
        img_make.main(str(in_path),str(save_path),str(pt_path),opt)
    elif mode=="test_con":
        img_make_con.main(str(in_path),str(save_path),opt)
    elif mode=="connect":
        connect.main(str(in_path),str(out_path),str(save_path))
    elif mode=="copy":
        copy.main(str(in_path),str(save_path))
