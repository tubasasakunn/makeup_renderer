from pathlib import Path
import shutil
import sys
from tqdm import tqdm
EXTS = [".jpg", ".png", ".JPG", ".jpeg", ".tif",".gif" ,".tiff", ".TIF"]
#めんどくさい構造のファイルのみをコピー
#ディレクトリ内のEXTSの拡張子のもののみをコピー
#ディレクトリinディレクトリで同じファイル名のときデータがかぶるのに配慮

def reGet(path):
    files=[]
    for path_i in path.iterdir():
        if path_i.is_dir():
            files.extend(reGet(path_i))
        else:
            files.append(path_i)
    return files

def rename_cp(file_path,dir_path):
    #if file_path.suffix in EXTS:
    if file_path.name=="all_img.png":
        fromPath=file_path
        toPath=dir_path/(str(file_path).replace('/', ''))
        shutil.copy(fromPath, toPath)


if __name__ == '__main__':
    args = sys.argv
    in_dir=Path(args[1])
    out_dir=Path(args[2])
    out_dir.mkdir(exist_ok=True) 
    files=reGet(in_dir)
    for file in tqdm(files):
        rename_cp(file,out_dir)
