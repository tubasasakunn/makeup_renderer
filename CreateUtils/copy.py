from pathlib import Path
import cv2

file_name="all_img.png"
def main(input_path,save_path):
    input_path=Path(input_path)
    save_path=Path(save_path)
    save_path.mkdir(exist_ok=True) 
    dirs=list(input_path.iterdir())
    for dir in dirs:
        try:
            img_name=str(dir/file_name)
            img=cv2.imread(img_name)
            cv2.imwrite(str(save_path/dir.name)+'.png',img)
        except:
            print(dir)