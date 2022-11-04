from pathlib import Path

from pkg_resources import safe_name
import cv2
make_name="Makeup/2.png"
trans_name="test.png"


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation) for im in im_list]
    return cv2.hconcat(im_list_resize)

def main(input_path,output_path,save_path,mode=None):
    input_path=Path(input_path)
    output_path=Path(output_path)
    save_path=Path(save_path)

    inputs=list(input_path.iterdir())
    for p in inputs:
        try:
            in_img=cv2.imread(str(p/'A_in.png'))
            out_img=cv2.imread(str(p/'A_out.png'))
            trans_input_img=cv2.imread(str(p/'B_in.png'))
            trans_output_img=cv2.imread(str(p/'B_in.png'))

            make_img=cv2.imread(str(output_path/p.name/make_name))
            trans_img=cv2.imread(str(output_path/p.name/trans_name))

            save_img=hconcat_resize_min((in_img,out_img,make_img,trans_input_img,trans_output_img,trans_img))
            save_name=str(save_path/p.name)+'.png'
            print(save_name)
            cv2.imwrite(save_name,save_img)
        except:
            print(p)

    for p in inputs:
        try:
            in_img=cv2.imread(str(p/'A_in.png'))
            out_img=cv2.imread(str(p/'A_out.png'))
            trans_input_img=cv2.imread(str(p/'B_in.png'))
            trans_output_img=cv2.imread(str(p/'B_in.png'))

            make_img=cv2.imread(str(output_path/(p.name+'_inv')/make_name))
            trans_img=cv2.imread(str(output_path/(p.name+'_inv')/trans_name))

            save_img=hconcat_resize_min((in_img,out_img,make_img,trans_input_img,trans_output_img,trans_img))
            save_name=str(save_path/p)+'.png'
            cv2.imwrite(safe_name,save_img)
        except:
            print(p)