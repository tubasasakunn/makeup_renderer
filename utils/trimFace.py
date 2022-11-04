import numpy as np
import utils.utils as utils
import cv2
import copy

rate=1


class TrimFace:
    def __init__(self,in_img,out_img,mask,landmark):


        #回転でボケるため
        in_img=cv2.resize(in_img,dsize=None,fx=rate,fy=rate)
        out_img=cv2.resize(out_img,dsize=None,fx=rate,fy=rate)
        
        self.ori_img=in_img
        self.trimming_face(in_img,out_img,mask,landmark)


    
    def get_facemask(self,img,mask):
        h,w,c=img.shape
        parsing=mask
        mask=((parsing==1)*1 + (parsing==10)*1+(parsing==12)*1+(parsing==13)*1).astype(np.uint8)
        mask=cv2.resize(mask,dsize=(w,h))
        mask=np.array([mask,mask,mask]).transpose(1,2,0)
        return mask

    def trimming_face(self,in_img,out_img,mask,landmark):

        self.ori_h, self.ori_w = in_img.shape[:2]
        h, w = in_img.shape[:2]
        landmark=utils.get_landmark(in_img)
        mask=self.get_facemask(in_img,mask)
        FACE_COD=utils.FACE_OVAL
        size=np.array((w,h))
        x_axis=(landmark[FACE_COD[9]]-landmark[FACE_COD[27]])[:2]*size
        y_axis=(landmark[FACE_COD[0]]-landmark[FACE_COD[18]])[:2]*size
        center=np.array([landmark[x] for x in FACE_COD]).mean(axis=0)[:2]*size

        x_norm=np.linalg.norm(x_axis, ord=2)*0.5
        y_norm=np.linalg.norm(y_axis, ord=2)*0.5
        angle=np.degrees(np.arctan(x_axis[1]/x_axis[0]))
        
        M = cv2.getRotationMatrix2D(center, angle, 1)
        in_img=cv2.warpAffine(in_img, M, (w, h))
        out_img=cv2.warpAffine(out_img, M, (w, h))
        mask=cv2.warpAffine(mask, M, (w, h))


        print(size)



        in_face=in_img[int(max(center[1]-y_norm,0)):int(min(center[1]+y_norm,in_img.shape[0])),int(max(center[0]-x_norm,0)):int(min(center[0]+x_norm,in_img.shape[1])),:]
        out_face=out_img[int(max(center[1]-y_norm,0)):int(min(center[1]+y_norm,in_img.shape[0])),int(max(center[0]-x_norm,0)):int(min(center[0]+x_norm,in_img.shape[1])),:]
        self.get_mask=mask[int(max(center[1]-y_norm,0)):int(min(center[1]+y_norm,in_img.shape[0])),int(max(center[0]-x_norm,0)):int(min(center[0]+x_norm,in_img.shape[1])),:]
        #out_face=out_img[int(center[1]-y_norm):int(center[1]+y_norm),int(center[0]-x_norm):int(center[0]+x_norm),:]
        self.mask=mask
        self.in_img = in_img
        self.out_img = out_img
        self.in_face = in_face
        self.out_face = out_face
        self.angle=angle
        self.center=center
        self.x_norm=x_norm
        self.y_norm=y_norm
        self.landmark=landmark
        self.mean=np.mean(self.in_face[np.nonzero(self.in_face*(self.get_mask==1))])
        self.std=np.std(self.in_face[np.nonzero(self.in_face*(self.get_mask==1))])
        #self.mean=np.mean(self.in_face*(self.get_mask==1))
        #self.std=np.std(self.in_face*(self.get_mask==1))
        

    def get_face(self):
        input= 128+60*((self.in_face*self.get_mask)-self.mean)/self.std
        output= 128+60*((self.out_face*self.get_mask)-self.mean)/self.std
        #return self.in_face*self.get_mask,self.out_face*self.get_mask
        cv2.imwrite("res.png",input)

        return input.astype(np.float32),output.astype(np.float32)

    def set_face(self,in_face):
        in_face=self.std*(in_face-128)/60+self.mean
        in_face=in_face.clip(0,255)
        #self.in_img[int(self.center[1]-self.y_norm):int(self.center[1]+self.y_norm),int(self.center[0]-self.x_norm):int(self.center[0]+self.x_norm),:]=in_face
        in_face=in_face*self.get_mask+self.in_face*(1-self.get_mask)

        in_img=self.in_img.copy()
        in_img[int(max(self.center[1]-self.y_norm,0)):int(min(self.center[1]+self.y_norm,self.in_img.shape[0])),int(max(self.center[0]-self.x_norm,0)):int(min(self.center[0]+self.x_norm,self.in_img.shape[1])),:]=in_face
        M = cv2.getRotationMatrix2D(self.center, -1*self.angle, 1)
        h, w = self.in_img.shape[:2]
        img=cv2.warpAffine(in_img, M, (w, h))
        mask=cv2.warpAffine(self.mask, M, (w, h))
        #margin=np.sqrt(self.y_norm**2+self.x_norm**2+0.000001)
        original=self.ori_img.copy()
        #original[int(max(self.center[1]-margin,0)):int(min(self.center[1]+margin,original.shape[0])),int(max(self.center[0]-margin,0)):int(min(self.center[0]+margin,original.shape[1])),:] = img[int(max(self.center[1]-margin,0)):int(min(self.center[1]+margin,original.shape[0])),int(max(self.center[0]-margin,0)):int(min(self.center[0]+margin,original.shape[1])),:]
        original=img*mask+original*(1-mask)

        img=cv2.resize(original,dsize=None,fx=1/rate,fy=1/rate)
        return img

