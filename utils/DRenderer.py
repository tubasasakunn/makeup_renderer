from cgi import test
from wsgiref.simple_server import WSGIRequestHandler
from matplotlib.image import thumbnail
from numpy.lib.shape_base import _put_along_axis_dispatcher
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch
import copy
from utils.CubicSpline import cubicSpline
import utils.utils as utils
import kornia
import torchvision

def normalize(x, width):
    return x * (width - 1) 

def dnorm(x,y):
    return torch.sqrt(x*x+y*y+0.0001)





class Renderer():
    def __init__(self, canvas,renderer,mode="Gaussian",device='cpu',break_time=1):
        self.device=device
        self.w,self.h = canvas.shape[0:2]
        self.renderer=renderer
        #mode="original"
        self.mode=mode
        self.break_time=break_time
        if self.renderer=='powder':
            self.d=7
            self.d_shape=7
            self.d_color=3
            self.d_alpha=1
    
    def rotatemat(self,theta):
        sin=torch.sin(theta)
        cos=torch.cos(theta)
        mat=torch.stack((torch.stack((cos,sin,torch.tensor((0)).to(self.device))),torch.stack((-1*sin,cos,torch.tensor((0)).to(self.device)))))
        return mat
        
    def cod2canvas(self,x,y,thickness,x_cod,y_cod):
        canvas_x=x_cod-x+0.001
        canvas_y=y_cod-y+0.001
        canvas=torch.pow(canvas_x,2)+torch.pow(canvas_y,2)
        canvas=canvas+0.0001
        if self.mode=="original":
            thickness=normalize(thickness,(self.h+self.w)//20)
            canvas=torch.sqrt(canvas)
            canvas=utils.dclamp((thickness-canvas)/thickness,min=0,max=1)
        elif self.mode=="Gaussian":
            thickness=normalize(thickness,(self.h+self.w)//60)
            canvas=torch.exp(-1*canvas/(2*thickness*thickness+0.0001))
            canvas=canvas/(canvas.max()+0.001)
        elif self.mode=="Line":
            thickness=normalize(thickness,(self.h+self.w)//60)
            canvas=torch.exp(-1*canvas/(2*thickness*thickness+0.0001))
            canvas=canvas/(canvas.max()+0.001)
        elif self.mode=="Stick":
            thickness=normalize(thickness,(self.h+self.w)//2)
            canvas=torch.exp(-1*canvas/(2*thickness*thickness+0.0001))
            canvas=canvas/(canvas.max()+0.001)
        return canvas

    def blur(self,canvas,thickness):
        if self.mode=="original":
            thickness=normalize(thickness,(self.h+self.w)//20)
        elif self.mode=="Gaussian":
            thickness=normalize(thickness,(self.h+self.w)//60)
        
        k_size=max(51,int(thickness)*40+1)
        k_size=min((min(canvas.shape))*2+-3,k_size)
        sigma=0.3*(k_size/2-1)+0.8
        gauss = kornia.filters.GaussianBlur2d((k_size, k_size), (sigma, sigma))
        canvas=canvas.unsqueeze(0)
        canvas=canvas.unsqueeze(0)
        canvas=gauss(canvas)
        return canvas

    def maek_params(self):
        if self.renderer=='powder':
            return self.make_params_area()
        if self.renderer=='pen':
            return self.make_params_area()
        if self.renderer=='stick':
            return self.make_params_area()

    def draw_stroke(self,params):
        if self.renderer=='powder':
            return self.powder(params)
        if self.renderer=='pen':
            return self.pen(params)
        if self.renderer=='stick':
            return self.stick(params)



    def make_params_area(self):
        start_x=torch.rand(1)#.uniform(0.1,0.35)
        start_y=torch.rand(1)#.uniform(0.1,0.35)
        end_x=torch.rand(1)#.uniform(0.65,0.99)
        end_y=torch.rand(1)#.uniform(0.35,0.65)
        middle_x=torch.rand(1)#(start_x+end_x)/2+ torch.random.uniform(-0.2,0.2)
        middle_y=torch.rand(1)#(start_y+end_y)/2+torch.random.uniform(-0.2,0.2)
        thickness=torch.rand(1)#.uniform(0.01,0.4)
        return torch.cat((start_x,start_y,end_x,end_y,middle_x,middle_y,thickness))

    def make_area(self,params):
        canvas=torch.zeros([self.w,self.h]).to(self.device)
        canvas_x,canvas_y=torch.meshgrid(torch.arange(self.w),torch.arange(self.h))
        canvas_x=canvas_x.to(self.device)
        canvas_y=canvas_y.to(self.device)


        #start_x,start_y,end_x,end_y,middle_x,middle_y,thickness=params
        start_x=params[0]
        start_y=params[1]
        end_x=params[2]
        end_y=params[3]
        middle_x=params[4]
        middle_y=params[5]
        thickness=params[6]
        start_p=torch.stack((start_x,start_y,torch.tensor((1)).to(self.device)))
        middle_p=torch.stack((middle_x,middle_y,torch.tensor((1)).to(self.device)))
        end_p=torch.stack((end_x,end_y,torch.tensor((1)).to(self.device)))
        
        x_axis_vec=torch.Tensor([1,0]).to(self.device)
        vec=torch.stack([end_x-start_x,end_y-start_y])
        vec=vec/dnorm(vec[0],vec[1])#start->end unit vec


        theta=torch.arccos(torch.dot(x_axis_vec,vec))

        mat=self.rotatemat(theta)
        inv_mat=self.rotatemat(-1*theta)
        
        local_start_p=torch.matmul(mat,start_p-start_p)
        local_end_p=torch.matmul(mat,end_p-start_p)
        local_middle_p=torch.matmul(mat,middle_p-start_p)

        spline_points=torch.stack((local_start_p,local_middle_p,local_end_p))
        
        spl = cubicSpline(spline_points)
        point_num=100
        for num in range(point_num):
            if num>point_num*self.break_time:
                break
            point_x=local_start_p[0]+local_end_p[0]*num/point_num
            point_y=spl.fit(point_x)
            point_x,point_y=torch.matmul(inv_mat,torch.stack((point_x,point_y,torch.tensor(1).to(self.device))))+start_p[0:2]   
            point_x=normalize(point_x,self.w)
            point_y=normalize(point_y,self.h)

            
            canvas_x_tmp=copy.deepcopy(canvas_x)
            canvas_y_tmp=copy.deepcopy(canvas_y)
            if self.mode=="Line" and num/point_num >0.5:
                min_t=0.93
                thickness=thickness*(2*(min_t-1)*num/point_num+2-min_t)
            canvas_tmp=self.cod2canvas(point_x,point_y,thickness,canvas_x_tmp,canvas_y_tmp)
            canvas=canvas_tmp+canvas
            


        canvas=utils.dclamp(canvas,min=0,max=1)
        return canvas


    def powder(self,params):
        thickness=params[6]
        canvas=self.make_area(params)
        canvas = self.blur(canvas,thickness)
        return canvas

    def pen(self,params):
        self.mode="Line"
        canvas=self.make_area(params)
        canvas=canvas.unsqueeze(0)
        canvas=canvas.unsqueeze(0)
        return canvas

    def stick(self,params):
        self.mode="Stick"
        canvas=self.make_area(params)
        canvas=canvas.unsqueeze(0)
        canvas=canvas.unsqueeze(0)
        return canvas


if __name__ == '__main__':
    for i in range(11):
        canvas=np.zeros([128,128,3])
        rdrr=Renderer(canvas,'area')
        start_x=torch.tensor([0.6])
        start_y=torch.tensor([0.1])
        end_x=torch.tensor([0.6])
        end_y=torch.tensor([0.9])
        middle_x=torch.tensor([i/10+0.001],requires_grad = True)
        middle_y=torch.tensor([0.2],requires_grad = True)
        thickness=torch.tensor([0.2],requires_grad = True)
        params=torch.cat((start_x,start_y,end_x,end_y,middle_x,middle_y,thickness))
        #params=rdrr.make_params_area()
        canvas=rdrr.draw_stroke(params)
        torchvision.utils.save_image(canvas,"testDrender"+str(i)+".png")