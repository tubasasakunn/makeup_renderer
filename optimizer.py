from torch import optim
import torch

class MakeOptimizer:
    def __init__(self,opt):
        self.opt=opt
    
    def make(self,params_dict,epoch,key=None,i=None):
        max_epoch=self.opt["epoch"]
        rate=epoch/max_epoch

        if self.opt["optimizer"]=="Foundation" or self.opt["optimizer"]=="Lipstick":
            RGB_c=params_dict["RGB"]
            alpha_c=params_dict["alpha"]
            optimizer = optim.Adam([RGB_c,alpha_c],lr=0.1)
            params_name=["RGB","alpha"]
            params_list=[RGB_c,alpha_c]
            params_dict=dict(zip(params_name,params_list))
        elif self.opt["optimizer"]=="Eye" or self.opt["optimizer"]=="Face":
            start_x=params_dict["start_x"]
            start_y=params_dict["start_y"]
            end_x=params_dict["end_x"]
            end_y=params_dict["end_y"]
            middle_x=params_dict["middle_x"]
            middle_y=params_dict["middle_y"]
            thickness=params_dict["thickness"]
            RGB_c=params_dict["RGB"]
            alpha_c=params_dict["alpha"]

            if self.opt["optimizer"]=="Face":
                if rate<0.5:
                    optimizer = optim.Adam([start_x,start_y,end_x,end_y,middle_x,middle_y,RGB_c,alpha_c],lr=0.005)
                else:
                    optimizer = optim.SGD([start_x,start_y,end_x,end_y,middle_x,middle_y,thickness,RGB_c,alpha_c],lr=0.5)
            elif self.opt["optimizer"]=="Eye":
                if self.opt["Params"][key]["renderMode"]=='pen':
                    if rate<0.05:
                        optimizer = optim.Adam([end_x,end_y,thickness,RGB_c],lr=0.01)
                    else:
                        optimizer = optim.Adam([start_x,start_y,end_x,end_y,middle_x,middle_y,thickness,RGB_c],lr=0.001)
                elif self.opt["Params"][key]["renderMode"]=='powder':
                    if rate<-1:
                        optimizer = optim.Adam([RGB_c],lr=0.005)
                    elif rate<0.5:
                        optimizer = optim.Adam([start_x,start_y,end_x,end_y,middle_x,middle_y,RGB_c,alpha_c],lr=0.005)
                    else:
                        optimizer = optim.SGD([start_x,start_y,end_x,end_y,middle_x,middle_y,thickness,RGB_c,alpha_c],lr=0.5)
                    '''
                    if i==None:
                        if (thickness>0.7).max()==1 and epoch==max_epoch-1:
                            mask=torch.ones(alpha_c.shape)
                            mask[thickness>0.7]=0
                            alpha_c=alpha_c*mask
                            optimizer = optim.Adam([RGB_c],lr=0.001)
                    else:    
                        if thickness[i]>0.7 and epoch==max_epoch-1:
                            mask=torch.ones(alpha_c.shape)
                            mask[i]=0
                            alpha_c=alpha_c*mask
                            alpha_c=torch.tensor(alpha_c,requires_grad = True)
                            optimizer = optim.Adam([RGB_c],lr=0.001)
                    '''


            params_name=["start_x","start_y","end_x","end_y","middle_x","middle_y","thickness","RGB","alpha"]
            params_list=[start_x,start_y,end_x,end_y,middle_x,middle_y,thickness,RGB_c,alpha_c]
            params_dict=dict(zip(params_name,params_list))
        

        return optimizer,params_dict
