import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms
from pathlib import Path
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter



def get_model(params_num=11,device="cuda"):
    net = models.vgg16(pretrained=True)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=params_num)
    net=net.to(device)
    return net

class Dataset(data.Dataset):
    def __init__(self, path,device="cuda"):
        self.paths=list(Path(path).iterdir())
        self.device=device
        
    def __len__(self):
        return len(self.paths)
    
    def str2list(self,moji):
        moji_list=moji.replace("]","").replace("[","").split(',')
        float_list=[float(i) for i in moji_list]
        return float_list

    def __getitem__(self, idx):
        path=self.paths[idx]
        params=torch.tensor(self.str2list(str(path.stem)))
        img=read_image(str(path))/255
        params=params.to(self.device)
        img=img.to(self.device)
        return img,params

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"
    print("use device is %s"%(device))
    max_epoch=2000
    training_data=Dataset("./dataset/dataset_Face",device)
    test_data=Dataset("./dataset/dataset_Face_test",device)
    train_dataloader = DataLoader(training_data, batch_size=2, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=2, shuffle=True)
    net=get_model(11,device)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    writer = SummaryWriter("./log")
    writer_i=0
    for epoch in range(max_epoch):
        #train
        for i,(img,params) in enumerate(train_dataloader):
            #学習
            optimizer.zero_grad()
            predict=net(img)
            loss = criterion(predict, params)
            loss.backward()
            optimizer.step()

            #確認用
            writer_i=writer_i+1
            writer.add_scalar("loss",loss,writer_i)
            print("epoch=%d iter=%d loss=%f"%(epoch,i,loss))

        #test
        for i,(img,params) in enumerate(test_dataloader):
            predict=net(img)
            loss = criterion(predict, params)
            print("test epoch=%d iter=%d loss=%f"%(epoch,i,loss))
    writer.close()

main()