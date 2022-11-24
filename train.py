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
import model

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256,64,3)
        self.conv2 = nn.Conv2d(64,32,3)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.fc1 = nn.Linear(6272, 800)
        self.fc2 = nn.Linear(800, 120)
        self.fc3 = nn.Linear(120, 11)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        x=self.pool(self.relu(self.conv1(x)))
        x=self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size()[0], -1)
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.relu(self.fc3(x))
        return x



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

def main(dataset_path,max_epoch=2000,name="test"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"
    print("use device is %s"%(device))

    name=Path(name)
    name.mkdir(exist_ok=True) 
    
    image_dataset=Dataset(dataset_path,device)
    train_dataset, valid_dataset = torch.utils.data.random_split(
    image_dataset, 
    [int(len(image_dataset)*0.7), len(image_dataset)-int(len(image_dataset)*0.7)]
)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=True)
    psgan_net=model.Generator().to(device)
    psgan_net.load_state_dict(torch.load("model/G.pth"))
    net=BaseModel().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    writer = SummaryWriter(str(name/"log"))
    writer_i=0
    for epoch in range(max_epoch):
        #train
        for i,(img,params) in enumerate(train_dataloader):
            #学習
            optimizer.zero_grad()
            latent=psgan_net(img)#[bacth_size,3,256,256]->[bacth_size,256,64,64]
            predict=net(latent)#[bacth_size,256,64,64]->[batch_size,11]
            loss = criterion(predict, params)
            loss.backward()
            optimizer.step()

            #確認用
            writer_i=writer_i+1
            writer.add_scalar("loss",loss,writer_i)
            print("epoch=%d iter=%d loss=%f"%(epoch,i,loss))

        #test
        for i,(img,params) in enumerate(valid_dataloader):
            latent=psgan_net(img)#[bacth_size,3,256,256]->[bacth_size,256,64,64]
            predict=net(latent)
            loss = criterion(predict, params)
            print("test epoch=%d iter=%d loss=%f"%(epoch,i,loss))
            
        
        if epoch%100==0:
            torch.save(net.state_dict(),str(name/"log"/"model_last.pth"))
        if epoch%200==0:
            torch.save(net.state_dict(),str(name/"log"/"model_%05d.pth")%(epoch))

        
    writer.close()
if __name__ == '__main__':
    res=Path("result")
    res.mkdir(exist_ok=True)
    dataset_path=Path("dataset")
    paths=list(dataset_path.iterdir())
    for path in paths:
        print(path)
        main(str(path),2000,str(res/path.name))
