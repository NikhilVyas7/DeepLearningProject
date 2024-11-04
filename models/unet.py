import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms.functional as TF

class Contract(nn.Module):

    def __init__(self,in_channels,out_channels):
        super(Contract,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3)
        self.relu = nn.ReLU() #Modifies input
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
    def forward(self,x):

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.pool(x)
    

class Expand(nn.Module):
    #Consider adding padding to Expand input to limit size discrepancy
    def __init__(self,in_channels,out_channels):
        super(Expand,self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2)
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3)
        self.relu = nn.ReLU() #Modifies input
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3)
    def forward(self,x,copy):
        _, _, h, w = x.size() #Determine expected h, w size 
        #pad x to be same size as 
        x = self.up_conv(x) 
        diffY = copy.size()[2] -  x.size()[2]
        diffX = copy.size()[3] -  x.size()[3]
        x = F.pad(x,(diffX //2, diffX - diffX //2, diffY //2, diffY - diffY //2))
        #Padding x up to copy size instead of bringing it down
        x = torch.cat((x,copy),dim=1)#Concatenate by channels

        x = self.relu(self.conv1(x))

        x = self.relu(self.conv2(x))

        return x

class UNetOutput(nn.Module):
        #Like Expand, but ensures output matches image input size
    def __init__(self,in_channels,num_classes):
        super(UNetOutput,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1) #Don't change size
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels,num_classes,kernel_size=3,padding=1)
    def forward(self,x,copy):
        #Interpolate x size up to copy size.
        #Apply conv relu, and conv relu to get to channel size of output
        _, _, h, w= copy.size()
        x = F.interpolate(x,(h,w))
        x = torch.cat((x,copy),dim=1)
        return self.conv2(self.relu(self.conv1(x)))


class UNet(nn.Module):

    def __init__(self,num_classes=10,channels=None):
        #There will be 6 upsamples, 
        super(UNet,self).__init__()
        if channels == None:
            channels = [64,128,256,512,1024]
        contract_channels = [3] + channels #3,64,128,256,512,1024
        self.contract_channels = contract_channels
        self.contracts = nn.ModuleList([Contract(contract_channels[i],contract_channels[i+1]) for i in range(len(contract_channels)-1)])
        expand_channels = list(reversed(channels))#1024,512,256,128,64
        self.expand_channels = expand_channels
        self.expands = nn.ModuleList([Expand(expand_channels[i],expand_channels[i+1]) for i in range(len(expand_channels)-1)])
        #self.expands[i] will match with self.contracts[len(contracts)-2-i]
        #The expand going from 1024 to 512 has to match with the output from contract going from 256 to 512

        self.output = UNetOutput(in_channels=expand_channels[-1]+3,num_classes=num_classes)


    def forward(self,x):
        intermediate_outputs = {3:x}
        for i, module in enumerate(self.contracts):
            intermediate_outputs[self.contract_channels[i+1]] =  module(intermediate_outputs[self.contract_channels[i]])
        out = intermediate_outputs[self.contract_channels[-1]]
        for i, module in enumerate(self.expands):
            channel = self.expand_channels[i+1]#output of this modules
            out = module(out,intermediate_outputs[channel])
        
        out = self.output(out,x)
        return out
if __name__ == "__main__":
    print("Testing model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_input = torch.randn((2,3,1000,750)).to(device) #Literally 16 times smaller, so can probably do batch sizes of 16
    model = UNet().to(device)
    #model = nn.DataParallel(model,device_ids=[0,1])
    print("Thru model..")
    out = model.forward(sample_input)
    print(out.shape)