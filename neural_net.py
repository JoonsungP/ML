import torch
import torch.nn as nn

res_chan = 64

class edrn_core(nn.Module):
    def __init__(self,nvar,n_stn=439,n_block=16):
        super(edrn_core, self).__init__()
        self.n_block = n_block
        self.init_conv = nn.Conv2d(nvar,res_chan,kernel_size=8,padding=2,bias=False)          # 9x9x64 before EDRN core (8x8x64 in here)
        self.res_block = nn.Sequential(
            nn.Conv2d(res_chan,res_chan,kernel_size=3,padding=1,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(res_chan,res_chan,kernel_size=3,padding=1,bias=False))

        #self.upsampling = nn.Sequential(
        #    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.out_conv = nn.Conv2d(res_chan,res_chan*nvar,kernel_size=5,padding=2,bias=False)
        self.fc = nn.Sequential(
            nn.Linear(res_chan*nvar*32*32,n_stn),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.init_conv(x)
        long_skip = x
        for i in range(self.n_block):
            x_res = self.res_block(x)
            x = torch.add(x, x_res)

        x = torch.add(x,long_skip)
        #x = self.upsampling(x)
        x = self.out_conv(x)
        x = torch.flatten(x,1)
        x = self.fc(x)

        return x



