# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

## 27x27
# %%
class cnn_test(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=24, out_channels=64, kernel_size=8, stride=1, padding=2, groups=1, bias=False)  # N:27->24
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8, stride=1, padding=2, groups=1, bias=False)  # N:24->21
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8, stride=1, padding=2, groups=1, bias=False)  # N:21->18
        #self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*18*18, 64*9*9)
        self.fc2 = nn.Linear(64*9*9, 16*9*9)
        self.fc3 = nn.Linear(16*9*9, 438)
   
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
    
        return x
# %%
