import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    def __init__(self, C_in, C_1, C_2):
        super(InceptionBlock, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_1, 1, 1)
        self.conv2 = nn.Conv2d(C_in, C_2, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(C_1)
        self.bn2 = nn.BatchNorm2d(C_2)
        
    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x)))
        return torch.cat([x1, x2], dim=1)
    
class InceptionMnist(nn.Module):
    def __init__(self, C_in, num_classes):
        super(InceptionMnist, self).__init__()
        self.conv1 = nn.Conv2d(C_in, 96, 3, 1)
        self.S1 = nn.Sequential(InceptionBlock(96, 32, 32), InceptionBlock(64, 32, 48), nn.Conv2d(80, 160, 3, 2), nn.BatchNorm2d(160), nn.ReLU())
        self.S2 = nn.Sequential(InceptionBlock(160, 112, 48), InceptionBlock(160, 96, 64), InceptionBlock(160, 80, 80), InceptionBlock(160, 48, 96), nn.Conv2d(144, 240, 3, 2), nn.BatchNorm2d(240), nn.ReLU())
        self.S3 = nn.Sequential(InceptionBlock(240, 176, 160), InceptionBlock(336, 176, 160))        
        self.global_max_pool = nn.MaxPool2d(1)
        self.linear = nn.Linear(8400, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.S1(x)
        x = self.S2(x)
        x = self.S3(x)
        x = self.global_max_pool(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x
        
        