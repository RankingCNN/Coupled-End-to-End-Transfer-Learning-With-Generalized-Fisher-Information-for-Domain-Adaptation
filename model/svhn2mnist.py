import torch.nn as nn
import torch.nn.functional as F
from .grad_reverse import grad_reverse
import torch


def clip_relu(x):
	return torch.clamp(x, 0,1)

class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=3, padding=1)
        
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=3, padding=1)
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = x.view(x.size(0), 8192)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        return x


class Predictor(nn.Module):
    def __init__(self, prob=0.5):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2_fc = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 10)
        self.bn_fc3 = nn.BatchNorm1d(10)
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = self.fc3(x)
        return x
   
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(3072, 8192)
        self.bn1_fc = nn.BatchNorm1d(8192)
        self.decov3 = nn.ConvTranspose2d(128,64,4,stride=2, padding=1)
        self.decov3_bn = nn.BatchNorm2d(64)
        self.decov2 = nn.ConvTranspose2d(64, 3, 2, stride = 2)
        
    def set_lambda(self, lambd):
        self.lambd = lambd    
        
    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = x.view(x.size(0), 128, 8,8)
        x = F.relu(self.decov3_bn(self.decov3(x)))
        x = self.decov2(x)
        x = clip_relu(x)
        return x


'''
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(3072,3072)
        self.fc2 = nn.Linear(3072, 8192)
        self.conv1 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2)
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv3 = nn.Conv2d(64,64,kernel_size=5, stride=1, padding=2)
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(x.size(0), 128, 8,8)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.up1(x)
        x = F.relu(self.conv3(x))
        x = self.up2(x)
        x = self.conv4(x)
        x = clip_relu(x)#rev for 0,1 inputs
        return x
'''        
        
        
        
        
    
        
        
        
        
    

