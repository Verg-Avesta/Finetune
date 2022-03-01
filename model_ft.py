import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F

class Resnet50T(nn.Module):
    def __init__(self):
        super(Resnet50T, self).__init__()
        self.m = models.resnet50(pretrained=True)
        for param in self.m.parameters():
            param.requires_grad = False
        
    
    def forward(self, x):
        x = self.m(x)
        return x

class Resnet50S(nn.Module):
    def __init__(self):
        super(Resnet50S, self).__init__()
        self.m = models.resnet50(pretrained=True)
        for param in self.m.parameters():
            param.requires_grad = False
        self.m.fc = nn.Linear(2048, 256)
        self.fc1 = nn.Linear(256, 1000)
        
    
    def forward(self, x):
        x = self.m(x)
        x = F.relu(self.fc1(x))
        return x

def loss_fn_kd(outputs, labels, teacher_outputs):
    alpha = 0.5
    T = 1
    
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss
