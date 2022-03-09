import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import timm

# Resnet50 teacher model
class Resnet50T(nn.Module):
    def __init__(self):
        super(Resnet50T, self).__init__()
        self.m = models.resnet50(pretrained=True)
        for param in self.m.parameters():
            param.requires_grad = False    
    
    def forward(self, x):
        x = self.m(x)
        return x


# Resnet50 student model
class Resnet50S(nn.Module):
    def __init__(self):
        super(Resnet50S, self).__init__()
        self.m = models.resnet50(pretrained=True)
        for param in self.m.parameters():
            param.requires_grad = False
        self.m.fc = nn.Linear(2048, 256)
        self.fc1 = nn.Linear(256, 1000)
        
    def forward(self, x):
        x = F.relu(self.m(x))
        x = self.fc1(x)
        return x


# Resnet101 teacher model
class Resnet101T(nn.Module):
    def __init__(self):
        super(Resnet101T, self).__init__()
        self.m = models.resnet101(pretrained=True)
        for param in self.m.parameters():
            param.requires_grad = False  
    
    def forward(self, x):
        x = self.m(x)
        return x


# Resnet101 student model
class Resnet101S(nn.Module):
    def __init__(self):
        super(Resnet101S, self).__init__()
        self.m = models.resnet101(pretrained=True)
        for param in self.m.parameters():
            param.requires_grad = False
        self.m.fc = nn.Linear(2048, 256)
        self.fc1 = nn.Linear(256, 1000)       
    
    def forward(self, x):
        x = F.relu(self.m(x))
        x = self.fc1(x)
        return x


# Vision transformer teacher model
class ViTT(nn.Module):
    def __init__(self):
        super(ViTT, self).__init__()
        self.m = timm.create_model('vit_base_patch16_224', pretrained=True)
        for param in self.m.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.m(x)
        return x


# Vision transformer student model
class ViTS(nn.Module):
    def __init__(self):
        super(ViTS, self).__init__()
        self.m = timm.create_model('vit_base_patch16_224', pretrained=True)
        for param in self.m.parameters():
            param.requires_grad = False
        self.m.head = nn.Linear(768, 256)
        self.fc1 = nn.Linear(256, 1000)
    
    def forward(self, x):
        x = F.relu(self.m(x))
        x = self.fc1(x)
        return x

# Maybe unable to download model
# DeiT teacher model
class DeiTT(nn.Module):
    def __init__(self):
        super(DeiTT, self).__init__()
        self.m = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
        for param in self.m.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.m(x)
        return x


# DeiT student model
class DeiTS(nn.Module):
    def __init__(self):
        super(DeiTS, self).__init__()
        self.m = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
        for param in self.m.parameters():
            param.requires_grad = False
        self.m.head_dist = nn.Linear(768, 256)
        self.fc1 = nn.Linear(256, 1000)
    
    def forward(self, x):
        x = F.relu(self.m(x))
        x = self.fc1(x)
        return x


# loss function, KL divergence and cross entropy
def loss_fn_kd(outputs, labels, teacher_outputs):
    alpha = 0.5
    T = 1
    
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss
