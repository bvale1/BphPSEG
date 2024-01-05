import torch
import torch.nn as nn
import torch.nn.functional as F

# To compare with the built-in cross entropy loss
# to check I understood how to implement custom loss in pytorch
class CrossEntropyLoss(nn.Module):
    # Binary Cross Entropy Loss
    def __init__(self, weight=torch.tensor([1.0, 1.0])):
        # len(weight) must equal the number of classes
        nn.Module.__init__(self)
        # [num_classes] -> [1, num_classes, 1, 1]
        weight /= torch.linalg.norm(weight) # normalise
        self.weight = weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
    def forward(self, y_hat, y):
        # expected inputs to be of shape (batch_size, num_classes, height, width)
        y_hat = F.softmax(y_hat, dim=1)
        loss = - y * torch.log(y_hat)
        loss *= self.weight
        return torch.mean(loss)


class FocalLoss(nn.Module):
    # Binary Focal Loss TODO: fix this
    def __init__(self, weight=torch.tensor([1.0, 1.0]), gamma=2.0):
        nn.Module.__init__(self)
        # [num_classes] -> [1, num_classes, 1, 1]
        weight /= torch.linalg.norm(weight) # normalise
        self.weight = weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.gamma = gamma
        
    def forward(self, y_hat, y):
        # expected inputs to be of shape (batch_size, num_classes, height, width)
        y_hat = F.softmax(y_hat, dim=1)
        loss = - y * (((1 - y_hat)**self.gamma) * torch.log(y_hat))
        loss *= self.weight
        return torch.mean(loss)
    
