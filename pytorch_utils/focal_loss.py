import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    # Binary Focal Loss TODO: fix this
    def __init__(self, weight=None, 
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def __call__(self, input_tensor, target_tensor):
        print(f'input_tensor: {input_tensor.shape}, target_tensor: {target_tensor.shape}')
        input_tensor = torch.flatten(input_tensor, start_dim=2, end_dim=3)
        target_tensor = torch.flatten(target_tensor, start_dim=2, end_dim=3)
        target_tensor = target_tensor[:,1,:]
        print(f'input_tensor: {input_tensor.shape}, target_tensor: {target_tensor.shape}')
        prob = F.log_softmax(input_tensor, dim=1)
        prob = torch.exp(prob)
        print(f'prob: {prob.shape}, prob.dtype: {prob.dtype}, target_tensor: {target_tensor.shape}, target_tensor.dtype: {target_tensor.dtype}')
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * prob, 
            target_tensor, 
            weight=self.weight,
            reduction = self.reduction
        )