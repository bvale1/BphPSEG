import torch


def remove_dropout(model):
    '''
    Remove all Dropout layers from a model no matter how deeply
    nested each layer is.
    BatchNorm and Dropout layers cause problems when performing regression tasks.
    '''
    for name, m in model.named_children():
        if isinstance(m, torch.nn.Dropout):
            setattr(model, name, torch.nn.Identity())
        elif hasattr(m, 'children'):
            remove_dropout(m)
            

def remove_batchnorm(model):
    '''
    Remove all BatchNorm layers from a model no matter how deeply
    nested each layer is.
    BatchNorm and Dropout layers cause problems when performing regression tasks.
    '''
    for name, m in model.named_children():
        if isinstance(m, torch.nn.BatchNorm2d):
            setattr(model, name, torch.nn.Identity())
        elif hasattr(m, 'children'):
            remove_batchnorm(m)
            
            
def reset_weights(m):
    '''
    Reset all the parameters of a model no matter how deeply nested each
    layer is.
    '''
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()
    else:
        for layer in m.children():
            reset_weights(layer)