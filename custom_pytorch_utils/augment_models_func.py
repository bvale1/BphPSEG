import torch


def remove_dropout(module):
    '''
    Remove all Dropout layers from a model no matter how deeply
    nested each layer is.
    BatchNorm and Dropout layers cause problems when performing regression tasks.
    '''
    for name, m in module.named_children():
        if isinstance(m, torch.nn.Dropout):
            setattr(module, name, torch.nn.Identity())
        elif hasattr(m, 'children'):
            remove_dropout(m)
            

def remove_batchnorm(module):
    '''
    Remove all BatchNorm and LayerNorm layers from a model no matter how deeply
    nested each layer is.
    BatchNorm and Dropout layers cause problems when performing regression tasks.
    '''
    for name, m in module.named_children():
        if isinstance(m, torch.nn.BatchNorm2d):# or isinstance(m, torch.nn.LayerNorm):
            setattr(module, name, torch.nn.Identity())
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