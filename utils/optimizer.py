import torch.optim as optim
from lion_pytorch import Lion

def get_optimizer(optimizer_name, model_params, lr, weight_decay=1e-6):
    """
    Returns the optimizer based on the optimizer name.
    
    :param optimizer_name: Name of the optimizer ('adam', 'adamw', 'sgd', 'lion')
    :param model_params: Parameters of the model
    :param lr: Learning rate
    :param weight_decay: Weight decay (default is 1e-6)
    :return: Optimizer object
    """
    if optimizer_name.lower() == 'lion':
        return Lion(params=model_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(params=model_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(params=model_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        return optim.Adam(params=model_params, lr=lr, weight_decay=weight_decay)