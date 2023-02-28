from torch import optim as optim


def build_optimizer(config, model):
    """
    Builds optimizer for the model from the config.
    """
    optimizer_type = config.TRAIN.OPTIMIZER.NAME.lower()
    lr = config.TRAIN.LR
    momentum = config.TRAIN.OPTIMIZER.MOMENTUM
    eps = config.TRAIN.OPTIMIZER.EPS
    betas = config.TRAIN.OPTIMIZER.BETAS
    parameters = model.parameters()

    if optimizer_type == "sgd":
        optimizer = optim.SGD(
            parameters, lr=lr, momentum=momentum, nesterov=True
        )
    elif optimizer_type == "adam":
        optimizer = optim.Adam(parameters, lr=lr, eps=eps, betas=betas)
    elif optimizer_type == "adamw":
        optimizer = optim.AdamW(parameters, lr=lr, eps=eps, betas=betas)
    else:
        raise NotImplementedError(f"Unknown optimizer: {optimizer_type}")

    return optimizer
