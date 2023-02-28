from torch.utils.data import DataLoader

from data.datasets import CIFAR10Dataset, MediumImagenetDataset


def build_loader(config):
    if config.DATA.DATASET == "cifar10":
        dataset_train = CIFAR10Dataset(train=True)
        dataset_val = CIFAR10Dataset(train=False)
    elif config.DATA.DATASET == "medium_imagenet":
        dataset_train = MediumImagenetDataset(config, train=True)
        dataset_val = MediumImagenetDataset(config, train=False)
    else:
        raise NotImplementedError

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=True,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
    )

    data_loader_val = DataLoader(
        dataset_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=False,
    )

    return dataset_train, dataset_val, data_loader_train, data_loader_val
