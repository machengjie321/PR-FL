from os.path import join
import torch
import torchvision
import torchvision.transforms as transforms
from bases.vision.data_loader import DataLoader
from bases.vision.transforms import Flatten, OneHot, DataToTensor
from configs import femnist, celeba, cifar10, imagenet100,mnist
from bases.vision.datasets import FEMNIST, CelebA, TinyImageNet

__all__ = ["get_data", "get_data_loader"]


def get_config_by_name(name: str):
    if name.lower() == "femnist":
        return femnist
    elif name.lower() == "celeba":
        return celeba
    elif name.lower() == "cifar10":
        return cifar10
    elif name.lower() in ["tinyimagenet", "tiny-imagenet-200", "imagenet"]:
        return TinyImageNet
    elif name.lower() == "mnist":
        return mnist
    else:
        raise ValueError("{} is not supported.".format(name))


def get_data(name: str, data_type, transform=None, target_transform=None, user_list=None):
    dataset = get_config_by_name(name)

    if dataset == femnist:
        assert data_type in ["train", "test"]
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])
        if target_transform is None:
            target_transform = transforms.Compose(
                [DataToTensor(dtype=torch.long), OneHot(dataset.NUM_CLASSES, to_float=True)])

        return FEMNIST(root=join("datasets", "FEMNIST"), train=data_type == "train", download=True, transform=transform,
                       target_transform=target_transform, user_list=user_list)
    if dataset == mnist:
        assert data_type in ["train", "test"]
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomRotation((-10, 10)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))])

        train_folder = join("datasets", "MNIST", "train")
        test_folder = join("datasets", "MNIST", "test")
        import os
        dl = True
        if os.path.isdir(train_folder) and os.path.isdir(test_folder):
            dl = False

        from torchvision import datasets
        if data_type == 'train':

            data = datasets.MNIST(root=train_folder,
                                    train=True,
                                    transform=transform,
                                    download=dl)
        else:
            data = datasets.MNIST(root=test_folder,
                                   train=False,
                                   transform=transform,
                                   download=dl)
        return data

    elif dataset == celeba:
        assert data_type in ["train", "test"]


        if transform is None:
            transform = transforms.Compose([transforms.Resize((84, 84)),
                                            transforms.ToTensor()])
        if target_transform is None:
            target_transform = transforms.Compose([DataToTensor(dtype=torch.long)])

        return CelebA(root=join("datasets", "CelebA"), train=data_type == "train", download=True, transform=transform,
                      target_transform=target_transform, user_list=user_list)

    elif dataset == cifar10:
        assert data_type in ["train", "test"]
        if transform is None:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
            if data_type == "train":
                transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.RandomCrop(32, 4),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean, std)])

            else:
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean, std)])
        if target_transform is None:
            target_transform = transforms.Compose([DataToTensor(dtype=torch.long),
                                                   OneHot(dataset.NUM_CLASSES, to_float=True)])

        return torchvision.datasets.CIFAR10(root=join("datasets", "CIFAR10"), train=data_type == "train", download=True,
                                            transform=transform, target_transform=target_transform)

    elif dataset == TinyImageNet:
        assert data_type in ["train", "test", "val"]
        if transform is None:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            if data_type == "train":
                transform = transforms.Compose([transforms.RandomResizedCrop(imagenet100.IMAGE_SIZE),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean, std)])
            else:
                transform = transforms.Compose([transforms.Resize(64),
                                                transforms.CenterCrop(imagenet100.IMAGE_SIZE),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean, std)])

        return TinyImageNet(root=join("datasets", "TinyImageNet"), data_type = "train", transform=transform,
                           target_transform=None,download=True)

    else:
        raise ValueError("{} dataset is not supported.".format(name))


def get_data_loader(name: str, data_type, batch_size=None, shuffle: bool = False, sampler=None, transform=None,
                    target_transform=None, subset_indices=None, num_workers=8, pin_memory=True, user_list=None):
    assert data_type in ["train", "val", "test"]
    if data_type == "train":
        assert batch_size is not None, "Batch size for training data is required"
    if shuffle is True:
        assert sampler is None, "Cannot shuffle when using sampler"

    data = get_data(name, data_type=data_type, transform=transform, target_transform=target_transform,
                    user_list=user_list)

    if subset_indices is not None:
        data = torch.utils.data.Subset(data, subset_indices)
    if data_type != "train" and batch_size is None:
        batch_size = len(data)

    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers,
                      pin_memory=pin_memory)
