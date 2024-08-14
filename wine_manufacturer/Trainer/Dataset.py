import torchvision.transforms as transforms
import torchvision
import torch
from torch.utils.data import DataLoader


class MNISTDataset:
    def __init__(self, train=True, n_size_1d = 14, batch_size=64):

        transform = transforms.Compose([
            transforms.Resize(n_size_1d),
            transforms.ToTensor(),
        ])

        self.data = torchvision.datasets.MNIST(root="./adversarial_example/Dataset/MNIST", train=train, download=True, transform=transform)

        self.loader = DataLoader(self.data, batch_size=batch_size, shuffle=False, num_workers=2)


    def __len__(self):
        return len(self.data)

    def get_raw_data(self):
        return self.data

    def get_data(self):

        return self.loader

class CIFAR10Dataset:
    def __init__(self, train=True, batch_size=64):

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225]),
        ])

        self.data = torchvision.datasets.CIFAR10(root="./Dataset/CIFAR10", train=train, download=True, transform=transform)

        self.loader = DataLoader(self.data, batch_size=batch_size, shuffle=False, num_workers=2)


    def __len__(self):
        return len(self.data)

    def get_data(self):

        return self.loader
