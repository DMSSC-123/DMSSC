import numpy as np
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

features_save_dir = "features"
features_suffix = '_features_list.pt'


class CustomDataset(Dataset):
    def __init__(self, x, y, transform=None):

        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]
        label = self.y[idx]

        if self.transform is None:
            return img, label

        img = self.transform(img)
        return img, label


def load_raw_image(name, img_size):
    if name == 'fashion_mnist':
        return fmnist_dataset(img_size)
    elif name == 'cifar100':
        return cifar100_dataset(img_size)


def cifar100_dataset(img_size):
    norm_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    dt = datasets.CIFAR100(
        root="../datasets",
        download=True,
        train=True,
        transform=norm_transform
    )

    x = dt.data[:3000]/255.
    x = x.astype(np.float32)
    y = dt.targets[:3000]

    dt = CustomDataset(x, y, transform=norm_transform)

    return dt


def fmnist_dataset(img_size):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dt = datasets.FashionMNIST(
        root="../datasets",
        download=True,
        train=True,
        transform=transform
    )

    x = dt.data[:1000].view(1000, 1, 28, 28) / 255.
    x = x.repeat(1, 3, 1, 1)
    y = dt.targets[:1000]
    dt = CustomDataset(x, y, transform)

    return dt


