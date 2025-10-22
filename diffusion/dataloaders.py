import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_cifar10_dataloaders(train_config, with_label: bool = False, **kwargs):
    data_transform = transforms.Compose(
        [
            transforms.Resize(
                (train_config.img_size, train_config.img_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root=train_config.data_dir, download=True, train=True, transform=data_transform
    )

    if train_config.debug:
        train_dataset = torch.utils.data.Subset(train_dataset, range(100))

    def collate_fn(batch):
        imgs, labels = zip(*batch)
        if with_label:
            imgs = torch.stack(imgs, dim=0)
            labels = torch.tensor(labels)
            return imgs, labels
        else:
            imgs = torch.stack(imgs, dim=0)
            return imgs

    loader_config = dict(
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if train_config.device.type == "cuda" else False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    loader_config.update(kwargs)

    dataloader = DataLoader(train_dataset, **loader_config)

    print(f"Debug: {train_config.debug} | Total {len(train_dataset)} images | {len(dataloader)} steps")
    return dataloader
