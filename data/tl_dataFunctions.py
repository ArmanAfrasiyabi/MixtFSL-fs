from data.utils import ImageJitter
from torchvision.transforms import Compose
import torchvision.transforms as Transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import Dataset
import os
from data.utils import TransformLoader


def ar_transform(args, aug):
    norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    jitter_param = dict(Brightness=0.4, Contrast=0.4, Color=0.4, )
    if aug:
        transforms = Compose([Transforms.RandomResizedCrop(args.img_size),
                              ImageJitter(jitter_param),
                              Transforms.RandomHorizontalFlip(),
                              Transforms.ToTensor(),
                              Transforms.Normalize(norm_mean, norm_std)])
    else:
        transforms = Compose([Transforms.RandomResizedCrop(args.img_size),
                              ImageJitter(jitter_param),
                              Transforms.ToTensor(),
                              Transforms.Normalize(norm_mean, norm_std)])
        TransformLoader_class = TransformLoader(args.img_size)
        transforms = TransformLoader_class.get_composed_transform(aug=False)
    return transforms


def ar_base_DataLaoder(args, aug, section='base', shuffle=True):
    data_path = args.benchmarks_dir + args.dataset + '/' + section + '/'
    # transforms = ar_transform(args, aug)
    trans_loader = TransformLoader(args.img_size)
    transforms = trans_loader.get_composed_transform(aug)
    dataset = ImageFolder(root=data_path, transform=transforms)
    return DataLoader(dataset=dataset,
                      batch_size=args.batch_size,
                      num_workers=args.num_workers,
                      shuffle=shuffle,
                      drop_last=False)


def ar_base_DataLaoder_viz(args, aug, section='base', shuffle=True):
    data_path = args.benchmarks_dir + args.dataset + '/' + section + '/'
    trans_loader = TransformLoader(args.img_size)
    transform = trans_loader.get_composed_transform(aug)
    dataset = ImageFolder(root=data_path, transform=transform)
    return DataLoader(dataset=dataset,
                      batch_size=args.batch_size,
                      num_workers=args.num_workers,
                      shuffle=shuffle,
                      drop_last=False)


def ar_base_underFolder_DataLaoder(args, aug, section='base_undreFolder'):
    data_path = args.benchmarks_dir + args.dataset + '/' + section + '/'
    # transforms = ar_transform(args, aug)
    trans_loader = TransformLoader(args.img_size)
    transforms = trans_loader.get_composed_transform(aug)

    loaderList = []
    for i in range(args.n_base_class):
        dataset = ImageFolder(root=data_path, transform=transforms)
        loaderList.append(DataLoader(dataset=dataset,
                                     batch_size=args.n_shot,
                                     num_workers=args.num_workers,
                                     shuffle=True,
                                     drop_last=False))

    return loaderList
