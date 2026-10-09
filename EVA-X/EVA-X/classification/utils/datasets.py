# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
from PIL import ImageOps

from torchvision import datasets, transforms

from timm.data import create_transform
from utils.dataloader_med import ChestX_ray14, Covidx, CheXpert
from competition.dataset import CompetitionDataset


class ResizePadToSquare:
    """Resize the long edge to the target size and pad the short edge."""

    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, image):
        width, height = image.size
        scale = self.size / max(width, height)
        resized = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            PIL.Image.BICUBIC,
        )
        pad_width = self.size - resized.width
        pad_height = self.size - resized.height
        padding = (
            pad_width // 2,
            pad_height // 2,
            pad_width - pad_width // 2,
            pad_height - pad_height // 2,
        )
        return ImageOps.expand(resized, border=padding, fill=self.fill)


def build_square_resize(args, size):
    if getattr(args, 'resize_mode', 'crop') == 'pad':
        return ResizePadToSquare(size)
    return transforms.Resize(size, interpolation=PIL.Image.BICUBIC)


def build_square_crop(args):
    if getattr(args, 'resize_mode', 'crop') == 'pad':
        return []
    return [transforms.CenterCrop(args.input_size)]

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_dataset_chest_xray(split, args):
    is_train = (split == 'train')
    transform = build_transform(is_train, args)
    positive_transform = None
    if is_train and args.dataset == 'chexpert' and getattr(args, 'consolidation_pos_aug', False):
        positive_transform = build_consolidation_positive_transform(args)
    if args.dataset == 'chestxray':
        data_list = getattr(args, f'{split}_list')
        dataset = ChestX_ray14(args.data_path, data_list, augment=transform, num_class=14, data_pct=args.data_pct, seed=args.seed, mode='train' if is_train else 'test')
    elif args.dataset == 'covidx':
        print(args.dataset)
        dataset = Covidx(data_dir=args.data_path, phase=split, transform=transform, num_classes=args.nb_classes, data_pct=args.data_pct, seed=args.seed, rank=args.rank, train_list=args.train_list, test_list=args.test_list)
    elif args.dataset == 'competition':
        data_list = getattr(args, f'{split}_list')
        if not data_list:
            raise ValueError(f'--{split}_list is required for the competition dataset')
        dataset = CompetitionDataset(
            csv_path=data_list,
            image_root=args.data_path,
            transform=transform,
            training=is_train,
            labeled=True,
            pseudo_label_csv=getattr(args, 'pseudo_label_list', None) if is_train else None,
        )
    elif args.dataset == 'chexpert':
        if split == 'train':
            mode = 'train'
        else:
            mode = 'valid'
        data_list = getattr(args, f'{split}_list')
        dataset = CheXpert(csv_path=data_list, image_root_path=args.data_path,
                             use_upsampling=is_train and getattr(args, 'chexpert_upsampling', False),
                             use_frontal=True, mode=mode, class_index=args.class_index, transform=transform,
                             # Smooth uncertainty labels during training only; AUC evaluation needs binary targets.
                             use_rand_label=is_train and args.use_smooth_label, positive_transform=positive_transform,
                             full_train_epoch=getattr(args, 'chexpert_full_epoch', False))
    else:
        raise NotImplementedError
    print(dataset)
    if is_train:
        print('train transform: ', transform)
    else:
        print('test transform: ', transform)

    return dataset


def build_consolidation_positive_transform(args):
    if 'eva' in args.model:
        mean = (0.49185243, 0.49185243, 0.49185243)
        std = (0.28509309, 0.28509309, 0.28509309)
    else:
        mean = (0.5056, 0.5056, 0.5056)
        std = (0.252, 0.252, 0.252)

    crop_pct = 224 / 256 if args.input_size <= 224 else 1.0
    size = int(args.input_size / crop_pct)
    if getattr(args, 'resize_mode', 'crop') == 'pad':
        resize = build_square_resize(args, args.input_size)
        crop = []
    else:
        resize = transforms.Resize(size, interpolation=PIL.Image.BICUBIC)
        crop = [transforms.CenterCrop(args.input_size)]
    return transforms.Compose([
        resize,
        transforms.RandomAffine(
            degrees=5,
            translate=(0.03, 0.03),
            scale=(0.97, 1.03),
            fill=0,
        ),
        transforms.ColorJitter(brightness=0.10, contrast=0.10),
        *crop,
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def build_transform(is_train, args):

    if 'eva' in args.model:
            mean=(0.49185243, 0.49185243, 0.49185243)
            std=(0.28509309, 0.28509309, 0.28509309)
    else:
        mean = (0.5056, 0.5056, 0.5056)
        std = (0.252, 0.252, 0.252)


    aug_strategy = getattr(args, 'aug_strategy', 'default')
    if getattr(args, 'light_aug', False) and aug_strategy == 'default':
        aug_strategy = 'light'
    if is_train and args.dataset == 'chexpert':
        print(f'CheXpert augmentation strategy: {aug_strategy}')

    if is_train and args.dataset == 'chexpert' and aug_strategy in ('light', 'moderate'):
        if args.input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(args.input_size / crop_pct)
        if aug_strategy == 'moderate':
            degrees = 5
            translate = (0.03, 0.03)
            scale = (0.97, 1.03)
            jitter = 0.10
        else:
            degrees = 3
            translate = (0.02, 0.02)
            scale = (0.98, 1.02)
            jitter = 0.08
        resize = build_square_resize(args, args.input_size if getattr(args, 'resize_mode', 'crop') == 'pad' else size)
        return transforms.Compose([
            resize,
            transforms.RandomAffine(
                degrees=degrees,
                translate=translate,
                scale=scale,
                fill=0,
            ),
            transforms.ColorJitter(brightness=jitter, contrast=jitter),
            *build_square_crop(args),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    if args.dataset == 'competition':
        competition_transform = [ResizePadToSquare(args.input_size)]
        if is_train:
            competition_transform.extend([
                transforms.RandomAffine(
                    degrees=3,
                    translate=(0.02, 0.02),
                    scale=(0.98, 1.02),
                    fill=0,
                ),
                transforms.ColorJitter(brightness=0.08, contrast=0.08),
            ])
        competition_transform.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return transforms.Compose(competition_transform)

    # train transform
    if args.build_timm_transform and is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    if is_train:
        print('\033[91m' + 'The timm transform is NOT activated. Please make sure you meant it.' + '\033[0m')

    if args.dataset != 'chexpert':
        t = []
        if args.input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(args.input_size / crop_pct)
        t.append(
            transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))

    else:
        if 'tiny' in args.model:
            t = []
            crop_pct = 1.0
            size = int(args.input_size / crop_pct)
            t.append(
                build_square_resize(args, args.input_size if getattr(args, 'resize_mode', 'crop') == 'pad' else size),
            )
            t.extend(build_square_crop(args))
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(mean, std))
        else:
            # random resize and crop
            t = []
            if is_train:
                if args.input_size <= 224:
                    crop_pct = 224 / 256
                else:
                    crop_pct = 1.0
                size = int(args.input_size / crop_pct)
                t.append(
                    build_square_resize(args, args.input_size if getattr(args, 'resize_mode', 'crop') == 'pad' else size),
                )
                t.extend(build_square_crop(args))
            else:
                if getattr(args, 'resize_mode', 'crop') == 'pad':
                    t.append(build_square_resize(args, args.input_size))
                else:
                    t.append(transforms.Resize(int(256), interpolation=PIL.Image.BICUBIC))
                    t.append(transforms.CenterCrop(args.input_size))
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(mean, std))

    return transforms.Compose(t)

