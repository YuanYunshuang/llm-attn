from collections import OrderedDict

import numpy as np
import torch
from torchvision.transforms import v2
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class RemapLabels:
    def __init__(self, src_label, tgt_label):
        self.src_label = src_label
        self.tgt_label = tgt_label

    def __call__(self, data_dict):
        gt = np.array(data_dict['gt_img'])
        new_gt = np.zeros_like(gt)
        for i, l in enumerate(self.tgt_label):
            idx = self.src_label.index(l)
            new_gt[gt==idx+1] = i + 1
        data_dict['gt_img'] = Image.fromarray(new_gt)
        return data_dict


class FlipRotImage:
    def __init__(self, degrees=180):
        self.transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=(-degrees, degrees)),
        ])

    def __call__(self, data_dict):
        img_keys = [k for k in data_dict.keys() if 'img' in k]
        imgs = [data_dict[k] for k in img_keys]
        imgs = self.transforms(imgs)
        for i, k in enumerate(img_keys):
            data_dict[k] = imgs[i]
        return data_dict


class FormatImage:
    def __init__(self,):
        self.transforms_input = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

        self.transforms_gt = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=False),
        ])

    def __call__(self, data_dict):
        for k, v in data_dict.items():
            if 'img' in k:
                if 'input' in k:
                    data_dict[k] = self.transforms_input(v)
                elif 'gt' in k:
                    data_dict[k] = self.transforms_gt(v)
                    if data_dict[k].ndim == 3:
                        data_dict[k] /= 255
                else:
                    raise ValueError(f'Unexpected key {k}')
        # img_keys = [k for k in data_dict.keys() if 'img' in k]
        # imgs = [data_dict[k] for k in img_keys]
        # imgs = self.transforms(imgs)
        # for i, k in enumerate(img_keys):
        #     data_dict[k] = imgs[i]
        return data_dict

class LGLNFlipRotImage:
    def __init__(self, degrees=180):
        self.transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=(-degrees, degrees)),
        ])

    def __call__(self, data_dict):
        img_keys = [k for k in data_dict.keys() if 'img' in k]
        imgs = []
        cnt = []
        for k in img_keys:
            imgs.extend(data_dict[k])
            cnt.append(len(data_dict[k]))
        imgs = self.transforms(imgs)
        ptr = 0
        for i, k in enumerate(img_keys):
            data_dict[k] = imgs[ptr:ptr+cnt[i]]
            ptr += cnt[i]
        return data_dict


class LGLNFormatImage:
    def __init__(self,):
        self.transforms_input = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

        self.transforms_gt = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.int8, scale=False),
        ])

    def __call__(self, data_dict):
        for k, v in data_dict.items():
            if 'img' in k:
                if 'input' in k:
                    data_dict[k] = self.transforms_input(v)
                elif 'gt' in k:
                    data_dict[k] = self.transforms_gt(v)
                else:
                    raise ValueError(f'Unexpected key {k}')
        return data_dict


class LGLNViTTransform:
    def __init__(self, mode, degrees=10, color_jitter=0.4):
        if mode == 'train':
            self.transforms = v2.Compose([
                v2.RandomRotation(degrees=(-degrees, degrees)),
                v2.ColorJitter(color_jitter, color_jitter, color_jitter),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transforms = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    def __call__(self, data_dict):
        data_dict['img'] = self.transforms(data_dict['img'])
        return data_dict


class ImgGridMasking:
    def __init__(self, grid_size=64, img_size=384):
        self.grid_size = grid_size
        self.img_size = img_size
        num_patches = img_size // grid_size
        x = np.arange(num_patches)
        xy = np.stack(np.meshgrid(x, x), axis=-1)
        mask = np.abs(xy[..., 0] - xy[..., 1]) % 2 == 1
        self.mask = np.kron(mask, np.ones((grid_size, grid_size)))[..., np.newaxis]

    def __call__(self, data_dict):
        imgs = data_dict['img']
        if np.random.random() < 0.5:
            imgs = imgs * self.mask
        else:
            imgs = imgs * (1 - self.mask)
        data_dict['img'] = imgs
        return data_dict


class ImgRandomMasking:
    def __init__(self, grid_size=64, img_size=384, pos_ratio=0.8):
        self.grid_size = grid_size
        self.img_size = img_size
        num_patches = img_size // grid_size
        mask = np.random.rand(num_patches, num_patches) < pos_ratio
        self.mask = np.kron(mask, np.ones((grid_size, grid_size)))[..., np.newaxis]

    def __call__(self, data_dict):
        imgs = data_dict['img']
        if np.random.random() < 0.5:
            imgs = imgs * self.mask
        else:
            imgs = imgs * (1 - self.mask)
        data_dict['img'] = imgs
        return data_dict


class Pipeline(object):
    """Composes several processing modules together.
        Take care that these functions modify the input data directly.
    """

    def __init__(self, cfgs):
        self.processes = []
        if isinstance(cfgs, list):
            for cfg in cfgs:
                for k, v in cfg.items():
                    self.build_process(k, v)
        elif isinstance(cfgs, OrderedDict):
            for k, v in cfgs.items():
                self.build_process(k, v)
        else:
            raise NotImplementedError

    def build_process(self, k, v):
        cls = globals().get(k, None)
        assert cls is not None, f"Pipeline process node {k} not found."
        self.processes.append(cls(**v))

    def __call__(self, data_dict):
        for p in self.processes:
            p(data_dict)
        return data_dict