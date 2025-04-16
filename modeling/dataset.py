import logging
import glob
import os
import importlib

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from preprocess import Pipeline


CLASSES = ['Woodland', 'Grassland', 'Settlement', 'FlowingWater', 'StandingWater']

def get_dataloader(cfgs, mode='train', distributed=False):
    dataset = HamelnDataset(cfgs, mode)
    shuffle = cfgs.get('shuffle', True) if mode=='train' else False
    if distributed:
        shuffle = False
        sampler = DistributedSampler(dataset)
    else:
        sampler = None
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=cfgs[f'batch_size'],
                                             sampler=sampler,
                                             num_workers=cfgs['n_workers'],
                                             shuffle=shuffle,
                                             collate_fn=dataset.collate_batch,
                                             pin_memory=True,
                                             drop_last=True)
    return dataloader


class HamelnDataset(Dataset):

    def __init__(self,
                 cfgs,
                 mode='train',
    ):
        self.cfgs = cfgs
        self.root_dir = cfgs['data_path']
        self.mode = mode
        self.multi_cls = cfgs.get('multi_cls', False)
        self.label_error = cfgs.get('label_error', 0.0)
        if not self.multi_cls:
            assert 'cls_label' in cfgs
            self.cls_label = cfgs['cls_label']
        label_pattern = cfgs.get('label_pattern', None)

        if mode == 'inference':
            self.samples = []
            self.img_dir = self.root_dir
            for split in ['train', 'test']:
                self.samples.extend([os.path.join(split, x) for x in os.listdir(os.path.join(self.root_dir, split))])
        else:
            self.img_dir = os.path.join(self.root_dir, self.mode)
            self.samples = os.listdir(self.img_dir)
        if label_pattern is not None:
            samples = []
            if len(label_pattern) == 5:
                assert len(label_pattern) == len(self.samples[0].split('_')[-1].split('.')[0])
            else:
                assert len(label_pattern) == len(self.samples[0].split('_')[-2])
            for sample in self.samples:
                label = sample[:-4].split('_')[4]
                flag = True
                for pi in [i for i, p in enumerate(label_pattern) if p != '*']:
                    if label[pi] != label_pattern[pi]:
                        flag = False
                        break
                if flag:
                    samples.append(sample)
            self.samples = samples
        self.invert_label = np.random.random(len(self.samples)) < self.label_error
        pl_name = 'train' if mode == 'train' else 'test'
        self.pipeline = Pipeline(cfgs[f'{pl_name}_pipeline'])


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        filename = self.samples[item]
        patch, year, x , y, l = filename.split('.')[0].split('_')[:5]
        img = cv2.imread(os.path.join(self.img_dir, filename), cv2.IMREAD_UNCHANGED)
        if self.multi_cls:
            label = torch.Tensor([int(x) for x in l])
        else:
            if self.cls_label < 3 or len(l) == 4:
                label = int(l[self.cls_label])
            else:
                label = int(int(l[3]) or int(l[4]))
            if self.invert_label[item]:
                label = 1 - label

        data = {
            'patch': patch,
            'year': year,
            'x': x,
            'y': y,
            'img': img,
            'label': label,
        }
        data = self.pipeline(data)
        return data

    @staticmethod
    def collate_batch(data_list):
        data_dict = {}
        for k in data_list[0].keys():
            data = [x[k] for x in data_list]
            if isinstance(data[0], torch.Tensor):
                data_dict[k] = torch.stack(data, dim=0)
            elif isinstance(data[0], list) and isinstance(data[0][0], torch.Tensor):
                data = [torch.stack(x, dim=0) for x in data]
                data_dict[k] = torch.cat(data, dim=0)
            elif k == 'label':
                data_dict[k] = torch.tensor(data, dtype=torch.long)
            else:
                data_dict[k] = data

        return data_dict