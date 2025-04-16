import math
import os
import time
from functools import partial

import cv2
import numpy as np
import torch
import tqdm
from torchvision.utils import make_grid
from src.utils.logger import plt_to_image
from src.utils.common import unpackbits
from src.utils.train_utils import load_tensors_to_gpu


class Hooks:
    def __init__(self, cfg):
        self.hooks = []
        if cfg is None:
            return
        for hook_cfg in cfg:
            self.hooks.append(
                globals()[hook_cfg['type']](**hook_cfg)
            )

    def __call__(self, runner, hook_stage, **kwargs):
        for hook in self.hooks:
            getattr(hook, hook_stage)(runner, **kwargs)

    def set_logger(self, logger):
        for hook in self.hooks:
            hook.set_logger(logger)


class BaseHook:
    def __init__(self, **kwargs):
        pass

    def pre_iter(self, runner, **kwargs):
        pass

    def post_iter(self, runner, **kwargs):
        pass

    def pre_epoch(self, runner, **kwargs):
        pass

    def post_epoch(self, runner, **kwargs):
        pass

    def pre_train(self, runner, **kwargs):
        pass

    def post_train(self, runner, **kwargs):
        pass

    def set_logger(self, logger):
        self.logger = logger


class MemoryUsageHook(BaseHook):
    def __init__(self, device='cuda:0', **kwargs):
        super().__init__(**kwargs)
        self.device = device

    def post_iter(self, runner, **kwargs):
        memory = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
        torch.cuda.empty_cache()
        runner.logger.update(memory=memory)


class TrainTimerHook(BaseHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.elapsed_time = 0
        self.start_time = None
        self.mean_time_per_itr = None
        self.observations = 0

    def pre_epoch(self, runner, **kwargs):
        if self.start_time is None:
            self.start_time = time.time()
            self.last_time = time.time()

    def post_iter(self, runner, **kwargs):
        cur_time = time.time()
        self.elapsed_time = (cur_time - self.start_time) / 3600
        # total_run_iter = (runner.total_iter * (runner.epoch - runner.start_epoch)) + runner.iter
        # time_per_iter = self.elapsed_time / total_run_iter
        time_per_iter = (cur_time - self.last_time) / 3600
        m = self.observations
        if self.mean_time_per_itr is None:
            self.mean_time_per_itr = time_per_iter
        else:
            self.mean_time_per_itr = m / (m + 1) * self.mean_time_per_itr + 1 / (m + 1) * time_per_iter
        iter_remain = runner.total_iter * (runner.total_epochs - runner.epoch + 1) - runner.iter
        time_remain = self.mean_time_per_itr * iter_remain
        runner.logger.update(t_remain=time_remain, t_used=self.elapsed_time)
        self.last_time = cur_time
        self.observations += 1


class CheckPointsHook(BaseHook):
    def __init__(self, max_ckpt=3, epoch_every=None, iter_every=None, **kwargs):
        super().__init__(**kwargs)
        self.max_ckpt = max_ckpt
        self.epoch_every = epoch_every
        self.iter_every = iter_every

    def post_epoch(self, runner, **kwargs):
        if runner.gpu_id != 0:
            return
        self.save(runner, f'epoch{runner.epoch}.pth')
        if runner.epoch > self.max_ckpt:
            if (self.epoch_every is None or not
            (runner.epoch - self.max_ckpt) % self.epoch_every == 0):
                filename = os.path.join(
                    runner.logger.logdir,
                    f'epoch{runner.epoch - self.max_ckpt}.pth')
                if os.path.exists(filename):
                    os.remove(filename)

    def post_iter(self, runner, **kwargs):
        if runner.gpu_id != 0:
            return
        if self.iter_every is not None and runner.iter % self.iter_every == 0:
            self.save(runner, f'latest.pth')

    @staticmethod
    def save(runner, name):
        save_path = os.path.join(runner.logger.logdir, name)
        print(f'Saving checkpoint to {save_path}.')
        torch.save({
            'epoch': runner.epoch,
            'model': runner.model.state_dict(),
            # 'optimizer': runner.optimizer.state_dict(),
            'lr_scheduler': runner.lr_scheduler.state_dict(),
        }, save_path)


class ImgWriterHook(BaseHook):
    def __init__(self, max_n_img=32, log_every=50, **kwargs):
        super().__init__(**kwargs)
        self.max_n_img = max_n_img
        self.log_every = log_every

    def post_iter(self, runner, **kwargs):
        out_dict = runner.model.out_dict
        steps = (runner.epoch - 1) * runner.total_iter + runner.iter
        if steps % self.log_every == 0:
            for k, v in out_dict.items():
                if 'img' in k:
                    img = out_dict[k]
                    if img.ndim == 3:
                        img = img.unsqueeze(1)
                    n_img = min(len(img), self.max_n_img)
                    img = make_grid(img[:n_img])
                    runner.writer.add_image(f"train/{k.replace('_img', '')}", img, steps)


class SegPlotWriterHook(BaseHook):
    def __init__(self, max_n_img=32, log_every=50, **kwargs):
        super().__init__(**kwargs)
        self.max_n_img = max_n_img
        self.log_every = log_every

    def post_iter(self, runner, **kwargs):
        out_dict = runner.model.out_dict
        steps = (runner.epoch - 1) * runner.total_iter + runner.iter
        if steps % self.log_every == 0:
            import matplotlib.pyplot as plt  # this might use main thread if imported on the top
            rows = min(4, out_dict['input_img'].shape[0])
            cols = max(out_dict['conf_img'].shape[-1] + 2, 4)
            fig = plt.figure(figsize=(3*cols, 3*rows))
            axs = fig.subplots(rows, cols)
            for r in range(rows):
                axs[r, 0].imshow(out_dict['input_img'][r].permute(1, 2, 0).detach().cpu().numpy()[..., ::-1])
                axs[r, 0].set_title('input')
                axs[r, 1].imshow((out_dict['gt_img'][r] / 4).float().permute(1, 2, 0).detach().cpu().numpy(),
                                 cmap='jet', vmin=0, vmax=1)
                axs[r, 1].set_title('gt')
                axs[r, 2].imshow(1 - out_dict['unc_img'][r].detach().cpu().numpy(),
                                 cmap='hot', vmin=0, vmax=1)
                axs[r, 2].set_title('unc')
                confs = out_dict['conf_img'][r].permute(2, 0, 1).detach().cpu().numpy()
                for i, img in enumerate(confs[min(confs.shape[0] - 1, 1):, :, :]):
                    axs[r, 3 + i].imshow(img, cmap='jet', vmin=0, vmax=1)
                    axs[r, 3 + i].set_title(f'conf{i + 1}')
            plt.tight_layout()
            image = plt_to_image(plt)
            plt.close()
            runner.writer.add_image(f"train/img", image, steps, dataformats='HWC')


class LGLNSegPlotWriterHookV2(BaseHook):
    def __init__(self, max_n_img=32, log_every=100, **kwargs):
        super().__init__(**kwargs)
        self.max_n_img = max_n_img
        self.log_every = log_every
        self.classes = ['FlowingWater', 'Grassland', 'Settlement', 'StandingWater', 'Woodland']

    def post_iter(self, runner, **kwargs):
        out_dict = runner.model.out_dict
        steps = (runner.epoch - 1) * runner.total_iter + runner.iter
        if steps % self.log_every == 0:
            import matplotlib.pyplot as plt
            # this might use main thread if imported on the top
            years = np.array(out_dict['data_years'][0])
            n = max(len(years), 4)
            input_img = out_dict['input_img'][:n]
            conf_img = out_dict['conf_img'][:n]
            rows = min(4, input_img.shape[0])
            if len(input_img) > 4:
                inds = torch.arange(0, len(input_img), len(input_img) // rows + 1).tolist()
                if len(inds) < 4:
                    inds.append(len(input_img) - 1)
            else:
                inds = torch.arange(0, len(input_img), 1).tolist()
            input_img = input_img[inds]
            conf_img = conf_img[inds]
            years = years[inds]
            gt_img = unpackbits(out_dict['gt_img'][0].squeeze(), num_bits=5, bitorder='little').float()
            cols = conf_img.shape[-1] + 1
            fig = plt.figure(figsize=(3*cols, 3*(rows + 1)))
            axs = fig.subplots(rows + 1, cols)
            for r in range(rows):
                axs[r, 0].imshow(input_img[r].permute(1, 2, 0).detach().cpu().numpy()[::2, ::2, ::-1])
                axs[r, 0].set_title(years[r])
                confs = conf_img[r].permute(2, 0, 1).detach().cpu().numpy()
                for i, img in enumerate(confs):
                    axs[r, 1 + i].imshow(img[::2, ::2], cmap='jet', vmin=0, vmax=1)
                    axs[r, 1 + i].set_title(f'{self.classes[i]}')
            axs[-1, 0].imshow(
                np.zeros_like((out_dict['gt_img'][0].squeeze()[::2, ::2].float() / 20).detach().cpu().numpy()),
                cmap='jet', vmin=0, vmax=1)
            for l in range(conf_img.shape[-1]):
                axs[-1, l+1].imshow(gt_img[::2, ::2, l].detach().cpu().numpy(),
                                 cmap='jet', vmin=0, vmax=1)
                axs[-1, l+1].set_title(f'GT:{self.classes[l]}')
            plt.tight_layout()
            image = plt_to_image(plt)
            plt.close()
            runner.writer.add_image(f"train/img", image, steps, dataformats='HWC')


class LGLNSegPlotWriterHook(BaseHook):
    def __init__(self, max_n_img=32, log_every=100, **kwargs):
        super().__init__(**kwargs)
        self.max_n_img = max_n_img
        self.log_every = log_every
        self.classes = ['Woodland', 'Grassland', 'Settlement', 'FlowingWater', 'StandingWater']

    def post_iter(self, runner, **kwargs):
        out_dict = runner.model.out_dict
        steps = (runner.epoch - 1) * runner.total_iter + runner.iter
        if steps % self.log_every == 0:
            import matplotlib.pyplot as plt
            # this might use main thread if imported on the top
            years = [x[0] for x in out_dict['data_years']]
            input_img = out_dict['input_img'][:4].detach().cpu().numpy()
            conf_img = out_dict['conf_img'][:4].detach().cpu().numpy()
            gt_img = out_dict['gt_img'][:4].squeeze().detach().cpu().numpy()
            rows = 4
            cols = conf_img.shape[-1] + 1
            fig = plt.figure(figsize=(3*cols, 3*(rows + 1)))
            axs = fig.subplots(rows, cols)
            for r in range(rows):
                axs[r, 0].imshow(input_img[r].transpose(1, 2, 0)[::2, ::2])
                axs[r, 0].set_title(years[r])
                confs = conf_img[r].transpose(2, 0, 1)
                labels = gt_img[r]
                for i, img in enumerate(confs[1:]):
                    axs[r, 1 + i].imshow(img[::2, ::2], cmap='jet', vmin=0, vmax=1)
                    axs[r, 1 + i].set_title(f'{self.classes[i]}')
                    axs[r, -1].set_axis_off()
                axs[r, -1].imshow((labels[::2, ::2] / 5), cmap='jet', vmin=0, vmax=1)
            plt.tight_layout()
            image = plt_to_image(plt)
            plt.close()
            runner.writer.add_image(f"train/img", image, steps, dataformats='HWC')


class SaveSegResultHook(BaseHook):
    def __init__(self, save_every=4, **kwargs):
        super().__init__(**kwargs)
        self.save_every = save_every

    def set_logger(self, logger):
        self.log_dir = os.path.join(logger.logdir, 'seg_result')
        os.makedirs(self.log_dir, exist_ok=True)

    def post_iter(self, runner, **kwargs):
        out_dict = runner.model.out_dict
        for i, f in enumerate(out_dict['filename']):
            if i % self.save_every == 0:
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(15, 5))
                axs = fig.subplots(1, 4)
                axs[0].imshow(out_dict['input_img'][0].permute(1, 2, 0).detach().cpu().numpy())
                axs[1].imshow(out_dict['conf_img'][0, :, :, 1].detach().cpu().numpy(), cmap='jet')
                axs[2].imshow(1 - out_dict['unc_img'][0].detach().cpu().numpy(), cmap='hot')
                axs[3].imshow((out_dict['gt_img'][0] > 0).float().permute(1, 2, 0).detach().cpu().numpy(), cmap='jet')
                plt.savefig(os.path.join(self.log_dir, f))
                plt.close()


class HolisticSegResultHook(BaseHook):
    def __init__(self, input_size=512, overlap=24, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.overlap = overlap
        self.margin = overlap // 2
        self.maps = set()
        self.max_xy = [0, 0]
        self.conf_channels = kwargs.get('conf_channels',
                                        ['bg', 'agr', 'for', 'wat', 'set'])

    def set_logger(self, logger):
        self.log_dir = os.path.join(logger.logdir, 'holistic_seg_result')
        os.makedirs(self.log_dir, exist_ok=True)

    def post_iter(self, runner, **kwargs):
        out_dict = runner.model.out_dict
        self.maps = self.maps | set(['_'.join(x.split('_')[:2]) for x in out_dict['filename']])
        for i, f in enumerate(out_dict['filename']):
            x, y = f.split('.')[0].split('_')[2:4]
            self.max_xy = [max(int(x), self.max_xy[0]), max(int(y), self.max_xy[1])]
            conf = out_dict['conf_img'][i, :, :].detach().cpu().numpy()
            unc = out_dict['unc_img'][i].detach().cpu().numpy()
            for c, k in enumerate(self.conf_channels):
                cv2.imwrite(os.path.join(self.log_dir, f.replace('.png', f'_{k}.png')),
                            conf[..., c] * 255)
            cv2.imwrite(os.path.join(self.log_dir, f.replace('.png', f'_unc.png')), unc * 255)

    def post_epoch(self, runner, **kwargs):
        names = os.listdir(self.log_dir)
        for m in self.maps:
            for k in self.conf_channels + ['unc']:
                self.stitch(k, [n for n in names if (f'{k}.png' in n) and (m in n)])

    def stitch(self, key, names):
        patch, year, x, y = names[0].split('.')[0].split('_')[:4]
        name = f'{patch}_{year}'
        fullmap = np.zeros((self.max_xy[0] + self.input_size, self.max_xy[1] + self.input_size))
        print(f"Stitching images for '{name}_{key}'...")
        for n in tqdm.tqdm(names):
            x, y = n.split('.')[0].split('_')[2:4]
            x = int(x) + self.margin * bool(int(x))
            y = int(y) + self.margin * bool(int(y))
            x0 = min(x, self.margin)
            y0 = min(y, self.margin)
            sx = self.input_size - self.margin - x0
            sy = self.input_size - self.margin - y0
            f = os.path.join(self.log_dir, n)
            res = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            os.remove(f)
            try:
                fullmap[x:x + sx, y:y + sy] = res[x0:-self.margin, y0:-self.margin]
            except:
                print('d')

        cv2.imwrite(os.path.join(self.log_dir, f'{name}_{key}.png'), fullmap)


class LGLNEvalHook(BaseHook):
    def __init__(self, input_size=1024, overlap=0, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.overlap = overlap
        self.margin = overlap // 2
        self.maps = set()
        self.max_xy = [0, 0]
        self.conf_channels = ['FlowingWater', 'Grassland', 'Settlement', 'StandingWater', 'Woodland']
        self.ious = []
        self.accs = []

    def set_logger(self, logger):
        self.log_dir = os.path.join(logger.logdir, 'holistic_seg_result')
        os.makedirs(self.log_dir, exist_ok=True)

    def post_iter(self, runner, **kwargs):
        out_dict = runner.model.out_dict
        maps = []
        for p, years in zip(out_dict['patch'], out_dict['data_years']):
            maps.extend([f'{p}_{y}' for y in years])
        self.maps = self.maps | set(maps)
        for i, patch in enumerate(out_dict['patch']):
            x, y = out_dict['x'][i], out_dict['y'][i]
            self.max_xy = [max(int(x), self.max_xy[0]), max(int(y), self.max_xy[1])]
            for j, year in enumerate(out_dict['data_years'][i]):
                cls = out_dict['conf_img'][j, :, :].argmax(-1).cpu().numpy()
                cv2.imwrite(os.path.join(self.log_dir, f'{patch}_{year}_{x}_{y}.png'), cls * 50)

    def post_epoch(self, runner, **kwargs):
        names = os.listdir(self.log_dir)
        for m in self.maps:
            self.stitch([n for n in names if m in n])

    def stitch(self, names):
        patch, year, x, y = names[0].split('.')[0].split('_')[:4]
        name = f'{patch}_{year}'
        fullmap = np.zeros((self.max_xy[0] + self.input_size, self.max_xy[1] + self.input_size))
        print(f"Stitching images for '{name}'...")
        for n in tqdm.tqdm(names):
            x, y = n.split('.')[0].split('_')[2:4]
            x = int(x)
            y = int(y)
            sx = self.input_size
            sy = self.input_size
            f = os.path.join(self.log_dir, n)
            res = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            os.remove(f)
            fullmap[x:x + sx, y:y + sy] = res

        cv2.imwrite(os.path.join(self.log_dir, f'{name}.png'), fullmap)


class LGLNPseudoLabelHook(BaseHook):
    def __init__(self, unc=0.5, **kwargs):
        super().__init__(**kwargs)
        self.unc = unc

    def set_logger(self, logger):
        pass

    def post_train(self, runner, **kwargs):
        dataset = runner.dataloader.dataset
        pseudo_labels, has_new_labels = dataset.pre_update_pseudo_labels()
        if has_new_labels:
            runner.model.eval()
            print("Generating pseudo labels for:")
            for k, v in pseudo_labels.items():
                print(f"{k}: {', '.join([str(x) for x in v['years']])}")
            with torch.no_grad():
                for data in tqdm.tqdm(dataset, total=len(dataset)):
                    data = dataset.collate_batch([data])
                    load_tensors_to_gpu(data, runner.gpu_id)
                    data = runner.model.forward(data, epoch=runner.epoch, itr=runner.iter, gpu_id=runner.gpu_id)
                    patch, x, y, year = data['patch'][0], data['x'][0], data['y'][0], data['data_years'][0][0]
                    conf_img = data['conf_img'].squeeze()
                    max_conf, plabel = conf_img.max(-1)
                    # ignore = (max_conf < self.pos) & (max_conf > self.neg)
                    ignore = data['unc_img'].squeeze() > self.unc
                    plabel[ignore] = -1
                    idx = pseudo_labels[patch]['years'].index(year)
                    sz = pseudo_labels[patch]['data'][idx][x:x + dataset.crop_size, y:y + dataset.crop_size].shape
                    pseudo_labels[patch]['data'][idx][x:x+dataset.crop_size, y:y+dataset.crop_size] = \
                        plabel.cpu().numpy()[:sz[0], :sz[1]].astype(np.int8)

            # cv2.imwrite("/home/yuan/Downloads/tmp.jpg", pseudo_labels[patch]['data'][0].astype(np.uint8) * 50)
            dataset.post_update_pseudo_labels(pseudo_labels)


class CCVitResultHook(BaseHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.results = []

    def set_logger(self, logger):
        self.log_dir = os.path.join(logger.logdir, 'cls_result')
        os.makedirs(self.log_dir, exist_ok=True)

    def draw_attn_on_img(self, img, attn, ax, title=None):
        # Resize attention map to match the image size (512x512)
        attn = np.clip(attn * 1000, 0, 1)
        s = img.shape[0] // attn.shape[0]
        attention_map_resized = np.kron(attn, np.ones((s, s)))

        ax.imshow(img, extent=(0, img.shape[0], 0, img.shape[1]))  # Show the image
        ax.imshow(attention_map_resized, cmap='jet', alpha=0.5, extent=(0, img.shape[0], 0, img.shape[1]))
        ax.set_title(title)

    def post_iter(self, runner, **kwargs):
        out_dict = runner.model.out_dict
        vit = runner.model.vit
        import matplotlib.pyplot as plt

        attn = vit.cross_attn.attn_weight[:, 0, 0, 1:].cpu().numpy()
        s = int(math.sqrt(attn.shape[-1]))
        attn = attn.reshape(-1, s, s)

        for i, (patch, year, x, y, img, conf, label) in enumerate(zip(
            out_dict['patch'],
            out_dict['year'],
            out_dict['x'],
            out_dict['y'],
            out_dict['img'],
            out_dict['conf'],
            out_dict['label']
        )):
            if i % 5 != 0: continue
            img = img.cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5
            fig = plt.figure(figsize=(8, 4))
            axs = fig.subplots(1, 2)
            axs[0].imshow(img)
            self.draw_attn_on_img(img[::2, ::2], attn[i], axs[1], title='attn weights')
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, f'{patch}_{year}_{x}_{y}_{conf[1]:.2f}_{label:1d}.jpg'))
            plt.close()
            # cv2.imwrite(os.path.join(self.log_dir, f'{patch}_{year}_{x}_{y}_{conf[1]:.2f}_{label:1d}.jpg'),
            #             (img.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255)
        self.results.append(torch.stack([out_dict['conf'].argmax(-1), out_dict['label']], dim=-1))

    def post_epoch(self, runner, **kwargs):
        res = torch.cat(self.results, dim=0)
        acc = (res[:, 0] == res[:, 1]).sum() / len(res)
        print(f"Accuracy: {acc * 100:.2f}%")


class CVitResultHook(BaseHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.results = []

    def set_logger(self, logger):
        self.log_dir = os.path.join(logger.logdir, 'cls_result')
        os.makedirs(self.log_dir, exist_ok=True)

    def draw_attn_on_img(self, img, attn, ax, title=None):
        # Resize attention map to match the image size (512x512)
        attn = np.clip(attn * 1000, 0, 1)
        s = img.shape[0] // attn.shape[0]
        attention_map_resized = np.kron(attn, np.ones((s, s)))

        ax.imshow(img, extent=(0, img.shape[0], 0, img.shape[1]))  # Show the image
        ax.imshow(attention_map_resized, cmap='jet', alpha=0.5, extent=(0, img.shape[0], 0, img.shape[1]))
        ax.set_title(title)

    def post_iter(self, runner, **kwargs):
        out_dict = runner.model.out_dict
        vit = runner.model.vit
        attn_weights = {}
        import matplotlib.pyplot as plt
        for i, msb in enumerate(vit.blocks): # multi scale block
            for j, sab in enumerate(msb.blocks): # self attention block
                attn_weight = sab[0].attn.attn_weight
                s = int(math.sqrt(attn_weight.shape[-1] - 1))
                attn_weights[f'msb{i}_sab{j}'] = attn_weight[:, 0, 0, 1:].view(-1, s, s)
            for k, cab in enumerate(msb.blocks): # cross attention block
                attn_weight = cab[0].attn.attn_weight
                attn_weights[f'msb{i}_cab{j}'] = attn_weight[:, 0, 0, 1:].view(-1, s, s)

        for i, (patch, year, x, y, img, conf, label) in enumerate(zip(
            out_dict['patch'],
            out_dict['year'],
            out_dict['x'],
            out_dict['y'],
            out_dict['img'],
            out_dict['conf'],
            out_dict['label']
        )):
            if i % 5 != 0: continue
            img = img.cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5
            fig = plt.figure(figsize=(10, 4))
            axs = fig.subplots(2, 5)
            axs[0, 0].imshow(img)
            attns = [(k, v[i].cpu().numpy()) for k, v in attn_weights.items()]
            for j in range(1, 10):
                row = j // 5
                col = j % 5
                self.draw_attn_on_img(img[::2, ::2], attns[j-1][1], axs[row, col], title=attns[j-1][0])
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, f'{patch}_{year}_{x}_{y}_{conf[1]:.2f}_{label:1d}.jpg'))
            plt.close()
            # cv2.imwrite(os.path.join(self.log_dir, f'{patch}_{year}_{x}_{y}_{conf[1]:.2f}_{label:1d}.jpg'),
            #             (img.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255)
        self.results.append(torch.stack([out_dict['conf'].argmax(-1), out_dict['label']], dim=-1))

    def post_epoch(self, runner, **kwargs):
        res = torch.cat(self.results, dim=0)
        acc = (res[:, 0] == res[:, 1]).sum() / len(res)
        print(f"Accuracy: {acc * 100:.2f}%")










