import math
import torch
from torch import nn

import vit


def get_model(cfgs):
    return VitClassifier(**cfgs)


def pred_to_conf_unc(preds, activation='relu', edl=True):
    if callable(activation):
        evidence = activation(preds)
    elif activation == 'sigmoid':
        evidence = preds.sigmoid()
        evidence = torch.stack((evidence, 1 - evidence), dim=-1)
    elif activation == 'softmax':
        evidence = preds.softmax(dim=-1)
    else:
        evidence = preds

    # use entropy as uncertainty
    entropy = -evidence * torch.log(evidence)
    unc = entropy.sum(dim=-1) / math.log(evidence.shape[-1])
    # conf = torch.sqrt(evidence * (1 - unc.unsqueeze(-1)))
    conf = evidence
    if activation == 'sigmoid':
        unc = unc[..., 0]
        conf = conf[..., 0]
    return conf, unc


class VitClassifier(nn.Module):
    def __init__(self, vit_type, loss_cfg, drop_patch=False, **kwargs):
        super().__init__()
        use_fused_attn = True if kwargs.get('mode', 'train') == 'train' else False
        self.vit = getattr(vit, f"crossvit_{vit_type}")(fused_attn=use_fused_attn, drop_patch=drop_patch)
        self.loss_fn = build_loss(**loss_cfg)

    def forward(self, data, **kwargs):
        x = data['img']
        out = self.vit(x, **kwargs)
        data['pred'] = out
        data['conf'], data['unc'] = pred_to_conf_unc(
            data['pred'], self.loss_fn.activation, False)
        self.out_dict = data
        return data

    def loss(self, data, epoch, **kwargs):
        pred = data['pred']
        gt = data['label'].reshape(len(pred), -1)
        total_loss = self.loss_fn(pred, gt, temp=epoch)
        loss_dict = {
            'total_loss': total_loss
        }
        return total_loss, loss_dict