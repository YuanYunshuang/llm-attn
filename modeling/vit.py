# Modified from https://github.com/IBM/CrossViT/blob/main/models/crossvit.py#L41
from itertools import repeat
import collections.abc
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub
from functools import partial

from src.utils.vit_util import trunc_normal_, DropPath, Mlp, Block, compute_num_patches, DropPatch

IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(repeat(x, 2))


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class DownConv_v2(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels):
        super(DownConv_v2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Sequential(
            conv3x3(self.in_channels, self.out_channels),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            conv3x3(self.out_channels, self.out_channels, stride=2),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        # tic = timeit.default_timer()
        x = self.conv(x)
        # print('conv_blk:', timeit.default_timer() - tic)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, multi_conv=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # can be replace by multiple convs
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_weight = None

    def forward(self, x):
        """1st token of x cross attend all tokens of in x."""
        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        self.attn_weight = attn

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class MultiScaleBlock(nn.Module):

    def __init__(self, dim, patches, depth, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, fused_attn=True):
        super().__init__()

        num_branches = len(dim)
        self.num_branches = num_branches
        # different branch could have different embedding size, the first one is the base
        self.blocks = nn.ModuleList()
        for d in range(num_branches):
            tmp = []
            for i in range(depth[d]):
                tmp.append(
                    Block(dim=dim[d], num_heads=num_heads[d], mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias,
                          proj_drop=drop, attn_drop=attn_drop, drop_path=drop_path[i], norm_layer=norm_layer,
                          fused_attn=fused_attn))
            if len(tmp) != 0:
                self.blocks.append(nn.Sequential(*tmp))

        if len(self.blocks) == 0:
            self.blocks = None

        self.projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[d] == dim[(d+1) % num_branches] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [norm_layer(dim[d]), act_layer(), nn.Linear(dim[d], dim[(d+1) % num_branches])]
            self.projs.append(nn.Sequential(*tmp))

        self.fusion = nn.ModuleList()
        for d in range(num_branches):
            d_ = (d+1) % num_branches
            nh = num_heads[d_]
            if depth[-1] == 0:  # backward capability:
                self.fusion.append(CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                       drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=norm_layer,
                                                       has_mlp=False))
            else:
                tmp = []
                for _ in range(depth[-1]):
                    tmp.append(CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=norm_layer,
                                                   has_mlp=False))
                self.fusion.append(nn.Sequential(*tmp))

        self.revert_projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[(d+1) % num_branches] == dim[d] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [norm_layer(dim[(d+1) % num_branches]), act_layer(), nn.Linear(dim[(d+1) % num_branches], dim[d])]
            self.revert_projs.append(nn.Sequential(*tmp))

    def forward(self, x):
        outs_b = [block(x_) for x_, block in zip(x, self.blocks)]
        # only take the cls token out
        proj_cls_token = [proj(x[:, 0:1]) for x, proj in zip(outs_b, self.projs)]
        # cross attention
        outs = []
        for i in range(self.num_branches):
            tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
            tmp = self.fusion[i](tmp)
            reverted_proj_cls_token = self.revert_projs[i](tmp[:, 0:1, ...])
            tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
            outs.append(tmp)
        return outs


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=(224, 224), patch_size=(8, 16), in_chans=3, num_classes=1000, embed_dim=(192, 384), depth=([1, 3, 1], [1, 3, 1], [1, 3, 1]),
                 num_heads=(6, 12), mlp_ratio=(2., 2., 4.), qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, multi_conv=False, fused_attn=True):
        super().__init__()

        self.num_classes = num_classes
        if not isinstance(img_size, list):
            img_size = to_2tuple(img_size)
        self.img_size = img_size

        num_patches = compute_num_patches(img_size, patch_size)
        self.num_branches = len(patch_size)

        self.patch_embed = nn.ModuleList()
        if hybrid_backbone is None:
            self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i])) for i in range(self.num_branches)])
            for im_s, p, d in zip(img_size, patch_size, embed_dim):
                self.patch_embed.append(PatchEmbed(img_size=im_s, patch_size=p, in_chans=in_chans, embed_dim=d, multi_conv=multi_conv))
        else:
            self.pos_embed = nn.ParameterList()
            from src.utils.vit_util import T2T, get_sinusoid_encoding
            tokens_type = 'transformer' if hybrid_backbone == 't2t' else 'performer'
            for idx, (im_s, p, d) in enumerate(zip(img_size, patch_size, embed_dim)):
                self.patch_embed.append(T2T(im_s, tokens_type=tokens_type, patch_size=p, embed_dim=d))
                self.pos_embed.append(nn.Parameter(data=get_sinusoid_encoding(n_position=1 + num_patches[idx], d_hid=embed_dim[idx]), requires_grad=False))

            del self.pos_embed
            self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i])) for i in range(self.num_branches)])

        self.cls_token = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, embed_dim[i])) for i in range(self.num_branches)])
        self.pos_drop = nn.Dropout(p=drop_rate)

        total_depth = sum([sum(x[-2:]) for x in depth])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]  # stochastic depth decay rule
        dpr_ptr = 0
        self.blocks = nn.ModuleList()
        for idx, block_cfg in enumerate(depth):
            curr_depth = max(block_cfg[:-1]) + block_cfg[-1]
            dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]
            blk = MultiScaleBlock(embed_dim, num_patches, block_cfg, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_,
                                  norm_layer=norm_layer, fused_attn=fused_attn)
            dpr_ptr += curr_depth
            self.blocks.append(blk)

        self.norm = nn.ModuleList([norm_layer(embed_dim[i]) for i in range(self.num_branches)])
        self.head = nn.ModuleList([nn.Linear(embed_dim[i], num_classes) if num_classes > 0 else nn.Identity() for i in range(self.num_branches)])

        for i in range(self.num_branches):
            if self.pos_embed[i].requires_grad:
                trunc_normal_(self.pos_embed[i], std=.02)
            trunc_normal_(self.cls_token[i], std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        out = {'cls_token'}
        if self.pos_embed[0].requires_grad:
            out.add('pos_embed')
        return out

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B, C, H, W = x.shape
        xs = []
        for i in range(self.num_branches):
            x_ = torch.nn.functional.interpolate(x, size=(self.img_size[i], self.img_size[i]), mode='bicubic') if H != self.img_size[i] else x
            tmp = self.patch_embed[i](x_)
            cls_tokens = self.cls_token[i].expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            tmp = torch.cat((cls_tokens, tmp), dim=1)
            tmp = tmp + self.pos_embed[i]
            tmp = self.pos_drop(tmp)
            xs.append(tmp)

        for blk in self.blocks:
            xs = blk(xs)

        # NOTE: was before branch token section, move to here to assure all branch token are before layer norm
        xs = [self.norm[i](x) for i, x in enumerate(xs)]
        out = [x[:, 0] for x in xs]

        return out

    def forward(self, x):
        xs = self.forward_features(x)
        ce_logits = [self.head[i](x) for i, x in enumerate(xs)]
        ce_logits = torch.mean(torch.stack(ce_logits, dim=0), dim=0)
        return ce_logits


class ConvVitV1(nn.Module):
    def __init__(self, img_size=384, in_channel=3, out_channel=512, depth=6, num_classes=2, norm_layer=nn.LayerNorm,
                 drop_patch=False, **kwargs):
        super().__init__()
        blocks = []
        for _ in range(depth):
            out_dim = max(16, in_channel * 2)
            blocks.append(DownConv(in_channel, out_dim))
            in_channel = out_dim
        self.conv_blocks = nn.Sequential(*blocks)
        self.proj_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        down_ratio = 2 ** depth
        num_patches = (img_size // down_ratio) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, out_channel))
        self.drop_patch = DropPatch(0.2) if drop_patch else None
        self.cross_attn = CrossAttention(out_channel, num_heads=1, qkv_bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, out_channel))

        self.norm = norm_layer(out_channel)
        self.head = nn.Linear(out_channel, num_classes)

    @torch.jit.ignore
    def no_weight_decay(self):
        out = {'cls_token'}
        if self.pos_embed[0].requires_grad:
            out.add('pos_embed')
        return out

    def forward_features(self, x, **kwargs):
        B, C, H, W = x.shape
        # tic = timeit.default_timer()
        for blk in self.conv_blocks:
            x = blk(x)[0]
        # print('blk:',timeit.default_timer() - tic)

        # tic = timeit.default_timer()
        x = self.proj_conv(x).flatten(2, 3).permute(0, 2, 1) #BLD
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        # print('proj:', timeit.default_timer() - tic)

        # tic = timeit.default_timer()
        if self.drop_patch:
            x = self.drop_patch(x, kwargs.get('drop_patch_mask', None))
        x = self.cross_attn(x)
        # print('cross_attn:', timeit.default_timer() - tic)
        out = self.norm(x)[:, 0]

        return out

    def forward(self, x, **kwargs):
        x = self.forward_features(x, **kwargs)
        ce_logits = self.head(x)
        return ce_logits


class ConvVit(nn.Module):
    def __init__(self, img_size=384, in_channel=3, out_channel=512, depth=6, num_classes=2, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        blocks = []
        for _ in range(depth):
            out_dim = max(16, in_channel * 2)
            blocks.append(DownConv(in_channel, out_dim))
            in_channel = out_dim
        self.conv_blocks = nn.Sequential(*blocks)
        self.proj_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        down_ratio = 2 ** depth
        num_patches = (img_size // down_ratio) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, out_channel))
        self.self_attn = Block(out_channel, 4)
        self.cross_attn = CrossAttention(out_channel, num_heads=1, qkv_bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, out_channel))

        self.norm = norm_layer(out_channel)
        self.head = nn.Linear(out_channel, num_classes)

    @torch.jit.ignore
    def no_weight_decay(self):
        out = {'cls_token'}
        if self.pos_embed[0].requires_grad:
            out.add('pos_embed')
        return out

    def forward_features(self, x):
        B, C, H, W = x.shape
        for blk in self.conv_blocks:
            x = blk(x)[0]
        x = self.proj_conv(x).flatten(2, 3).permute(0, 2, 1) #BLD
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        x = torch.cat([x[:, :1], self.self_attn(x[:, 1:])], dim=1)
        x = self.cross_attn(x)

        out = self.norm(x)[:, 0]

        return out

    def forward(self, x):
        x = self.forward_features(x)
        ce_logits = self.head(x)
        return ce_logits


class ConvVitMultiCls(nn.Module):
    def __init__(self, img_size=384, in_channel=3, out_channel=512, depth=6, num_classes=4, norm_layer=nn.LayerNorm,
                 drop_patch=False, **kwargs):
        super().__init__()
        blocks = []
        for _ in range(depth):
            out_dim = max(16, in_channel * 2)
            blocks.append(DownConv(in_channel, out_dim))
            in_channel = out_dim
        self.conv_blocks = nn.Sequential(*blocks)
        self.proj_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        down_ratio = 2 ** depth
        num_patches = (img_size // down_ratio) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, out_channel))
        self.drop_patch = DropPatch(0.2) if drop_patch else None
        self.cross_attn = nn.ModuleList([CrossAttention(out_channel, num_heads=1, qkv_bias=False) for _ in range(num_classes)])
        self.cls_token = nn.Parameter(torch.zeros(num_classes, 1, out_channel))

        self.norm = nn.ModuleList([norm_layer(out_channel) for _ in range(num_classes)])
        self.head = nn.ModuleList([nn.Linear(out_channel, 1) for _ in range(num_classes)])

    @torch.jit.ignore
    def no_weight_decay(self):
        out = {'cls_token'}
        if self.pos_embed[0].requires_grad:
            out.add('pos_embed')
        return out

    def forward_features(self, x, **kwargs):
        B, C, H, W = x.shape
        # tic = timeit.default_timer()
        for blk in self.conv_blocks:
            x = blk(x)
        # print('blk:',timeit.default_timer() - tic)

        x = self.proj_conv(x).flatten(2, 3).permute(0, 2, 1) #BLD

        xs = []
        for ct in self.cls_token:
            cls_token = ct.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x_ = torch.cat((cls_token, x), dim=1)
            x_ = x_ + self.pos_embed
            xs.append(x_)

        if self.drop_patch:
            xs = self.drop_patch(xs, kwargs.get('drop_patch_mask', None))

        outs = []
        for ca, norm, x_ in zip(self.cross_attn, self.norm, xs):
            outs.append(norm(ca(x_))[:, 0])
        return outs

    def forward(self, x, **kwargs):
        x = self.forward_features(x, **kwargs)
        ce_logits = torch.cat([h(x_) for x_, h in zip(x, self.head)], dim=-1)
        return ce_logits


class PatchConv(nn.Module):
    def __init__(self, img_size=384, in_channel=3, out_channel=512, depth=6, num_classes=2, norm_layer=nn.LayerNorm,
                 drop_patch=False, patch_size=128, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        blocks = []
        for _ in range(depth):
            out_dim = max(16, in_channel * 2)
            blocks.append(DownConv(in_channel, out_dim))
            in_channel = out_dim
        self.conv_blocks = nn.Sequential(*blocks)
        self.proj_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        down_ratio = 2 ** depth
        num_patches = (img_size // down_ratio) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, out_channel))
        self.drop_patch = DropPatch(0.2) if drop_patch else None
        self.cross_attn = CrossAttention(out_channel, num_heads=1, qkv_bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, out_channel))

        self.norm = norm_layer(out_channel)
        self.head = nn.Linear(out_channel, num_classes)

    @torch.jit.ignore
    def no_weight_decay(self):
        out = {'cls_token'}
        if self.pos_embed[0].requires_grad:
            out.add('pos_embed')
        return out

    def forward_features(self, x, **kwargs):
        B, C, H, W = x.shape
        n = self.patch_size # patch size
        Hn, Wn = H // n, W // n
        patches = x.unfold(2, n, n).unfold(3, n, n) # Shape: (B, C, H/n, W/n, n, n)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B, H/n, W/n, C, n, n)
        patches = patches.view(B * Hn * Wn, C, n, n)
        # tic = timeit.default_timer()
        for blk in self.conv_blocks:
            patches = blk(patches)[0]
        # print('blk:',timeit.default_timer() - tic)


        x = self.proj_conv(patches)
        b, c, h, w = x.shape
        x = x.view(B, Hn, Wn, c, h, w)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous() # (B, C, H/n, h, W/n, w)
        x = x.view(B, c, Hn*h, Wn*w).view(B, c, Hn*h*Wn*w).permute(0, 2, 1).contiguous()
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        # print('proj:', timeit.default_timer() - tic)

        # tic = timeit.default_timer()
        if self.drop_patch:
            x = self.drop_patch(x, kwargs.get('drop_patch_mask', None))
        x = self.cross_attn(x)
        # print('cross_attn:', timeit.default_timer() - tic)
        out = self.norm(x)[:, 0]

        return out

    def forward(self, x, **kwargs):
        x = self.forward_features(x, **kwargs)
        ce_logits = self.head(x)
        return ce_logits


def _cfg(url: str = '', **kwargs) -> Dict[str, Any]:
    return {
        'url': url,
        'num_classes': 2,
        'input_size': (3, 256, 256),
        'pool_size': None,
        'crop_pct': 0.9,
        'interpolation': 'bicubic',
        'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN,
        'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj',
        'classifier': 'head',
        **kwargs,
    }

def crossvit_tiny_256(**kwargs):
    model = VisionTransformer(img_size=[256, 256], num_classes=2,
                              patch_size=[16, 16], embed_dim=[96, 192], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
                              num_heads=[3, 3], mlp_ratio=[4, 4, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


def crossvit_tiny_512(**kwargs):
    model = VisionTransformer(img_size=[512, 512], num_classes=2,
                              patch_size=[16, 16], embed_dim=[96, 192], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
                              num_heads=[3, 3], mlp_ratio=[4, 4, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

def crossvit_conv_384(**kwargs):
    model = ConvVit(**kwargs)
    model.default_cfg = _cfg()
    return model

def crossvit_conv_384_v1(**kwargs):
    model = ConvVitV1(**kwargs)
    model.default_cfg = _cfg()
    return model

def crossvit_conv_384_multi_cls(**kwargs):
    model = ConvVitMultiCls(**kwargs)
    model.default_cfg = _cfg()
    return model

def crossvit_patch_conv_384(**kwargs):
    model = PatchConv(**kwargs)
    model.default_cfg = _cfg()
    return model