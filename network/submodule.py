import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BasicConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        deconv=False,
        is_3d=False,
        bn=True,
        relu=True,
        **kwargs
    ):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(
                    in_channels, out_channels, bias=False, **kwargs
                )
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(
                    in_channels, out_channels, bias=False, **kwargs
                )
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)  # , inplace=True)
        return x


class Conv2x(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        deconv=False,
        is_3d=False,
        concat=True,
        keep_concat=True,
        bn=True,
        relu=True,
        keep_dispc=False,
    ):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_3d = is_3d
        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(
                in_channels,
                out_channels,
                deconv,
                is_3d,
                bn=True,
                relu=True,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            )
        else:
            self.conv1 = BasicConv(
                in_channels,
                out_channels,
                deconv,
                is_3d,
                bn=True,
                relu=True,
                kernel_size=kernel,
                stride=2,
                padding=1,
            )

        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(
                out_channels * 2,
                out_channels * mul,
                False,
                is_3d,
                bn,
                relu,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        else:
            self.conv2 = BasicConv(
                out_channels,
                out_channels,
                False,
                is_3d,
                bn,
                relu,
                kernel_size=3,
                stride=1,
                padding=1,
            )

    def forward(self, x, rem):
        x = self.conv1(x)
        x = F.interpolate(x, size=(rem.shape[-2], rem.shape[-1]), mode="nearest")
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


class BasicConv_IN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        deconv=False,
        is_3d=False,
        IN=True,
        relu=True,
        **kwargs
    ):
        super(BasicConv_IN, self).__init__()

        self.relu = relu
        self.use_in = IN
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(
                    in_channels, out_channels, bias=False, **kwargs
                )
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(
                    in_channels, out_channels, bias=False, **kwargs
                )
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_in:
            x = self.IN(x)
        if self.relu:
            x = nn.LeakyReLU()(x)  # , inplace=True)
        return x


class Conv2x_IN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        deconv=False,
        is_3d=False,
        concat=True,
        keep_concat=True,
        IN=True,
        relu=True,
        keep_dispc=False,
    ):
        super(Conv2x_IN, self).__init__()
        self.concat = concat
        self.is_3d = is_3d
        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv_IN(
                in_channels,
                out_channels,
                deconv,
                is_3d,
                IN=True,
                relu=True,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            )
        else:
            self.conv1 = BasicConv_IN(
                in_channels,
                out_channels,
                deconv,
                is_3d,
                IN=True,
                relu=True,
                kernel_size=kernel,
                stride=2,
                padding=1,
            )

        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv_IN(
                out_channels * 2,
                out_channels * mul,
                False,
                is_3d,
                IN,
                relu,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        else:
            self.conv2 = BasicConv_IN(
                out_channels,
                out_channels,
                False,
                is_3d,
                IN,
                relu,
                kernel_size=3,
                stride=1,
                padding=1,
            )

    def forward(self, x, rem):
        x = self.conv1(x)
        x = F.interpolate(x, size=(rem.shape[-2], rem.shape[-1]), mode="nearest")
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


def build_gwc_volume(
    left: torch.Tensor, right: torch.Tensor, max_disp: int, num_groups: int
) -> torch.Tensor:
    B, C, H, W = left.shape
    assert C % num_groups == 0

    left = left.view(B, num_groups, -1, H, W)
    right = right.view(B, num_groups, -1, H, W)
    volume = []
    for i in range(max_disp):
        v = (left[..., i:] * right[..., : W - i]).mean(dim=2, keepdim=True)
        zeros = torch.zeros([B, num_groups, 1, H, i]).to(v)
        volume.append(torch.cat([zeros, v], dim=-1))
        # volume.append(F.pad(v, (i, 0), "constant", 0))
    return torch.cat(volume, dim=2)


def norm_correlation(fea1, fea2):
    cost = torch.mean(
        (
            (fea1 / (torch.norm(fea1, 2, 1, True) + 1e-05))
            * (fea2 / (torch.norm(fea2, 2, 1, True) + 1e-05))
        ),
        dim=1,
        keepdim=True,
    )
    return cost


def build_norm_correlation_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = norm_correlation(
                refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i]
            )
        else:
            volume[:, :, i, :, :] = norm_correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    return volume


def correlation(fea1, fea2):
    cost = torch.sum((fea1 * fea2), dim=1, keepdim=True)
    return cost


def build_correlation_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = correlation(
                refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i]
            )
        else:
            volume[:, :, i, :, :] = correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    return volume


def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, :] = refimg_fea[:, :, :, :]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


def disparity_regression(x: torch.Tensor, max_disp: int) -> torch.Tensor:
    """Disparity regression layer
    Args:
        x (Tensor): cost volume with shape (N, D, H, W)
        maxdisp (int): maximum disparity
    Returns:
        Tensor: disparity map with shape (N, 1, H, W)
    """
    assert x.dim() == 4
    disp = torch.arange(0, max_disp).to(x).view(1, max_disp, 1, 1)
    return x.mul(disp).sum(1, keepdim=True)


class FeatureAtt(nn.Module):
    def __init__(self, cv_chan: int, feat_chan: int) -> None:
        super().__init__()

        self.feat_att = nn.Sequential(
            BasicConv(feat_chan, feat_chan // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_chan // 2, cv_chan, 1),
        )

    def forward(self, cv: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        feat_att = self.feat_att(feat).unsqueeze(2)
        cv = torch.sigmoid(feat_att) * cv
        return cv


def context_upsample(disp_low: torch.Tensor, up_weights: torch.Tensor) -> torch.Tensor:
    ###
    # cv (b,1,h,w)
    # sp (b,9,4*h,4*w)
    ###
    b, c, h, w = disp_low.shape
    disp_unfold = F.unfold(disp_low.reshape(b, c, h, w), 3, 1, 1).reshape(b, -1, h, w)
    disp_unfold = F.interpolate(disp_unfold, (h * 4, w * 4), mode="nearest").reshape(
        b, 9, h * 4, w * 4
    )
    return disp_unfold.mul(up_weights).sum(1)
