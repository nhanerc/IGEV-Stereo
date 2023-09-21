from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicMultiUpdateBlock(nn.Module):
    def __init__(
        self,
        corr_levels: int,
        corr_radius: int,
        n_gru_layers,
        hidden_dims: Tuple[int, int, int],
    ):
        super().__init__()
        self.n_gru_layers = n_gru_layers
        self.encoder = BasicMotionEncoder(corr_levels, corr_radius)
        encoder_output_dim = 128

        self.gru04 = ConvGRU(
            hidden_dims[2],
            encoder_output_dim + hidden_dims[1] * (self.n_gru_layers > 1),
        )
        self.gru08 = ConvGRU(
            hidden_dims[1], hidden_dims[0] * (self.n_gru_layers == 3) + hidden_dims[2]
        )
        self.gru16 = ConvGRU(hidden_dims[0], hidden_dims[1])
        self.disp_head = DispHead(hidden_dims[2], hidden_dim=256, output_dim=1)

        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 32, 3, padding=1), nn.ReLU(inplace=True)
        )

    def forward(
        self,
        net: List[torch.Tensor],
        inp: List[List[torch.Tensor]],
        corr: torch.Tensor,
        disp: torch.Tensor,
        iter04: bool = True,
        iter08: bool = True,
        iter16: bool = True,
        update: bool = True,
    ):
        if iter16:
            net[2] = self.gru16(net[2], inp[2], pool2x(net[1]))
        if iter08:
            x = pool2x(net[0])
            if self.n_gru_layers > 2:
                x = torch.cat([x, interp(net[2], net[1].shape[2:])], dim=1)
            net[1] = self.gru08(net[1], inp[1], x)

        if iter04:
            x = self.encoder(disp, corr)  # motion_features
            if self.n_gru_layers > 1:
                x = torch.cat([x, interp(net[1], net[0].shape[2:])], dim=1)
            net[0] = self.gru04(net[0], inp[0], x)

        if not update:
            return net

        delta_disp = self.disp_head(net[0])
        mask_feat_4 = self.mask_feat_4(net[0])
        return net, mask_feat_4, delta_disp


class DispHead(nn.Module):
    def __init__(
        self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 1
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim: int, input_dim: int, kernel_size: int = 3):
        super().__init__()
        params = {
            "in_channels": hidden_dim + input_dim,
            "out_channels": hidden_dim,
            "kernel_size": kernel_size,
            "padding": kernel_size // 2,
        }
        self.convz = nn.Conv2d(**params)  # type: ignore
        self.convr = nn.Conv2d(**params)  # type: ignore
        self.convq = nn.Conv2d(**params)  # type: ignore

    def forward(
        self,
        h: torch.Tensor,
        context_zrq: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        x: torch.Tensor,
    ):
        cz, cr, cq = context_zrq
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx) + cz)
        r = torch.sigmoid(self.convr(hx) + cr)
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)) + cq)
        h = (1 - z) * h + z * q
        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, corr_levels: int, corr_radius: int) -> None:
        super().__init__()
        cor_planes = corr_levels * (2 * corr_radius + 1) * (8 + 1)
        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 64, 128 - 1, 3, padding=1)

    def forward(self, disp: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        corr_model = nn.Sequential(
            self.convc1,
            nn.ReLU(),
            self.convc2,
            nn.ReLU(),
        )
        corr = corr_model(corr)

        disp_model = nn.Sequential(
            self.convd1,
            nn.ReLU(),
            self.convd2,
            nn.ReLU(),
        )
        disp_ = disp_model(disp)
        cor_disp = torch.cat([corr, disp_], dim=1)
        out = F.relu(self.conv(cor_disp))
        return torch.cat([out, disp], dim=1)


def pool2x(x: torch.Tensor) -> torch.Tensor:
    """Average pool with kernel size 3, stride 2, padding 1"""
    return F.avg_pool2d(x, 3, stride=2, padding=1)


def pool4x(x: torch.Tensor) -> torch.Tensor:
    """Average pool with kernel size 5, stride 4, padding 1"""
    return F.avg_pool2d(x, 5, stride=4, padding=1)


def interp(x: torch.Tensor, size: torch.Size) -> torch.Tensor:
    """Interpolate x to the size of dest"""
    return F.interpolate(x, size, mode="bilinear", align_corners=True)
