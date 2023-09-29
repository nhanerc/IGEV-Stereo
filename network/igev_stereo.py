from typing import Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from .update import BasicMultiUpdateBlock
from .extractor import MultiBasicEncoder, Feature
from .geometry import CGEV
from .hourglass import hourglass
from .submodule import (
    disparity_regression,
    FeatureAtt,
    BasicConv_IN,
    Conv2x_IN,
    Conv2x,
    BasicConv,
    context_upsample,
    build_gwc_volume,
)


try:
    autocast = torch.cuda.amp.autocast
except:

    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class IGEVStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_dims = args.hidden_dims
        self.n_gru_layers = len(self.hidden_dims)
        self.corr_levels = args.corr_levels
        self.corr_radius = args.corr_radius
        self.n_downsample = args.n_downsample
        self.max_disp = args.max_disp
        self.iters = args.iters
        self.mixed_precision = args.mixed_precision

        context_dims = self.hidden_dims
        self.cnet = MultiBasicEncoder(
            output_dim=[self.hidden_dims, context_dims],
            norm_fn="batch",
            downsample=self.n_downsample,
        )
        self.update_block = BasicMultiUpdateBlock(
            self.corr_levels, self.corr_radius, self.n_gru_layers, self.hidden_dims
        )

        self.context_zqr_convs = nn.ModuleList(
            [
                nn.Conv2d(context_dims[i], self.hidden_dims[i] * 3, 3, padding=3 // 2)
                for i in range(self.n_gru_layers)
            ]
        )

        self.feature = Feature()

        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
        )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48),
            nn.ReLU(),
        )

        self.spx = nn.Sequential(
            nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1),
        )
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24),
            nn.ReLU(),
        )

        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(
            nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1),
        )

        self.conv = BasicConv_IN(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)

        self.corr_stem = BasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att = FeatureAtt(8, 96)
        self.cost_agg = hourglass(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample_disp(
        self, disp: torch.Tensor, mask_feat_4: torch.Tensor, stem_2x: torch.Tensor
    ) -> torch.Tensor:
        with autocast(enabled=self.mixed_precision):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)
            spx_pred = self.spx_gru(xspx)
            spx_pred = F.softmax(spx_pred, 1)
            up_disp = context_upsample(disp * 4.0, spx_pred).unsqueeze(1)
        return up_disp

    def forward(self, images: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Estimate disparity between pair of frames"""
        # image: [B, 6, H, W]
        assert images.shape[1] == 6
        B = images.shape[0]
        images = torch.cat(images.split(3, dim=1), dim=0)
        images = 2.0 * images / 255.0 - 1.0
        with autocast(enabled=self.mixed_precision):
            features = self.feature(images)
            stem_2 = self.stem_2(images)
            stem_4 = self.stem_4(stem_2)
            features[0] = torch.cat([features[0], stem_4], dim=1)

            match_left, match_right = self.desc(self.conv(features[0])).split(
                [B, B], dim=0
            )
            gwc_volume = build_gwc_volume(
                match_left, match_right, self.max_disp // 4, 8
            )

            gwc_volume = self.corr_stem(gwc_volume)
            gwc_volume = self.corr_feature_att(gwc_volume, features[0][:B, ...])
            geo_encoding_volume = self.cost_agg(
                gwc_volume, [feature[:B, ...] for feature in features]
            )

            # Init disp from geometry encoding volume
            prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)
            disp = disparity_regression(prob, self.max_disp // 4)  # init disp

            del prob, gwc_volume

            if self.training:
                xspx = self.spx_4(features[0][:B, ...])
                xspx = self.spx_2(xspx, stem_2[:B, ...])
                spx_pred = self.spx(xspx)
                spx_pred = F.softmax(spx_pred, 1)

            cnet_list = self.cnet(images[:B, ...], num_layers=self.n_gru_layers)
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            inp_list = [
                conv(x).split(c, dim=1)
                for c, x, conv in zip(
                    self.hidden_dims, inp_list, self.context_zqr_convs
                )
            ]

        geo_fn = CGEV(
            match_left,
            match_right,
            geo_encoding_volume,
            radius=self.corr_radius,
            num_levels=self.corr_levels,
        )
        b = B
        h, w = match_left.shape[-2:]
        coords = (
            torch.arange(w)
            .float()
            .to(match_left.device)
            .reshape(1, 1, w, 1)
            .repeat(b, h, 1, 1)
        )

        # GRUs iterations to update disparity
        if self.training:
            disp_preds = [disp]
        disp.detach_()
        for itr in range(self.iters):
            geo_feat = geo_fn(disp, coords)
            with autocast(enabled=self.mixed_precision):
                net_list, mask_feat_4, delta_disp = self.update_block(
                    net_list,
                    inp_list,
                    geo_feat,
                    disp,
                    iter16=self.n_gru_layers == 3,
                    iter08=self.n_gru_layers >= 2,
                )
            disp = disp + delta_disp

            if self.training or itr == self.iters - 1:
                disp_up = self.upsample_disp(disp, mask_feat_4, stem_2[:B])
            if self.training:
                disp_preds.append(disp_up)

        if not self.training:
            return disp_up

        disp_preds[0] = context_upsample(disp_preds[0] * 4.0, spx_pred).unsqueeze(1)
        return disp_preds
