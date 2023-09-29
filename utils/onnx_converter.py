import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parents[1].as_posix()
sys.path.append(ROOT_DIR)

from network import IGEVStereo


class Wrapper(nn.Module):
    def __init__(self, model: nn.Module, negative: bool = False) -> None:
        super().__init__()
        self.model = model
        self.negative = negative

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        disp = self.model(images)
        if self.negative:
            return -disp
        return disp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--restore_ckpt",
        help="restore checkpoint",
        default="./pretrained_models/middlebury.pth",
    )
    parser.add_argument(
        "--save_onnx_path",
        help="save onnx model path",
        default="./pretrained_models/middlebury.onnx",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=32,
        help="number of flow-field updates during forward pass",
    )
    parser.add_argument(
        "--hidden_dims",
        nargs="+",
        type=int,
        default=[128] * 3,
        help="hidden state and context dimensions",
    )
    parser.add_argument(
        "--corr_levels",
        type=int,
        default=2,
        help="number of levels in the correlation pyramid",
    )
    parser.add_argument(
        "--corr_radius", type=int, default=4, help="width of the correlation pyramid"
    )
    parser.add_argument(
        "--n_downsample",
        type=int,
        default=2,
        help="resolution of the disparity field (1/2^K)",
    )
    parser.add_argument(
        "--max_disp", type=int, default=192, help="max disp of geometry encoding volume"
    )
    parser.add_argument(
        "--img_size", nargs="+", type=int, default=[608, 800], help="image size (HxW)"
    )
    parser.add_argument(
        "--mixed_precision",
        default=False,
        action="store_true",
        help="use mixed precision",
    )
    parser.add_argument(
        "--negative",
        default=False,
        action="store_true",
        help="use negative disparity",
    )
    args = parser.parse_args()

    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        checkpoint = torch.load(args.restore_ckpt, map_location="cpu")
        model.load_state_dict(checkpoint, strict=True)

    model = Wrapper(model.module, args.negative)
    model.eval()

    h, w = args.img_size
    images = torch.rand(1, 6, h, w)
    # model(images)
    torch.onnx.export(
        model,
        images,
        args.save_onnx_path,
        input_names=["images"],
        output_names=["disparity"],
        verbose=False,
        opset_version=16,
        do_constant_folding=True,
    )
