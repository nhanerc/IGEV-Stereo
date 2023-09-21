import torch
import torch.nn.functional as F


def grid_sample(img: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Wrapper for grid_sample, uses pixel coordinates

    Args:
        img (torch.Tensor): Input image with shape (N, C, Hi, Wi).
        grid (torch.Tensor): Coordinates with shape (N, Ho, Wo, 2).
    Returns:
        torch.Tensor: Output image with shape (N, C, Ho, Wo).
    """

    return F.grid_sample(
        img, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )


def bilinear_sampler(img: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Grid sample for 4D tensors with bilinear interpolation, zero padding and alignment corners.
    Ref: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/GridSampler.cpp

    Args:
        img (torch.Tensor): Input image with shape (N, C, Hi, Wi).
        grid (torch.Tensor): Grid with shape (N, Ho, Wo, 2).
    Returns:
        torch.Tensor: Output image with shape (N, C, Ho, Wo).

    Sanity check:
        grid = torch.rand([1000, 12, 34, 2])
        img = torch.randn([1000, 8, 100, 200])
        out1 = torch.nn.functional.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        out2 = bilinear_sampler(img, grid)
        print(torch.abs(out1 - out2).max())
    """
    N, C, Hi, Wi = img.shape
    Ho, Wo = grid.shape[1:3]
    img = img.view(N, C, -1)
    grid = grid.view(N, -1, 2)

    # align_corners=True and unormalize grid [-1, 1] to [0, W-1] and [0, H-1]
    grid = (grid + 1) * torch.tensor([Wi - 1, Hi - 1]) / 2
    delta = grid - grid.floor()  # bilinear interpolation

    # (N, Ho, Wo, 4) => (x0, y0, x1, y1)
    grid = grid.floor().long()
    grid = torch.cat([grid, grid + 1], dim=-1)
    # (N, Ho, Wo, 4) => (dx0, dy0, dx1, dy1)
    delta = torch.cat([delta, 1 - delta], dim=-1)
    # 0 => dx1 * dy1, 1 => dx0 * dy1
    # 3 => dx1 * dy0, 2 => dx0 * dy0
    # (dy1, dx0, dy0, dx1) * (dx1, dy1, dx0, dy0) = (dx1 * dy1, dx0 * dy1, dx0 * dy0, dx1 * dy0)
    delta = delta.roll(1, dims=-1) * delta.roll(2, dims=-1)

    is_inside = torch.logical_and(grid >= 0, grid < torch.tensor([Wi, Hi, Wi, Hi]))
    is_inside = torch.logical_and(is_inside, is_inside.roll(-1, dims=-1))
    delta.mul_(is_inside)  # if outside, delta = 0
    # delta = torch.where(is_inside, delta, 0)
    grid.clamp_(torch.tensor([0, 0, 0, 0]), torch.tensor([Wi, Hi, Wi, Hi]) - 1)

    # Expand dimension to (N, C, Ho, Wo, 4)
    delta = delta.unsqueeze(1)  # .expand(-1, C, -1, -1)

    # Get the 4 corners
    x = torch.tensor([0, 2, 2, 0]).view(1, 1, 4).expand(N, Ho * Wo, -1)
    y = torch.tensor([1, 1, 3, 3]).view(1, 1, 4).expand(N, Ho * Wo, -1)
    indices = grid.gather(-1, y) * Wi + grid.gather(-1, x)
    indices = indices.view(N, 1, -1).expand(-1, C, -1)
    return (
        img.gather(-1, indices).view(N, C, -1, 4).mul(delta).sum(-1).view(N, C, Ho, Wo)
    )


class CGEV:
    """Combined Geographic Encoding Volume (CGEV)

    Args:
        fmap1 (torch.Tensor): Feature map from the left image, shape [b, c, h, w1]
        fmap2 (torch.Tensor): Feature map from the right image, shape [b, c, h, w2]
        volume (torch.Tensor): Cost volume, shape [b, c, d, h, w]
        num_levels (int, optional): Number of levels in the pyramid. Defaults to 2.
        radius (int, optional): Radius of the correlation. Defaults to 4.

    Returns:
        torch.Tensor: CGEV, shape [b, c * (2 * radius + 1) * num_levels, h, w]
    """

    def __init__(
        self,
        fmap1: torch.Tensor,
        fmap2: torch.Tensor,
        volume: torch.Tensor,
        num_levels: int = 2,
        radius: int = 4,
    ):
        self.radius = radius

        # all pairs correlation
        corr = torch.einsum("aijk,aijh->ajkh", fmap1, fmap2)
        # [b, h, w1, w2] => [b * h * w1, 1, 1, w2]
        w2 = corr.shape[-1]
        corr = corr.flatten(0, 2).view(-1, 1, 1, w2)

        # [b, c, d, h, w] => [b * h * w, c, 1, d]
        c, d = volume.shape[1:3]
        volume = volume.permute(0, 3, 4, 1, 2).flatten(0, 2).view(-1, c, 1, d)

        self.volume_pyramid = [volume]
        self.corr_pyramid = [corr]
        for i in range(num_levels - 1):
            self.volume_pyramid.append(
                F.avg_pool2d(self.volume_pyramid[i], [1, 2], stride=[1, 2])
            )
            self.corr_pyramid.append(
                F.avg_pool2d(self.corr_pyramid[i], [1, 2], stride=[1, 2])
            )
        # This ensures that the last second dimension (height) for each volume and corr pyramid is 1

    def __call__(self, disp: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """CGEV forward

        Args:
            disp (torch.Tensor): Disparity map, shape [b, 1, h, w]
            coords (torch.Tensor): Coordinates, shape [b, h, w, 1]

        Returns:
            torch.Tensor: CGEV, shape [b, c * (2 * radius + 1) * num_levels, h, w]
        """
        b, _, h, w = disp.shape
        disp = disp.flatten().view(-1, 1, 1, 1)
        coords = coords.flatten().view(-1, 1, 1, 1)
        dx = (
            torch.linspace(-self.radius, self.radius, 2 * self.radius + 1)
            .view(1, 1, -1, 1)
            .to(disp)
        )
        y0 = torch.zeros_like(disp).repeat(
            1, 1, 2 * self.radius + 1, 1
        )  # Only unique value

        pyramid = []
        for i, (volume, corr) in enumerate(zip(self.volume_pyramid, self.corr_pyramid)):
            x0 = 2 * (dx + disp / (1 << i)) / (volume.shape[-1] - 1) - 1
            disp_lvl = torch.cat([x0, y0], dim=-1)
            volume = bilinear_sampler(volume, disp_lvl).view(b, h, w, -1)
            pyramid.append(volume)

            x0 = (
                2 * (coords / (1 << i) - disp / (1 << i) + dx) / (corr.shape[-1] - 1)
                - 1
            )
            coords_lvl = torch.cat([x0, y0], dim=-1)
            corr = bilinear_sampler(corr, coords_lvl).view(b, h, w, -1)
            pyramid.append(corr)
        return torch.cat(pyramid, dim=-1).permute(0, 3, 1, 2)
