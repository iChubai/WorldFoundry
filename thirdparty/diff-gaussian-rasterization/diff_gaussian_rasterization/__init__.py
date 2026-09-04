#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
    return_extra_outputs=False,
):
    rasterize_function = (
        _RasterizeGaussiansWithAux
        if return_extra_outputs
        else _RasterizeGaussians
    )
    return rasterize_function.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

def _rasterize_gaussians_forward(
    ctx,
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    # Restructure arguments the way that the C++ lib expects them
    args = (
        raster_settings.bg,
        means3D,
        colors_precomp,
        opacities,
        scales,
        rotations,
        raster_settings.scale_modifier,
        cov3Ds_precomp,
        raster_settings.viewmatrix,
        raster_settings.projmatrix,
        raster_settings.tanfovx,
        raster_settings.tanfovy,
        raster_settings.image_height,
        raster_settings.image_width,
        sh,
        raster_settings.sh_degree,
        raster_settings.campos,
        raster_settings.prefiltered,
        raster_settings.debug,
    )

    # The extension always returns auxiliary buffers; the public API decides
    # whether to expose them.
    if raster_settings.debug:
        cpu_args = cpu_deep_copy_tuple(args)  # Copy them before corruption
        try:
            outputs = _C.rasterize_gaussians(*args)
        except Exception as ex:
            torch.save(cpu_args, "snapshot_fw.dump")
            print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
            raise ex
    else:
        outputs = _C.rasterize_gaussians(*args)

    (
        num_rendered,
        color,
        depth,
        median_depth,
        final_opacity,
        radii,
        geomBuffer,
        binningBuffer,
        imgBuffer,
    ) = outputs

    # median_depth and final_opacity are forward-only diagnostics and do not
    # alter the canonical color/depth backward implementation.
    ctx.raster_settings = raster_settings
    ctx.num_rendered = num_rendered
    ctx.save_for_backward(
        colors_precomp,
        means3D,
        scales,
        rotations,
        cov3Ds_precomp,
        radii,
        sh,
        geomBuffer,
        binningBuffer,
        imgBuffer,
    )
    return color, radii, depth, median_depth, final_opacity


def _rasterize_gaussians_backward(ctx, grad_out_color, grad_depth):
    num_rendered = ctx.num_rendered
    raster_settings = ctx.raster_settings
    (
        colors_precomp,
        means3D,
        scales,
        rotations,
        cov3Ds_precomp,
        radii,
        sh,
        geomBuffer,
        binningBuffer,
        imgBuffer,
    ) = ctx.saved_tensors

    args = (
        raster_settings.bg,
        means3D,
        radii,
        colors_precomp,
        scales,
        rotations,
        raster_settings.scale_modifier,
        cov3Ds_precomp,
        raster_settings.viewmatrix,
        raster_settings.projmatrix,
        raster_settings.tanfovx,
        raster_settings.tanfovy,
        grad_out_color,
        grad_depth,
        sh,
        raster_settings.sh_degree,
        raster_settings.campos,
        geomBuffer,
        num_rendered,
        binningBuffer,
        imgBuffer,
        raster_settings.debug,
    )

    if raster_settings.debug:
        cpu_args = cpu_deep_copy_tuple(args)  # Copy them before corruption
        try:
            gradients = _C.rasterize_gaussians_backward(*args)
        except Exception as ex:
            torch.save(cpu_args, "snapshot_bw.dump")
            print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
            raise ex
    else:
        gradients = _C.rasterize_gaussians_backward(*args)

    (
        grad_means2D,
        grad_colors_precomp,
        grad_opacities,
        grad_means3D,
        grad_cov3Ds_precomp,
        grad_sh,
        grad_scales,
        grad_rotations,
    ) = gradients

    return (
        grad_means3D,
        grad_means2D,
        grad_sh,
        grad_colors_precomp,
        grad_opacities,
        grad_scales,
        grad_rotations,
        grad_cov3Ds_precomp,
        None,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        color, radii, depth, _, _ = _rasterize_gaussians_forward(ctx, *args)
        return color, radii, depth

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, grad_depth):
        return _rasterize_gaussians_backward(ctx, grad_out_color, grad_depth)


class _RasterizeGaussiansWithAux(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        return _rasterize_gaussians_forward(ctx, *args)

    @staticmethod
    def backward(
        ctx,
        grad_out_color,
        grad_radii,
        grad_depth,
        grad_median_depth,
        grad_final_opacity,
    ):
        # Auxiliary-output gradients are deliberately ignored. The canonical
        # color/depth gradient path remains unchanged.
        return _rasterize_gaussians_backward(ctx, grad_out_color, grad_depth)

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings, return_extra_outputs=False):
        super().__init__()
        self.raster_settings = raster_settings
        self.return_extra_outputs = return_extra_outputs

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings,
            return_extra_outputs=self.return_extra_outputs,
        )
