# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass

import numpy as np
import torch

from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from worldfoundry.core.geometry.transforms import closed_form_inverse_se3 as closed_form_inverse_se3


@dataclass
class FrameDiskCache:
    root_dir: str
    compress: bool = False

    def __post_init__(self):
        os.makedirs(self.root_dir, exist_ok=True)

    def __call__(self, frame_idx: int, frame: dict, pred_cpu: dict):
        def clone_tensor_map(src: dict) -> dict:
            cloned = {}
            for key, value in src.items():
                if isinstance(value, torch.Tensor):
                    cloned[key] = value.detach().clone().cpu()
                else:
                    cloned[key] = value
            return cloned

        view_payload = {
            key: tensor.detach().clone().cpu() for key, tensor in frame.items() if isinstance(tensor, torch.Tensor)
        }

        pred_payload = clone_tensor_map(pred_cpu)

        meta = {"frame_idx": int(frame_idx)}

        camera_pose_enc = pred_payload.get("camera_pose")
        img_tensor = view_payload.get("img")

        if isinstance(camera_pose_enc, torch.Tensor) and isinstance(img_tensor, torch.Tensor):
            B = camera_pose_enc.shape[0]
            H, W = img_tensor.shape[-2:]

            pose_enc = camera_pose_enc.unsqueeze(1)
            extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, (H, W))

            extrinsics = extrinsics[:, 0]  # (B, 3, 4)
            extrinsics_h = torch.eye(4, dtype=extrinsics.dtype, device=extrinsics.device).repeat(B, 1, 1)
            extrinsics_h[:, :3, :4] = extrinsics

            cam_to_world = closed_form_inverse_se3(extrinsics_h)

            view_payload["camera_pose"] = cam_to_world
            view_payload["world_to_cam"] = extrinsics_h

            pred_payload["extrinsic_world_to_cam"] = extrinsics_h

            if intrinsics is not None:
                intrinsics = intrinsics[:, 0]
                view_payload["intrinsic"] = intrinsics
                pred_payload["intrinsic"] = intrinsics

            meta["image_hw"] = torch.tensor([H, W], dtype=torch.int32)

        payload = {
            "pred": pred_payload,
            "view": view_payload,
            "meta": meta,
        }

        file_name = f"{frame_idx:06d}.pt"
        file_path = os.path.join(self.root_dir, file_name)
        torch.save(payload, file_path, pickle_protocol=4)


def unproject_depth_map_to_point_map(
    depth_map: np.ndarray, extrinsics_cam: np.ndarray, intrinsics_cam: np.ndarray
) -> np.ndarray:
    """
    Unproject a batch of depth maps to 3D world coordinates.

    Args:
        depth_map (np.ndarray): Batch of depth maps of shape (S, H, W, 1) or (S, H, W)
        extrinsics_cam (np.ndarray): Batch of camera extrinsic matrices of shape (S, 3, 4)
        intrinsics_cam (np.ndarray): Batch of camera intrinsic matrices of shape (S, 3, 3)

    Returns:
        np.ndarray: Batch of 3D world coordinates of shape (S, H, W, 3)
    """
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()
    if isinstance(extrinsics_cam, torch.Tensor):
        extrinsics_cam = extrinsics_cam.cpu().numpy()
    if isinstance(intrinsics_cam, torch.Tensor):
        intrinsics_cam = intrinsics_cam.cpu().numpy()

    world_points_list = []
    for frame_idx in range(depth_map.shape[0]):
        cur_world_points, _, _ = depth_to_world_coords_points(
            depth_map[frame_idx].squeeze(-1), extrinsics_cam[frame_idx], intrinsics_cam[frame_idx]
        )
        world_points_list.append(cur_world_points)
    world_points_array = np.stack(world_points_list, axis=0)

    return world_points_array


def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    eps=1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a depth map to world coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).
        extrinsic (np.ndarray): Camera extrinsic matrix of shape (3, 4). OpenCV camera coordinate convention, cam from world.

    Returns:
        tuple[np.ndarray, np.ndarray]: World coordinates (H, W, 3) and valid depth mask (H, W).
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # Multiply with the inverse of extrinsic matrix to transform to world coordinates
    # extrinsic_inv is 4x4 (note closed_form_inverse_OpenCV is batched, the output is (N, 4, 4))
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]

    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world  # HxWx3, 3x3 -> HxWx3
    # world_coords_points = np.einsum("ij,hwj->hwi", R_cam_to_world, cam_coords_points) + t_cam_to_world

    return world_coords_points, cam_coords_points, point_mask


def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a depth map to camera coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: Camera coordinates (H, W, 3)
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    return cam_coords
