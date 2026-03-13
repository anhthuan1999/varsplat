""" Camera tracking with uncertainty-aware weighting (Section 3.3). """
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from scipy.spatial.transform import Rotation as R
import time
from src.entities.arguments import OptimizationParams
from src.entities.losses import l1_loss
from src.entities.gaussian_model import GaussianModel
from src.entities.logger import Logger
from src.entities.datasets import BaseDataset
from src.entities.visual_odometer import VisualOdometer
from src.utils.gaussian_model_utils import build_rotation
from src.utils.tracker_utils import (compute_camera_opt_params,
                                     extrapolate_poses, multiply_quaternions,
                                     transformation_to_quaternion)
from src.utils.utils import (get_render_settings, np2torch,
                             render_gaussian_model, torch2np)
from src.entities.losses import *

eps = 1e-6

class Tracker(object):
    def __init__(self, config: dict, dataset: BaseDataset, logger: Logger, config_var: dict) -> None:
        """ Initializes the Tracker with a given configuration, dataset, and logger.
        Args:
            config: Configuration dictionary specifying hyperparameters and operational settings.
            dataset: The dataset object providing access to the sequence of frames.
            logger: Logger object for logging the tracking process.
        """
        self.dataset = dataset
        self.logger = logger
        self.config = config
        self.filter_alpha = self.config["filter_alpha"]
        self.filter_outlier_depth = self.config["filter_outlier_depth"]
        self.alpha_thre = self.config["alpha_thre"]
        self.soft_alpha = self.config["soft_alpha"]
        self.mask_invalid_depth_in_color_loss = self.config["mask_invalid_depth"]
        self.w_color_loss = self.config["w_color_loss"]
        self.transform = torchvision.transforms.ToTensor()
        self.opt = OptimizationParams(ArgumentParser(description="Training script parameters"))
        self.frame_depth_loss = []
        self.frame_color_loss = []
        self.odometry_type = self.config["odometry_type"]
        self.help_camera_initialization = self.config["help_camera_initialization"]
        self.init_err_ratio = self.config["init_err_ratio"]
        self.enable_exposure = self.config["enable_exposure"]
        self.odometer = VisualOdometer(self.dataset.intrinsics, self.config["odometer_method"])

        self.tau_track = config_var["tau_track"]
        self.w_limit = config_var["w_limit"]

        # Initialize log file
        log_file_path = Path(self.logger.output_path) / "tracker.log"
        self.log_file = open(log_file_path, "a", buffering=1)  # Line buffered

    def _log(self, message: str, print_to_console: bool = True) -> None:
        """Write message to log file and optionally print to console.
        Also writes to main timing log if available.
        Args:
            message: The message to log.
            print_to_console: Whether to also print to console.
        """
        # Write to tracker-specific log file
        self.log_file.write(message + "\n")
        self.log_file.flush()
        # Also write to main timing log via logger (log_and_print adds its own newline)
        if hasattr(self.logger, 'log_and_print'):
            self.logger.log_and_print(message)
        elif print_to_console:
            print(message)
    
    def __del__(self):
        """Close log file when tracker is destroyed."""
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()

    def compute_losses(self, gaussian_model: GaussianModel, render_settings: dict,
                       opt_cam_rot: torch.Tensor, opt_cam_trans: torch.Tensor,
                       gt_color: torch.Tensor, gt_depth: torch.Tensor, depth_mask: torch.Tensor,
                       exposure_ab=None) -> tuple:
        """ Computes the tracking losses with respect to ground truth color and depth.
        Args:
            gaussian_model: The current state of the Gaussian model of the scene.
            render_settings: Dictionary containing rendering settings such as image dimensions and camera intrinsics.
            opt_cam_rot: Optimizable tensor representing the camera's rotation.
            opt_cam_trans: Optimizable tensor representing the camera's translation.
            gt_color: Ground truth color image tensor.
            gt_depth: Ground truth depth image tensor.
            depth_mask: Binary mask indicating valid depth values in the ground truth depth image.
        Returns:
            A tuple containing losses and renders
        """
        rel_transform = torch.eye(4).cuda().float()
        rel_transform[:3, :3] = build_rotation(F.normalize(opt_cam_rot[None]))[0]
        rel_transform[:3, 3] = opt_cam_trans

        pts = gaussian_model.get_xyz()
        pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
        pts4 = torch.cat((pts, pts_ones), dim=1)
        transformed_pts = (rel_transform @ pts4.T).T[:, :3]

        quat = F.normalize(opt_cam_rot[None])
        _rotations = multiply_quaternions(gaussian_model.get_rotation(), quat.unsqueeze(0)).squeeze(0)

        render_dict = render_gaussian_model(gaussian_model, render_settings,
                                            override_means_3d=transformed_pts, override_rotations=_rotations)
        rendered_color, rendered_depth = render_dict["color"], render_dict["depth"]
        rendered_variance = render_dict.get("variance", None)
        if self.enable_exposure:
            rendered_color = torch.clamp(torch.exp(exposure_ab[0]) * rendered_color + exposure_ab[1], 0, 1.)
        alpha_mask = render_dict["alpha"] > self.alpha_thre

        tracking_mask = torch.ones_like(alpha_mask).bool()
        tracking_mask &= depth_mask
        depth_err = torch.abs(rendered_depth - gt_depth) * depth_mask

        if self.filter_alpha:
            tracking_mask &= alpha_mask
        if self.filter_outlier_depth and torch.median(depth_err) > 0:
            tracking_mask &= depth_err < 50 * torch.median(depth_err)

        color_loss = l1_loss(rendered_color, gt_color, agg="none")
        depth_loss = l1_loss(rendered_depth, gt_depth, agg="none") * tracking_mask

        if self.soft_alpha:
            alpha = render_dict["alpha"] ** 3
            color_loss *= alpha
            depth_loss *= alpha
            if self.mask_invalid_depth_in_color_loss:
                color_loss *= tracking_mask
        else:
            color_loss *= tracking_mask


        v = rendered_variance.detach()
        m = tracking_mask[0]
        logv = torch.log(v + eps)
        mu = torch.stack([logv[c][m].median() for c in range(3)])
        mu = mu.view(3, 1, 1)
        w = torch.exp(-(logv - mu) / self.tau_track)
        w_rgb = torch.clamp(w, min=1 / self.w_limit, max=self.w_limit)

        color_loss = (w_rgb * color_loss * tracking_mask)


        color_loss = color_loss.sum()
        depth_loss = depth_loss.sum()

        return color_loss, depth_loss, rendered_color, rendered_depth, alpha_mask

    def track(self, frame_id: int, gaussian_model: GaussianModel, prev_c2ws: np.ndarray) -> np.ndarray:
        """
        Updates the camera pose estimation for the current frame based on the provided image and depth, using either ground truth poses,
        constant speed assumption, or visual odometry.
        Args:
            frame_id: Index of the current frame being processed.
            gaussian_model: The current Gaussian model of the scene.
            prev_c2ws: Array containing the camera-to-world transformation matrices for the frames (0, i - 2, i - 1)
        Returns:
            The updated camera-to-world transformation matrix for the current frame.
        """
        _, image, depth, gt_c2w = self.dataset[frame_id]

        if (self.help_camera_initialization or self.odometry_type == "odometer") and self.odometer.last_rgbd is None:
            _, last_image, last_depth, _ = self.dataset[frame_id - 1]
            self.odometer.update_last_rgbd(last_image, last_depth)

        if self.odometry_type == "gt":
            return gt_c2w
        elif self.odometry_type == "const_speed":
            init_c2w = extrapolate_poses(prev_c2ws[1:])
        elif self.odometry_type == "odometer":
            odometer_rel = self.odometer.estimate_rel_pose(image, depth)
            init_c2w = prev_c2ws[-1] @ odometer_rel
        elif self.odometry_type == "previous":
            init_c2w = prev_c2ws[-1]

        last_c2w = prev_c2ws[-1]
        last_w2c = np.linalg.inv(last_c2w)
        init_rel = init_c2w @ np.linalg.inv(last_c2w)
        init_rel_w2c = np.linalg.inv(init_rel)
        reference_w2c = last_w2c
        render_settings = get_render_settings(
            self.dataset.width, self.dataset.height, self.dataset.intrinsics, reference_w2c)
        opt_cam_rot, opt_cam_trans = compute_camera_opt_params(init_rel_w2c)
        if self.enable_exposure:
            exposure_ab = torch.nn.Parameter(torch.tensor(
                0.0, device="cuda")), torch.nn.Parameter(torch.tensor(0.0, device="cuda"))
        else:
            exposure_ab = None
        gaussian_model.training_setup_camera(opt_cam_rot, opt_cam_trans, self.config, exposure_ab)

        gt_color = self.transform(image).cuda()
        gt_depth = np2torch(depth, "cuda")
        depth_mask = gt_depth > 0.0
        gt_trans = np2torch(gt_c2w[:3, 3])
        gt_quat = np2torch(R.from_matrix(gt_c2w[:3, :3]).as_quat(canonical=True)[[3, 0, 1, 2]])
        num_iters = self.config["iterations"]
        current_min_loss = float("inf")

        self._log(f"\nTracking frame {frame_id}")
        # Initial loss check
        color_loss, depth_loss, _, _, _ = self.compute_losses(gaussian_model, render_settings, opt_cam_rot,
                                                              opt_cam_trans, gt_color, gt_depth, depth_mask, 
                                                              exposure_ab)
        if len(self.frame_color_loss) > 0 and (
            color_loss.item() > self.init_err_ratio * np.median(self.frame_color_loss)
            or depth_loss.item() > self.init_err_ratio * np.median(self.frame_depth_loss)
        ):
            num_iters *= 2
            self._log(f"Higher initial loss, increasing num_iters to {num_iters}")
            if self.help_camera_initialization and self.odometry_type != "odometer":
                _, last_image, last_depth, _ = self.dataset[frame_id - 1]
                self.odometer.update_last_rgbd(last_image, last_depth)
                odometer_rel = self.odometer.estimate_rel_pose(image, depth)
                init_c2w = last_c2w @ odometer_rel
                init_rel = init_c2w @ np.linalg.inv(last_c2w)
                init_rel_w2c = np.linalg.inv(init_rel)
                opt_cam_rot, opt_cam_trans = compute_camera_opt_params(init_rel_w2c)
                gaussian_model.training_setup_camera(opt_cam_rot, opt_cam_trans, self.config, exposure_ab)
                render_settings = get_render_settings(
                    self.dataset.width, self.dataset.height, self.dataset.intrinsics, last_w2c)
                self._log(f"re-init with odometer for frame {frame_id}")

        start_time = time.time()
        for iter in range(num_iters):
            color_loss, depth_loss, _, _, _, = self.compute_losses(
                gaussian_model, render_settings, opt_cam_rot, opt_cam_trans, gt_color, gt_depth, depth_mask, exposure_ab)

            total_loss = (self.w_color_loss * color_loss + (1 - self.w_color_loss) * depth_loss)
            total_loss.backward()
            gaussian_model.optimizer.step()
            # gaussian_model.scheduler.step(total_loss, epoch=iter)
            gaussian_model.optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                if total_loss.item() < current_min_loss:
                    current_min_loss = total_loss.item()
                    best_w2c = torch.eye(4)
                    best_w2c[:3, :3] = build_rotation(F.normalize(opt_cam_rot[None].clone().detach().cpu()))[0]
                    best_w2c[:3, 3] = opt_cam_trans.clone().detach().cpu()

                cur_quat, cur_trans = F.normalize(opt_cam_rot[None].clone().detach()), opt_cam_trans.clone().detach()
                cur_rel_w2c = torch.eye(4)
                cur_rel_w2c[:3, :3] = build_rotation(cur_quat)[0]
                cur_rel_w2c[:3, 3] = cur_trans
                if iter == num_iters - 1:
                    cur_w2c = torch.from_numpy(reference_w2c) @ best_w2c
                else:
                    cur_w2c = torch.from_numpy(reference_w2c) @ cur_rel_w2c
                cur_c2w = torch.inverse(cur_w2c)
                cur_cam = transformation_to_quaternion(cur_c2w)
                if (gt_quat * cur_cam[:4]).sum() < 0:  # for logging purpose
                    gt_quat *= -1
                if iter == num_iters - 1:
                    self.frame_color_loss.append(color_loss.item())
                    self.frame_depth_loss.append(depth_loss.item())
                    self.logger.log_tracking_iteration(
                        frame_id, cur_cam, gt_quat, gt_trans, total_loss, color_loss, depth_loss, iter, num_iters,
                        wandb_output=True, print_output=True)
                elif iter % 20 == 0:
                    self.logger.log_tracking_iteration(
                        frame_id, cur_cam, gt_quat, gt_trans, total_loss, color_loss, depth_loss, iter, num_iters,
                        wandb_output=False, print_output=True)

        end_time = time.time()
        self._log(f"Tracking time for frame {frame_id}: {end_time - start_time} seconds")
        self._log(f"Average tracking time iteration: {(end_time - start_time) / num_iters} seconds")

        final_c2w = torch.inverse(torch.from_numpy(reference_w2c) @ best_w2c)
        final_c2w[-1, :] = torch.tensor([0., 0., 0., 1.], dtype=final_c2w.dtype, device=final_c2w.device)
        return torch2np(final_c2w), exposure_ab
