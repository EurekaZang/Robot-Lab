# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_inv, quat_apply


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm

"""
MDP terminations.
"""


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    return env.episode_length_buf >= env.max_episode_length


def command_resample(env: ManagerBasedRLEnv, command_name: str, num_resamples: int = 1) -> torch.Tensor:
    """Terminate the episode based on the total number of times commands have been re-sampled.

    This makes the maximum episode length fluid in nature as it depends on how the commands are
    sampled. It is useful in situations where delayed rewards are used :cite:`rudin2022advanced`.
    """
    command: CommandTerm = env.command_manager.get_term(command_name)
    return torch.logical_and((command.time_left <= env.step_dt), (command.command_counter == num_resamples))


"""
Root terminations.
"""


def bad_orientation(
    env: ManagerBasedRLEnv, limit_angle: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's orientation is too far from the desired orientation limits.

    This is computed by checking the angle between the projected gravity vector and the z-axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() > limit_angle


def root_height_below_minimum(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height


"""
Joint terminations.
"""


def joint_pos_out_of_limit(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the asset's joint positions are outside of the soft joint limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids is None:
        asset_cfg.joint_ids = slice(None)

    limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    out_of_upper_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] > limits[..., 1], dim=1)
    out_of_lower_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] < limits[..., 0], dim=1)
    return torch.logical_or(out_of_upper_limits, out_of_lower_limits)


def joint_pos_out_of_manual_limit(
    env: ManagerBasedRLEnv, bounds: tuple[float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's joint positions are outside of the configured bounds.

    Note:
        This function is similar to :func:`joint_pos_out_of_limit` but allows the user to specify the bounds manually.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids is None:
        asset_cfg.joint_ids = slice(None)
    # compute any violations
    out_of_upper_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] > bounds[1], dim=1)
    out_of_lower_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] < bounds[0], dim=1)
    return torch.logical_or(out_of_upper_limits, out_of_lower_limits)


def joint_vel_out_of_limit(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the asset's joint velocities are outside of the soft joint limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute any violations
    limits = asset.data.soft_joint_vel_limits
    return torch.any(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) > limits[:, asset_cfg.joint_ids], dim=1)


def joint_vel_out_of_manual_limit(
    env: ManagerBasedRLEnv, max_velocity: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's joint velocities are outside the provided limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute any violations
    return torch.any(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) > max_velocity, dim=1)


def joint_effort_out_of_limit(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when effort applied on the asset's joints are outside of the soft joint limits.

    In the actuators, the applied torque are the efforts applied on the joints. These are computed by clipping
    the computed torques to the joint limits. Hence, we check if the computed torques are equal to the applied
    torques.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # check if any joint effort is out of limit
    out_of_limits = torch.isclose(
        asset.data.computed_torque[:, asset_cfg.joint_ids], asset.data.applied_torque[:, asset_cfg.joint_ids]
    )
    return torch.any(out_of_limits, dim=1)


"""
Contact sensor.
"""


def illegal_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    )


def too_close(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Determines if the robot is too close to any obstacle based on LiDAR data.

    This function processes the flattened LiDAR observation to check for critical proximity.
    It identifies elements corresponding to distance measurements (which, in a 0-indexed
    flattened vector where elements are [sin, cos, dist], are located at indices
    2, 5, 8, ..., up to the last element). If any of these distance measurements
    falls below a predefined threshold (0.1 meters), it indicates the robot is
    too close to an obstacle.

    Args:
        env (ManagerBasedRLEnv): The reinforcement learning environment instance.
                                 (This argument is included to maintain the function
                                 signature consistent with Isaac Lab's termination
                                 function requirements; its properties are not directly
                                 used within this function's logic.)
        lidar_data (torch.Tensor): A tensor of shape (num_envs, 1080) containing
                                   the flattened LiDAR observation data.
                                   Expected distance values are at 0-indexed positions
                                   2, 5, 8, ..., 1079.
                                   (Based on the previous description, the data is
                                   arranged as [sin(alpha), cos(alpha), distance],
                                   so distance is the third element in each triplet.)

    Returns:
        torch.Tensor: A boolean tensor of shape (num_envs,).
                      Returns `False` for an environment if the robot is
                      "too close" (i.e., the termination condition is met).
                      Returns `True` if the robot is NOT "too close" (i.e.,
                      the task should continue).
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    hit_points_w = sensor.data.ray_hits_w
    sensor_pos_w = sensor.data.pos_w
    sensor_quat_w = sensor.data.quat_w

    num_envs = hit_points_w.shape[0]
    num_points = hit_points_w.shape[1]

    relative_points_w = hit_points_w - sensor_pos_w.unsqueeze(1)
    sensor_quat_inv_w = quat_inv(sensor_quat_w)
    
    # Reshape for batch processing
    # Flatten the points to (num_envs * num_points, 3)
    relative_points_flat = relative_points_w.view(-1, 3)
    # Repeat quaternions for each point
    sensor_quat_inv_repeated = sensor_quat_inv_w.repeat_interleave(num_points, dim=0)
    
    # Apply quaternion rotation
    points_local_flat = quat_apply(sensor_quat_inv_repeated, relative_points_flat)
    # Reshape back to (num_envs, num_points, 3)
    points_local = points_local_flat.view(num_envs, num_points, 3)

    distances = torch.linalg.norm(points_local, dim=-1)

    angles_rad = torch.atan2(points_local[..., 1], points_local[..., 0])

    angles_deg = torch.rad2deg(angles_rad)
    bin_indices = ((angles_deg % 360) + 360) % 360
    bin_indices = bin_indices.long()
    # Initialize the final scan tensor with the maximum distance
    final_scan = torch.full(
        (num_envs, 360), fill_value=sensor.cfg.max_distance, device=env.device, dtype=torch.float32
    )
    # Iterate over each environment and update the final scan
    for env_idx in range(num_envs):
        env_distances = distances[env_idx]
        env_bin_indices = bin_indices[env_idx]
        # Filter out invalid distance values
        valid_mask = torch.isfinite(env_distances) & (env_distances > 0)
        if valid_mask.any():
            valid_distances = env_distances[valid_mask]
            valid_bins = env_bin_indices[valid_mask]
            # Use scatter_reduce to update the minimum distance
            final_scan[env_idx].scatter_reduce_(
                dim=0,
                index=valid_bins,
                src=valid_distances,
                reduce="amin",
                include_self=False
            )
    scan_angles_deg = torch.arange(360, device=env.device, dtype=torch.float32)
    scan_angles_rad = torch.deg2rad(scan_angles_deg)

    sin_alpha = torch.sin(scan_angles_rad)
    cos_alpha = torch.cos(scan_angles_rad)

    sin_alpha_b = sin_alpha.expand(num_envs, -1)
    cos_alpha_b = cos_alpha.expand(num_envs, -1)

    observation = torch.stack([sin_alpha_b, cos_alpha_b, final_scan], dim=-1)
    flattened_observation = torch.flatten(observation, start_dim=1)
    TOO_CLOSE_THRESHOLD = 0.2 # meters

    distances = flattened_observation[:, 2::3]
    is_too_close_per_bin = distances < TOO_CLOSE_THRESHOLD
    any_bin_too_close_per_env = torch.any(is_too_close_per_bin, dim=1)
    return any_bin_too_close_per_env
