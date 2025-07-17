"""
Cartpole with PPO_CYCLE_DEBUG logging for IsaacGymEnvs
This is a modified version of the Cartpole task with debug logging
to compare with DNNE exported code behavior.
"""

import numpy as np
import os
import torch
from typing import Tuple, Dict

from isaacgym import gymutil, gymtorch
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask


class Cartpole(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.reset_dist = self.cfg["env"]["resetDist"]

        self.max_push_effort = self.cfg["env"]["maxEffort"]
        self.max_episode_length = 500

        self.cfg["env"]["numObservations"] = 4
        self.cfg["env"]["numActions"] = 1
        
        # Enable PPO_CYCLE_DEBUG logging if set
        self.ppo_cycle_debug = os.environ.get('PPO_CYCLE_DEBUG', '0') == '1'
        if self.ppo_cycle_debug:
            print(f"[PPO_CYCLE_DEBUG] Cartpole.__init__: num_envs={cfg['env']['numEnvs']}, device={rl_device}")

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        
        if self.ppo_cycle_debug:
            print(f"[PPO_CYCLE_DEBUG] Cartpole initialized: DOFs={self.num_dof}")

    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis = self.cfg["sim"]["up_axis"]

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/cartpole.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True

        cartpole_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(cartpole_asset)

        pose = gymapi.Transform()
        if self.up_axis == 'z':
            pose.p.z = 2.0
            # asset is rotated from y-up to z-up
            pose.r = gymapi.Quat.from_euler_zyx(0.0, np.pi/2, 0.0)
        else:
            pose.p.y = 2.0

        self.cartpole_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            cartpole_handle = self.gym.create_actor(env_ptr, cartpole_asset, pose, "cartpole", i, 1, 0)

            dof_props = self.gym.get_actor_dof_properties(env_ptr, cartpole_handle)
            dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
            dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE
            dof_props['stiffness'][:] = 0.0
            dof_props['damping'][:] = 0.0
            self.gym.set_actor_dof_properties(env_ptr, cartpole_handle, dof_props)

            self.envs.append(env_ptr)
            self.cartpole_handles.append(cartpole_handle)

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)

        self.obs_buf[env_ids, 0] = self.dof_pos[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 1] = self.dof_vel[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 2] = self.dof_pos[env_ids, 1].squeeze()
        self.obs_buf[env_ids, 3] = self.dof_vel[env_ids, 1].squeeze()

        return self.obs_buf

    def reset_idx(self, env_ids):
        positions = 0.2 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)
        velocities = 0.5 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)
        
        if self.ppo_cycle_debug and len(env_ids) > 0:
            print(f"[PPO_CYCLE_DEBUG] Cartpole.reset_idx: Resetting {len(env_ids)} environments")
            print(f"[PPO_CYCLE_DEBUG] Reset positions: min={positions.min().item():.4f}, max={positions.max().item():.4f}")
            print(f"[PPO_CYCLE_DEBUG] Reset velocities: min={velocities.min().item():.4f}, max={velocities.max().item():.4f}")

        self.dof_pos[env_ids, :] = positions[:]
        self.dof_vel[env_ids, :] = velocities[:]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                               gymtorch.unwrap_tensor(self.dof_state),
                                               gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        if self.ppo_cycle_debug and self.progress_buf[0] < 5:  # Log first 5 steps
            print(f"[PPO_CYCLE_DEBUG] Cartpole.pre_physics_step: action shape={actions.shape}")
            print(f"[PPO_CYCLE_DEBUG] Actions: min={actions.min().item():.4f}, max={actions.max().item():.4f}, mean={actions.mean().item():.4f}")
            
        actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)
        actions_tensor[::self.num_dof] = actions.to(self.device).squeeze() * self.max_push_effort
        forces = gymtorch.unwrap_tensor(actions_tensor)
        self.gym.set_dof_actuation_force_tensor(self.sim, forces)

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)
        
        if self.ppo_cycle_debug and self.progress_buf[0] <= 2:  # Log first couple steps
            print(f"[PPO_CYCLE_DEBUG] Cartpole.post_physics_step: progress={self.progress_buf[0]}")
            print(f"[PPO_CYCLE_DEBUG] Observations shape: {self.obs_buf.shape}")
            print(f"[PPO_CYCLE_DEBUG] Obs: min={self.obs_buf.min().item():.4f}, max={self.obs_buf.max().item():.4f}, mean={self.obs_buf.mean().item():.4f}")
            print(f"[PPO_CYCLE_DEBUG] Rewards: min={self.rew_buf.min().item():.4f}, max={self.rew_buf.max().item():.4f}, mean={self.rew_buf.mean().item():.4f}")
            print(f"[PPO_CYCLE_DEBUG] Dones: {self.reset_buf.sum().item()} environments done")

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_cartpole_reward(
            self.obs_buf, self.reset_buf, self.progress_buf, self.max_episode_length,
            self.reset_dist
        )


@torch.jit.script
def compute_cartpole_reward(obs_buf, reset_buf, progress_buf, max_episode_length: float, reset_dist: float):
    # type: (Tensor, Tensor, Tensor, float, float) -> Tuple[Tensor, Tensor]

    # reward is combo of angle deviated from upright, velocity of cart, and velocity of pole moving
    reward = 1.0 - obs_buf[:, 2] * obs_buf[:, 2] - 0.01 * torch.abs(obs_buf[:, 1]) - 0.005 * torch.abs(obs_buf[:, 3])

    # adjust reward for reset agents
    reward = torch.where(torch.abs(obs_buf[:, 0]) > reset_dist, torch.ones_like(reward) * -2.0, reward)
    reward = torch.where(torch.abs(obs_buf[:, 2]) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)

    reset = torch.where(torch.abs(obs_buf[:, 0]) > reset_dist, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(torch.abs(obs_buf[:, 2]) > np.pi / 2, torch.ones_like(reset_buf), reset)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    return reward, reset