# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from isaacsim.core.utils.torch.rotations import quat_rotate
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import  Articulation, ArticulationCfg, RigidObjectCfg, RigidObject, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import subtract_frame_transforms
from torch.utils.tensorboard.writer import SummaryWriter

@configclass
class FrankaPickPlaceEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333 # 5.0 in franka Lift
    decimation = 2
    action_space = 9         # the dimension of the action space for each environment
    observation_space = 30    # the dimension of the observation space from each environment instance
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        # dt=1 / 120,
        dt = 0.01, # Franka lift
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # robot
    # Articulation USD Path: http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/Franka/franka_instanceable.usd
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "panda_finger_joint.*": 0.035,
            },
            pos=(0.5, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    
    # cube
    dexcube = RigidObjectCfg(
        prim_path="/World/envs/env_.*/dexcube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )


    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )


    # markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    frame_marker_cfg.prim_path = "/Visuals/FrameMarker"

    # scaling factors
    action_scale = 7.5
    dof_velocity_scale = 0.1
    
    # reward scales
    dist_reward_scale = 1.0
    target_reward_scale = 50
    lift_reward_scale = 10
    grasp_alignment_scale = 1
    target_alignment_scale = 0
    velocity_penalty_scale = -0.0001 * 10
    action_penalty_scale = -0.0001


class FrankaPickPlaceEnv(DirectRLEnv):

    cfg: FrankaPickPlaceEnvCfg

    def __init__(self, cfg: FrankaPickPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # helper function to get the pose of certain links. From franka_cabinet_env
        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        #  
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
        
        # defining robot limits
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits, device=self.device)
        
        # defining robot grasp
        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_link7")),
            self.device,
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_leftfinger")),
            self.device,
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_rightfinger")),
            self.device,
        )

        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0 # average of two finger positions
        finger_pose[3:7] = lfinger_pose[3:7] # rotation taken as the same as one of the fingers
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        robot_local_grasp_pose_rot, robot_local_grasp_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        robot_local_grasp_pose_pos += torch.tensor([0, 0, 0], device=self.device)
        self.robot_local_grasp_pos = robot_local_grasp_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))
        
        # instantiate finger positions
        self.left_finger_pos  = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_finger_pos = torch.zeros((self.num_envs, 3), device=self.device)
        
        # instantiate grasp pose
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)

        # instantiate hand and cube positions for distance, and direction from hand to cube
        self.cube_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.cube_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.cube_vel = torch.zeros((self.num_envs, 6), device=self.device)
        
        # instantiate target position and rotations
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_rot = torch.zeros((self.num_envs, 4), device=self.device)
        
        
        # defining axes for the alignment
        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.gripper_y_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.cube_z_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.cube_y_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.target_z_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.target_y_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        
        
        # misc
        self.stage_flag = 0
        self.num_envs_towards_target = 0
        self.closed_finger_dist = torch.ones((self.num_envs,1), device=self.device)
        self.writer = SummaryWriter(log_dir='/home/chris/Repositories/SemesterThesis/tensorboard_logs/')
        self.global_step = 0
        
        self.dist_reward_scale = self.cfg.dist_reward_scale
        self.target_reward_scale = self.cfg.target_reward_scale
        self.lift_reward_scale = self.cfg.lift_reward_scale
        self.velocity_penalty_scale = self.cfg.velocity_penalty_scale
        self.action_penalty_scale = self.cfg.action_penalty_scale
        self.grasp_alignment_scale = self.cfg.grasp_alignment_scale
        self.target_alignment_scale = self.cfg.target_alignment_scale

    def _setup_scene(self):
        
        self._robot     = Articulation(self.cfg.robot)
        self._dexcube   = RigidObject(self.cfg.dexcube)
        self.scene.articulations["robot"]   = self._robot
        self.scene.rigid_objects["dexcube"] = self._dexcube
    
        # loading terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        # add markers
        self.ee_marker = VisualizationMarkers(self.cfg.frame_marker_cfg.replace(prim_path="/Visuals/endEffector"))
        self.cube_marker = VisualizationMarkers(self.cfg.frame_marker_cfg.replace(prim_path="Visuals/cube"))
        self.target_marker = VisualizationMarkers(self.cfg.frame_marker_cfg.replace(prim_path="Visuals/target"))

    # pre-physics steps

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0,1.0)

        scaling = self.dt * self.actions * self.cfg.action_scale
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * scaling
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        
    # need to manually set to train or play here.
    # TODO: automatically select train or play
    def _apply_action(self):
        
        task = "play"
        if task == "train":
            self._robot.set_joint_position_target(self.robot_dof_targets)
        elif task == "play":
            cube_target_dist = torch.norm(self.cube_pos - self.target_pos, p=2, dim=1)
            total_velocity = torch.sum(torch.square(self._robot.data.joint_vel), dim=1)
            
            mask = (cube_target_dist < 0.2) & (total_velocity < 2.0)
            env_ids_open_fingers = torch.nonzero(mask).squeeze(-1)
            if env_ids_open_fingers.numel() > 0:
                self._compute_intermediate_values(env_ids_open_fingers)

                finger_joint_indices = self._robot.find_joints("panda_finger_joint.*")[0]
                for idx in finger_joint_indices:
                    self.robot_dof_targets[env_ids_open_fingers, idx] = self.robot_dof_upper_limits[idx]

            self._robot.set_joint_position_target(self.robot_dof_targets)

 
    def _get_observations(self) -> dict:

        grasp_to_cube = self.cube_pos - self.robot_grasp_pos
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel
        # object position in robot root frame
        object_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self.cube_pos
        )
        
        # pos and vel of joints relative to the defaults
        joint_pos_rel = self._robot.data.joint_pos[:, :] - self._robot.data.default_joint_pos[:, :]
        joint_vel_rel = self._robot.data.joint_vel[:, :] - self._robot.data.default_joint_vel[:, :]
        
        choice = "lift"
        if choice == "lift":
            obs = torch.cat([
                joint_pos_rel,              # 9 elements
                joint_vel_rel,              # 9 elements
                object_pos_b,               # 3 elements
                self.actions,               # 9 elements
                ], dim=-1)
            
        elif choice == "OG":
            obs = torch.cat([
                self.robot_grasp_pos,       # 3 elements
                self.cube_pos,              # 3 elements <<
                grasp_to_cube,              # 3 elements <<
                joint_pos,                  # 9 elements <<
                joint_vel                   # 9 elements <<
            ], dim=-1)
        
        # observations = {"policy": torch.clamp(obs, -5.0, 5.0)}
        observations = {"policy": obs}
        return observations


    def _get_rewards(self) -> torch.Tensor:
        total_reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._compute_intermediate_values()   # need to compute intermediate values first
        
        cube_grasp_dist = torch.norm(self.cube_pos - self.robot_grasp_pos, p=2, dim=1)
        cube_target_dist = torch.norm(self.cube_pos - self.target_pos, p=2, dim=1)
        fingers_dist = torch.norm(self.left_finger_pos - self.right_finger_pos, p=2, dim=1)
        
        num_grasps = torch.sum(cube_grasp_dist < 0.1)
        num_lifts = torch.sum(self.cube_pos[:, 2] > 0.04)
        num_reached_target = torch.sum(cube_target_dist < 0.2)
        
        if num_grasps < 0.75 * self.num_envs and self.stage_flag == 0: # stage one: getting hand close
            self._reward_scheduler(1)
            self.stage_flag = 1
            
        elif num_grasps > 0.75 * self.num_envs and self.stage_flag == 1: # stage two: lifting
            self._reward_scheduler(2)
            self.stage_flag = 2
            
        elif num_lifts > 0.75 * self.num_envs and self.stage_flag == 2: # stage three: getting to target
            self._reward_scheduler(3)
            self.stage_flag = 3
        
        # for logging to see progression from one stage to the next
        if self.stage_flag == 1:
            self.num_envs_towards_target = num_grasps
        elif self.stage_flag == 2:
            self.num_envs_towards_target = num_lifts
        elif self.stage_flag == 3:
            self.num_envs_towards_target = num_reached_target

        
        # distance reward: grasp-cube
        dist_reward = 1.0 - torch.tanh(cube_grasp_dist / 0.1)
        total_reward += dist_reward * self.dist_reward_scale
        
        # lifting reward
        lift_reward = torch.where(self.cube_pos[:, 2] > 0.04, 1.0, 0.0) # cube z-position at spawn [0.0240]
        total_reward += lift_reward * self.lift_reward_scale
        
        # distance reward: cube-target
        target_reward = 1.0 - torch.tanh(cube_target_dist / 0.15)
        total_reward += target_reward * self.target_reward_scale
        
        # alignment reward: gripper forward with cube z
        axis1 = tf_vector(self.robot_grasp_rot, self.gripper_forward_axis)
        axis2 = tf_vector(self.cube_rot, self.cube_z_axis)
        dot1 = torch.bmm(axis1.view(self.num_envs, 1, 3), axis2.view(self.num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        grasp_alignment_reward = -1*(torch.sign(dot1) * dot1**2) # multiply by -1 if vectors point away from each other
        total_reward += grasp_alignment_reward * self.grasp_alignment_scale
        
        # alignment reward: gripper forward with target z
        axis3 = tf_vector(self.robot_grasp_rot, self.gripper_forward_axis)
        axis4 = tf_vector(self.target_rot, self.target_z_axis)
        dot2 = torch.bmm(axis3.view(self.num_envs, 1, 3), axis4.view(self.num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        target_alignment_reward = -1*(torch.sign(dot2) * dot2**2) # multiply by -1 if vectors point away from each other
        total_reward += target_alignment_reward * self.target_alignment_scale
        
        # action penalty
        action_penalty = torch.sum(torch.square(self.actions), dim=-1)
        total_reward += action_penalty * self.action_penalty_scale
        
        # velocity penalty
        vel_penalty = torch.sum(torch.square(self._robot.data.joint_vel), dim=1)
        total_reward += vel_penalty * self.velocity_penalty_scale


        # logging rewards
        self.extras["log"] = {
            "dist_reward": (self.dist_reward_scale * dist_reward).mean(),
            "target_reward": (self.target_reward_scale * target_reward).mean(),
            "lifting_reward": (self.lift_reward_scale * lift_reward).mean(),
            "grasp_align_reward": (self.grasp_alignment_scale * grasp_alignment_reward).mean(),
            "target_align_reward": (self.target_alignment_scale * target_alignment_reward).mean(),
            "velocity_penalty": (self.velocity_penalty_scale * vel_penalty).mean(),
            "action_penalty": (self.action_penalty_scale * action_penalty).mean(),
            "stage": (self.stage_flag),
            "num_envs_towards_target": (self.num_envs_towards_target),
            "finger_dist": (fingers_dist).mean(),
        }
        
        # Log rewards to TensorBoard
        current_step = self.global_step
        
        self.writer.add_scalar('Rewards/Distance', (self.dist_reward_scale * dist_reward).mean().item(), current_step)
        self.writer.add_scalar('Rewards/Cube_to_Target', (self.target_reward_scale * target_reward).mean().item(), current_step)
        self.writer.add_scalar('Rewards/Lifting', (self.lift_reward_scale * lift_reward).mean().item(), current_step)
        self.writer.add_scalar('Rewards/Total_reward', (self.num_envs_towards_target), current_step)
        self.writer.add_scalar('Rewards/Grasp_align', (self.grasp_alignment_scale * grasp_alignment_reward).mean().item(), current_step)
        self.writer.add_scalar('Rewards/Target_align', (self.target_alignment_scale * target_alignment_reward).mean().item(), current_step)
        
        self.writer.add_scalar('Penalties/Velocity', (self.velocity_penalty_scale * vel_penalty).mean().item(), current_step)
        self.writer.add_scalar('Penalties/Action', (self.action_penalty_scale * action_penalty).mean().item(), current_step)
        
        self.writer.add_scalar('Properties/Finger_dist', (fingers_dist).mean().item(), current_step)
        self.writer.add_scalar('Properties/Envs_at_target', (self.num_envs_towards_target), current_step)
        self.writer.add_scalar('Properties/Stage', (self.stage_flag), current_step)
        
        self.global_step += 1
        if self.global_step % 100 == 0:
            self.writer.flush()
        
        return total_reward



    def _reward_scheduler(self, stage_idx, env_ids: torch.Tensor | None = None):

        if stage_idx == 1: # getting hand close
            self.dist_reward_scale = 1
            self.target_reward_scale = 0
            self.lift_reward_scale = 0
            self.grasp_alignment_scale = 0.1
            
        elif stage_idx == 2: # lifting
            self.dist_reward_scale = 1
            self.target_reward_scale = 0
            self.lift_reward_scale = 10
            self.grasp_alignment_scale = 0.5
            
        elif stage_idx == 3: # approaching target
            self.dist_reward_scale = 0
            self.target_reward_scale = 40
            self.lift_reward_scale = 0.5
            # self.velocity_penalty_scale *= 10
            self.grasp_alignment_scale = 0
            self.target_alignment_scale = 5
        
        return


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        
        # # checking if cube has fallen over
        # cube_local_z = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))
        # cube_up_world = quat_rotate(self.cube_rot, cube_local_z)  # quat_rotate(q,v)
        # world_z = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))
        # dot_product = torch.sum(cube_up_world * world_z, dim=1)  # Shape: (num_envs,)
        # terminated = dot_product < 0.05
        
        # check if cube is in air
        terminated = self.cube_pos[:,2] > 2.0
        
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        # resetting robot
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125, 0.125, (len(env_ids), self._robot.num_joints),self.device)
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos, device=self.device)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # resetting cube
        cube_init_state = self._dexcube.data.default_root_state[env_ids].clone()
        
        # randomize cube local position
        cube_local_pos = torch.zeros((len(env_ids), 3), device=self.device)
        cube_local_pos[:, :2] += sample_uniform(-0.1, 0.1, (len(env_ids), 2), self.device)

        # cube position relative to its environment origin
        cube_init_state[:, :3] = self.scene.env_origins[env_ids] + cube_local_pos

        # set velocities to zero
        cube_init_state[:, 7:] = 0.0
        self._dexcube.write_root_state_to_sim(cube_init_state, env_ids=env_ids)
        
        # resetting target with random offsets in x,y,z
        target_offsets = sample_uniform(-0.1, 0.1, (len(env_ids), 3), self.device)
        self.target_pos[env_ids] = (
            self.scene.env_origins[env_ids]
            + torch.tensor([0.35, 0.35, 0.4], device=self.device)
            + target_offsets
        )
        self.target_rot[env_ids] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        
        self._compute_intermediate_values(env_ids)     

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]
        
        # tf_combine(rot1, pos1, rot2, pos2)
        # computes the combined transformation by applying the first transformation to the second
        # result is a new rotation and position representing the second frame expressed in the coordinate
        # system of the first frame.
        self.robot_grasp_rot[env_ids], self.robot_grasp_pos[env_ids] = tf_combine(
            hand_rot, hand_pos, self.robot_local_grasp_rot[env_ids], self.robot_local_grasp_pos[env_ids]
            )

        self.cube_pos = self._dexcube.data.body_link_state_w[:, 0, :3]  # (num_envs, 3)
        self.cube_rot = self._dexcube.data.body_link_state_w[:, 0, 3:7]  # (num_envs, 4)
        self.cube_vel = self._dexcube.data.body_link_state_w[:, 0, 7:]  # (num_envs, 6)
        
        self.left_finger_pos[env_ids] = self._robot.data.body_pos_w[env_ids, self.left_finger_link_idx]
        self.right_finger_pos[env_ids] = self._robot.data.body_pos_w[env_ids, self.right_finger_link_idx]
        
        self.ee_marker.visualize(self.robot_grasp_pos, self.robot_grasp_rot)
        self.cube_marker.visualize(self.cube_pos,self.cube_rot)
        self.target_marker.visualize(self.target_pos, self.target_rot)
