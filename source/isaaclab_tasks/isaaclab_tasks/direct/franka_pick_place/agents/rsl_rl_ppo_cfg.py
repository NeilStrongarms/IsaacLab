# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class FrankaPickPlacePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "franka_pick_place_direct"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
)

	



# # Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# from omni.isaac.lab.utils import configclass

# from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
#     RslRlOnPolicyRunnerCfg,
#     RslRlPpoActorCriticCfg,
#     RslRlPpoAlgorithmCfg,
# )


# @configclass
# class FrankaPickPlacePPORunnerCfg(RslRlOnPolicyRunnerCfg):
#     num_steps_per_env = 16
#     max_iterations = 1500
#     save_interval = 50
#     experiment_name = "franka_pick_place_direct"
#     empirical_normalization = True
#     policy = RslRlPpoActorCriticCfg(
#         init_noise_std=1.0,
#         actor_hidden_dims=[256, 128, 64],
#         critic_hidden_dims=[256, 128, 64],
#         activation="elu",
#     )
#     algorithm = RslRlPpoAlgorithmCfg(
#         value_loss_coef=1.0,
#         use_clipped_value_loss=True,
#         clip_param=0.2,
#         entropy_coef=0.0,
#         num_learning_epochs=8,
#         num_mini_batches=8,
#         learning_rate=5.0e-4,
#         schedule="adaptive",
#         gamma=0.99,
#         lam=0.95,
#         desired_kl=0.008,
#         max_grad_norm=1.0,
#     )


# from omni.isaac.lab.utils import configclass

# from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
#     RslRlOnPolicyRunnerCfg,
#     RslRlPpoActorCriticCfg,
#     RslRlPpoAlgorithmCfg,
# )


# @configclass
# class FrankaPickPlacePPORunnerCfg(RslRlOnPolicyRunnerCfg):
#     num_steps_per_env = 16
#     max_iterations = 2000
#     save_interval = 100
#     experiment_name = "franka_pick_place_direct"
#     empirical_normalization = True  # Keep observation normalization enabled

#     policy = RslRlPpoActorCriticCfg(
#         init_noise_std=1.0,
#         actor_hidden_dims=[256, 256, 128, 64],
#         critic_hidden_dims=[256, 256, 128, 64],
#         activation="elu",
#     )

#     algorithm = RslRlPpoAlgorithmCfg(
#         value_loss_coef=1.0,
#         use_clipped_value_loss=True,
#         clip_param=0.1,
#         entropy_coef=0.01,
#         num_learning_epochs=10,
#         num_mini_batches=16,
#         learning_rate=3e-4,
#         schedule="linear",
#         gamma=0.996,
#         lam=0.95,
#         desired_kl=0.01,
#         max_grad_norm=0.5,
#     )