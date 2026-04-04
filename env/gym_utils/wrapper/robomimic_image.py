"""
Environment wrapper for Robomimic environments with image observations.

Also return done=False since we do not terminate episode early.

Modified from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env/robomimic/robomimic_image_wrapper.py

"""

import numpy as np
import gym
from gym import spaces
import imageio


class RobomimicImageWrapper(gym.Env):
    def __init__(
        self,
        env,
        shape_meta: dict,
        normalization_path=None,
        low_dim_keys=[
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ],
        image_keys=[
            "agentview_image",
            "robot0_eye_in_hand_image",
        ],
        depth_keys=None,
        clamp_obs=False,
        init_state=None,
        render_hw=(256, 256),
        render_camera_name="agentview",
        impedance_mode="fixed",
        control_obs=False,
        controller_configs=None,
    ):
        self.env = env
        self.init_state = init_state
        self.has_reset_before = False
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.video_writer = None
        self.clamp_obs = clamp_obs

        # Setting up controller config parameters.
        self.impedance_mode = impedance_mode
        self.control_obs = control_obs
        self.controller_configs = controller_configs
        self.default_damping = self.controller_configs["damping"]
        self.default_stiffness = self.controller_configs["kp"]
        self.damping_exp_scale = self.controller_configs["damping_limits"][1] / self.default_damping
        self.stiffness_exp_scale = self.controller_configs["kp_limits"][1] / self.default_stiffness

        # set up normalization
        self.normalize = normalization_path is not None
        if self.normalize:
            normalization = np.load(normalization_path)
            self.obs_min = normalization["obs_min"]
            self.obs_max = normalization["obs_max"]
            self.action_min = normalization["action_min"]
            self.action_max = normalization["action_max"]

        # setup spaces
        low = np.full(env.action_dimension, fill_value=-1)
        high = np.full(env.action_dimension, fill_value=1)
        self.action_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype,
        )
        self.low_dim_keys = low_dim_keys
        self.image_keys = image_keys
        self.depth_keys = depth_keys
        self.obs_keys = low_dim_keys + image_keys
        if depth_keys is not None:
            self.obs_keys += depth_keys
        observation_space = spaces.Dict()
        for key, value in shape_meta["obs"].items():
            shape = value["shape"]
            if key.endswith("rgb"):
                min_value, max_value = 0, 255 # rgb not normalized until later
                d_type = np.uint8
            elif key.endswith("state"):
                min_value, max_value = -1, 1
                d_type = np.float32
            elif key.endswith("depth"):
                min_value, max_value = 0, 255 # depth not normalized until later, for consistency with rgb
                d_type = np.uint8
            else:
                raise RuntimeError(f"Unsupported type {key}")
            
            # Add two more entries for damping and kp, if needed - only for state obs.
            if self.control_obs and key.endswith("state"):
                shape = [s + 2 for s in shape]  # add two dimensions for damping and kp
            this_space = spaces.Box(
                low=min_value,
                high=max_value,
                shape=shape,
                dtype=d_type,
            )
            observation_space[key] = this_space
        self.observation_space = observation_space

    def normalize_obs(self, obs):
        obs = 2 * (
            (obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5
        )  # -> [-1, 1]
        if self.clamp_obs:
            obs = np.clip(obs, -1, 1)
        return obs

    def unnormalize_action(self, action):
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min

    def get_observation(self, raw_obs):
        obs = {"rgb": None, "state": None}  # stack rgb if multiple cameras
        if self.depth_keys is not None:
            obs["depth"] = None # stack depth if multiple cameras
        for key in self.obs_keys:
            if key in self.image_keys:
                if obs["rgb"] is None:
                    obs["rgb"] = raw_obs[key]
                else:
                    obs["rgb"] = np.concatenate(
                        [obs["rgb"], raw_obs[key]], axis=0
                    )  # C H W
            elif self.depth_keys is not None and key in self.depth_keys:
                if obs["depth"] is None:
                    obs["depth"] = raw_obs[key]
                else:
                    obs["depth"] = np.concatenate(
                        [obs["depth"], raw_obs[key]], axis=-1 # depth channel is last, not first
                    )  # C H W
            else:
                if obs["state"] is None:
                    obs["state"] = raw_obs[key]
                else:
                    obs["state"] = np.concatenate([obs["state"], raw_obs[key]], axis=-1)
        if self.normalize:
            obs["state"] = self.normalize_obs(obs["state"])
        obs["rgb"] *= 255  # [0, 1] -> [0, 255], in float64
        if self.depth_keys is not None:
            obs["depth"] *= 255  # [0, 1] -> [0, 255], in float32
            obs["depth"] = np.transpose(obs["depth"], (2, 0, 1))  # moving depth channel to the front for consistency with rgb

        # Include controller state in observation if specified.
        if self.control_obs:
            # Grab kp and kd directly from environment - assume all joints have the same gains for now.
            kp = self.env.env.robots[0].controller.kp[0:1]
            kd = self.env.env.robots[0].controller.kd[0:1]

            damping = kd / (2 * np.sqrt(kp) ) # Damping ratio formula.

            # Exponential normalization.
            kp_obs = np.log(kp / self.default_stiffness) / np.log(self.stiffness_exp_scale)
            damping_obs = np.log(damping / self.default_damping) / np.log(self.damping_exp_scale)
            obs["state"] = np.concatenate([obs["state"], damping_obs, kp_obs], axis=-1)
        return obs

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def reset(self, options={}, **kwargs):
        """Ignore passed-in arguments like seed"""
        # Close video if exists
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None

        # Start video if specified
        if "video_path" in options:
            self.video_writer = imageio.get_writer(options["video_path"], fps=30)

        # Call reset
        new_seed = options.get(
            "seed", None
        )  # used to set all environments to specified seeds
        if self.init_state is not None:
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset()
                self.has_reset_before = True

            # always reset to the same state to be compatible with gym
            raw_obs = self.env.reset_to({"states": self.init_state})
        elif new_seed is not None:
            self.seed(seed=new_seed)
            raw_obs = self.env.reset()
        else:
            # random reset
            raw_obs = self.env.reset()
        return self.get_observation(raw_obs)

    def step(self, action):
        # Parse action based on impedance mode.
        if self.impedance_mode == "variable":
            damping, kp, delta = action[:6], action[6:12], action[12:]
            # Un-normalizing/scaling damping and kp actions.
            damping = np.power(self.damping_exp_scale, damping) * self.default_damping
            kp = np.power(self.stiffness_exp_scale, kp) * self.default_stiffness
        elif self.impedance_mode == "variable_kp":
            kp, delta = action[:6], action[6:]
            # Un-normalizing/scaling kp action.
            kp = np.power(self.stiffness_exp_scale, kp) * self.default_stiffness
        else:
            delta = action

        # Un-normalizing operational space action (delta), if necessary.
        if self.normalize:
            delta = self.unnormalize_action(delta)

        # Re-combining full action based on impedance mode.
        if self.impedance_mode == "variable":
            action = np.concatenate([damping, kp, delta], axis=0)
        elif self.impedance_mode == "variable_kp":
            action = np.concatenate([kp, delta], axis=0)
        else:
            action = delta

        # Stepping environment.
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.get_observation(raw_obs)

        # Adding delta, damping, and stiffness to info dict.
        info["delta"] = delta # TODO: don't actually need this? already in action?
        # Grabbing current damping and stiffness from the controller.
        controller_kp = self.env.env.robots[0].controller.kp
        controller_kd = self.env.env.robots[0].controller.kd
        info["damping"] = controller_kd / (2 * np.sqrt(controller_kp) )
        info["stiffness"] = controller_kp

        # Adding ee forces to info dict.
        force, torque = self.get_ee_forces()
        info["ee_force"] = force
        info["ee_torque"] = torque

        # render if specified
        if self.video_writer is not None:
            video_img = self.render(mode="rgb_array")
            self.video_writer.append_data(video_img)

        return obs, reward, False, info

    def render(self, mode="rgb_array"):
        h, w = self.render_hw
        return self.env.render(
            mode=mode,
            height=h,
            width=w,
            camera_name=self.render_camera_name,
        )

    def get_ee_forces(self):
        force = self.env.env.robots[0].ee_force
        torque = self.env.env.robots[0].ee_torque
        return force, torque


if __name__ == "__main__":
    import os
    from omegaconf import OmegaConf
    import json

    os.environ["MUJOCO_GL"] = "egl"

    cfg = OmegaConf.load("cfg/robomimic/finetune/can/ft_ppo_diffusion_mlp_img.yaml")
    shape_meta = cfg["shape_meta"]

    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    import matplotlib.pyplot as plt

    wrappers = cfg.env.wrappers
    obs_modality_dict = {
        "low_dim": (
            wrappers.robomimic_image.low_dim_keys
            if "robomimic_image" in wrappers
            else wrappers.robomimic_lowdim.low_dim_keys
        ),
        "rgb": (
            wrappers.robomimic_image.image_keys
            if "robomimic_image" in wrappers
            else None
        ),
    }
    if obs_modality_dict["rgb"] is None:
        obs_modality_dict.pop("rgb")
    ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)

    with open(cfg.robomimic_env_cfg_path, "r") as f:
        env_meta = json.load(f)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=True,
    )
    env.env.hard_reset = False

    wrapper = RobomimicImageWrapper(
        env=env,
        shape_meta=shape_meta,
        image_keys=["robot0_eye_in_hand_image"],
    )
    wrapper.seed(0)
    obs = wrapper.reset()
    print(obs.keys())
    img = wrapper.render()
    wrapper.close()
    plt.imshow(img)
    plt.savefig("test.png")
