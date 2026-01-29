# HumanoidEnv.py
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import utils
from gymnasium.spaces import Box, Dict


class HumanoidEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 67,
    }

    def __init__(
        self,
        reset_noise_scale=0.1,
        action_smoothness_coeff=0.1,
        action_magnitude_coeff=0.05,
        forward_reward_weight=2.1,
        **kwargs
    ):
        utils.EzPickle.__init__(self, **kwargs)
        self.last_action = None

        # Physics enhancement parameters
        self.reset_noise_scale = reset_noise_scale
        self.action_smoothness_coeff = action_smoothness_coeff
        self.action_magnitude_coeff = action_magnitude_coeff
        self.forward_reward_weight = forward_reward_weight

        self.observation_space = Dict({
            "proprio": Box(low=-np.inf, high=np.inf, shape=(376,), dtype=np.float32),
        })

        MujocoEnv.__init__(
            self,
            "humanoid.xml",
            5,
            observation_space=self.observation_space,
            default_camera_config={
                "trackbodyid": 1,
                "distance": 4.5,
                "lookat": np.array((0.0, 0.0, 1.0)),
            },
            **kwargs,
        )

    def _get_obs(self):
        data = self.data
        position = data.qpos.flat.copy()
        velocity = data.qvel.flat.copy()
        com_inertia = data.cinert.flat.copy()
        com_velocity = data.cvel.flat.copy()
        actuator_forces = data.qfrc_actuator.flat.copy()
        external_contact_forces = data.cfrc_ext.flat.copy()

        proprio = np.concatenate((
            position[2:], velocity, com_inertia, com_velocity, actuator_forces, external_contact_forces
        ))
        return {"proprio": proprio.astype(np.float32)}

    @property
    def is_healthy(self):
        # loose alive condition
        min_z, max_z = 0.3, 2.5
        z = self.data.qpos[2]
        return min_z < z < max_z

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        # Base rewards
        ctrl_cost = -0.05 * np.sum(np.square(action))
        alive_bonus = 1.0

        # Action smoothness penalty: penalize large changes between consecutive actions
        # Handle first step when last_action might be None
        if self.last_action is None:
            self.last_action = np.zeros_like(action)
        action_diff = action - self.last_action
        smoothness_penalty = -self.action_smoothness_coeff * np.sum(np.square(action_diff))

        # Action magnitude bonus: encourage using sufficient action magnitude
        # Prevents "standing still" hacking by rewarding meaningful movements
        action_magnitude = np.mean(np.abs(action))
        magnitude_bonus = self.action_magnitude_coeff * np.clip(action_magnitude, 0, 2.0)

        # Forward velocity reward: encourage forward movement
        forward_velocity = self.data.qvel[0]
        forward_reward = self.forward_reward_weight * forward_velocity

        # Total safety reward (physics only, diversity handled by discriminator)
        terminated = not self.is_healthy
        safety_reward = ctrl_cost + alive_bonus + smoothness_penalty + magnitude_bonus + forward_reward

        obs = self._get_obs()
        self.last_action = action.copy()

        info = {
            "safety_reward": float(safety_reward),
            "ctrl_cost": float(ctrl_cost),
            "smoothness_penalty": float(smoothness_penalty),
            "magnitude_bonus": float(magnitude_bonus),
            "action_magnitude": float(action_magnitude),
            "forward_reward": float(forward_reward),
            "forward_velocity": float(forward_velocity),
        }
        return obs, float(safety_reward), terminated, False, info

    def reset_model(self):
        # Increased noise scale for better exploration of initial states
        noise_low = -self.reset_noise_scale
        noise_high = self.reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)
        self.set_state(qpos, qvel)
        # Initialize last_action as zero vector for smoothness calculation
        self.last_action = np.zeros(self.model.nu, dtype=np.float32)
        return self._get_obs()

