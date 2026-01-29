# SkillWrapper.py
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from collections import deque


class SkillWrapper(gym.Wrapper):
    """
    Wraps a dict-observation env to a flat Box obs:
        obs = [proprio(376), skill_one_hot(num_skills)]
    Reward = intrinsic (DIAYN) + environment (safety/physics).
    Maintains a trajectory buffer for discriminator.
    """
    def __init__(
        self,
        env,
        discriminator,
        num_skills: int = 32,
        env_reward_weight: float = 0.1,
        trajectory_length: int = 8
    ):
        super().__init__(env)
        self.discriminator = discriminator
        self.num_skills = num_skills
        self.current_skill = 0
        self.env_reward_weight = env_reward_weight
        self.trajectory_length = trajectory_length

        # Trajectory buffer: stores recent states
        self.trajectory_buffer = deque(maxlen=trajectory_length)

        # eval control (for recording)
        self.eval_mode = False
        self.fixed_skill = 0

        obs_dim = 376
        low = np.concatenate([np.full(obs_dim, -np.inf, dtype=np.float32), np.zeros(num_skills, dtype=np.float32)])
        high = np.concatenate([np.full(obs_dim,  np.inf, dtype=np.float32), np.ones(num_skills, dtype=np.float32)])
        self.observation_space = Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)

        if self.eval_mode:
            self.current_skill = int(self.fixed_skill)
        else:
            self.current_skill = int(np.random.randint(0, self.num_skills))

        proprio = obs_dict["proprio"]

        # Validate proprio dimension
        expected_dim = 376
        if proprio.shape[0] != expected_dim:
            raise ValueError(f"Expected proprio dimension {expected_dim}, got {proprio.shape[0]}")

        # Clear trajectory buffer and add initial state once
        # The buffer will naturally fill up over the first few steps
        self.trajectory_buffer.clear()
        self.trajectory_buffer.append(proprio.copy())

        info["skill_idx"] = self.current_skill
        info["raw_proprio"] = proprio
        return self._add_skill(proprio), info

    def step(self, action):
        obs_dict, safety_reward, term, trunc, info = self.env.step(action)
        proprio = obs_dict["proprio"]

        # Add current state to trajectory buffer
        self.trajectory_buffer.append(proprio.copy())

        # Build trajectory: pad with first state if buffer not full yet
        traj_list = list(self.trajectory_buffer)
        if len(traj_list) < self.trajectory_length:
            # Pad with the first state (repeat it at the beginning)
            first_state = traj_list[0]
            padding_needed = self.trajectory_length - len(traj_list)
            traj_list = [first_state] * padding_needed + traj_list

        # Get trajectory as numpy array: (trajectory_length, state_dim)
        trajectory = np.array(traj_list, dtype=np.float32)

        # Intrinsic reward from discriminator (drives diversity)
        intrinsic_reward = self.discriminator.get_reward(trajectory, self.current_skill)

        # Environment reward (drives physical plausibility)
        env_reward = float(safety_reward)

        # Combined reward: discriminator (main) + environment (regularization)
        total_reward = float(intrinsic_reward) + self.env_reward_weight * env_reward

        policy_input = self._add_skill(proprio)

        info = dict(info)
        info["skill_idx"] = self.current_skill
        info["raw_proprio"] = proprio
        info["intrinsic_reward"] = float(intrinsic_reward)
        info["env_reward"] = env_reward
        info["total_reward"] = total_reward

        return policy_input, total_reward, term, trunc, info

    def _add_skill(self, obs_np: np.ndarray) -> np.ndarray:
        skill_one_hot = np.zeros(self.num_skills, dtype=np.float32)
        skill_one_hot[self.current_skill] = 1.0
        return np.concatenate([obs_np.astype(np.float32), skill_one_hot], axis=0)

    def render(self):
        return self.env.render()
