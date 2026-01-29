# script.py
import os
import time
import torch
import numpy as np
import imageio
from tqdm import tqdm

# EGL for headless rendering
os.environ["MUJOCO_GL"] = "egl"

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, CheckpointCallback

from discriminator import DIAYNDiscriminator
from SkillWrapper import SkillWrapper
from HumanoidEnv import HumanoidEnv
from relabel_utils import relabel_replay_buffer


class DIAYNTrainCallback(BaseCallback):
    """
    Trains discriminator periodically from replay buffer samples using trajectories.
    """
    def __init__(self, discriminator, update_freq: int = 100, trajectory_length: int = 8):
        super().__init__()
        self.discriminator = discriminator
        self.update_freq = update_freq
        self.trajectory_length = trajectory_length

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            buffer_size = self.model.replay_buffer.size()
            if buffer_size > self.trajectory_length:  # Need at least trajectory_length samples
                # === ACTIVE LEARNING: Entropy-Based Sampling (PEBBLE/VPL) ===
                # Step 1: Build all valid trajectories and compute entropy
                sample_size = min(1024, buffer_size)  # Oversample for active selection
                candidate_indices = np.random.randint(self.trajectory_length - 1, buffer_size, size=sample_size)

                n_z = self.discriminator.num_skills

                candidate_trajectories = []
                candidate_skills = []

                for idx in candidate_indices:
                    start_idx = idx - (self.trajectory_length - 1)
                    if start_idx < 0:
                        continue

                    dones_in_range = self.model.replay_buffer.dones[start_idx:idx]
                    if np.any(dones_in_range):
                        continue

                    # Build trajectory
                    traj = []
                    for i in range(self.trajectory_length):
                        t_idx = start_idx + i
                        obs = self.model.replay_buffer.observations[t_idx]
                        if obs.ndim == 2:
                            obs = obs[0]
                        proprio = obs[:-n_z]
                        traj.append(proprio)

                    candidate_trajectories.append(np.array(traj))

                    # Get skill index
                    final_obs = self.model.replay_buffer.observations[idx]
                    if final_obs.ndim == 2:
                        final_obs = final_obs[0]
                    skill_one_hot = final_obs[-n_z:]
                    skill_idx = np.argmax(skill_one_hot)
                    candidate_skills.append(skill_idx)

                # Step 2: Compute entropy for each trajectory (uncertainty-based active learning)
                if len(candidate_trajectories) >= 256:
                    candidate_trajectories = np.array(candidate_trajectories, dtype=np.float32)
                    candidate_skills = np.array(candidate_skills, dtype=np.int64)

                    with torch.no_grad():
                        traj_tensor = torch.as_tensor(candidate_trajectories, dtype=torch.float32).to(self.model.device)
                        logits = self.discriminator.forward(traj_tensor)
                        # Use softmax and log_softmax separately for numerical stability
                        probs = torch.softmax(logits, dim=-1)
                        log_probs = torch.log_softmax(logits, dim=-1)
                        entropy = -torch.sum(probs * log_probs, dim=-1).cpu().numpy()

                    # Step 3: Select top-K highest entropy samples (most uncertain/informative)
                    target_samples = 256
                    if len(entropy) > target_samples:
                        top_k_indices = np.argsort(entropy)[-target_samples:]  # Highest entropy
                        trajectories = candidate_trajectories[top_k_indices]
                        skill_indices = candidate_skills[top_k_indices]
                        max_entropy = entropy[top_k_indices[-1]]
                        avg_entropy = entropy[top_k_indices].mean()
                    else:
                        trajectories = candidate_trajectories
                        skill_indices = candidate_skills
                        max_entropy = entropy.max()
                        avg_entropy = entropy.mean()

                    # Step 4: Update discriminator on high-uncertainty samples
                    loss = self.discriminator.update(trajectories, skill_indices)
                    self.logger.record("train/disc_loss", loss)
                    self.logger.record("train/disc_valid_samples", len(trajectories))
                    self.logger.record("train/disc_max_entropy", max_entropy)
                    self.logger.record("train/disc_avg_entropy", avg_entropy)
        return True


class PebbleRelabelCallback(BaseCallback):
    """
    PEBBLE essence:
      Every relabel_freq steps, relabel replay buffer rewards using the latest discriminator.
    """
    def __init__(
        self,
        discriminator,
        relabel_freq: int = 100_000,
        env_reward_weight: float = 0.1,
        avg_env_reward: float = 0.9,
        trajectory_length: int = 8
    ):
        super().__init__()
        self.discriminator = discriminator
        self.relabel_freq = relabel_freq
        self.env_reward_weight = env_reward_weight
        self.avg_env_reward = avg_env_reward
        self.trajectory_length = trajectory_length

    def _on_step(self) -> bool:
        if self.n_calls > 0 and self.n_calls % self.relabel_freq == 0:
            relabel_replay_buffer(
                self.model,
                self.discriminator,
                self.model.device,
                env_reward_weight=self.env_reward_weight,
                avg_env_reward=self.avg_env_reward,
                trajectory_length=self.trajectory_length
            )
        return True


def record_final_skills(env, model, num_skills: int, save_dir: str):
    """
    Record one rollout video per skill z = 0..num_skills-1.
    """
    print("\n>>> Recording Final Videos for All Skills... <<<")
    video_dir = os.path.join(save_dir, "skill_videos")
    os.makedirs(video_dir, exist_ok=True)

    env.eval_mode = True

    for z in tqdm(range(num_skills), desc="Recording Skills"):
        env.fixed_skill = z
        obs, _ = env.reset()
        frames = []
        done = False

        while not done:
            frame = env.render()
            frames.append(frame)

            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            done = bool(term or trunc)

        video_path = os.path.join(video_dir, f"skill_{z:02d}.mp4")
        imageio.mimsave(video_path, frames, fps=30)

    print(f"All videos saved to {video_dir}")


def run_pebble_diayn():
    num_skills = 32
    state_dim = 376
    trajectory_length = 8  # Length of trajectory for discriminator

    total_timesteps = 3_000_000
    relabel_freq = 200_000  # Increased from 100k to reduce instability

    # Reward weights
    env_reward_weight = 0.1  # Weight for environment reward (physics constraints)
    avg_env_reward = 1.8     # Estimated average environment reward for relabeling (matches README)

    run_name = f"PEBBLE_DIAYN_{int(time.time())}"
    log_dir = f"./logs/{run_name}"
    checkpoint_dir = f"./checkpoints/{run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"--- Starting PEBBLE-Enhanced DIAYN: {run_name} ---")

    # Env with physics enhancements
    base_env = HumanoidEnv(
        render_mode="rgb_array",
        reset_noise_scale=0.1,          # Larger initial state exploration
        action_smoothness_coeff=0.1,    # Smooth action transitions
        action_magnitude_coeff=0.05     # Encourage meaningful action magnitudes
    )
    base_env = TimeLimit(base_env, max_episode_steps=1000)

    # Discriminator with trajectory support
    discriminator = DIAYNDiscriminator(
        state_dim=state_dim,
        num_skills=num_skills,
        trajectory_length=trajectory_length
    )

    # Wrapper with environment reward and trajectory support
    env = SkillWrapper(
        base_env,
        discriminator,
        num_skills=num_skills,
        env_reward_weight=env_reward_weight,
        trajectory_length=trajectory_length
    )

    # SAC
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        buffer_size=1_000_000,
        batch_size=2048,
        learning_starts=10_000,
        ent_coef="auto",
    )
    discriminator.to(model.device)

    callbacks = CallbackList([
        DIAYNTrainCallback(
            discriminator,
            update_freq=100,
            trajectory_length=trajectory_length
        ),
        PebbleRelabelCallback(
            discriminator,
            relabel_freq=relabel_freq,
            env_reward_weight=env_reward_weight,
            avg_env_reward=avg_env_reward,
            trajectory_length=trajectory_length
        ),
        CheckpointCallback(save_freq=100_000, save_path=checkpoint_dir, name_prefix="pebble_diayn"),
    ])

    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    model.save(os.path.join(checkpoint_dir, "final_model"))
    torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, "discriminator.pth"))

    record_final_skills(env, model, num_skills, checkpoint_dir)
    env.close()


if __name__ == "__main__":
    run_pebble_diayn()
