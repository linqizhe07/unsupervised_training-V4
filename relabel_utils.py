# relabel_utils.py
import torch
import numpy as np


def _split_obs(obs_tensor: torch.Tensor, n_skills: int):
    """
    Supports obs shapes:
      (B, D) or (B, 1, D)
    Returns:
      proprio: (B, proprio_dim)
      skill_indices: (B,)
    """
    if obs_tensor.ndim == 3:
        # (B, 1, D) -> (B, D)
        obs_tensor = obs_tensor.squeeze(1)

    if obs_tensor.ndim != 2:
        raise ValueError(f"Unexpected obs_tensor shape: {tuple(obs_tensor.shape)}")

    proprio = obs_tensor[:, :-n_skills]
    skill_one_hot = obs_tensor[:, -n_skills:]
    skill_indices = torch.argmax(skill_one_hot, dim=-1)
    return proprio, skill_indices


def _build_trajectories(proprio_buffer, dones_buffer, trajectory_length: int, buffer_size: int):
    """
    Build trajectories from a buffer of states with episode boundary awareness.
    For each timestep t, create trajectory [t-L+1, ..., t] where L is trajectory_length.
    Trajectories that cross episode boundaries are marked invalid (filled with NaN).

    Args:
        proprio_buffer: (buffer_size, proprio_dim) numpy array
        dones_buffer: (buffer_size, 1) numpy array, episode termination flags
        trajectory_length: int, length of trajectory window
        buffer_size: int, number of valid entries

    Returns:
        trajectories: (buffer_size, trajectory_length, proprio_dim) numpy array
        valid_mask: (buffer_size,) boolean array indicating valid trajectories
    """
    if buffer_size == 0:
        raise ValueError("Cannot build trajectories from empty buffer")

    proprio_dim = proprio_buffer.shape[-1]
    trajectories = np.zeros((buffer_size, trajectory_length, proprio_dim), dtype=np.float32)
    valid_mask = np.ones(buffer_size, dtype=bool)

    for t in range(buffer_size):
        start_idx = t - (trajectory_length - 1)

        # Check for episode boundaries in the trajectory window
        if start_idx < 0:
            # Not enough history - pad with first state (acceptable for early buffer states)
            # But still check for episode boundaries in available history
            if t > 0 and np.any(dones_buffer[0:t]):
                valid_mask[t] = False
                trajectories[t] = np.nan  # Mark invalid
                continue

            # Build padded trajectory
            traj = []
            for i in range(trajectory_length):
                idx = start_idx + i
                if idx < 0:
                    traj.append(proprio_buffer[0])
                else:
                    traj.append(proprio_buffer[idx])
            trajectories[t] = np.array(traj, dtype=np.float32)
        else:
            # Check if any 'done' in [start_idx, t-1] (trajectory should not cross episodes)
            if np.any(dones_buffer[start_idx:t]):
                valid_mask[t] = False
                trajectories[t] = np.nan  # Mark invalid
                continue

            # Build normal trajectory
            traj = []
            for i in range(trajectory_length):
                idx = start_idx + i
                traj.append(proprio_buffer[idx])
            trajectories[t] = np.array(traj, dtype=np.float32)

    return trajectories, valid_mask


def relabel_replay_buffer(
    model,
    discriminator,
    device,
    batch_size: int = 4096,
    env_reward_weight: float = 0.1,
    avg_env_reward: float = 0.9,
    trajectory_length: int = 8
):
    """
    PEBBLE essence:
      When the reward model (here: DIAYN discriminator) changes,
      replay buffer rewards become stale. Relabel the whole buffer with the latest discriminator.

    This function rewrites replay_buffer.rewards:
      r = intrinsic_reward + env_reward_weight * avg_env_reward
    where intrinsic_reward = log p(z|trajectory) - log p(z) is recomputed from trajectories,
    and avg_env_reward is an estimate (since env reward doesn't change with discriminator).
    """
    print("--- [PEBBLE] Relabeling Replay Buffer with fresh Discriminator (trajectory-based)... ---")

    replay_buffer = model.replay_buffer
    pos = replay_buffer.pos
    buffer_size = replay_buffer.buffer_size if replay_buffer.full else pos

    if buffer_size == 0:
        print("--- [PEBBLE] Buffer is empty, skipping relabel. ---")
        return

    discriminator.eval()

    # First pass: extract all proprio states and skills
    print(f"--- [PEBBLE] Extracting {buffer_size} states... ---")
    all_obs = replay_buffer.observations[:buffer_size]  # (buffer_size, 1, D) or (buffer_size, D)
    all_obs_tensor = torch.as_tensor(all_obs, dtype=torch.float32, device=device)

    n_skills = discriminator.num_skills
    all_proprio, all_skill_indices = _split_obs(all_obs_tensor, n_skills)
    all_proprio_np = all_proprio.cpu().numpy()  # (buffer_size, proprio_dim)
    all_skill_indices_np = all_skill_indices.cpu().numpy()  # (buffer_size,)

    # Build trajectories with episode boundary checking
    print(f"--- [PEBBLE] Building trajectories (length={trajectory_length}) with boundary checks... ---")
    all_dones_np = replay_buffer.dones[:buffer_size]  # (buffer_size, 1)
    all_trajectories, valid_mask = _build_trajectories(
        all_proprio_np, all_dones_np, trajectory_length, buffer_size
    )

    num_valid = np.sum(valid_mask)
    num_invalid = buffer_size - num_valid
    print(f"--- [PEBBLE] Valid trajectories: {num_valid}/{buffer_size} ({num_invalid} cross episode boundaries) ---")

    # Second pass: compute rewards in batches (only for valid trajectories)
    print(f"--- [PEBBLE] Computing intrinsic rewards for valid trajectories... ---")
    start = 0
    with torch.no_grad():
        while start < buffer_size:
            end = min(start + batch_size, buffer_size)

            # Get batch validity mask
            batch_valid_mask = valid_mask[start:end]

            # Get batch of trajectories: (batch, trajectory_length, proprio_dim)
            traj_batch_np = all_trajectories[start:end]
            skill_batch = torch.as_tensor(
                all_skill_indices_np[start:end],
                dtype=torch.long,
                device=device
            )

            # For invalid trajectories, use fallback reward (env reward only)
            batch_rewards = np.full((end - start, 1), env_reward_weight * avg_env_reward, dtype=np.float32)

            # Compute discriminator rewards only for valid trajectories
            if np.any(batch_valid_mask):
                valid_trajs = traj_batch_np[batch_valid_mask]
                valid_skills = skill_batch[batch_valid_mask]

                traj_tensor = torch.as_tensor(valid_trajs, dtype=torch.float32, device=device)

                # Compute discriminator logits
                logits = discriminator.forward(traj_tensor)  # (num_valid, num_skills)
                log_probs = torch.log_softmax(logits, dim=-1)

                # Get log probability for the actual skill
                selected_log_probs = log_probs.gather(1, valid_skills.unsqueeze(1)).squeeze(1)

                # Intrinsic reward (discriminator): log p(z|trajectory) - log p(z)
                intrinsic_rewards = selected_log_probs + float(np.log(n_skills))

                # Add estimated environment reward (weighted)
                total_rewards = intrinsic_rewards + env_reward_weight * avg_env_reward

                # Update only valid trajectory rewards
                batch_rewards[batch_valid_mask] = total_rewards.detach().cpu().numpy().reshape(-1, 1)

            # SB3 rewards are typically (B, 1)
            replay_buffer.rewards[start:end] = batch_rewards
            start = end

    print(f"--- [PEBBLE] Relabeled {buffer_size} transitions ({num_valid} with discriminator, {num_invalid} with fallback). ---")
