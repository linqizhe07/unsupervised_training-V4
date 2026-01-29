# diagnose_training.py
"""
Diagnostic script to analyze training collapse and check discriminator/policy health.
Run this to understand what went wrong and whether to restart training.
"""
import os
import glob
import numpy as np
import torch
from stable_baselines3 import SAC
from discriminator import DIAYNDiscriminator
from HumanoidEnv import HumanoidEnv
from SkillWrapper import SkillWrapper
from gymnasium.wrappers import TimeLimit


def find_latest_checkpoint(checkpoint_dir):
    """Find the most recent checkpoint."""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "pebble_diayn_*_steps.zip"))
    if not checkpoints:
        return None
    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.split('_')[-2]))
    return checkpoints[-1]


def analyze_discriminator(discriminator, env, model, num_skills=32, num_episodes=10):
    """
    Test if discriminator can actually distinguish between skills.
    Returns accuracy and average confidence.
    """
    print("\n=== Discriminator Analysis ===")
    env.eval_mode = True

    correct_predictions = 0
    total_predictions = 0
    confidences = []

    for skill_idx in range(min(num_skills, 5)):  # Test first 5 skills
        env.fixed_skill = skill_idx

        for episode in range(2):  # 2 episodes per skill
            obs, _ = env.reset()
            done = False
            episode_states = []

            while not done and len(episode_states) < 1000:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(action)
                episode_states.append(info['raw_proprio'])
                done = term or trunc

            # Sample a few trajectories from this episode
            if len(episode_states) >= 8:
                for _ in range(3):  # 3 samples per episode
                    start_idx = np.random.randint(0, len(episode_states) - 7)
                    trajectory = np.array(episode_states[start_idx:start_idx+8], dtype=np.float32)

                    with torch.no_grad():
                        traj_tensor = torch.as_tensor(trajectory, device=discriminator.device).unsqueeze(0)
                        logits = discriminator.forward(traj_tensor)
                        probs = torch.softmax(logits, dim=-1)
                        predicted_skill = torch.argmax(probs, dim=-1).item()
                        confidence = probs[0, predicted_skill].item()

                    if predicted_skill == skill_idx:
                        correct_predictions += 1
                    total_predictions += 1
                    confidences.append(confidence)

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    avg_confidence = np.mean(confidences) if confidences else 0

    print(f"Discriminator Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")
    print(f"Average Confidence: {avg_confidence:.3f}")
    print(f"Random Baseline: {1/num_skills:.2%}")

    if accuracy < 0.3:
        print("⚠️  WARNING: Discriminator accuracy is very low! Skills are not being distinguished.")
    elif accuracy > 0.8:
        print("✓ Discriminator is working well.")
    else:
        print("⚠️  Discriminator accuracy is moderate. Skills may be partially learned.")

    return accuracy, avg_confidence


def analyze_rewards(env, model, discriminator, num_skills=32):
    """
    Test actual rewards for different skills.
    """
    print("\n=== Reward Analysis ===")
    env.eval_mode = True

    skill_rewards = []
    skill_episode_lengths = []

    for skill_idx in range(min(num_skills, 8)):  # Test first 8 skills
        env.fixed_skill = skill_idx
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            steps += 1
            done = term or trunc

        skill_rewards.append(total_reward)
        skill_episode_lengths.append(steps)
        print(f"Skill {skill_idx:2d}: Reward={total_reward:7.1f}, Length={steps:4d}")

    avg_reward = np.mean(skill_rewards)
    avg_length = np.mean(skill_episode_lengths)

    print(f"\nAverage Reward: {avg_reward:.1f}")
    print(f"Average Episode Length: {avg_length:.1f}")

    if avg_reward < -500:
        print("⚠️  CRITICAL: Average reward is very negative. Training has collapsed.")
    elif avg_reward < 0:
        print("⚠️  WARNING: Average reward is negative. Policy is not learning effectively.")
    else:
        print("✓ Average reward is positive.")

    return skill_rewards, skill_episode_lengths


def main():
    print("=== Training Diagnostic Tool ===\n")

    # Find checkpoint directory
    checkpoint_dirs = glob.glob("./checkpoints/PEBBLE_DIAYN_*")
    if not checkpoint_dirs:
        print("❌ No checkpoint directory found. Has training started?")
        return

    latest_dir = max(checkpoint_dirs, key=os.path.getmtime)
    print(f"Found checkpoint directory: {latest_dir}")

    # Find latest checkpoint
    latest_checkpoint = find_latest_checkpoint(latest_dir)
    if not latest_checkpoint:
        print("❌ No checkpoint files found.")
        return

    print(f"Latest checkpoint: {latest_checkpoint}")
    step_num = int(latest_checkpoint.split('_')[-2])
    print(f"Training steps: {step_num:,}")

    # Load discriminator
    discriminator_path = os.path.join(latest_dir, "discriminator.pth")
    if not os.path.exists(discriminator_path):
        print("❌ Discriminator checkpoint not found.")
        return

    num_skills = 32
    state_dim = 376
    trajectory_length = 8

    discriminator = DIAYNDiscriminator(
        state_dim=state_dim,
        num_skills=num_skills,
        trajectory_length=trajectory_length
    )
    discriminator.load_state_dict(torch.load(discriminator_path, map_location='cpu'))
    discriminator.eval()
    print("✓ Discriminator loaded")

    # Create environment
    base_env = HumanoidEnv(
        render_mode="rgb_array",
        reset_noise_scale=0.1,
        action_smoothness_coeff=0.1,
        action_magnitude_coeff=0.05
    )
    base_env = TimeLimit(base_env, max_episode_steps=1000)

    env = SkillWrapper(
        base_env,
        discriminator,
        num_skills=num_skills,
        env_reward_weight=0.1,
        trajectory_length=trajectory_length
    )
    print("✓ Environment created")

    # Load model
    try:
        model = SAC.load(latest_checkpoint, env=env)
        print("✓ Model loaded")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Run diagnostics
    accuracy, confidence = analyze_discriminator(discriminator, env, model, num_skills)
    skill_rewards, skill_lengths = analyze_rewards(env, model, discriminator, num_skills)

    # Final recommendations
    print("\n=== DIAGNOSIS & RECOMMENDATIONS ===")

    if accuracy < 0.3 and np.mean(skill_rewards) < -500:
        print("❌ TRAINING HAS COLLAPSED")
        print("\nRecommendations:")
        print("1. STOP current training immediately")
        print("2. The fixed issues in script.py:")
        print("   - avg_env_reward corrected: 0.9 → 1.8")
        print("   - relabel_freq increased: 100k → 200k")
        print("3. Restart training from scratch with the fixed configuration")
        print("4. Monitor tensorboard closely, especially around 2M steps")
    elif accuracy > 0.6 and np.mean(skill_rewards) > 0:
        print("✓ Training looks healthy")
        print("\nContinue training and monitor for stability.")
    else:
        print("⚠️  Training is struggling but may recover")
        print("\nConsider:")
        print("1. Continue for 500k more steps and re-evaluate")
        print("2. If no improvement, restart with fixed configuration")

    env.close()


if __name__ == "__main__":
    main()
