import os
import cv2
import time
import numpy as np
import collections
import torch
from stable_baselines3.common.callbacks import BaseCallback

class PebbleCollectorCallback(BaseCallback):
    def __init__(self, base_save_dir="./pebble_data", segment_len=150, max_segments=2000):
        super().__init__()
        self.base_save_dir = base_save_dir
        self.segment_len = segment_len
        
        # 内存 Buffer
        self.buffer = collections.deque(maxlen=max_segments)
        
        self.curr_proprio = []
        self.curr_frames = []
        
        self.current_round_path = os.path.join(self.base_save_dir, "phase1_warmup")
        os.makedirs(self.current_round_path, exist_ok=True)

    def set_round_name(self, round_name):
        self.current_round_path = os.path.join(self.base_save_dir, round_name)
        os.makedirs(self.current_round_path, exist_ok=True)
        print(f"--- [Callback] Switch data saving to: {self.current_round_path} ---")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])[0]
        proprio = infos.get("raw_proprio")
        current_skill_idx = infos.get("skill_idx", -1)
        
        frame = None
        try:
            if hasattr(self.training_env, 'envs'):
                frame = self.training_env.envs[0].render()
            else:
                frame = self.training_env.render()
        except Exception:
            pass

        if frame is not None and proprio is not None:
            try:
                small_frame = cv2.resize(frame, (128, 128))
                self.curr_frames.append(small_frame)
                self.curr_proprio.append(proprio)
            except Exception:
                pass

        if len(self.curr_proprio) >= self.segment_len:
            proprio_arr = np.array(self.curr_proprio, dtype=np.float32)
            frames_arr = np.array(self.curr_frames, dtype=np.uint8)
            
            seg_data = {
                "proprio": proprio_arr,
                "frames": frames_arr,
            }
            
            self.buffer.append(seg_data)
            
            if current_skill_idx >= 0:
                filename = f"skill_{current_skill_idx:02d}_latest.npz"
            else:
                timestamp = int(time.time() * 1000)
                filename = f"exploration_{timestamp}.npz"
                
            filepath = os.path.join(self.current_round_path, filename)
            try:
                np.savez_compressed(filepath, proprio=proprio_arr, frames=frames_arr)
            except Exception as e:
                pass
            
            self.curr_proprio = []
            self.curr_frames = []
            
        dones = self.locals.get("dones", [False])
        if dones[0]:
            self.curr_proprio = []
            self.curr_frames = []
            
        return True

    def sample_active_pairs(self, discriminator, batch_size=10, trajectory_length=8):
        """ [VPL] 基于熵 (Uncertainty) 的主动采样 - 全量扫描版 (适配轨迹判别器) """
        buffer_len = len(self.buffer)

        # 如果数据不够，直接返回
        if buffer_len < batch_size * 2:
            print(f"[Pebble] Not enough segments. Current: {buffer_len}, Need: {batch_size * 2}")
            return None, None, None, None

        # [CRITICAL FIX] 发文章级别要求：全量扫描 Buffer，不进行随机下采样
        candidate_indices = np.arange(buffer_len)

        entropies = []
        valid_indices = []

        device = discriminator.device

        # 批量计算或者循环计算均可，这里用循环保证显存安全
        with torch.no_grad():
            for idx in candidate_indices:
                # 采样 proprio (Time, Dim)
                raw_obs = self.buffer[idx]["proprio"]  # (150, 376)

                # 下采样以加速计算 (保持 stride=5)
                sampled_obs = raw_obs[::5]  # (30, 376)

                # 构建轨迹: 为每个时刻构建 trajectory_length 的滑动窗口
                # 从 sampled_obs 中构建轨迹
                num_timesteps = len(sampled_obs)
                if num_timesteps < trajectory_length:
                    # 如果不够长，跳过这个样本
                    continue

                # 为每个时刻构建轨迹 [t-L+1, ..., t]
                trajectories = []
                for t in range(num_timesteps):
                    traj = []
                    for i in range(trajectory_length):
                        t_idx = t - (trajectory_length - 1) + i
                        if t_idx < 0:
                            traj.append(sampled_obs[0])  # 用第一帧填充
                        else:
                            traj.append(sampled_obs[t_idx])
                    trajectories.append(np.array(traj))

                trajectories = np.array(trajectories, dtype=np.float32)  # (num_timesteps, trajectory_length, 376)
                traj_tensor = torch.as_tensor(trajectories, dtype=torch.float32).to(device)

                # 使用新的 discriminator.forward() 方法
                logits = discriminator.forward(traj_tensor)  # (num_timesteps, num_skills)
                log_probs = torch.log_softmax(logits, dim=-1)
                probs = torch.exp(log_probs)

                # Entropy = -sum(p * log p) with numerical stability
                entropy_per_step = -torch.sum(probs * log_probs, dim=-1)
                avg_entropy = entropy_per_step.mean().item()

                entropies.append(avg_entropy)
                valid_indices.append(idx)

        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2. 选择 Entropy 最高的 Top-K (真正的 Global Maxima)
        entropies = np.array(entropies)
        # argsort 是升序，取最后 batch_size 个即为最大值
        high_entropy_ptrs = np.argsort(entropies)[-batch_size:] 
        
        selected_indices_1 = [valid_indices[i] for i in high_entropy_ptrs]
        
        # 3. 对手样本：随机选择
        # (这里也可以改进为选差异最大的，但随机通常足够作为 baseline)
        selected_indices_2 = np.random.choice(buffer_len, size=batch_size, replace=False)

        # 4. 组装
        p1 = np.array([self.buffer[i]["proprio"] for i in selected_indices_1])
        p2 = np.array([self.buffer[i]["proprio"] for i in selected_indices_2])
        f1 = [self.buffer[i]["frames"] for i in selected_indices_1]
        f2 = [self.buffer[i]["frames"] for i in selected_indices_2]
            
        print(f"[Active Learning] Scanned {buffer_len} items. Sampled {batch_size} pairs. Max Entropy: {np.max(entropies):.4f}")
        return p1, p2, f1, f2