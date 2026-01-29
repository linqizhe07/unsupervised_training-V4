# 修复了已知问题
avg_env_reward配置错误：2.1-->0.9


Relabel太频繁且太激进：
每100k步就relabel整个replay buffer（1M transitions）
在2.5M步时发生了第25次relabel
突然改变了所有历史数据的奖励，导致policy训练目标剧变

判别器过拟合：
disc_loss持续下降，但这可能意味着判别器对某些模式过于自信
给出极端的负奖励，误导policy

# 直接运行
python script_stable.py

# 监控训练（在另一个终端）
tensorboard --logdir ./logs
监控指标:

train/disc_loss - 判别器损失（应逐渐下降）
train/disc_max_entropy - 最困惑样本熵（训练初期高）
train/disc_avg_entropy - 平均熵（随技能分化下降）
rollout/ep_rew_mean - Episode 平均奖励
预期训练时间: ~36-48 小时（3M 步，GPU 加速）

💡 这个 Pipeline 的独特性
非传统 DIAYN: 用轨迹而非单帧状态，防止"静态姿势欺骗"
真正的主动学习: 不是随机采样，而是基于熵的智能选择
工程鲁棒性: Episode 边界检查、数值稳定、准确的奖励估计
物理合理性: 混合内在奖励（多样性）和环境奖励（物理约束）
这是一个production-ready 的无监督技能学习系统 🎯
