# training_config.json Parameter Description 配置参数说明

## network_architecture (Neural network architecture) 神经网络结构
- state_size: State space dimension 状态空间维度
- action_size: Action space dimension 动作空间维度
- hidden_size: Number of hidden layer neurons 隐藏层神经元数量
- activation: Activation function type (e.g., tanh) 激活函数类型（如 tanh）

## training_parameters (Training parameters) 训练参数
- learning_rate: Learning rate 学习率
- gamma: Discount factor (weight for future rewards) 折扣因子（未来奖励的权重）
- epsilon: Initial exploration rate (ε-greedy policy) 初始探索率（ε-贪婪策略）
- epsilon_min: Minimum exploration rate 最小探索率
- epsilon_decay: Exploration rate decay coefficient 探索率衰减系数
- batch_size: Batch size for each training 每次训练的批量大小
- memory_size: Experience replay buffer size 经验回放池容量
- target_update_frequency: Target network update frequency 目标网络更新频率

## training_schedule (Training schedule) 训练计划
- episodes: Number of training episodes 训练回合数
- max_steps_per_episode: Maximum steps per episode 每回合最大步数
- evaluation_frequency: Evaluation frequency (evaluate every N episodes) 评估频率（每多少回合评估一次）
- save_frequency: Model saving frequency (save every N episodes) 模型保存频率（每多少回合保存一次）

## early_stopping (Early stopping control) 早停控制
- enabled: Whether to enable early stopping 是否启用早停机制
- patience: Number of consecutive episodes without improvement to stop training 连续多少回合无提升就停止训练
- min_episodes: Minimum number of episodes before early stopping is allowed 至少训练这么多回合才允许早停
- threshold: Average reward improvement threshold (for significant improvement) 平均奖励提升阈值（用于判断是否有显著提升）

## gpu_settings (GPU settings) GPU设置
- use_gpu: Whether to use GPU 是否使用GPU
- device: Device selection (e.g., auto, cuda, cpu) 设备选择（如 auto、cuda、cpu）
- mixed_precision: Whether to use mixed precision training 是否使用混合精度训练
- gradient_clipping: Gradient clipping threshold 梯度裁剪阈值

## model_saving (Model saving settings) 模型保存设置
- save_dir: Model saving directory 模型保存目录
- model_name: Model file name 模型文件名
- save_best_only: Only save the best model 仅保存最佳模型

## logging (Logging settings) 日志设置
- log_level: Log level (e.g., INFO) 日志级别（如 INFO）
- log_file: Log file name 日志文件名
- tensorboard_logging: Whether to enable TensorBoard logging 是否启用TensorBoard日志 