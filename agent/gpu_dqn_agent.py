import torch
import torch.nn as nn
from torch.optim.adam import Adam
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from collections import deque
import json
import os
from tqdm import tqdm
import csv


class DQNetwork(nn.Module):
    """GPU加速的神经网络"""
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.activation = nn.Tanh()
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


class GPUAcceleratedDQNAgent:
    """GPU加速的DQN智能体"""
    def __init__(self, simulator_config: str = "simulator_config.json", 
                 training_config: str = "training_config.json"):
        self.simulator_config = self._load_config(simulator_config)
        self.training_config = self._load_config(training_config)
        
        # 设备设置
        self.device = self._setup_device()
        print(f"[INFO] 使用设备: {self.device}")
        
        # 网络参数
        network_config = self.training_config["network_architecture"]
        self.state_size = network_config["state_size"]
        self.action_size = network_config["action_size"]
        self.hidden_size = network_config["hidden_size"]
        
        # 创建网络
        self.q_network = DQNetwork(self.state_size, self.action_size, self.hidden_size).to(self.device)
        self.target_network = DQNetwork(self.state_size, self.action_size, self.hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 训练参数
        train_params = self.training_config["training_parameters"]
        self.optimizer = Adam(self.q_network.parameters(), lr=train_params["learning_rate"])
        self.gamma = train_params["gamma"]
        self.epsilon = train_params["epsilon"]
        self.epsilon_min = train_params["epsilon_min"]
        self.epsilon_decay = train_params["epsilon_decay"]
        self.memory = deque(maxlen=train_params["memory_size"])
        self.batch_size = train_params["batch_size"]
        
        # 训练统计
        self.training_stats = {
            'episodes': 0,
            'total_reward': 0.0,
            'avg_reward': 0.0,
            'epsilon_history': [],
            'reward_history': [],
            'loss_history': []
        }
        
        # 模型保存设置
        self.save_dir = self.training_config["model_saving"]["save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)
        
    def _load_config(self, config_file: str) -> Dict:
        """加载配置文件"""
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _setup_device(self) -> torch.device:
        """设置计算设备"""
        gpu_settings = self.training_config["gpu_settings"]
        
        if gpu_settings["use_gpu"] and torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[INFO] GPU可用: {torch.cuda.get_device_name()}")
            print(f"[INFO] GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = torch.device("cpu")
            print("[INFO] 使用CPU训练")
        
        return device
    
    def _preprocess_state(self, env_state: Dict) -> torch.Tensor:
        """预处理单个状态"""
        queue_lengths = env_state.get('queue_lengths', {})
        queue_vector = []
        for lane in ['N_L', 'N_R', 'S_L', 'S_R', 'E_L', 'E_R', 'W_L', 'W_R']:
            queue_vector.append(queue_lengths.get(lane, 0))
        queue_vector = np.array(queue_vector) / 10.0
        
        # 相位编码（简化为2相位）
        phase = env_state.get('current_phase', 'NS_STRAIGHT_RIGHT')
        phase_mapping = {
            'NS_STRAIGHT_RIGHT': 0,
            'EW_STRAIGHT_RIGHT': 1
        }
        phase_encoded = np.zeros(10)  # 保持10维以兼容现有代码
        phase_encoded[phase_mapping.get(phase, 0)] = 1.0
        
        state = np.concatenate([queue_vector, phase_encoded])
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)

    def _preprocess_state_batch(self, env_states: List[Dict]) -> torch.Tensor:
        """批量预处理状态"""
        queue_vectors = []
        phase_encodeds = []
        phase_mapping = {
            'NS_STRAIGHT_RIGHT': 0,
            'EW_STRAIGHT_RIGHT': 1
        }
        
        for env_state in env_states:
            queue_lengths = env_state.get('queue_lengths', {})
            queue_vector = [queue_lengths.get(lane, 0) for lane in ['N_L', 'N_R', 'S_L', 'S_R', 'E_L', 'E_R', 'W_L', 'W_R']]
            queue_vector = np.array(queue_vector) / 10.0
            queue_vectors.append(queue_vector)
            
            phase = env_state.get('current_phase', 'NS_STRAIGHT_RIGHT')
            phase_encoded = np.zeros(10)
            phase_encoded[phase_mapping.get(phase, 0)] = 1.0
            phase_encodeds.append(phase_encoded)
        
        states = np.concatenate([np.stack(queue_vectors, axis=0), np.stack(phase_encodeds, axis=0)], axis=1)
        return torch.FloatTensor(states).to(self.device)
    
    def _get_action_duration(self, action: int) -> float:
        """将动作索引转换为绿灯持续时间"""
        signal_config = self.simulator_config["optimization"]["signal_control"]
        min_green = signal_config["green_min"]
        max_green = signal_config["green_max"]
        duration = min_green + (max_green - min_green) * action / (self.action_size - 1)
        return duration
    
    def _compute_reward(self, env_state: Dict) -> float:
        """计算奖励值"""
        objective_value = env_state.get('objective_value', 0)
        return -objective_value  # 负的目标函数值作为奖励
    
    def act(self, env_state: Dict) -> float:
        """选择动作"""
        state = self._preprocess_state(env_state)
        
        if np.random.random() <= self.epsilon:
            # 探索：随机选择动作
            action = random.randrange(self.action_size)
        else:
            # 利用：选择Q值最大的动作
            with torch.no_grad():
                q_values = self.q_network(state)
                action = int(q_values.argmax().item())
        
        return self._get_action_duration(action)
    
    def remember(self, state: Dict, action: float, reward: float, next_state: Dict, done: bool):
        """存储经验到回放缓冲区"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """经验回放训练"""
        if len(self.memory) < self.batch_size:
            return
        
        # 采样批次
        batch = random.sample(self.memory, self.batch_size)
        state_dicts, actions, rewards, next_state_dicts, dones = zip(*batch)
        
        # 预处理批次数据
        states = self._preprocess_state_batch(list(state_dicts))
        actions = torch.LongTensor([self._action_to_index(a) for a in actions]).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = self._preprocess_state_batch(list(next_state_dicts))
        dones = torch.BoolTensor(dones).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # 计算损失
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        gpu_settings = self.training_config["gpu_settings"]
        if gpu_settings["gradient_clipping"] > 0:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 
                                         max_norm=gpu_settings["gradient_clipping"])
        
        self.optimizer.step()
        
        # 更新探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 记录损失
        self.training_stats['loss_history'].append(loss.item())
    
    def _action_to_index(self, action: float) -> int:
        """将动作持续时间转换为索引"""
        signal_config = self.simulator_config["optimization"]["signal_control"]
        min_green = signal_config["green_min"]
        max_green = signal_config["green_max"]
        index = int((action - min_green) / (max_green - min_green) * (self.action_size - 1))
        return max(0, min(self.action_size - 1, index))
    
    def train(self, env, episodes: int = 500, max_steps: int = 2000):
        """训练智能体"""
        # 使用配置文件中的参数，如果未提供的话
        if episodes == 500:  # 使用默认值
            episodes = self.training_config["training_schedule"]["episodes"]
        if max_steps == 2000:  # 使用默认值
            max_steps = self.training_config["training_schedule"]["max_steps_per_episode"]

        # 打印关键训练参数
        train_params = self.training_config["training_parameters"]
        print("[INFO] 训练参数设置:")
        print(f"  learning_rate: {train_params['learning_rate']}")
        print(f"  gamma: {train_params['gamma']}")
        print(f"  epsilon_decay: {train_params['epsilon_decay']}")
        print(f"  batch_size: {train_params['batch_size']}")
        print(f"  memory_size: {train_params['memory_size']}")

        print(f"[INFO] 开始训练，使用设备: {self.device}")
        print(f"[INFO] 训练参数: {episodes} 回合, 每回合 {max_steps} 步")
        
        best_reward = float('-inf')
        
        # 早停参数
        early_stopping_config = self.training_config.get("early_stopping", {})
        patience = early_stopping_config.get("patience", 20)
        min_episodes = early_stopping_config.get("min_episodes", 30)
        threshold = early_stopping_config.get("threshold", 50)
        early_stopping_enabled = early_stopping_config.get("enabled", True)
        
        best_avg_reward = float('-inf')
        no_improve_count = 0
        
        for episode in tqdm(range(episodes), desc="Training"):
            env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                current_state = env.get_state()
                action = self.act(current_state)
                env.set_phase_duration(action)
                next_state = env.step()
                reward = self._compute_reward(next_state)
                total_reward += reward
                done = (step == max_steps - 1)
                
                self.remember(current_state, action, reward, next_state, done)
                
                # 定期进行经验回放
                if step % 4 == 0:
                    self.replay()
            
            # 定期更新目标网络
            if episode % self.training_config["training_parameters"]["target_update_frequency"] == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            # 更新训练统计
            self.training_stats['episodes'] += 1
            self.training_stats['total_reward'] += total_reward
            self.training_stats['avg_reward'] = self.training_stats['total_reward'] / self.training_stats['episodes']
            self.training_stats['epsilon_history'].append(self.epsilon)
            self.training_stats['reward_history'].append(total_reward)
            
            # 定期打印训练信息
            if episode % self.training_config["training_schedule"]["evaluation_frequency"] == 0:
                print(f"Episode: {episode}, Avg Reward: {self.training_stats['avg_reward']:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}")
            
            # 保存最佳模型
            if total_reward > best_reward and self.training_config["model_saving"]["save_best_only"]:
                best_reward = total_reward
                self.save_model(os.path.join(self.save_dir, "best_model.pth"))
            
            # 定期保存模型
            if episode % self.training_config["training_schedule"]["save_frequency"] == 0:
                self.save_model(os.path.join(self.save_dir, f"model_episode_{episode}.pth"))

            # 早停判断
            if early_stopping_enabled and episode >= min_episodes:
                if self.training_stats['avg_reward'] > best_avg_reward + threshold:
                    best_avg_reward = self.training_stats['avg_reward']
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"[Early Stopping] 连续{patience}回合平均奖励无提升，提前终止训练。最终Avg Reward: {self.training_stats['avg_reward']:.2f}")
                    break

        # 训练完成，清理模型文件，只保留最佳模型
        self._cleanup_model_files()
    
    def _cleanup_model_files(self):
        """清理训练过程中产生的模型文件，只保留最佳模型"""
        import glob
        
        # 保存最佳模型到最终位置
        best_model_path = os.path.join(self.save_dir, "best_model.pth")
        final_model_path = os.path.join(self.save_dir, "trained_model_gpu.pth")
        
        if os.path.exists(best_model_path):
            # 将最佳模型复制到最终位置
            import shutil
            shutil.copy2(best_model_path, final_model_path)
            print(f"[INFO] 最佳模型已保存为: {final_model_path}")
        
        # 删除训练过程中产生的临时模型文件
        pattern = os.path.join(self.save_dir, "model_episode_*.pth")
        temp_files = glob.glob(pattern)
        
        deleted_count = 0
        for file_path in temp_files:
            try:
                os.remove(file_path)
                deleted_count += 1
                print(f"[INFO] 已删除临时模型文件: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"[WARN] 删除文件失败 {file_path}: {e}")
        
        print(f"[INFO] 清理完成，删除了 {deleted_count} 个临时模型文件")
        print(f"[INFO] 保留的模型文件: trained_model_gpu.pth (最终模型)")
    
    def save_model(self, filename: str):
        """保存模型"""
        model_data = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_stats': self.training_stats,
            'configs': {
                'simulator_config': self.simulator_config,
                'training_config': self.training_config
            }
        }
        torch.save(model_data, filename)
        print(f"[INFO] 模型已保存到: {filename}")
    
    def load_model(self, filename: str):
        """加载模型"""
        model_data = torch.load(filename, map_location=self.device)
        self.q_network.load_state_dict(model_data['q_network_state_dict'])
        self.target_network.load_state_dict(model_data['target_network_state_dict'])
        self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
        self.epsilon = model_data['epsilon']
        self.training_stats = model_data['training_stats']
        print(f"[INFO] 模型已从 {filename} 加载")
    
    def get_training_stats(self) -> Dict:
        """获取训练统计信息"""
        return self.training_stats.copy() 