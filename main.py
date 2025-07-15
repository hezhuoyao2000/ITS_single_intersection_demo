#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Traffic Signal Control System Main Program - Two-Phase Optimization
智能交通信号控制系统主程序 - 两相位优化版本

author: [Your Name]
作者: [您的姓名]
date: [Current Date]
日期: [当前日期]
description: Use reinforcement learning to optimize traffic signal control, minimizing vehicle delay and queue length
描述: 使用强化学习优化交通信号控制，最小化车辆延误和排队长度
"""

import json
import time
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
import os

from env.traffic_env import TrafficEnvironment
from agent.gpu_dqn_agent import GPUAcceleratedDQNAgent
from visualization.visualizer import TrafficVisualizer
from log.result_logger import log_and_save_results


def load_config(config_file: str) -> Dict:
    """Load configuration file
    加载配置文件"""
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_system_info():
    """Print system information
    打印系统信息"""
    print("=" * 70)
    print("Smart Traffic Signal Control System - Two-Phase Optimization 智能交通信号控制系统 - 两相位优化版本")
    print("=" * 70)
    print("System Features: 系统特点:")
    print("- Simplified two-phase signal system - 简化的两相位信号系统")
    print("- Dynamic signal control based on reinforcement learning - 基于强化学习的动态信号控制")
    print("- GPU-accelerated DQN training - GPU加速的DQN训练")
    print("- Real-time vehicle flow simulation - 实时车辆流量模拟")
    print("- Visualized traffic state - 可视化交通状态")
    print("=" * 70)



def demo_basic_simulation():
    """Basic simulation demo (using the latest trained DQN agent model)
    基础模拟演示（使用最新训练好的DQN智能体模型）"""
    print("\nStarting basic traffic simulation demo (AI model control)...\n开始基础交通模拟演示（AI模型控制）...")
    
    # Create environment and visualizer
    env = TrafficEnvironment()
    visualizer = TrafficVisualizer()
    
    # Load DQN agent and latest model
    agent = GPUAcceleratedDQNAgent()
    model_path = os.path.join("models", "trained_model_gpu.pth")
    try:
        agent.load_model(model_path)
        print("[INFO] Loaded trained AI model for control 已加载训练好的AI模型进行控制")
    except FileNotFoundError:
        print("[WARN] Trained model not found, using random policy 未找到训练好的模型，使用随机策略")
        agent.epsilon = 1.0
    
    # Run simulation
    metrics_history = []
    simulation_time = 3600  # 1 hour
    print(f"Running {simulation_time} seconds of traffic simulation... 运行 {simulation_time} 秒的交通模拟...")
    
    for step in range(simulation_time):
        # Use AI model to decide green light duration
        action = agent.act(env.get_state())
        env.set_phase_duration(action)
        state = env.step()
        metrics_history.append(state)
        
        # Update display every 10 seconds
        if step % 10 == 0:
            print(f"Time: {state['current_time']:.1f}s, Phase: {state['current_phase']}, Green: {action:.1f}s, Queue: {state['total_queue_length']} vehicles, Avg Delay: {state['average_delay']:.1f}s 时间: {state['current_time']:.1f}s, 相位: {state['current_phase']}, 绿灯时间: {action:.1f}s, 排队: {state['total_queue_length']}辆, 平均延误: {state['average_delay']:.1f}s")
    
    # Show final statistics
    final_state = metrics_history[-1]
    print(f"\nSimulation complete! 模拟完成!")
    print(f"Total vehicles: {final_state['stats']['total_vehicles']} 总车辆数: {final_state['stats']['total_vehicles']}")
    print(f"Average delay: {final_state['average_delay']:.2f}s 平均延误: {final_state['average_delay']:.2f}秒")
    print(f"Objective value: {final_state['objective_value']:.2f} 目标函数值: {final_state['objective_value']:.2f}")
    print(f"Throughput: {final_state['stats']['throughput']} vehicles 吞吐量: {final_state['stats']['throughput']}辆")

    # Read signal parameters
    with open('simulator_config.json', 'r', encoding='utf-8') as f:
        sim_config = json.load(f)
    sim_params = sim_config["optimization"]["signal_control"]
    # Read objective function weight parameters
    delay_weight = sim_config["optimization"]["objective_weights"]["delay"]
    queue_weight = sim_config["optimization"]["objective_weights"]["queue"]
    # Read training parameters
    with open('training_config.json', 'r', encoding='utf-8') as f:
        train_config = json.load(f)
    train_params = train_config["training_parameters"]
    train_schedule = train_config.get("training_schedule")
    early_stopping = train_config.get("early_stopping")

    # Call unified logging and saving function
    log_and_save_results(train_params, sim_params, delay_weight, queue_weight, final_state, agent.epsilon, '1.basic_simulation_result.csv', train_schedule, early_stopping)

    # Plot performance metrics
    visualizer.plot_performance_metrics(metrics_history)
    return env, visualizer, metrics_history


def demo_ai_training():
    """AI training demo
    AI训练演示"""
    print("\nStarting AI agent training demo...")
    
    # Create environment and agent
    env = TrafficEnvironment()
    agent = GPUAcceleratedDQNAgent()
    
    # Training parameters
    episodes = 500  # Reduce training time for demonstration
    max_steps = 2000
    print(f"Training parameters: {episodes} episodes, {max_steps} steps per episode")
    print("Starting training...")
    
    # Train agent
    start_time = time.time()
    agent.train(env, episodes=episodes, max_steps=max_steps)
    training_time = time.time() - start_time
    print(f"Training complete! Time: {training_time:.2f} seconds")
    
    # Display training statistics
    stats = agent.get_training_stats()
    print(f"Training episodes: {stats['episodes']}")
    print(f"Average reward: {stats['avg_reward']:.2f}")
    print(f"Final exploration rate: {agent.epsilon:.3f}")
    
    # Model is automatically saved and cleaned up during training
    print(f"[INFO] Training complete, final model saved as: models/trained_model_gpu.pth")
    
    # Automatically execute basic simulation after training
    print(f"\n[INFO] Starting basic simulation after training...")
    demo_basic_simulation()

    return agent


def demo_ai_performance():
    """AI performance demo
    AI性能演示"""
    print("\nStarting AI performance demo...")
    
    env = TrafficEnvironment()
    agent = GPUAcceleratedDQNAgent()
    
    # Try to load trained model
    model_path = os.path.join("models", "trained_model_gpu.pth")
    try:
        agent.load_model(model_path)
        print("Loaded trained GPU model")
    except FileNotFoundError:
        print("Trained model not found, using random policy")
        agent.epsilon = 1.0
    
    metrics_history = []
    simulation_time = 3600
    print(f"Running {simulation_time} seconds of AI control simulation... 运行 {simulation_time} 秒的AI控制模拟...")
    print(f"[INFO] Inference device: {agent.device}")
    
    for step in range(simulation_time):
        current_state = env.get_state()
        action = agent.act(current_state)
        env.set_phase_duration(action)
        state = env.step()
        metrics_history.append(state)
        
        if step % 10 == 0:
            print(f"Time: {state['current_time']:.1f}s, Phase: {state['current_phase']}, Green: {action:.1f}s, Queue: {state['total_queue_length']} vehicles, Avg Delay: {state['average_delay']:.1f}s 时间: {state['current_time']:.1f}s, 相位: {state['current_phase']}, 绿灯时间: {action:.1f}s, 排队: {state['total_queue_length']}辆, 平均延误: {state['average_delay']:.1f}s")
    
    final_state = metrics_history[-1]
    print(f"\nAI control simulation complete! 模拟完成!")
    print(f"Total vehicles: {final_state['stats']['total_vehicles']} 总车辆数: {final_state['stats']['total_vehicles']}")
    print(f"Average delay: {final_state['average_delay']:.2f}s 平均延误: {final_state['average_delay']:.2f}秒")
    print(f"Objective value: {final_state['objective_value']:.2f} 目标函数值: {final_state['objective_value']:.2f}")
    print(f"Throughput: {final_state['stats']['throughput']} vehicles 吞吐量: {final_state['stats']['throughput']}辆")

    # Unified parameter retrieval
    train_params = agent.training_config["training_parameters"]
    sim_params = agent.simulator_config["optimization"]["signal_control"]
    delay_weight = agent.simulator_config["optimization"]["objective_weights"]["delay"]
    queue_weight = agent.simulator_config["optimization"]["objective_weights"]["queue"]

    # Get training configuration
    train_schedule = agent.training_config.get("training_schedule")
    early_stopping = agent.training_config.get("early_stopping")
    
    # Call unified logging and saving function
    log_and_save_results(train_params, sim_params, delay_weight, queue_weight, final_state, agent.epsilon, '3.AI_simulation_result.csv', train_schedule, early_stopping)
    
    return env, metrics_history


def compare_strategies():
    """Compare performance of different strategies
    比较不同策略的性能"""
    print("\nStarting strategy performance comparison...")
    
    strategies = {
        "Fixed Time (15s)": 15.0,
        "Fixed Time (20s)": 20.0,
        "Fixed Time (30s)": 30.0
    }
    
    results = {}
    
    for strategy_name, green_time in strategies.items():
        print(f"\nTesting strategy: {strategy_name}")
        
        env = TrafficEnvironment()
        env.set_phase_duration(green_time)
        
        metrics_history = []
        simulation_time = 600  # 10 minutes test
        
        for step in range(simulation_time):
            state = env.step()
            metrics_history.append(state)
        
        final_state = metrics_history[-1]
        results[strategy_name] = {
            'avg_delay': final_state['average_delay'],
            'total_queue': final_state['total_queue_length'],
            'objective_value': final_state['objective_value'],
            'total_vehicles': final_state['stats']['total_vehicles'],
            'throughput': final_state['stats']['throughput']
        }
        
        print(f"   Average delay: {final_state['average_delay']:.2f}s")
        print(f"   Total queue: {final_state['total_queue_length']} vehicles")
        print(f"   Objective value: {final_state['objective_value']:.2f}")
        print(f"   Throughput: {final_state['stats']['throughput']} vehicles")
    
    # Plot comparison results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    strategy_names = list(results.keys())
    avg_delays = [results[s]['avg_delay'] for s in strategy_names]
    total_queues = [results[s]['total_queue'] for s in strategy_names]
    objective_values = [results[s]['objective_value'] for s in strategy_names]
    throughputs = [results[s]['throughput'] for s in strategy_names]
    
    # Average delay
    axes[0, 0].bar(strategy_names, avg_delays, color='red', alpha=0.7)
    axes[0, 0].set_title('Average Delay', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Seconds')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Total queue length
    axes[0, 1].bar(strategy_names, total_queues, color='blue', alpha=0.7)
    axes[0, 1].set_title('Total Queue Length', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Vehicles')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Objective function value
    axes[1, 0].bar(strategy_names, objective_values, color='green', alpha=0.7)
    axes[1, 0].set_title('Objective Function Value', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Throughput
    axes[1, 1].bar(strategy_names, throughputs, color='purple', alpha=0.7)
    axes[1, 1].set_title('Throughput', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Vehicles')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return results


def interactive_demo():
    """Interactive demo (using AI model control)
    交互式演示（使用AI模型控制）"""
    print("\nStarting interactive demo (AI control)...")
    
    env = TrafficEnvironment()
    visualizer = TrafficVisualizer()
    agent = GPUAcceleratedDQNAgent()
    model_path = os.path.join("models", "trained_model_gpu.pth")
    try:
        agent.load_model(model_path)
        print("[INFO] Loaded trained AI model for control 已加载训练好的AI模型进行控制")
    except FileNotFoundError:
        print("[WARN] Trained model not found, using random policy 未找到训练好的模型，使用随机策略")
        agent.epsilon = 1.0
    
    print("Press 'q' to exit demo")
    print("Press 's' to switch signal phase (not supported yet)")
    print("Press 'r' to reset environment (not supported yet)")
    
    try:
        while True:
            # Use AI agent to decide green light duration
            action = agent.act(env.get_state())
            env.set_phase_duration(action)
            state = env.step()
            
            # Update display
            visualizer.update_display(state)
            
            # Check user input
            if plt.waitforbuttonpress(timeout=0.1):
                break
                
    except KeyboardInterrupt:
        print("\nDemo ended")
    
    plt.close()


def main():
    """Main function
    主函数"""
    print_system_info()
    
    while True:
        print("\nPlease select demo mode:")
        print("1. Basic traffic simulation")
        print("2. AI agent training")
        print("3. AI performance demo")
        print("4. Strategy performance comparison")
        print("5. Interactive demo")
        print("6. Exit")
        
        choice = input("\nPlease enter your choice (1-6): ").strip()
        
        if choice == '1':
            demo_basic_simulation()
        elif choice == '2':
            demo_ai_training()
        elif choice == '3':
            demo_ai_performance()
        elif choice == '4':
            compare_strategies()
        elif choice == '5':
            interactive_demo()
        elif choice == '6':
            print("Thank you for using the Smart Traffic Signal Control System!")
            break
        else:
            print("Invalid choice, please try again")


if __name__ == "__main__":
    main()
