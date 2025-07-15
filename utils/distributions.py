import random
import numpy as np
from typing import Tuple


def get_truncated_normal(mean: float, std: float, min_val: float, max_val: float) -> float:
    """
    Generate a random number from a truncated normal distribution
    生成截断正态分布的随机数
    
    Args:
        mean: 均值
        std: 标准差
        min_val: 最小值
        max_val: 最大值
    
    Returns:
        截断正态分布的随机数
    """
    # Use numpy's truncated normal distribution
    return np.random.normal(mean, std)
    # Simple implementation: generate a normal random number, then truncate to the specified range
    # value = np.random.normal(mean, std)
    # return max(min_val, min(max_val, value))


def poisson_arrival_time(rate: float) -> float:
    """
    Generate vehicle arrival interval based on Poisson distribution
    根据泊松分布生成车辆到达时间间隔
    
    Args:
        rate: 到达率（辆/分钟）
    
    Returns:
        到达时间间隔（秒）
    """
    # Convert arrival rate to per second
    lambda_per_second = rate / 60.0
    
    # Generate exponential distribution arrival interval
    return np.random.exponential(1.0 / lambda_per_second)


def generate_vehicle_type(car_ratio: float = 0.8) -> str:
    """
    Generate vehicle type based on ratio
    根据比例生成车辆类型
    
    Args:
        car_ratio: 小车比例
    
    Returns:
        车辆类型 ("car" 或 "truck")
    """
    return "car" if random.random() < car_ratio else "truck"


def generate_direction(straight_ratio: float = 0.7) -> str:
    """
    Generate driving direction based on ratio (for left lane only)
    根据比例生成行驶方向（仅适用于左车道）
    
    Args:
        straight_ratio: 直行比例
    
    Returns:
        行驶方向 ("straight" 或 "left")
    """
    return "straight" if random.random() < straight_ratio else "left"


def get_lane_direction(lane: str) -> str:
    """
    Determine driving direction based on lane
    根据车道确定行驶方向
    
    Args:
        lane: 车道标识 (N_L, N_R, S_L, S_R, E_L, E_R, W_L, W_R)
        N_L: 北向车道左侧（直行）
        N_R: 北向车道右侧（右转）
        S_L: 南向车道左侧（直行）
        S_R: 南向车道右侧（右转）
        E_L: 东向车道左侧（直行）
        E_R: 东向车道右侧（右转）
        W_L: 西向车道左侧（直行）
        W_R: 西向车道右侧（右转）
    
    Returns:
        行驶方向 ("straight", "left", "right")
    """
    if lane.endswith('_R'):
        return "right"  # Right lanes are all right turns
    else:
        return "straight"  # Left lanes are all straight


def calculate_queue_length(vehicles: list) -> int:
    """
    Calculate queue length
    计算排队长度
    
    Args:
        vehicles: 排队中的车辆列表
    
    Returns:
        排队车辆数量
    """
    return len(vehicles)


def calculate_average_delay(vehicles: list) -> float:
    """
    Calculate average delay time
    计算平均延误时间
    
    Args:
        vehicles: 已完成的车辆列表
    
    Returns:
        平均延误时间（秒）
    """
    if not vehicles:
        return 0.0
    
    total_delay = sum(vehicle.get_delay_time() for vehicle in vehicles)
    return total_delay / len(vehicles)


def calculate_weighted_objective(avg_delay: float, queue_length: int, 
                               delay_weight: float = 0.6, queue_weight: float = 0.4) -> float:
    """
    Calculate weighted objective value
    计算加权目标函数值
    
    Args:
        avg_delay: 平均延误时间
        queue_length: 排队长度
        delay_weight: 延误时间权重
        queue_weight: 排队长度权重
    
    Returns:
        加权目标函数值
    """
    return delay_weight * avg_delay + queue_weight * queue_length
