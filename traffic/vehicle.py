import random
from enum import Enum
from typing import Optional


class VehicleType(Enum):
    """Vehicle type enumeration\n车辆类型枚举"""
    CAR = "car"
    TRUCK = "truck"


class Direction(Enum):
    """Driving direction enumeration\n行驶方向枚举"""
    STRAIGHT = "straight"
    LEFT = "left"
    RIGHT = "right"


class Vehicle:
    """Vehicle class\n车辆类"""
    
    def __init__(self, vehicle_id: int, vehicle_type: VehicleType, direction: Direction, 
                 arrival_time: float, lane: str):
        """
        Initialize vehicle\n初始化车辆
        Args:
            vehicle_id: Unique vehicle identifier\n车辆唯一标识
            vehicle_type: Vehicle type (car/truck)\n车辆类型（小车/大车）
            direction: Driving direction (straight/left/right)\n行驶方向（直行/左转/右转）
            arrival_time: Arrival time\n到达时间
            lane: Lane (N_L, N_R, S_L, S_R, E_L, E_R, W_L, W_R)\n所在车道（N_L, N_R, S_L, S_R, E_L, E_R, W_L, W_R）
        """
        self.vehicle_id = vehicle_id
        self.vehicle_type = vehicle_type
        self.direction = direction
        self.arrival_time = arrival_time
        self.lane = lane
        
        # Vehicle state\n车辆状态
        self.enter_queue_time = None  # Time entering queue\n进入排队时间
        self.exit_queue_time = None   # Time leaving queue\n离开排队时间
        self.passage_time = None      # Time to pass intersection\n通过路口时间
        self.waiting_time = 0.0  # New: accumulated waiting time\n新增：累计等待时间
        
        # Calculate time required to pass intersection\n计算车辆通过路口所需时间
        self._calculate_passage_time()
    
    def _calculate_passage_time(self):
        """Calculate time to pass intersection based on vehicle type and direction\n根据车辆类型和方向计算通过路口时间"""
        from utils.distributions import get_truncated_normal
        
        # Determine time distribution parameters\n确定时间分布参数
        if self.vehicle_type == VehicleType.CAR:
            if self.direction == Direction.STRAIGHT:
                mean, std = 7, 2
            elif self.direction == Direction.LEFT:
                mean, std = 5, 1
            else:  # RIGHT
                mean, std = 10, 2
        else:  # TRUCK
            if self.direction == Direction.STRAIGHT:
                mean, std = 10, 3
            elif self.direction == Direction.LEFT:
                mean, std = 8, 2
            else:  # RIGHT
                mean, std = 12, 3
        
        # Generate truncated normal distribution for passage time\n生成截断正态分布的通过时间
        self.passage_time = get_truncated_normal(mean, std, mean - 2*std, mean + 2*std)
    
    def enter_queue(self, time: float):
        """Vehicle enters queue\n车辆进入排队"""
        self.enter_queue_time = time
        self.waiting_time = 0.0  # Reset waiting time when entering queue\n进入队列时等待时间归零
    
    def exit_queue(self, time: float):
        """Vehicle leaves queue\n车辆离开排队"""
        self.exit_queue_time = time
    
    def get_delay_time(self) -> float:
        """Calculate delay time (from arrival to leaving queue)\n计算延误时间（从到达时间到离开排队时间）"""
        if self.exit_queue_time is None or self.enter_queue_time is None:
            return 0.0
        return self.exit_queue_time - self.arrival_time
    
    def get_queue_time(self) -> float:
        """Calculate queue time (from entering to leaving queue)\n计算排队时间（从进入排队到离开排队）"""
        if self.exit_queue_time is None or self.enter_queue_time is None:
            return 0.0
        return self.exit_queue_time - self.enter_queue_time
    
    def get_passage_time(self) -> float:
        """Get time required for vehicle to pass intersection\n获取车辆通过路口所需时间"""
        return self.passage_time if self.passage_time is not None else 0.0
    
    def get_waiting_time(self) -> float:
        """Get vehicle waiting time\n获取车辆等待时间"""
        return self.waiting_time
    
    def update_waiting_time(self, time_step: float):
        """Update vehicle waiting time (for environment step)\n更新车辆等待时间（用于环境步进）"""
        # This method is mainly used to update waiting time for vehicles in the environment\n这个方法主要用于环境中的车辆等待时间更新
        if self.enter_queue_time is not None and self.exit_queue_time is None:
            self.waiting_time += time_step
    
    def complete_passage(self, completion_time: float):
        """Complete passing intersection\n完成通过路口"""
        self.exit_queue_time = completion_time
    
    def get_lane(self) -> str:
        """Get vehicle lane\n获取车辆所在车道"""
        return self.lane
    
    def __str__(self):
        return f"Vehicle(id={self.vehicle_id}, type={self.vehicle_type.value}, " \
               f"direction={self.direction.value}, lane={self.lane})"
    
    def __repr__(self):
        return self.__str__()
