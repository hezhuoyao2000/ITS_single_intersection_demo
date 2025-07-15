import json
import random
from typing import List, Dict, Tuple
from traffic.vehicle import Vehicle, VehicleType, Direction
from utils.distributions import poisson_arrival_time, generate_vehicle_type, get_lane_direction


class FlowGenerator:
    """Traffic flow generator\n车流生成器"""
    
    def __init__(self, simulator_config: str = "simulator_config.json"):
        """
        Initialize traffic flow generator\n初始化车流生成器
        Args:
            simulator_config: Simulator config file path\n模拟器配置文件路径
        """
        self.config = self._load_config(simulator_config)
        self.vehicle_id_counter = 0
        self.lanes = ["N_L", "N_R", "S_L", "S_R", "E_L", "E_R", "W_L", "W_R"]
        
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration file\n加载配置文件"""
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def generate_vehicle_arrival(self, lane: str, current_time: float) -> Vehicle:
        """
        Generate a vehicle arrival event for the specified lane\n为指定车道生成一个车辆到达事件
        Args:
            lane: Lane identifier\n车道标识
            current_time: Current time\n当前时间
        Returns:
            Generated vehicle object\n生成的车辆对象
        """
        # Get lane arrival rate\n获取车道到达率
        arrival_rate = self.config["traffic_flow"]["arrival_rates"][lane]
        
        # Generate arrival interval\n生成到达时间间隔
        arrival_interval = poisson_arrival_time(arrival_rate)
        arrival_time = current_time + arrival_interval
        
        # Generate vehicle type\n生成车辆类型
        car_ratio = self.config["traffic_flow"]["vehicle_types"]["car"]["ratio"]
        vehicle_type_str = generate_vehicle_type(car_ratio)
        vehicle_type = VehicleType(vehicle_type_str)
        
        # Determine driving direction\n确定行驶方向
        direction_str = get_lane_direction(lane)
        direction = Direction(direction_str)
        
        # Create vehicle object\n创建车辆对象
        vehicle = Vehicle(
            vehicle_id=self.vehicle_id_counter,
            vehicle_type=vehicle_type,
            direction=direction,
            arrival_time=arrival_time,
            lane=lane
        )
        
        self.vehicle_id_counter += 1
        return vehicle
    
    def generate_vehicles_for_time_period(self, start_time: float, end_time: float) -> List[Vehicle]:
        """
        Generate vehicle arrival events for all lanes in the specified time period\n为指定时间段生成所有车道的车辆到达事件
        Args:
            start_time: Start time\n开始时间
            end_time: End time\n结束时间
        Returns:
            List of vehicles, sorted by arrival time\n车辆列表，按到达时间排序
        """
        vehicles = []
        
        for lane in self.lanes:
            current_time = start_time
            while current_time < end_time:
                vehicle = self.generate_vehicle_arrival(lane, current_time)
                if vehicle.arrival_time < end_time:
                    vehicles.append(vehicle)
                    current_time = vehicle.arrival_time
                else:
                    break
        
        # Sort by arrival time\n按到达时间排序
        vehicles.sort(key=lambda v: v.arrival_time)
        return vehicles
    
    def get_lane_arrival_rate(self, lane: str) -> float:
        """Get arrival rate for the specified lane\n获取指定车道的到达率"""
        return self.config["traffic_flow"]["arrival_rates"][lane]
    
    def get_all_lanes(self) -> List[str]:
        """Get all lane identifiers\n获取所有车道标识"""
        return self.lanes.copy()
    
    def get_vehicle_type_ratio(self) -> Dict[str, float]:
        """Get vehicle type ratios\n获取车辆类型比例"""
        vehicle_types = self.config["traffic_flow"]["vehicle_types"]
        return {
            "car": vehicle_types["car"]["ratio"],
            "truck": vehicle_types["truck"]["ratio"]
        }
    
    def get_direction_ratio(self) -> Dict[str, float]:
        """Get direction ratios (simplified version)\n获取行驶方向比例（简化版本）"""
        return {
            "straight": 0.7,
            "right": 0.3
        }
