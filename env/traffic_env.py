import json
import time
from typing import List, Dict, Tuple, Optional
from enum import Enum
from traffic.vehicle import Vehicle, VehicleType, Direction
from traffic.flow_generator import FlowGenerator
from utils.distributions import calculate_queue_length, calculate_average_delay, calculate_weighted_objective


class SignalPhase(Enum):
    """Simplified two-phase signal system\n简化的两相位信号系统"""
    NS_STRAIGHT_RIGHT = "NS_STRAIGHT_RIGHT"  # North-South straight and right turn\n南北方向直行和右转
    EW_STRAIGHT_RIGHT = "EW_STRAIGHT_RIGHT"  # East-West straight and right turn\n东西方向直行和右转


class TrafficEnvironment:
    """Simplified traffic environment class - two-phase system\n简化的交通环境类 - 两相位系统"""
    
    def __init__(self, simulator_config: str = "simulator_config.json"):
        """
        Initialize traffic environment\n初始化交通环境
        Args:
            simulator_config: Simulator config file path\n模拟器配置文件路径
        """
        self.config = self._load_config(simulator_config)
        self.flow_generator = FlowGenerator(simulator_config)
        
        # Environment state\n环境状态
        self.current_time = 0.0
        self.current_phase = SignalPhase.NS_STRAIGHT_RIGHT
        self.phase_start_time = 0.0
        self.phase_duration = 0.0
        
        # Vehicle queues (grouped by lane)\n车辆队列（按车道分组）
        self.queues = {lane: [] for lane in self.flow_generator.get_all_lanes()}
        self.completed_vehicles = []
        
        # Signal light state\n信号灯状态
        self.signal_states = self._initialize_signal_states()
        
        # Statistics\n统计信息
        self.stats = {
            'total_vehicles': 0,
            'total_delay': 0.0,
            'total_queue_time': 0.0,
            'phase_changes': 0,
            'throughput': 0
        }
    
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration file\n加载配置文件"""
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _initialize_signal_states(self) -> Dict[str, str]:
        """Initialize signal light state\n初始化信号灯状态"""
        return {
            'N_L': 'red', 'N_R': 'red',
            'S_L': 'red', 'S_R': 'red',
            'E_L': 'red', 'E_R': 'red',
            'W_L': 'red', 'W_R': 'red'
        }
    
    def reset(self):
        """Reset environment state\n重置环境状态"""
        self.current_time = 0.0
        self.current_phase = SignalPhase.NS_STRAIGHT_RIGHT
        self.phase_start_time = 0.0
        self.phase_duration = 0.0
        
        # Clear queues\n清空队列
        for lane in self.queues:
            self.queues[lane] = []
        self.completed_vehicles = []
        
        # Reset signal lights\n重置信号灯
        self.signal_states = self._initialize_signal_states()
        
        # Reset statistics\n重置统计信息
        self.stats = {
            'total_vehicles': 0,
            'total_delay': 0.0,
            'total_queue_time': 0.0,
            'phase_changes': 0,
            'throughput': 0
        }
    
    def set_phase_duration(self, duration: float):
        """
        Set current phase duration\n设置当前相位持续时间
        
        Args:
            duration: Duration (seconds)\n持续时间（秒）
        """
        self.phase_duration = duration
    
    def get_current_phase_info(self) -> Tuple[SignalPhase, float, float]:
        """
        Get current phase information\n获取当前相位信息
        
        Returns:
            (Current phase, phase start time, phase duration)\n(当前相位, 相位开始时间, 相位持续时间)
        """
        return self.current_phase, self.phase_start_time, self.phase_duration
    
    def get_signal_states(self) -> Dict[str, str]:
        """
        Get current signal light state\n获取当前信号灯状态
        
        Returns:
            信号灯状态字典
        """
        return self.signal_states.copy()
    
    def get_queue_lengths(self) -> Dict[str, int]:
        """
        Get queue lengths for each lane\n获取各车道排队长度
        
        Returns:
            排队长度字典
        """
        return {lane: len(queue) for lane, queue in self.queues.items()}
    
    def get_total_queue_length(self) -> int:
        """
        Get total queue length\n获取总排队长度
        
        Returns:
            总排队车辆数
        """
        return sum(len(queue) for queue in self.queues.values())
    
    def get_average_delay(self) -> float:
        """
        Get average delay time\n获取平均延误时间
        
        Returns:
            平均延误时间（秒）
        """
        if not self.completed_vehicles:
            return 0.0
        return sum(v.get_delay_time() for v in self.completed_vehicles) / len(self.completed_vehicles)
    
    def get_objective_value(self) -> float:
        """
        Get objective value\n获取目标函数值
        
        Returns:
            加权目标函数值
        """
        avg_delay = self.get_average_delay()
        total_queue = self.get_total_queue_length()
        
        delay_weight = self.config["optimization"]["objective_weights"]["delay"]
        queue_weight = self.config["optimization"]["objective_weights"]["queue"]
        
        return calculate_weighted_objective(avg_delay, total_queue, delay_weight, queue_weight)
    
    def _update_signal_states(self, phase: SignalPhase):
        """
        Update signal light state based on phase\n根据相位更新信号灯状态
        
        Args:
            phase: Current phase\n当前相位
        """
        # Reset all signal lights to red
        for lane in self.signal_states:
            self.signal_states[lane] = 'red'
        
        # Set green lights based on phase
        if phase == SignalPhase.NS_STRAIGHT_RIGHT:
            # North-South green lights
            self.signal_states['N_L'] = 'green'
            self.signal_states['N_R'] = 'green'
            self.signal_states['S_L'] = 'green'
            self.signal_states['S_R'] = 'green'
        elif phase == SignalPhase.EW_STRAIGHT_RIGHT:
            # East-West green lights
            self.signal_states['E_L'] = 'green'
            self.signal_states['E_R'] = 'green'
            self.signal_states['W_L'] = 'green'
            self.signal_states['W_R'] = 'green'
    
    def _get_next_phase(self, current_phase: SignalPhase) -> SignalPhase:
        """
        Get next phase\n获取下一个相位
        
        Args:
            current_phase: Current phase\n当前相位
            
        Returns:
            下一个相位
        """
        if current_phase == SignalPhase.NS_STRAIGHT_RIGHT:
            return SignalPhase.EW_STRAIGHT_RIGHT
        else:
            return SignalPhase.NS_STRAIGHT_RIGHT
    
    def _get_phase_duration(self, phase: SignalPhase) -> float:
        """
        Get phase duration\n获取相位持续时间
        
        Args:
            phase: Phase\n相位
            
        Returns:
            持续时间（秒）
        """
        if phase == SignalPhase.NS_STRAIGHT_RIGHT:
            return self.phase_duration
        else:
            return self.phase_duration
    
    def _process_vehicle_arrivals(self, time_step: float):
        """
        Process vehicle arrivals\n处理车辆到达
        
        Args:
            time_step: Time step\n时间步长
        """
        # Generate new vehicles
        new_vehicles = self.flow_generator.generate_vehicles_for_time_period(
            self.current_time, self.current_time + time_step
        )
        
        # Add vehicles to their respective lane queues
        for vehicle in new_vehicles:
            if vehicle.arrival_time <= self.current_time + time_step:
                lane = vehicle.lane
                if lane in self.queues:
                    vehicle.enter_queue(self.current_time)
                    self.queues[lane].append(vehicle)
                    self.stats['total_vehicles'] += 1
    
    def _process_vehicle_departures(self, time_step: float):
        """
        Process vehicle departures\n处理车辆离开
        
        Args:
            time_step: Time step\n时间步长
        """
        # Get current green lanes
        green_lanes = [lane for lane, state in self.signal_states.items() if state == 'green']
        
        # Process vehicle departures for green lanes
        for lane in green_lanes:
            if self.queues[lane]:
                # Calculate number of vehicles that can depart
                vehicle = self.queues[lane][0]
                passage_time = vehicle.get_passage_time()
                
                # If vehicle can complete passage
                if vehicle.get_waiting_time() + time_step >= passage_time:
                    # Vehicle departs
                    completed_vehicle = self.queues[lane].pop(0)
                    completed_vehicle.complete_passage(self.current_time)
                    self.completed_vehicles.append(completed_vehicle)
                    self.stats['throughput'] += 1
    
    def step(self, time_step: float = 1.0) -> Dict:
        """
        Environment step\n环境步进
        
        Args:
            time_step: Time step\n时间步长
            
        Returns:
            环境状态字典
        """
        # Process vehicle arrivals
        self._process_vehicle_arrivals(time_step)
        
        # Check if phase needs to be changed
        phase_elapsed = self.current_time - self.phase_start_time
        if phase_elapsed >= self.phase_duration:
            # Change phase
            self.current_phase = self._get_next_phase(self.current_phase)
            self.phase_start_time = self.current_time
            self.stats['phase_changes'] += 1
        
        # Update signal light state
        self._update_signal_states(self.current_phase)
        
        # Process vehicle departures
        self._process_vehicle_departures(time_step)
        
        # Update time
        self.current_time += time_step
        
        # Update vehicle waiting times
        for lane_queues in self.queues.values():
            for vehicle in lane_queues:
                vehicle.update_waiting_time(time_step)
        
        # Return current state
        return self.get_state()
    
    def get_state(self) -> Dict:
        """
        Get current environment state\n获取当前环境状态
        
        Returns:
            状态字典
        """
        queue_lengths = self.get_queue_lengths()
        total_queue = self.get_total_queue_length()
        avg_delay = self.get_average_delay()
        objective_value = self.get_objective_value()
        
        return {
            'current_time': self.current_time,
            'current_phase': self.current_phase.value,
            'signal_states': self.signal_states.copy(),
            'queue_lengths': queue_lengths,
            'total_queue_length': total_queue,
            'average_delay': avg_delay,
            'objective_value': objective_value,
            'stats': self.stats.copy(),
            'completed_vehicles_count': len(self.completed_vehicles)
        }
