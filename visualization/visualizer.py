import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.text import Text
import numpy as np
from typing import Dict, List, Optional
import time


class TrafficVisualizer:
    """Traffic visualizer - supports two-phase system\n交通可视化器 - 支持两相位系统"""
    
    def __init__(self):
        """Initialize visualizer\n初始化可视化器"""
        self.fig, self.ax = plt.subplots(figsize=(16, 14))
        self.setup_intersection()
    
    def setup_intersection(self):
        """Set up intersection layout\n设置路口布局"""
        self.ax.clear()
        self.ax.set_xlim(-12, 12)
        self.ax.set_ylim(-12, 12)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        # Draw roads\n绘制道路
        self._draw_roads()
        # Draw lane markers\n绘制车道标识
        self._draw_lanes()
        # Draw traffic light positions\n绘制信号灯位置
        self._draw_traffic_lights()
        # Draw vehicle direction arrows\n绘制车辆行进方向箭头
        self._draw_direction_arrows()
        # Draw intersection title\n绘制路口标题
        self.ax.set_title('Smart Traffic Signal Control System - Two-Phase Optimization', 
                         fontsize=18, fontweight='bold', pad=25)
    
    def _draw_roads(self):
        """Draw roads\n绘制道路"""
        # North-south road\n南北方向道路
        self.ax.add_patch(patches.Rectangle((-2.5, -10), 5, 20, 
                                          facecolor='gray', alpha=0.3))
        # East-west road\n东西方向道路
        self.ax.add_patch(patches.Rectangle((-10, -2.5), 20, 5, 
                                          facecolor='gray', alpha=0.3))
        # Road center lines\n道路中心线
        self.ax.plot([0, 0], [-10, 10], 'white', linewidth=3)
        self.ax.plot([1.2, 1.2], [-10, 10], 'gray', linewidth=3)
        self.ax.plot([-1.2, -1.2], [-10, 10], 'gray', linewidth=3)
        self.ax.plot([-10, 10], [0, 0], 'white', linewidth=3)
        self.ax.plot([-10, 10], [1.2, 1.2], 'gray', linewidth=3)
        self.ax.plot([-10, 10], [-1.2, -1.2], 'gray', linewidth=3)
        # Intersection center\n路口中心
        self.ax.add_patch(patches.Circle((0, 0), 1, facecolor='yellow', alpha=0.5))
    
    def _draw_lanes(self):
        """Draw lane markers\n绘制车道标识"""
        # Lane marker positions - adjusted for traffic rules layout\n车道标识位置 - 修正为符合交通规则的布局
        lane_positions = {
            # Northbound lane (north to south)\n北向车道（从北向南）
            'N_R': (0.5, 8), 'N_L': (2, 8),
            # Southbound lane (south to north)\n南向车道（从南向北）
            'S_L': (-2, -8), 'S_R': (-0.5, -8),
            # Eastbound lane (east to west)\n东向车道（从东向西）
            'E_R': (8, -0.5), 'E_L': (8, -2),
            # Westbound lane (west to east)\n西向车道（从西向东）
            'W_R': (-8, 0.5), 'W_L': (-8, 2)
        }
        for lane, pos in lane_positions.items():
            self.ax.text(pos[0], pos[1], lane, fontsize=12, 
                        ha='center', va='center', 
                        bbox=dict(boxstyle="round,pad=0.4", 
                                facecolor="lightblue", alpha=0.8))
    
    def _draw_traffic_lights(self):
        """Draw traffic light positions\n绘制信号灯位置"""
        light_positions = {
            # Northbound lane traffic light\n北向车道信号灯
            'N_R': (0.5, 2), 'N_L': (2, 2),
            # Southbound lane (south to north)\n南向车道（从南向北）
            'S_L': (-2, -2), 'S_R': (-0.5, -2),
            # Eastbound lane (east to west)\n东向车道（从东向西）
            'E_R': (2, -0.5), 'E_L': (2, -2),
            # Westbound lane (west to east)\n西向车道（从西向东）
            'W_R': (-2, 0.5), 'W_L': (-2, 2)
        }
        for lane, pos in light_positions.items():
            self.ax.add_patch(patches.Circle(pos, 0.4, 
                                           facecolor='red', alpha=0.8))
            self.ax.text(pos[0], pos[1], lane, fontsize=9, 
                        ha='center', va='center', color='white', fontweight='bold')
    
    def _draw_direction_arrows(self):
        """Draw vehicle direction arrows - adjusted for traffic rules\n绘制车辆行进方向箭头 - 修正为符合交通规则的方向"""
        # Define arrow positions and directions - two lanes per direction, left for straight, right for right turn\n定义箭头位置和方向 - 每个方向的两条车道，左侧直行，右侧右转
        arrow_configs = {
            # Northbound lane (north to south) - left for straight, right for right turn\n北向车道（从北向南）- 左侧直行，右侧右转
            'N_L': {'pos': (2, 4), 'direction': 'down', 'color': 'blue', 'label': 'Straight 直行'},
            'N_R': {'pos': (1, 4), 'direction': 'left_down', 'color': 'green', 'label': 'Right Turn 右转'},
            # Southbound lane (south to north) - left for straight, right for right turn\n南向车道（从南向北）- 左侧直行，右侧右转
            'S_L': {'pos': (-2, -4), 'direction': 'up', 'color': 'blue', 'label': 'Straight 直行'},
            'S_R': {'pos': (-1, -4), 'direction': 'right_up', 'color': 'green', 'label': 'Right Turn 右转'},
            # Eastbound lane (east to west) - left for straight, right for right turn\n东向车道（从东向西）- 左侧直行，右侧右转
            'E_L': {'pos': (4, -2), 'direction': 'left', 'color': 'blue', 'label': 'Straight 直行'},
            'E_R': {'pos': (4, -1), 'direction': 'left_up', 'color': 'green', 'label': 'Right Turn 右转'},
            # Westbound lane (west to east) - left for straight, right for right turn\n西向车道（从西向东）- 左侧直行，右侧右转
            'W_L': {'pos': (-4, 2), 'direction': 'right', 'color': 'blue', 'label': 'Straight 直行'},
            'W_R': {'pos': (-4, 1), 'direction': 'right_down', 'color': 'green', 'label': 'Right Turn 右转'}
        }
        for lane, config in arrow_configs.items():
            pos = config['pos']
            direction = config['direction']
            color = config['color']
            label = config['label']
            # Draw direction arrow\n绘制方向箭头
            if direction == 'down':
                self._draw_arrow(pos[0], pos[1], 0, -1, color, label)
            elif direction == 'up':
                self._draw_arrow(pos[0], pos[1], 0, 1, color, label)
            elif direction == 'left':
                self._draw_arrow(pos[0], pos[1], -1, 0, color, label)
            elif direction == 'right':
                self._draw_arrow(pos[0], pos[1], 1, 0, color, label)
            elif direction == 'right_down':
                self._draw_arrow(pos[0], pos[1], 0.7, -0.7, color, label)
            elif direction == 'right_up':
                self._draw_arrow(pos[0], pos[1], 0.7, 0.7, color, label)
            elif direction == 'left_down':
                self._draw_arrow(pos[0], pos[1], -0.7, -0.7, color, label)
            elif direction == 'left_up':
                self._draw_arrow(pos[0], pos[1], -0.7, 0.7, color, label)
    
    def _draw_arrow(self, x, y, dx, dy, color, label):
        """Draw arrow\n绘制箭头"""
        # Draw arrow body\n绘制箭头主体
        self.ax.arrow(x, y, dx, dy, head_width=0.3, head_length=0.4, 
                     fc=color, ec=color, alpha=0.8, linewidth=2)
        # Add direction label\n添加方向标签
        label_x = x + dx * 0.5
        label_y = y + dy * 0.5
        self.ax.text(label_x, label_y, label, fontsize=8, 
                    ha='center', va='center', color=color, 
                    fontweight='bold', bbox=dict(boxstyle="round,pad=0.2", 
                                               facecolor="white", alpha=0.8))
    
    def update_signal_lights(self, signal_states: Dict[str, str]):
        """Update signal light states\n更新信号灯状态"""
        light_positions = {
            # Northbound lane traffic light\n北向车道信号灯
            'N_R': (0.5, 2), 'N_L': (2, 2),
            # Southbound lane (south to north)\n南向车道（从南向北）
            'S_L': (-2, -2), 'S_R': (-0.5, -2),
            # Eastbound lane (east to west)\n东向车道（从东向西）
            'E_R': (2, -0.5), 'E_L': (2, -2),
            # Westbound lane (west to east)\n西向车道（从西向东）
            'W_R': (-2, 0.5), 'W_L': (-2, 2)
        }
        # Remove old signal lights\n清除旧的信号灯
        for artist in self.ax.get_children():
            if isinstance(artist, patches.Circle) and artist.get_radius() == 0.4:
                artist.remove()
        # Draw new signal lights\n绘制新的信号灯
        for lane, state in signal_states.items():
            if lane in light_positions:
                pos = light_positions[lane]
                color_map = {'red': 'red', 'yellow': 'yellow', 'green': 'green'}
                color = color_map.get(state, 'red')
                self.ax.add_patch(patches.Circle(pos, 0.4, 
                                               facecolor=color, alpha=0.8))
                self.ax.text(pos[0], pos[1], lane, fontsize=9, 
                            ha='center', va='center', color='white', fontweight='bold')
    
    def update_queues(self, queue_lengths: Dict[str, int]):
        """Update queued vehicles display\n更新排队车辆显示"""
        # Remove old vehicle markers\n清除旧的车辆标记
        for artist in self.ax.get_children():
            if isinstance(artist, patches.Rectangle) and artist.get_width() == 0.3:
                artist.remove()
        # Vehicle position definitions\n车辆位置定义
        vehicle_positions = {
            'N_L': [(2, 7-i*0.6) for i in range(8)],
            'N_R': [(0.5, 7-i*0.6) for i in range(8)],
            'S_L': [(-2, -7+i*0.6) for i in range(8)],
            'S_R': [(-0.5, -7+i*0.6) for i in range(8)],
            'E_L': [(7-i*0.6, -2) for i in range(8)],
            'E_R': [(7-i*0.6, -0.5) for i in range(8)],
            'W_L': [(-7+i*0.6, 2) for i in range(8)],
            'W_R': [(-7+i*0.6, 0.5) for i in range(8)]
        }
        # Draw queued vehicles\n绘制排队车辆
        for lane, count in queue_lengths.items():
            if count > 0 and lane in vehicle_positions:
                positions = vehicle_positions[lane]
                for i in range(min(count, len(positions))):
                    pos = positions[i]
                    # Choose vehicle color based on lane direction\n根据车道方向选择车辆颜色
                    if 'L' in lane:  # Left lane (straight)\n左车道（直行）
                        color = 'blue'
                    else:  # Right lane (right turn)\n右车道（右转）
                        color = 'green'
                    self.ax.add_patch(patches.Rectangle(
                        (pos[0]-0.15, pos[1]-0.15), 0.3, 0.3,
                        facecolor=color, alpha=0.8, edgecolor='black', linewidth=1
                    ))
    
    def update_moving_vehicles(self, moving_vehicles: List[Dict]):
        """Update moving vehicles display\n更新移动车辆显示"""
        # Remove old moving vehicle markers\n清除旧的移动车辆标记
        for artist in self.ax.get_children():
            if isinstance(artist, patches.Circle) and artist.get_radius() == 0.18:
                artist.remove()
        # Draw moving vehicles\n绘制移动中的车辆
        for v in moving_vehicles:
            x, y = v.get('x', 0), v.get('y', 0)
            color = 'orange'  # Or distinguish color by direction/type\n或根据方向/类型区分颜色
            self.ax.add_patch(patches.Circle((x, y), 0.18, facecolor=color, alpha=0.9, edgecolor='black', linewidth=1))
    
    def update_stats(self, stats: Dict):
        """Update statistics display - moved to bottom of interface\n更新统计信息显示 - 移到界面下方"""
        # Remove old statistics\n清除旧的统计信息
        for artist in self.ax.get_children():
            if isinstance(artist, Text) and (artist.get_text().startswith('Statistics') or 
                                           artist.get_text().startswith('统计')):
                artist.remove()
        # Show statistics - moved to bottom left\n显示统计信息 - 移到下方左侧
        stats_text = f"""Statistics:
Time: {stats.get('current_time', 0):.1f}s
Phase: {stats.get('current_phase', 'N/A')}
Total Queue: {stats.get('total_queue_length', 0)} vehicles
Avg Delay: {stats.get('average_delay', 0):.1f}s
Objective: {stats.get('objective_value', 0):.2f}
Completed: {stats.get('completed_vehicles_count', 0)} vehicles
Throughput: {stats.get('stats', {}).get('throughput', 0)} vehicles"""
        self.ax.text(-11, -11, stats_text, fontsize=14, 
                    bbox=dict(boxstyle="round,pad=0.8", 
                             facecolor="lightyellow", alpha=0.9))
    
    def update_phase_info(self, phase_info: Dict):
        """Update phase information display - moved to bottom of interface\n更新相位信息显示 - 移到界面下方"""
        # Remove old phase info\n清除旧的相位信息
        for artist in self.ax.get_children():
            if isinstance(artist, Text) and artist.get_text().startswith('Phase'):
                artist.remove()
        # Show phase info - moved to bottom right\n显示相位信息 - 移到下方右侧
        phase_text = f"""Phase Info:
Current: {phase_info.get('current_phase', 'N/A')}
Duration: {phase_info.get('phase_duration', 0):.1f}s
Elapsed: {phase_info.get('phase_elapsed', 0):.1f}s"""
        self.ax.text(6, -11, phase_text, fontsize=14,
                    bbox=dict(boxstyle="round,pad=0.8",
                             facecolor="lightgreen", alpha=0.9))
    
    def update_display(self, env_state: Dict):
        """Update display\n更新显示"""
        self.update_signal_lights(env_state.get('signal_states', {}))
        self.update_queues(env_state.get('queue_lengths', {}))
        self.update_stats(env_state)
        # Add phase info\n添加相位信息
        phase_info = {
            'current_phase': env_state.get('current_phase', 'N/A'),
            'phase_duration': env_state.get('phase_duration', 0),
            'phase_elapsed': env_state.get('current_time', 0) % 60  # Simplified phase time\n简化的相位时间
        }
        self.update_phase_info(phase_info)
        plt.pause(0.01)
    
    def create_animation(self, env, duration: float = 60.0, time_step: float = 1.0):
        """Create animation\n创建动画"""
        frames = int(duration / time_step)
        def animate(frame):
            # Run environment\n运行环境
            state = env.step(time_step)
            self.update_display(state)
            return []
        anim = FuncAnimation(self.fig, animate, frames=frames, 
                           interval=time_step*1000, blit=False, repeat=False)
        return anim
    
    def plot_performance_metrics(self, metrics_history: List[Dict]):
        """Plot performance metrics charts\n绘制性能指标图表"""
        if not metrics_history:
            return
        # Set Chinese font\n设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        # Extract data\n提取数据
        times = [m.get('current_time', 0) for m in metrics_history]
        queue_lengths = [m.get('total_queue_length', 0) for m in metrics_history]
        avg_delays = [m.get('average_delay', 0) for m in metrics_history]
        objective_values = [m.get('objective_value', 0) for m in metrics_history]
        # Queue length\n排队长度
        axes[0, 0].plot(times, queue_lengths, 'b-', linewidth=2)
        axes[0, 0].set_title('Total Queue Length', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Vehicles')
        axes[0, 0].grid(True, alpha=0.3)
        # Average delay\n平均延误
        axes[0, 1].plot(times, avg_delays, 'r-', linewidth=2)
        axes[0, 1].set_title('Average Delay', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Seconds')
        axes[0, 1].grid(True, alpha=0.3)
        # Objective function value\n目标函数值
        axes[1, 0].plot(times, objective_values, 'g-', linewidth=2)
        axes[1, 0].set_title('Objective Function Value', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Objective Value')
        axes[1, 0].grid(True, alpha=0.3)
        # Throughput\n吞吐量
        throughputs = [m.get('stats', {}).get('throughput', 0) for m in metrics_history]
        axes[1, 1].plot(times, throughputs, 'purple', linewidth=2)
        axes[1, 1].set_title('Throughput', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Vehicles')
        axes[1, 1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def show(self):
        """Show figure\n显示图形"""
        plt.show()
    
    def save_animation(self, filename: str, env, duration: float = 60.0, 
                      time_step: float = 1.0):
        """Save animation\n保存动画"""
        anim = self.create_animation(env, duration, time_step)
        anim.save(filename, writer='pillow', fps=int(1/time_step))
        print(f"Animation saved to: {filename} 动画已保存到: {filename}")
