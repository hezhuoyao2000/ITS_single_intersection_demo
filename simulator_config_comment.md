# simulator_config.json Parameter Description 配置参数说明

## intersection (Intersection-related parameters) 交叉口相关参数
- type: Intersection type (e.g., cross_intersection) 交叉口类型（如 cross_intersection）
- lanes_per_direction: Number of lanes per direction 每个方向的车道数
- intersection_width: Intersection width (meters) 交叉口宽度（米）
- vehicle_gap: Vehicle gap (meters) 车辆间隔（米）

## signal_phases (Signal phase settings) 信号相位设置
- phase1/phase2: Each signal phase setting 每个信号相位的设置
  - name: Phase name 相位名称
  - description: Phase description 相位描述
  - lanes: Lanes involved in this phase 本相位涉及的车道
  - yellow_duration: Yellow light duration (seconds) 黄灯时长（秒）
  - all_red_duration: All-red duration (seconds) 全红时长（秒）

## traffic_flow (Traffic flow parameters) 交通流参数
- arrival_rates: Arrival rates for each lane (vehicles/hour), e.g., N_L, N_R, S_L, S_R, E_L, E_R, W_L, W_R 各车道到达率（辆/小时），如 N_L、N_R、S_L、S_R、E_L、E_R、W_L、W_R
- vehicle_types: Vehicle types and parameters 车辆类型及参数
  - car/truck: Vehicle type 车辆类型
    - ratio: Ratio of this vehicle type 该类型车辆比例
    - straight_time: Time required for straight movement through intersection (mean, std, unit: seconds) 直行通过交叉口所需时间（mean 均值，std 标准差，单位：秒）
    - right_turn_time: Time required for right turn through intersection (mean, std, unit: seconds) 右转通过交叉口所需时间（mean 均值，std 标准差，单位：秒）

## optimization (Optimization objectives and signal control parameters) 优化目标与信号控制参数
- objective_weights: Objective function weights 目标函数权重
  - delay: Delay weight 延误权重
  - queue: Queue weight 排队权重
- signal_control: Signal control parameters 信号控制参数
  - green_min: Minimum green light duration (seconds) 最小绿灯时长（秒）
  - green_max: Maximum green light duration (seconds) 最大绿灯时长（秒）
  - action_discretization: Number of action discretizations (action space size) 动作离散化数（动作空间大小）

## simulation (Simulation parameters) 仿真参数
- time_step: Simulation time step (seconds) 仿真步长（秒）
- simulation_duration: Total simulation duration (seconds) 仿真总时长（秒）
- evaluation_window: Evaluation window (seconds) 评估窗口（秒） 