# Intelligent Traffic Signal Control System - Two-Phase Optimized Version

## Project Overview

This is a reinforcement learning-based intelligent traffic signal control system, specifically designed for optimizing signal control at intersections. The system adopts a simplified two-phase signal control strategy and utilizes a GPU-accelerated DQN algorithm to achieve intelligent signal control, minimizing vehicle delay time and queue length.

AAttention！ this project is just a school assignment that's very simple and immature with many issues， at least it can run, it's for sutdents, no matter what you can get from this piece of shit, I well update and reform this project if possible. im just a rookie.

## System Features

- **Simplified Two-Phase Signal System**: Alternating passage for north-south and east-west directions
- **Reinforcement Learning-Based Dynamic Control**: Intelligent signal control using GPU-accelerated DQN algorithm
- **Real-Time Traffic Flow Simulation**: Vehicle arrivals simulated based on Poisson distribution
- **Traffic State Visualization**: Real-time display of traffic status and performance metrics
- **Parametric Configuration**: Separate management of simulator and training parameters
- **GPU-Accelerated Training**: Supports NVIDIA GPU-accelerated training

## System Architecture

### Three Core Modules

1. **Traffic Simulator Module** (`env/traffic_env.py`)
   - Simplified two-phase traffic environment simulation
   - Signal light control logic
   - Vehicle queuing and passage simulation

2. **Visualization Module** (`visualization/visualizer.py`)
   - Intersection layout visualization
   - Real-time traffic status display
   - Performance metrics charts

3. **Reinforcement Learning Module** (`agent/gpu_dqn_agent.py`)
   - GPU-accelerated DQN agent implementation
   - State and action space definition
   - Training and inference functions

### Auxiliary Modules

- **Vehicle Class** (`traffic/vehicle.py`): Vehicle object definition and attribute management
- **Traffic Flow Generator** (`traffic/flow_generator.py`): Vehicle arrival simulation based on Poisson distribution
- **Utility Functions** (`utils/distributions.py`): Probability distribution and calculation tools

## Intersection Design

### Geometric Layout
- Cross intersection, 4 lanes in total (2 per direction)
- Each direction has 2 lanes (left lane for straight, right lane for right turn)
- Left-hand traffic rule

### Signal Phases
Simplified two-phase signal system:

**Phase 1 - North-South Passage:**
- Green light for north-south (left lane straight, right lane right turn allowed simultaneously)
- Red light for east-west

**Phase 2 - East-West Passage:**
- Green light for east-west (left lane straight, right lane right turn allowed simultaneously)
- Red light for north-south

### Lane Configuration
- **Left Lane**: Straight only
- **Right Lane**: Right turn only

## Traffic Flow Simulation

### Vehicle Arrivals
- Vehicle arrivals simulated using Poisson distribution
- Arrival rates for each lane are configurable
- Straight lane arrival rate > right turn lane arrival rate

### Vehicle Types
- **Car**: 80%
- **Truck**: 20%

### Passage Time
Vehicle passage time at the intersection is simulated using truncated normal distribution:
- Car straight: mean 6s, std 1s
- Truck straight: mean 8s, std 1.5s
- Car right turn: mean 3s, std 1s
- Truck right turn: mean 4s, std 1.5s

## Optimization Objective

### Objective Function
Minimize the weighted sum:
```
Objective = w_delay × Average Delay Time + w_queue × Queue Length
```

### Weight Settings
- Delay time weight (w_delay): 0.6
- Queue length weight (w_queue): 0.4

## Configuration Files

### Simulator Configuration (`simulator_config.json`)
Contains all traffic simulation-related parameters:
- Intersection geometry parameters
- Signal phase settings
- Traffic flow parameters
- Optimization objective weights

### Training Configuration (`training_config.json`)
Contains all reinforcement learning training-related parameters:
- Network architecture parameters
- Training hyperparameters
- GPU settings
- Model saving settings

## Installation and Running

### Environment Requirements
- Python 3.7+
- NVIDIA GPU (recommended) or CPU
- Dependencies listed in `requirements.txt`

### Installation Steps
```bash
# Clone the project
git clone [project address]

# Install dependencies
pip install -r requirements.txt

# Create model save directory
mkdir models
```

### Run the Program
```bash
python main.py
```

## Usage Instructions

### Demo Modes

1. **Basic Traffic Simulation**: Run traffic simulation with fixed green light time
2. **AI Agent Training**: Train DQN agent to optimize signal control
3. **AI Performance Demo**: Use trained AI model for signal control
4. **Strategy Performance Comparison**: Compare performance of different signal control strategies
5. **Interactive Demo**: Real-time interactive traffic simulation

### Feature Highlights

- **Easy to Use**: Suitable for beginners to understand and use
- **Centralized Parameters**: Simulator and training parameters managed separately
- **Visualization Friendly**: Real-time display of traffic status and performance metrics
- **Modular Design**: Independent modules for easy extension and maintenance
- **GPU Acceleration**: Supports GPU-accelerated training for improved efficiency

## Technical Implementation

### Reinforcement Learning Algorithm
- **Algorithm**: Deep Q-Network (DQN)
- **State Space**: 8 lane queue lengths + phase encoding
- **Action Space**: 10 discrete green light time options
- **Reward Function**: Negative value of the objective function

### Neural Network Structure
- Input layer: 18 neurons (8 queue lengths + 10 phase encodings)
- Hidden layer: 64 neurons
- Output layer: 10 neurons (corresponding to 10 actions)

### Training Parameters
- Learning rate: 0.001
- Discount factor: 0.95
- Exploration rate: 1.0 → 0.01 (decay)
- Experience replay: 5000 samples
- Batch size: 128

### GPU Optimization
- Supports CUDA acceleration
- Mixed precision training
- Gradient clipping
- Batch processing optimization

## Performance Metrics

### Evaluation Metrics
- **Average Delay Time**: Total time from vehicle arrival to passage
- **Queue Length**: Number of queued vehicles per lane
- **Throughput**: Number of vehicles passing through the intersection per unit time
- **Objective Function Value**: Weighted composite metric

### Visualization Metrics
- Real-time traffic status display
- Performance trend charts
- Strategy comparison charts

## Extension Suggestions

1. **More Complex Neural Networks**: Use CNN or RNN
2. **Multi-Intersection Coordination**: Extend to coordinated control of multiple intersections
3. **Real-Time Data Integration**: Integrate real traffic data
4. **Pedestrian Consideration**: Include pedestrians and non-motorized vehicles
5. **Emergency Vehicles**: Consider priority passage for emergency vehicles
6. **Adaptive Learning**: Implement online learning and adaptation

## Author

[Your Name]

## License

[License Information] 