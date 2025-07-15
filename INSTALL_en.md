# Intelligent Traffic Signal Control System - Installation Guide

## Environment Requirements

- Python 3.7 or higher
- NVIDIA GPU (recommended for GPU-accelerated training) or CPU
- At least 4GB RAM

## Installation Steps

### 1. Clone the Project

```bash
git clone [your GitHub repository address]
cd ass2_code
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Create a virtual environment using venv
python -m venv venv

# Activate the virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Test PyTorch GPU support
python tests/PyTorch_GPUtest.py

# Run system tests
python tests/test_system.py
```

## Running the Project

### Start the Main Program

```bash
python main.py
```

### Available Demo Modes

1. **Basic Traffic Simulation** - Use the AI model for traffic control
2. **AI Agent Training** - Train a new DQN agent
3. **AI Performance Demo** - Demonstrate AI control effects
4. **Strategy Performance Comparison** - Compare different control strategies
5. **Interactive Demo** - Real-time interactive traffic simulation

## Project Structure

```
ass2_code/
├── agent/                 # Reinforcement learning agent
├── env/                   # Traffic environment simulation
├── traffic/               # Traffic flow generation
├── visualization/         # Visualization module
├── utils/                 # Utility functions
├── tests/                 # Test files
├── models/                # Model save directory
├── log/                   # Log module
├── main.py                # Main program
├── requirements.txt       # Dependency list
├── simulator_config.json  # Simulator configuration
├── training_config.json   # Training configuration
└── README.md              # Project documentation
```

## Configuration Files

The project uses two main configuration files:

- `simulator_config.json` - Traffic simulator parameters
- `training_config.json` - Reinforcement learning training parameters

## Troubleshooting

### Common Issues

1. **CUDA Errors**
   - Ensure the correct version of PyTorch is installed
   - Check if your NVIDIA drivers are up to date

2. **Out of Memory**
   - Reduce the batch_size parameter
   - Lower the memory_size parameter

3. **Dependency Installation Failure**
   - Upgrade pip: `pip install --upgrade pip`
   - Use a mirror in China: `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/`

### Getting Help

If you encounter issues, please check:
1. Python version is 3.7+
2. All dependencies are installed correctly
3. Configuration files exist and are properly formatted

## Development Environment Setup

### Code Formatting

```bash
# Install black code formatter
pip install black

# Format code
black .
```

### Type Checking

```bash
# Install mypy type checker
pip install mypy

# Run type checking
mypy .
```

## License

[Add your license information here] 