# 智能交通信号控制系统 - 安装指南

## 环境要求

- Python 3.7 或更高版本
- NVIDIA GPU (推荐，用于GPU加速训练) 或 CPU
- 至少 4GB 内存

## 安装步骤

### 1. 克隆项目

```bash
git clone [你的GitHub仓库地址]
cd ass2_code
```

### 2. 创建虚拟环境（推荐）

```bash
# 使用venv创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 验证安装

```bash
# 测试PyTorch GPU支持
python tests/PyTorch_GPUtest.py

# 运行系统测试
python tests/test_system.py
```

## 运行项目

### 启动主程序

```bash
python main.py
```

### 可用的演示模式

1. **基础交通模拟** - 使用AI模型进行交通控制
2. **AI智能体训练** - 训练新的DQN智能体
3. **AI性能演示** - 展示AI控制效果
4. **策略性能比较** - 比较不同控制策略
5. **交互式演示** - 实时交互式交通模拟

## 项目结构

```
ass2_code/
├── agent/                 # 强化学习智能体
├── env/                   # 交通环境模拟
├── traffic/              # 交通流生成
├── visualization/        # 可视化模块
├── utils/               # 工具函数
├── tests/               # 测试文件
├── models/              # 模型保存目录
├── log/                 # 日志模块
├── main.py              # 主程序
├── requirements.txt     # 依赖列表
├── simulator_config.json # 模拟器配置
├── training_config.json  # 训练配置
└── README.md           # 项目说明
```

## 配置文件

项目使用两个主要的配置文件：

- `simulator_config.json` - 交通模拟器参数
- `training_config.json` - 强化学习训练参数

## 故障排除

### 常见问题

1. **CUDA错误**
   - 确保安装了正确版本的PyTorch
   - 检查NVIDIA驱动是否最新

2. **内存不足**
   - 减少batch_size参数
   - 降低memory_size参数

3. **依赖安装失败**
   - 更新pip: `pip install --upgrade pip`
   - 使用国内镜像: `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/`

### 获取帮助

如果遇到问题，请检查：
1. Python版本是否为3.7+
2. 所有依赖是否正确安装
3. 配置文件是否存在且格式正确

## 开发环境设置

### 代码格式化

```bash
# 安装black代码格式化工具
pip install black

# 格式化代码
black .
```

### 类型检查

```bash
# 安装mypy类型检查工具
pip install mypy

# 运行类型检查
mypy .
```

## 许可证

[在此添加你的许可证信息] 