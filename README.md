# NT8 Reinforcement Learning Trading Strategy

Automated trading strategy for NinjaTrader 8 using PyTorch reinforcement learning with DeepSeek-R1 reasoning capabilities. Focuses on ES and MES futures with multi-timeframe analysis (1min, 5min, 15min).

## 🎯 Features

- **Reinforcement Learning**: PPO-based agent with continuous position sizing
- **Multi-Timeframe Analysis**: Combines 1min, 5min, and 15min data
- **Deep Reasoning**: DeepSeek-R1:8b validates decisions and provides insights
- **Continuous Learning**: Model adapts to market conditions over time
- **Paper Trading**: Safe testing environment before live trading
- **GPU Accelerated**: Fast training and inference with CUDA support

## 📋 Requirements

- Python 3.10 or higher
- NinjaTrader 8 (with license)
- GPU (recommended, NVIDIA with CUDA support)
- Ollama installed and running (for DeepSeek-R1:8b)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository (if using git)
# git clone <repository-url>
# cd NT8-RL

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install PyTorch with CUDA (GPU Users)

If you have an NVIDIA GPU, install the CUDA version of PyTorch:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Setup Ollama and DeepSeek

```bash
# Pull DeepSeek-R1:8b model (if not already done)
ollama pull deepseek-r1:8b

# Verify Ollama is running
ollama list
```

### 4. Extract Historical Data from NT8

First, export historical data from NinjaTrader 8:

1. In NT8, go to Tools → Historical Data Manager
2. Export data for ES or MES (choose timeframe: 1min, 5min, 15min)
3. Save as CSV in `data/raw/` directory

Or use the NT8 strategy to stream real-time data (see NT8 Strategy Setup below).

### 5. Train the RL Model

```bash
# Basic training
python src/train.py --config configs/train_config.yaml

# With GPU
python src/train.py --config configs/train_config.yaml --device cuda

# Continue from checkpoint (recommended - preserves training progress)
python src/train.py --config configs/train_config.yaml --checkpoint models/checkpoint_1000.pt
```

**Training Notes:**
- Episodes are limited to 10,000 steps by default (configurable via `max_episode_steps`)
- This ensures episodes complete properly and metrics are tracked correctly
- Episodes restart automatically after reaching the limit
- Checkpoints can be used to resume training without losing progress

### 6. Backtest

```bash
# Run backtest
python src/backtest.py --model models/best_model.pt --data data/processed/test_data.csv
```

## 📁 Project Structure

```
NT8-RL/
├── docs/                    # Documentation
│   ├── plans/              # Planning documents
│   └── tutorials/          # Tutorial guides
├── src/                    # Source code
│   ├── data_extraction.py  # NT8 data handling
│   ├── trading_env.py      # RL environment
│   ├── rl_agent.py         # RL agent implementation
│   ├── train.py            # Training script
│   ├── backtest.py         # Backtesting
│   ├── live_trading.py     # Live execution
│   ├── reasoning_engine.py # DeepSeek-R1 reasoning
│   └── ...
├── nt8_strategy/           # NT8 C# strategies
├── models/                 # Saved model checkpoints
├── data/                   # Data files
│   ├── raw/               # Raw NT8 exports
│   └── processed/         # Processed data
├── logs/                   # Training logs
├── configs/                # Configuration files
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔧 Configuration

Create a configuration file `configs/train_config.yaml`:

```yaml
# Training Configuration
model:
  algorithm: "PPO"
  learning_rate: 0.0003
  batch_size: 64
  gamma: 0.99
  gae_lambda: 0.95

environment:
  instrument: "ES"
  timeframes: [1, 5, 15]  # minutes
  state_features: 200
  action_space: "continuous"  # -1.0 to 1.0

training:
  total_timesteps: 1000000
  save_freq: 10000
  eval_freq: 5000
  device: "cuda"  # or "cpu"

risk_management:
  max_position_size: 1.0
  max_drawdown: 0.20
  stop_loss_atr_multiplier: 2.0
```

## 🌐 Production UI (NEW!)

**Don't want to deal with command-line?** We now have a beautiful web interface that automates everything!

### Quick Start with UI

1. **Install Frontend Dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Start the System**:
   ```bash
   # Option 1: Use the startup script (recommended)
   python start_ui.py
   
   # Option 2: Manual start
   # Terminal 1: Backend API
   python src/api_server.py
   
   # Terminal 2: Frontend UI
   cd frontend && npm run dev
   ```

3. **Open in Browser**: Navigate to `http://localhost:3200`

The UI provides a step-by-step wizard that automates:
- ✅ Environment setup and dependency installation
- ✅ Data file upload (drag-and-drop CSV files)
- ✅ Model training with real-time progress
- ✅ Backtesting with instant results
- ✅ Paper trading controls
- ✅ Live trading (with safety warnings)
- ✅ Performance monitoring dashboard

**See [UI Guide](./docs/UI_GUIDE.md) for detailed documentation.**

---

## 📚 Tutorials for Beginners

Since you're new to PyTorch/RL, here are key concepts:

### Understanding Reinforcement Learning

1. **Agent**: The RL model that makes decisions
2. **Environment**: The trading market (simulated or real)
3. **State**: Current market conditions (price, volume, indicators)
4. **Action**: Trading decision (position size: -1.0 to 1.0)
5. **Reward**: Profit/loss and risk-adjusted return
6. **Policy**: The strategy the agent learns

### Key Files Explained

- **`trading_env.py`**: Simulates the trading environment. When you call `env.step(action)`, it:
  - Executes the action (buy/sell/hold)
  - Calculates reward (profit/loss)
  - Returns new state and reward

- **`rl_agent.py`**: The PPO agent that:
  - Observes market state
  - Decides position size (action)
  - Learns from rewards
  - Updates its policy (trading strategy)

- **`train.py`**: Training loop that:
  - Runs episodes (trading sessions)
  - Collects experiences
  - Updates the agent
  - Saves checkpoints

### Running Your First Training

```bash
# 1. Make sure you have data
ls data/processed/

# 2. Start training (watch the logs)
python src/train.py --config configs/train_config.yaml

# 3. Monitor with TensorBoard
tensorboard --logdir logs/
# Then open http://localhost:6006 in browser
```

## 🔌 NT8 Strategy Setup

### 1. Copy C# Strategy to NT8

1. Copy `nt8_strategy/RLTradingStrategy.cs` to your NT8 strategies folder:
   - Usually: `Documents\NinjaTrader 8\bin\Custom\Strategies\`

2. Compile in NT8:
   - Tools → Compile
   - Fix any errors (usually import paths)

### 2. Start Python Server

```bash
# Start the socket server
python src/nt8_bridge_server.py --port 8888
```

### 3. Configure Strategy in NT8

1. Open a chart in NT8
2. Right-click → Strategies → RLTradingStrategy
3. Configure:
   - Instrument: ES 12-24 (or MES)
   - Timeframe: 1 minute (primary)
   - Paper trading: Enable
   - Server IP: localhost
   - Server Port: 8888

### 4. Start Strategy

- The strategy will connect to Python server
- Market data streams to Python
- Python sends trading signals back
- NT8 executes trades (paper or live)

## 🧠 Reasoning Engine

The reasoning engine uses DeepSeek-R1:8b to:

1. **Pre-Trade Analysis**: Validates RL recommendations before execution
2. **Post-Trade Reflection**: Analyzes completed trades for learning
3. **Market Regime Detection**: Identifies market conditions

### Testing Reasoning Engine

```bash
# Test reasoning engine
python src/reasoning_engine.py
```

### Query DeepSeek for Recommendations

```bash
# Get AI recommendations
python src/query_deepseek.py
```

## 📊 Monitoring

### TensorBoard

```bash
# View training metrics
tensorboard --logdir logs/
```

### Live Performance

```bash
# Monitor live trading
python src/monitoring.py --mode live
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 🐛 Troubleshooting

### GPU Issues

```bash
# Check if PyTorch sees GPU
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
```

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama if needed
```

### NT8 Connection Issues

1. Check firewall settings
2. Verify Python server is running
3. Check port 8888 is not blocked
4. Review NT8 strategy logs

## 📖 Documentation

- [Main Recommendations](./docs/plans/RL.md)
- [Implementation Plan](./docs/IMPLEMENTATION_PLAN.md)
- [Reasoning Architecture](./docs/architecture_reasoning.md)

## 🤝 Contributing

This is a personal project, but suggestions and improvements are welcome!

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading involves risk. Past performance does not guarantee future results. Always test thoroughly in paper trading before using real money.

## 📝 License

[Add your license here]

## 🙏 Acknowledgments

- NinjaTrader for the trading platform
- PyTorch team for the deep learning framework
- Stable-Baselines3 team for RL algorithms
- DeepSeek for the reasoning model

