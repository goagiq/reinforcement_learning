# NT8 RL Trading System - Production UI Guide

## Overview

The Production UI is a comprehensive web interface that automates all the steps outlined in `PROJECT_COMPLETE.md`, making the system much more accessible and user-friendly.

## Quick Start

### 1. Start the Backend API Server

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac

# Start the API server
python src/api_server.py
```

The API server will start on `http://localhost:8200`.

### 2. Start the Frontend

```bash
cd frontend
npm install  # First time only
npm run dev
```

The frontend will start on `http://localhost:3200`.

### 3. Open in Browser

Navigate to `http://localhost:3200` and the setup wizard will guide you through:

1. **Environment Setup** - Automatically checks and installs dependencies
2. **Data Upload** - Drag-and-drop your CSV files
3. **Train Model** - Configure and start training
4. **Backtest** - Validate your model
5. **Paper Trading** - Test with live data
6. **Go Live** - Start automated trading

## What the UI Automates

### Previously Manual Steps (from PROJECT_COMPLETE.md)

#### ❌ Before (Manual)
```bash
# Step 1: Create virtual environment
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Step 2: Export data from NT8, save files manually
# Save as: data/raw/ES_1min.csv, ES_5min.csv, ES_15min.csv

# Step 3: Train model
python src/train.py --config configs/train_config.yaml --device cuda

# Step 4: Backtest
python src/backtest.py --model models/best_model.pt --episodes 20

# Step 5: Paper Trading
# Terminal 1:
python src/nt8_bridge_server.py

# Terminal 2:
python src/live_trading.py --model models/best_model.pt

# Step 6: Continuous Learning
python src/automated_learning.py --mode all
```

#### ✅ After (Automated via UI)

1. **Setup Wizard** - One-click environment checks and dependency installation
2. **File Upload** - Drag-and-drop CSV files directly in the browser
3. **Training Panel** - Configure parameters, start/stop training with real-time progress
4. **Backtest Panel** - Select model, set episodes, view results instantly
5. **Trading Panel** - Start bridge server and trading system with a single click
6. **Monitoring Panel** - Real-time performance metrics and notifications

## Key Features

### Setup Wizard

- ✅ Automatic environment detection
- ✅ One-click dependency installation
- ✅ Configuration file validation
- ✅ Step-by-step progress tracking
- ✅ Real-time activity log

### Training Management

- ✅ Device selection (CPU/CUDA)
- ✅ Training parameter configuration
- ✅ Real-time training progress via WebSocket
- ✅ Training log viewer
- ✅ Model list and management

### Backtesting

- ✅ Model selection dropdown
- ✅ Episodes configuration
- ✅ Results display with key metrics:
  - Sharpe Ratio
  - Win Rate
  - Profit Factor
  - Max Drawdown
  - Total P&L

### Trading Control

- ✅ NT8 bridge server management
- ✅ Paper/Live trading mode toggle
- ✅ Model selection
- ✅ Start/Stop controls
- ✅ Real-time trading log
- ✅ Safety warnings for live trading

### Performance Monitoring

- ✅ Real-time metrics dashboard
- ✅ Auto-refresh every 5 seconds
- ✅ Key performance indicators:
  - Total P&L
  - Sharpe Ratio
  - Sortino Ratio
  - Win Rate
  - Profit Factor
  - Max Drawdown
- ✅ Performance targets reference

## WebSocket Real-Time Updates

The UI uses WebSocket connections for real-time updates on:

- Training progress and completion
- Backtest results
- Trading status changes
- System notifications
- Performance metric updates

## API Endpoints

All operations are exposed via REST API:

- `GET /api/setup/check` - Check environment setup
- `POST /api/setup/install-dependencies` - Install Python dependencies
- `POST /api/data/upload` - Upload historical data files
- `GET /api/data/list` - List uploaded files
- `POST /api/training/start` - Start model training
- `GET /api/training/status` - Get training status
- `POST /api/training/stop` - Stop training
- `POST /api/backtest/run` - Run backtest
- `GET /api/models/list` - List trained models
- `POST /api/trading/start-bridge` - Start NT8 bridge server
- `POST /api/trading/start` - Start trading
- `POST /api/trading/stop` - Stop trading
- `GET /api/trading/status` - Get trading status
- `GET /api/monitoring/performance` - Get performance metrics
- `POST /api/learning/start` - Start continuous learning
- `GET /api/learning/status` - Get learning status

## Security Notes

⚠️ **Important**: In production:

1. Restrict CORS origins in `src/api_server.py`:
   ```python
   allow_origins=["http://your-domain.com"]
   ```

2. Add authentication/authorization
3. Use HTTPS for all connections
4. Secure WebSocket connections (WSS)
5. Validate all user inputs
6. Implement rate limiting

## Troubleshooting

### Backend API not responding

- Check that `src/api_server.py` is running
- Verify port 8200 is not in use
- Check for Python import errors in terminal

### Frontend cannot connect

- Verify backend is running on port 8200
- Check browser console for errors
- Verify CORS settings allow your origin

### WebSocket connection fails

- Check backend WebSocket endpoint is accessible
- Verify firewall settings
- Check browser WebSocket support

### File upload fails

- Check `data/raw` directory exists
- Verify file permissions
- Check file size limits

## Next Steps

After setup:

1. Upload your historical data (ES_1min.csv, ES_5min.csv, ES_15min.csv)
2. Train your first model
3. Run backtests to validate performance
4. Start paper trading to test with live data
5. Monitor performance metrics
6. Once satisfied, switch to live trading (with caution!)

## Comparison: Manual vs UI

| Task | Manual | UI |
|------|--------|-----|
| Setup Environment | Multiple terminal commands | One-click wizard |
| Upload Data | Manual file copying | Drag-and-drop upload |
| Train Model | Command line with flags | GUI with parameter inputs |
| Monitor Training | Terminal output | Real-time WebSocket updates |
| Run Backtest | Command line execution | Click button, see results |
| Start Trading | Multiple terminal windows | Single click start |
| View Performance | Check log files | Real-time dashboard |

The UI makes the system accessible to users who may not be comfortable with command-line interfaces, while still providing all the power and flexibility of the underlying system.

