# NinjaTrader 8 Connection Guide

This guide will help you connect NinjaTrader 8 with the Python RL trading system for data collection and training.

## Quick Overview

The connection works in two directions:
- **NT8 → Python**: Market data streams from NT8 to Python
- **Python → NT8**: Trading signals sent from Python to NT8 for execution

## Step-by-Step Connection

### Step 1: Install the NT8 Strategy

1. **Locate the strategy file**:
   - File: `nt8_strategy/RLTradingStrategy.cs`
   - Location in project: `D:\NT8-RL\nt8_strategy\RLTradingStrategy.cs`

2. **Copy to NT8 Strategies folder**:
   ```
   Copy to: Documents\NinjaTrader 8\bin\Custom\Strategies\RLTradingStrategy.cs
   ```

3. **Compile in NinjaTrader 8**:
   - Open NinjaTrader 8
   - Go to: **Tools → Compile**
   - Fix any compilation errors (if any)
   - You should see: "Compilation successful"

### Step 2: Start the Bridge Server

You have **two options** to start the bridge server:

#### Option A: Using the Web UI (Recommended)

1. Open your browser: `http://localhost:3200`
2. Navigate to the **Trading** tab
3. In the "NT8 Bridge Server" section, click **Start Bridge**
4. Wait for status to show "Bridge Server Running"
5. The server will listen on port **8888**

#### Option B: Using Command Line

```bash
# Activate virtual environment (if using venv)
.venv\Scripts\activate

# Start bridge server
python src/nt8_bridge_server.py
```

You should see:
```
NT8 Bridge Server started on localhost:8888
Waiting for NT8 strategy to connect...
```

### Step 3: Configure Strategy in NinjaTrader 8

1. **Open a Chart** in NinjaTrader 8:
   - Recommended instrument: **ES 12-24** (E-mini S&P 500)
   - Or **MES 12-24** (Micro E-mini S&P 500)
   - Recommended timeframe: **1 minute**

2. **Add the Strategy**:
   - Right-click on the chart
   - Select: **Strategies → RLTradingStrategy**

3. **Configure Strategy Parameters**:
   - In the strategy configuration window, you'll see the properties defined in the code:
   - **Parameters Section** (in NT8 UI):
     - **Server Host**: `localhost` (default) - Python bridge server hostname
     - **Server Port**: `8888` (default) - Python bridge server port
     - **Enable Auto Trading**: `False` (recommended for data collection/testing)
     - **Max Position Size**: `1.0` - Maximum contracts to trade
     - **Update Frequency**: `1` - Send data every N bars (default: every bar)
   - **Note**: These parameters come from `[NinjaScriptProperty]` attributes in the C# code

4. **Enable Paper Trading** (Recommended for testing):
   - In NT8, make sure Paper Trading is enabled
   - Or set the strategy to use a Sim101 account

5. **Click OK** to apply

### Step 4: Start the Strategy

1. In the chart, you should see the strategy panel
2. Right-click the strategy → **Enable**
3. The strategy will attempt to connect to the Python bridge server

### Step 5: Verify Connection

**In Python (bridge server output)**:
```
NT8 strategy connected from ('127.0.0.1', XXXXX)
```

**In NT8**:
- Check the strategy log/status
- Should show "Connected to server" or similar

**In Web UI**:
- Go to Trading tab
- Bridge status should show connection established

## Data Collection for Training

### Method 1: Export Historical Data from NT8

1. In NT8, go to: **Tools → Historical Data Manager**
2. Select your instrument (ES or MES)
3. Export data for timeframes: **1min, 5min, 15min**
4. Save as CSV files to: `D:\NT8-RL\data\raw\`
5. Name files: `ES_1min.csv`, `ES_5min.csv`, `ES_15min.csv`

### Method 2: Collect Live Data via Bridge

1. Keep bridge server running
2. Keep NT8 strategy running
3. Data will stream to Python
4. Use the data collection features in `live_trading.py` to save experiences

## Starting Training

Once you have data, start training:

### Via Web UI:
1. Go to **Training** tab
2. Configure:
   - Device: CPU or CUDA (GPU)
   - Total Timesteps: 1,000,000 (recommended)
3. Click **Start Training**
4. Monitor progress in real-time

### Via Command Line:
```bash
python src/train.py --config configs/train_config.yaml --device cuda
```

## Troubleshooting

### Bridge Server Won't Start

**Error**: Port 8888 already in use
- **Solution**: Check if another instance is running
- Kill the process: `python stop_ui.py` or manually kill port 8888

### NT8 Can't Connect

**Error**: Connection refused
- **Solution**: 
  1. Verify bridge server is running (check Web UI or terminal)
  2. Check firewall settings
  3. Verify `localhost` and port `8888` are correct in NT8 strategy settings

### No Data Flowing

**Issue**: Strategy running but no data
- **Solution**:
  1. Check strategy is enabled in NT8
  2. Verify market data is available (check NT8 connection)
  3. Check bridge server logs for errors

### Strategy Compilation Errors

**Error**: Missing Newtonsoft.Json
- **Solution**: NT8 should include this by default, but if missing:
  1. Check NT8 version compatibility
  2. Verify all using statements are correct
  3. Check NT8 documentation for JSON support

## Connection Architecture

```
┌─────────────────┐         TCP Socket         ┌──────────────────┐
│  NinjaTrader 8  │◄────────── Port 8888 ──────►│  Python Bridge   │
│                 │      (JSON messages)        │     Server       │
│ RLTradingStrategy│                             │                  │
└─────────────────┘                             └──────────────────┘
        │                                                 │
        │ Market Data                                    │ Trading Signals
        │ (Bars, OHLCV)                                  │ (Position, Size)
        │                                                 │
        ▼                                                 ▼
```

## Next Steps

Once connected:

1. **Collect Data**: Let the system run to collect market data
2. **Train Model**: Use collected data to train your RL agent
3. **Backtest**: Test trained model on historical data
4. **Paper Trade**: Test with live data in paper trading mode
5. **Go Live**: Once validated, enable live trading

## Important Notes

- **Always test in paper trading first** before going live
- **Monitor connections** - if bridge disconnects, restart both systems
- **Check logs** regularly for errors or warnings
- **Data format** - Ensure NT8 sends data in expected JSON format
- **Port conflicts** - Only one instance of bridge server can run at a time

## Quick Reference

| Component | Port/Path | Purpose |
|-----------|-----------|---------|
| Bridge Server | 8888 | NT8 ↔ Python communication |
| Backend API | 8200 | Web UI backend |
| Frontend UI | 3200 | Web interface |
| Ollama | 11434 | Reasoning engine (optional) |
| Strategy File | `Documents\NinjaTrader 8\bin\Custom\Strategies\` | NT8 strategy location |

## Support

If you encounter issues:
1. Check the logs in both NT8 and Python
2. Verify all ports are available
3. Ensure firewall allows localhost connections
4. Check that all services are running

Good luck with your trading system! 🚀

