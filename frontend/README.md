# NT8 RL Trading System - Frontend UI

Production-ready web interface for the NT8 RL Trading System.

## Features

- **Step-by-step Setup Wizard**: Guides you through environment setup, data upload, and configuration
- **Model Training**: Start, monitor, and stop RL model training with real-time progress
- **Backtesting**: Run backtests on trained models with detailed results
- **Paper Trading**: Test strategies with live data without risking real money
- **Live Trading**: Start automated trading with real capital (use with caution)
- **Performance Monitoring**: Real-time metrics and performance tracking
- **WebSocket Updates**: Real-time notifications and progress updates

## Quick Start

### Prerequisites

- Node.js 18+ and npm
- Python backend API server running (see main README)

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

This will start the Vite dev server at `http://localhost:3200` with hot module replacement.

### Production Build

```bash
npm run build
```

The built files will be in the `dist` directory.

### Preview Production Build

```bash
npm run preview
```

## Architecture

- **Framework**: React 18 with Vite
- **Styling**: Tailwind CSS
- **Icons**: Lucide React
- **HTTP Client**: Axios
- **Real-time**: WebSocket connection to backend API

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── SetupWizard.jsx      # Initial setup wizard
│   │   ├── Dashboard.jsx         # Main dashboard
│   │   ├── TrainingPanel.jsx    # Model training interface
│   │   ├── BacktestPanel.jsx     # Backtesting interface
│   │   ├── TradingPanel.jsx     # Trading control panel
│   │   └── MonitoringPanel.jsx  # Performance monitoring
│   ├── App.jsx                   # Main app component
│   ├── main.jsx                  # Entry point
│   └── index.css                 # Global styles
├── index.html
├── package.json
├── vite.config.js
└── tailwind.config.js
```

## API Endpoints

The frontend communicates with the backend API at `http://localhost:8200`:

- `/api/setup/*` - Setup and environment checks
- `/api/data/*` - Data upload and management
- `/api/training/*` - Model training control
- `/api/backtest/*` - Backtesting operations
- `/api/trading/*` - Trading system control
- `/api/monitoring/*` - Performance metrics
- `/api/models/*` - Model management
- `/ws` - WebSocket connection for real-time updates

## Features Overview

### Setup Wizard

1. **Environment Setup**: Checks virtual environment, dependencies, and configuration
2. **Data Upload**: Drag-and-drop CSV file upload for historical market data
3. **Model Training**: Configure and start training
4. **Backtest**: Validate model performance
5. **Paper Trading**: Test with live data
6. **Go Live**: Start automated trading

### Dashboard

- Overview of system status
- Quick start guide
- Recent activity log
- Real-time notifications

### Training Panel

- Device selection (CPU/CUDA)
- Training parameters configuration
- Real-time training progress
- Training log viewer
- Model list

### Backtest Panel

- Model selection
- Episodes configuration
- Results display with key metrics
- Backtest log

### Trading Panel

- NT8 bridge server control
- Model selection
- Paper/Live trading mode toggle
- Trading status monitoring
- Real-time trading log

### Monitoring Panel

- Performance metrics dashboard
- Real-time metric updates
- Performance targets reference
- Auto-refresh functionality

## Development Notes

- The frontend uses a proxy configuration to connect to the backend API
- WebSocket connection is established on component mount
- All API calls use Axios with error handling
- Components are organized by functionality
- Styling uses Tailwind CSS utility classes

## Troubleshooting

### Cannot connect to API

- Ensure the backend API server is running on port 8200
- Check CORS settings in `src/api_server.py`
- Verify proxy configuration in `vite.config.js`

### WebSocket connection fails

- Check that the backend supports WebSocket connections
- Verify the WebSocket endpoint is `/ws`
- Check browser console for connection errors

### Build errors

- Ensure all dependencies are installed: `npm install`
- Check Node.js version (18+ required)
- Clear cache and rebuild: `rm -rf node_modules && npm install`

## Contributing

When adding new features:

1. Create component in `src/components/`
2. Add API endpoint in `src/api_server.py` if needed
3. Update routing in `Dashboard.jsx` if adding new tab
4. Add WebSocket message handlers if real-time updates needed
5. Style with Tailwind CSS classes

