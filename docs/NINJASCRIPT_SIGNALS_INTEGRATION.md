# NinjaScript Signals Integration

## Overview

This document describes the integration of RL trading signals with NinjaTrader 8's trade management tools via dedicated NinjaScript signals.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Python RL Trading System                       │
│                                                             │
│  ┌──────────────┐      ┌──────────────────┐              │
│  │  RL Agent    │──────│  Signal          │              │
│  │  (Primary)   │      │  Calculator      │              │
│  └──────────────┘      └──────────────────┘              │
│         │                      │                           │
│         │                      │                           │
│         └──────────┬───────────┘                            │
│                    │                                         │
│         ┌──────────▼───────────┐                            │
│         │  Swarm Orchestrator  │                            │
│         │  - Warren Buffett    │                            │
│         │  - Elliott Wave      │                            │
│         │  - Markov Regime      │                            │
│         └──────────────────────┘                            │
└────────────────────┬──────────────────────────────────────┘
                     │
                     │ TCP Socket (localhost:8888)
                     │ JSON Messages with signals
                     │
┌────────────────────▼──────────────────────────────────────┐
│              NinjaTrader 8                                 │
│                                                             │
│  ┌──────────────────────────────────────────┐            │
│  │  RLSignalIndicator.cs                    │            │
│  │  - Receives signals from Python          │            │
│  │  - Exposes Signal_Trend and Signal_Trade │            │
│  └──────────────────┬───────────────────────┘            │
│                     │                                       │
│         ┌───────────▼───────────┐                          │
│         │  Trade Management Tool │                          │
│         │  - Reads Signal_Trend │                          │
│         │  - Reads Signal_Trade │                          │
│         │  - Executes trades    │                          │
│         └───────────────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

## Signal Definitions

### Signal_Trend

Indicates the overall trend direction and strength:

- **2** = Uptrend strong (action > 0.5, confidence >= 0.7)
- **1** = Uptrend weak (action > 0.05, but not strong)
- **-1** = Downtrend weak (action < -0.05, but not strong)
- **-2** = Downtrend strong (action < -0.5, confidence >= 0.7)
- **0** = No significant trend

### Signal_Trade

Indicates specific trading signals with context:

- **3** = Uptrend strengthening (trend is getting stronger)
- **2** = Uptrend pullback (temporary reversal in uptrend)
- **1** = Uptrend start (new uptrend beginning)
- **0** = No signal
- **-1** = Downtrend start (new downtrend beginning)
- **-2** = Downtrend pullback (temporary reversal in downtrend)
- **-3** = Downtrend strengthening (trend is getting stronger)

## Integration Points

### 1. Python Side (`src/signal_calculator.py`)

The `SignalCalculator` class maps RL agent actions and swarm recommendations to signal values:

```python
from src.signal_calculator import SignalCalculator

calculator = SignalCalculator(
    action_change_threshold=0.15,  # Minimum change to trigger update
    pullback_detection_bars=3,      # Bars to look back for pullback
    trend_strength_threshold=0.5   # Threshold for strong vs weak trend
)

signal_trend, signal_trade = calculator.calculate_signals(
    rl_action=rl_action_value,
    rl_confidence=rl_confidence,
    swarm_recommendation=swarm_recommendation,
    current_position=current_position,
    markov_regime=markov_regime,
    markov_regime_confidence=markov_confidence
)
```

### 2. Live Trading Integration (`src/live_trading.py`)

Signals are calculated and included in trade signals sent to NT8:

```python
signal = {
    "action": "buy" if position_change > 0 else "sell",
    "position_size": target_position,
    "confidence": min(abs(target_position), 1.0),
    "signal_trend": signal_trend,      # Added
    "signal_trade": signal_trade,       # Added
    "timestamp": datetime.now().isoformat()
}

bridge_server.send_trade_signal(signal)
```

### 3. NinjaScript Indicator (`nt8_strategy/RLSignalIndicator.cs`)

The indicator receives signals and exposes them as properties:

```csharp
// Public properties accessible by trade management tools
public int Signal_Trend { get; }
public int Signal_Trade { get; }
```

## Agent Integration

### Warren Buffett (Contrarian Agent)

The contrarian agent detects market extremes (greed/fear) and influences signals:

- **Greedy Market (Top)**: Contrarian says SELL → Boosts downward signals
- **Fearful Market (Bottom)**: Contrarian says BUY → Boosts upward signals
- **Confidence >= 0.6**: Signals are adjusted by up to 20% based on contrarian confidence

### Markov Regime Analyzer

Markov regime analysis provides market context:

- **Bull Regime**: Favors long positions, increases confidence for upward signals
- **Bear Regime**: Favors short positions, increases confidence for downward signals
- **Confidence >= 0.6**: Regime information is incorporated into signal calculation

### Elliott Wave Agent

Elliott Wave patterns influence signal strength:

- **High Confidence (>= 0.6)**: Boosts signals in the direction of the wave pattern
- **Phase Information**: Used to determine trend strength

## Configuration

Add to `configs/train_config_adaptive.yaml`:

```yaml
signals:
  action_change_threshold: 0.15    # Minimum action change to trigger signal update
  pullback_detection_bars: 3      # Bars to look back for pullback detection
  trend_strength_threshold: 0.5   # Threshold for strong vs weak trend
```

## Setup Instructions

### 1. Python Side

No additional setup required - signals are automatically calculated and sent when using `LiveTradingSystem`.

### 2. NinjaTrader 8 Side

1. Copy `nt8_strategy/RLSignalIndicator.cs` to:
   ```
   Documents\NinjaTrader 8\bin\Custom\Indicators\
   ```

2. Compile in NT8:
   - Tools → Compile
   - Check for errors

3. Add to Chart:
   - Right-click chart → Indicators
   - Select "RL Signal Indicator"
   - Configure Server Host and Port (default: localhost:8888)

4. Use in Trade Management Tool:
   ```csharp
   // Access signals from the indicator
   RLSignalIndicator indicator = ...; // Get indicator instance
   int trend = indicator.Signal_Trend;
   int trade = indicator.Signal_Trade;
   
   // Use in your trade management logic
   if (trend == 2 && trade == 3) {
       // Strong uptrend strengthening - enter long
   } else if (trend == -2 && trade == -3) {
       // Strong downtrend strengthening - enter short
   }
   ```

## Signal Update Behavior

- **Updates Only on Significant Changes**: Signals update only when action changes by >= `action_change_threshold` (default 0.15)
- **Pullback Detection**: Looks back `pullback_detection_bars` (default 3) to detect temporary reversals
- **Position State Consideration**: Signals consider current position (flat, long, short) when determining Signal_Trade values

## Example Signal Scenarios

### Scenario 1: Strong Uptrend Start
- RL Action: 0.6 (strong long)
- Confidence: 0.8
- Previous Action: 0.1
- **Result**: Signal_Trend = 2, Signal_Trade = 1 (uptrend start)

### Scenario 2: Uptrend Pullback
- RL Action: -0.2 (temporary short)
- Previous Actions: [0.5, 0.4, 0.3] (recent uptrend)
- **Result**: Signal_Trend = 1, Signal_Trade = 2 (uptrend pullback)

### Scenario 3: Trend Strengthening
- RL Action: 0.7 (increasing from 0.4)
- Confidence: 0.85
- **Result**: Signal_Trend = 2, Signal_Trade = 3 (uptrend strengthening)

### Scenario 4: Warren Buffett Contrarian Signal
- Market Condition: FEARFUL
- Contrarian Signal: BUY
- Contrarian Confidence: 0.7
- RL Action: 0.3
- **Result**: Action boosted to 0.4, Signal_Trend = 1, Signal_Trade = 1

## Troubleshooting

### Signals Not Updating

1. Check Python bridge server is running
2. Verify NT8 indicator is connected (check indicator output)
3. Ensure action changes exceed `action_change_threshold`
4. Check that signals are being sent in trade_signal messages

### Signals Always Zero

1. Verify RL agent is producing non-zero actions
2. Check that swarm agents are enabled and working
3. Ensure decision confidence meets minimum thresholds
4. Verify Markov regime report exists (if using)

### Connection Issues

1. Verify Server Host and Port match Python bridge server
2. Check firewall settings
3. Ensure Python server is listening on correct port
4. Check NT8 indicator output for connection errors

## Future Enhancements

1. **Real-time Markov Regime Detection**: Currently uses offline analysis - could add real-time regime detection
2. **Signal History**: Store signal history for backtesting trade management strategies
3. **Custom Signal Thresholds**: Allow per-instrument signal threshold configuration
4. **Signal Validation**: Add validation to ensure signals are consistent with market conditions

