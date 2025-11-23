# Metrics Synchronization Fix

## Problem Identified

The dashboard showed metrics that were not in sync:
- **Total Trades**: 0 (should show cumulative total)
- **Current Episode Trades**: 26 (correct)
- **Mean (Last 10 Episodes)**: All showing $0.00 and 0.0% (should show actual means)

## Root Cause

1. **Episode Trade Count Mismatch**:
   - `step_info` returned `"trades": self.state.trades_count` (cumulative, not reset per episode)
   - When episode ended, `train.py` used `step_info.get("trades", 0)` which was cumulative
   - This caused `total_trades` to be calculated incorrectly (using cumulative instead of episode count)

2. **Mean Calculations**:
   - Mean calculations had complex nested conditionals that could return 0.0 incorrectly
   - If `episode_pnls` was empty or had fewer than 10 episodes, means showed 0.0

## Fixes Applied

### 1. TradingEnvironment (`src/trading_env.py`)
- Added `"episode_trades": self.episode_trades` to `step_info`
- This provides the episode-specific trade count (resets each episode)
- Kept `"trades": self.state.trades_count` for backward compatibility

### 2. Trainer (`src/train.py`)
- Updated to use `episode_trades` from `step_info` when available
- Falls back to `trades` for backward compatibility
- Both in `current_episode_trades` update and episode end handling

### 3. API Server (`src/api_server.py`)
- Simplified mean calculations for clarity
- Fixed logic to properly handle cases with < 10 episodes
- Now correctly calculates means from available episodes

## Expected Behavior After Fix

- **Total Trades**: Will correctly accumulate across all episodes
- **Current Episode Trades**: Will show correct episode-specific count
- **Mean (Last 10 Episodes)**: Will show actual means from completed episodes
- **Metrics Sync**: All metrics will be synchronized and accurate

## Files Modified

1. **`src/trading_env.py`**: Added `episode_trades` to `step_info`
2. **`src/train.py`**: Updated to use `episode_trades` from `step_info`
3. **`src/api_server.py`**: Fixed mean calculation logic

