# Profitability Fix Implementation - Stop Loss & Risk/Reward Ratio

## Problem

**High win rate (37.7%) but losing money** - This indicates poor risk/reward ratio where average losses exceed average wins.

## Root Cause

Even with a 37.7% win rate, if `avg_loss > avg_win`, the system loses money:
- **Required**: `avg_win / avg_loss >= 1.5:1` (at minimum) for profitability
- **Current**: Likely `avg_win / avg_loss < 1.0:1` (losses are larger than wins)

## Fixes Implemented

### 1. Stop Loss Enforcement ✅

**Location**: `src/trading_env.py` - `step()` method

**Implementation**:
- Added `stop_loss_pct: 0.02` (2% stop loss) in config
- Enforces stop loss by checking if loss percentage exceeds threshold
- If stop loss hit, force closes position immediately
- Caps maximum loss per trade at 2% of entry price

**Code**:
```python
# CRITICAL FIX: Enforce stop loss to cap losses
if (self.state.position * price_change) < 0:  # Position is losing
    loss_pct = abs(price_change)
    
    # If loss exceeds stop loss, force close position
    if loss_pct >= self.stop_loss_pct:
        # Stop loss hit - force close position
        # ... (realize PnL, track win/loss, close position)
```

**Impact**: Prevents large losses from exceeding 2% per trade

### 2. Risk/Reward Ratio Check ✅

**Location**: `src/trading_env.py` - `step()` method (before quality filters)

**Implementation**:
- Calculates `avg_win` and `avg_loss` from recent trades
- Calculates `risk_reward_ratio = avg_win / avg_loss`
- Rejects trades if `risk_reward_ratio < min_risk_reward_ratio` (default 1.5:1)

**Code**:
```python
# CRITICAL FIX: Check risk/reward ratio before allowing trade
if abs(position_change) > self.action_threshold and self.state.trades_count > 0:
    avg_win = self.state.total_win_pnl / max(1, self.state.winning_trades)
    avg_loss = self.state.total_loss_pnl / max(1, self.state.losing_trades)
    
    if avg_loss > 0 and avg_win > 0:
        risk_reward_ratio = avg_win / avg_loss
        
        # Reject trades with poor risk/reward ratio
        if risk_reward_ratio < self.min_risk_reward_ratio:
            position_change = 0.0  # Reject trade
```

**Impact**: Only allows trades when historical performance shows `avg_win >= 1.5x avg_loss`

### 3. Track Average Win/Loss ✅

**Location**: `src/trading_env.py` - `TradeState` dataclass and `step()` method

**Implementation**:
- Added `total_win_pnl` and `total_loss_pnl` to `TradeState`
- Tracks cumulative PnL for winning and losing trades separately
- Calculates `avg_win` and `avg_loss` in `step()` info
- Exposes `risk_reward_ratio` in step info

**Code**:
```python
@dataclass
class TradeState:
    # ... existing fields ...
    total_win_pnl: float = 0.0  # Sum of all winning trade PnLs
    total_loss_pnl: float = 0.0  # Sum of all losing trade PnLs (absolute values)

# In step() method:
avg_win = self.state.total_win_pnl / max(1, self.state.winning_trades)
avg_loss = self.state.total_loss_pnl / max(1, self.state.losing_trades)
risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
```

**Impact**: Provides visibility into risk/reward ratio for monitoring

### 4. Tightened Quality Filters ✅

**Location**: `configs/train_config_adaptive.yaml`

**Changes**:
- `min_action_confidence`: 0.1 → 0.15 (increased)
- `min_quality_score`: 0.3 → 0.4 (increased)

**Impact**: Reduces trade count and improves trade quality

### 5. API Metrics Update ✅

**Location**: `src/api_server.py` and `src/train.py`

**Implementation**:
- Trainer now tracks `current_avg_win`, `current_avg_loss`, `current_risk_reward_ratio`
- API endpoint includes these in metrics response
- Frontend can display these metrics for monitoring

**Impact**: Provides visibility into profitability metrics

---

## Configuration Changes

### `configs/train_config_adaptive.yaml`

```yaml
environment:
  reward:
    # CRITICAL FIX: Stop loss and risk/reward ratio to fix profitability
    stop_loss_pct: 0.02  # Stop loss at 2% of entry price (caps maximum loss per trade)
    min_risk_reward_ratio: 1.5  # Minimum risk/reward ratio (1.5:1) - reject trades with poor R:R
    
    quality_filters:
      min_action_confidence: 0.15  # Increased from 0.1 to reduce trade count
      min_quality_score: 0.4  # Increased from 0.3 to reduce trade count
```

---

## Expected Impact

### Short-Term (Next 50 Episodes)
- **Trade Count**: Should decrease (filters tightened, R:R check active)
- **Win Rate**: May decrease slightly (fewer trades, but higher quality)
- **Avg Win / Avg Loss**: Should improve to >= 1.5:1
- **Profitability**: Should improve as R:R ratio improves

### Medium-Term (Next 200 Episodes)
- **Trade Count**: Should stabilize at 0.5-1.0 trades/episode
- **Win Rate**: Should stabilize around 40-50%
- **Risk/Reward Ratio**: Should be >= 1.5:1 consistently
- **Profitability**: Should be consistently positive

### Long-Term (500+ Episodes)
- **Trade Count**: 0.5-1.0 trades/episode (target: 300-800 total)
- **Win Rate**: 60-65%+ (target achieved)
- **Risk/Reward Ratio**: >= 2.0:1 (target)
- **Net Profit**: Strongly positive after commissions

---

## Monitoring

### Key Metrics to Watch

1. **Risk/Reward Ratio**: Should be >= 1.5:1
   - If < 1.5: System will reject trades (good!)
   - If >= 1.5: Trades allowed (good!)

2. **Average Win**: Should be >= 1.5x Average Loss
   - Monitor in dashboard
   - Alert if ratio drops below 1.5:1

3. **Stop Loss Hits**: Should see stop losses being triggered
   - Indicates system is capping losses
   - Good sign - prevents large losses

4. **Trade Count**: Should decrease
   - Fewer trades but higher quality
   - Target: 0.5-1.0 trades/episode

---

## Files Modified

1. **`src/trading_env.py`**:
   - Added stop loss enforcement in `step()` method
   - Added risk/reward ratio check before allowing trades
   - Added `total_win_pnl` and `total_loss_pnl` to `TradeState`
   - Track win/loss PnL in all trade closure paths
   - Calculate and expose `avg_win`, `avg_loss`, `risk_reward_ratio` in step info

2. **`src/train.py`**:
   - Track `current_avg_win`, `current_avg_loss`, `current_risk_reward_ratio` from step_info
   - Reset these metrics at episode start

3. **`src/api_server.py`**:
   - Include `avg_win`, `avg_loss`, `risk_reward_ratio` in training status response

4. **`configs/train_config_adaptive.yaml`**:
   - Added `stop_loss_pct: 0.02`
   - Added `min_risk_reward_ratio: 1.5`
   - Increased `min_action_confidence` to 0.15
   - Increased `min_quality_score` to 0.4

---

## Next Steps

1. **Monitor Risk/Reward Ratio**: Watch dashboard for `risk_reward_ratio` metric
2. **Monitor Average Win/Loss**: Ensure `avg_win >= 1.5x avg_loss`
3. **Watch Trade Count**: Should decrease as filters tighten
4. **Check Profitability**: Should improve as R:R ratio improves

---

## Success Criteria

- ✅ **Stop Loss Enforced**: Losses capped at 2% per trade
- ✅ **Risk/Reward Check**: Trades rejected if R:R < 1.5:1
- ✅ **Metrics Tracked**: `avg_win`, `avg_loss`, `risk_reward_ratio` available
- ⏭️ **Profitability**: Should improve over next 50-100 episodes
- ⏭️ **Trade Count**: Should decrease to 0.5-1.0 trades/episode

