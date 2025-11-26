# Volatility-Normalized Position Sizing - Complete ✅

## Status: COMPLETE

**Fix #2: Volatility-Normalized Position Sizing** has been implemented to ensure consistent risk per trade.

---

## Problem

**Before Fix:**
- Position sizes were absolute (-1.0 to 1.0) regardless of market volatility
- 1.0 position in 10% volatility = different risk than 1.0 in 1% volatility
- No Kelly Criterion or risk-per-trade basis
- Inconsistent risk per trade → unpredictable drawdowns

**Impact:**
- Different risk levels for same position size in different volatility regimes
- Poor risk-adjusted returns
- Unpredictable drawdowns

---

## Solution

Implemented volatility-normalized position sizing using ATR (Average True Range):

### 1. **ATR Calculation** ✅
- Calculates 14-period ATR for volatility measurement
- Uses True Range: max(high-low, |high-close_prev|, |low-close_prev|)
- ATR = Simple Moving Average of True Range

### 2. **Position Size Normalization** ✅
- Normalizes position sizes based on volatility
- Ensures consistent risk per trade (e.g., 1% of capital)
- Formula: `normalized_position = action_value * (risk_per_trade_pct / stop_distance_pct)`
- Higher volatility → smaller position (same risk)
- Lower volatility → larger position (same risk)

### 3. **Configuration** ✅
- Added `volatility_position_sizing` section to config
- Configurable parameters:
  - `enabled: true/false` (enable/disable)
  - `risk_per_trade_pct: 0.01` (1% risk per trade)
  - `atr_period: 14` (ATR period)
  - `atr_multiplier: 2.0` (Stop loss distance in ATR multiples)
  - `min_position_size: 0.01` (Minimum 1%)
  - `max_position_size: 1.0` (Maximum 100%)

---

## Implementation Details

### ATR Calculation Method
```python
def _calculate_atr(self, safe_step: int, period: int = 14) -> float:
    """
    Calculate Average True Range (ATR) for volatility measurement.
    """
    # Calculate True Range components
    # ATR = Simple Moving Average of True Range
```

### Position Size Normalization Method
```python
def _normalize_position_by_volatility(
    self, 
    action_value: float, 
    current_price: float, 
    safe_step: int
) -> float:
    """
    Normalize position size based on volatility to ensure consistent risk per trade.
    
    Formula:
    - Risk amount = capital * risk_per_trade_pct
    - Stop distance = ATR * atr_multiplier
    - Normalized position = action_value * (risk_per_trade_pct / stop_distance_pct)
    """
```

### Application in step() Method
```python
# Apply volatility-normalized position sizing
normalized_action_value = self._normalize_position_by_volatility(
    action_value, 
    current_price, 
    safe_step
)
new_position = normalized_action_value
```

---

## Impact

### Before Fix
- ❌ Fixed position size regardless of volatility
- ❌ Inconsistent risk per trade
- ❌ Unpredictable drawdowns

### After Fix
- ✅ Position sizes normalized by volatility
- ✅ Consistent risk per trade (1% default)
- ✅ Better risk-adjusted returns
- ✅ More predictable risk profile

### Example

**High Volatility (ATR = 3%):**
- Stop distance = 3% * 2.0 = 6%
- Normalized multiplier = 1% / 6% = 0.167
- Action value 1.0 → Position 0.167 (smaller)

**Low Volatility (ATR = 1%):**
- Stop distance = 1% * 2.0 = 2%
- Normalized multiplier = 1% / 2% = 0.5
- Action value 1.0 → Position 0.5 (larger)

**Result:** Same risk (1% of capital) in both cases!

---

## Configuration

### Enable/Disable
```yaml
volatility_position_sizing:
  enabled: true  # Enable volatility-normalized position sizing
```

### Adjust Risk Per Trade
```yaml
volatility_position_sizing:
  risk_per_trade_pct: 0.02  # 2% risk per trade (more aggressive)
```

### Adjust ATR Period
```yaml
volatility_position_sizing:
  atr_period: 20  # Use 20-period ATR (slower response)
```

### Adjust Stop Distance
```yaml
volatility_position_sizing:
  atr_multiplier: 3.0  # 3x ATR stop distance (wider stops)
```

---

## Files Modified

1. ✅ `configs/train_config_adaptive.yaml` - Added volatility_position_sizing configuration
2. ✅ `src/trading_env.py` - Added ATR calculation and position normalization methods

---

## Testing Recommendations

1. **Monitor Initialization:**
   - Look for: `"[FIX #2] Volatility position sizing: ENABLED (1.0% risk per trade, ATR=14)"`
   - Confirms volatility sizing is active

2. **Compare Position Sizes:**
   - High volatility periods → smaller positions
   - Low volatility periods → larger positions
   - Same action value → different positions (normalized by volatility)

3. **Verify Risk Consistency:**
   - Each trade should risk approximately 1% of capital
   - Risk per trade should be consistent across volatility regimes

---

## Expected Behavior

### When Enabled
- Position sizes adjust based on current volatility
- Higher volatility → smaller positions (same risk)
- Lower volatility → larger positions (same risk)
- Consistent risk per trade (1% default)

### When Disabled
- Position sizes use raw action values (-1.0 to 1.0)
- Same behavior as before fix
- No volatility normalization

---

## Status

✅ **COMPLETE** - Volatility-normalized position sizing is now implemented and configurable.

**Default:** ENABLED (`enabled: true` in config)

**Ready for testing!**

---

## Notes

- **Opt-in by default**: Currently enabled in config, but can be disabled
- **Backward compatible**: When disabled, behaves exactly as before
- **Standard practice**: This is standard risk management in professional trading
- **Improves learning**: Agent learns with consistent risk, better risk-adjusted returns

