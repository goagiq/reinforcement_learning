# Simplified Quality Filters Implementation

## Summary

Simplified quality filters have been implemented directly in `TradingEnvironment.step()` to filter low-quality trades during training. These filters mirror the `DecisionGate` logic used in live trading, ensuring consistency between training and production.

## Implementation Details

### Quality Filters Applied

1. **Action Confidence Check**: Rejects trades where `abs(action_value) < min_action_confidence` (default: 0.3)
2. **Quality Score Check**: Rejects trades where calculated quality score < `min_quality_score` (default: 0.5)
3. **Expected Value Check**: Rejects trades where `expected_value <= 0` (if `require_positive_expected_value` is enabled)

### Quality Score Calculation

The simplified quality score combines:
- **Confidence Component (30%)**: Action magnitude (0-1)
- **Recent Win Rate (30%)**: Win rate from last 50 trades
- **Market Conditions (20%)**: Price volatility (simplified proxy)
- **Action Threshold Ratio (20%)**: Action magnitude relative to threshold

### Expected Value Calculation

Expected value is calculated from recent trade performance:
- Uses last 50 trades (`recent_trades_window`)
- Calculates: `(win_rate * avg_win) - ((1 - win_rate) * avg_loss) - commission_cost`
- Returns `None` if insufficient data (< 10 trades)

### Configuration

Quality filters are configured in `configs/train_config_adaptive.yaml`:

```yaml
environment:
  reward:
    quality_filters:
      enabled: true  # Enable quality filters during training
      min_action_confidence: 0.3  # Minimum action magnitude (0-1)
      min_quality_score: 0.5  # Minimum quality score (0-1)
      require_positive_expected_value: true  # Reject trades with EV <= 0
```

## Files Modified

- **`src/trading_env.py`**:
  - Added quality filter configuration in `__init__`
  - Added `_calculate_simplified_quality_score()` method
  - Added `_calculate_expected_value_simplified()` method
  - Added quality filter checks in `step()` method (before trade execution)
  - Added PnL tracking for expected value calculation
  - Stores `action_value` for quality score calculation

- **`configs/train_config_adaptive.yaml`**:
  - Added `quality_filters` section under `environment.reward`

## Expected Impact

- **Trade Count**: Should reduce from ~4,945 to 300-800 (85-95% reduction)
- **Win Rate**: Should increase from 43.8% to 60-65%+
- **Net Profit**: Should transition from negative to consistently positive

## Testing

E2E tests have been implemented in `test_quality_filters.py`:
- ✅ Quality filters configuration
- ✅ TradingEnvironment imports
- ✅ Quality score calculation
- ✅ Expected value calculation
- ✅ Consecutive loss limit

All tests pass successfully.

## Next Steps

1. Monitor training metrics to verify trade count reduction
2. Adjust `min_action_confidence`, `min_quality_score` if system becomes too conservative
3. If no trades occur, temporarily reduce thresholds to re-enable activity, then gradually increase

