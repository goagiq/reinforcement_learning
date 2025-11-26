# Forecasting Enhancement Plan - Chronos Integration

**Date:** Current  
**Status:** Planning Phase  
**Priority:** High (Enhancement to profitable system)

---

## 🎯 Executive Summary

This plan integrates **Chronos-Bolt** time series forecasting into the trading system, enhancing predictive capabilities while building on the MARL evaluation recommendations. The implementation includes:

1. **Chronos-Bolt Forecasting** - Probabilistic multi-timeframe price predictions
2. **ForecastAgent** - New swarm agent providing forecast-based signals
3. **Regime-Aware RL State** - Adds regime features to RL agent (from MARL recommendations)
4. **Forecast-Regime Integration** - Uses forecasts to improve Markov regime detection
5. **Foundation for Regime Ensemble** - Sets up infrastructure for future regime-specific agents

**Key Decision:** Use **Chronos-Bolt-Base** (pretrained, probabilistic, fast, multi-timeframe) instead of XGBoost (requires training) or Chronos-2 (slower).

**Integration with MARL Recommendations:**
- ✅ Implements "Regime-Aware Single Agent" (Option 1 from MARL doc)
- ✅ Sets foundation for "Regime-Specific Ensemble" (Option 2 from MARL doc)
- ✅ Uses forecasts to enhance regime detection (enables better ensemble selection)

---

## 📋 Questions & Answers (For Future Reference)

### Question Set: Forecasting Requirements

| # | Question | Answer | Implication |
|---|----------|--------|-------------|
| 1 | Predict future price/returns directly? | **Yes** | Need forecasting model (not just signals) |
| 2 | Short-term predictions (1-5 bars)? | **Yes** | Focus on 1-5 bar horizon |
| 3 | Historical data available? | **Yes** | Same dataset as RL: `C:\Users\schuo\Documents\NinjaTrader 8\export` |
| 4 | Feed into Decision Gate? | **Yes** | Create new ForecastAgent for swarm |
| 5 | Probabilistic forecasts? | **Yes** | Need uncertainty estimates (for Markov integration) |
| 6 | <100ms latency requirement? | **Yes** | Need fast inference |
| 7 | System profitable? | **Yes** | Enhancement, not fix |
| 8 | Multi-timeframe predictions? | **Yes** | Predict for 1min, 5min, 15min |
| 9 | Willing to train model? | **No** | Use pretrained models |
| 10 | Automatic pattern discovery? | **Yes** | Model should learn from raw OHLCV |

---

## 🎯 Recommendation: Chronos-Bolt

Based on your answers, **Chronos-Bolt** is the optimal choice:

### Why Chronos-Bolt (Not Chronos-2 or XGBoost)?

| Requirement | Chronos-Bolt | Chronos-2 | XGBoost |
|------------|--------------|-----------|---------|
| **Probabilistic** | ✅ Yes | ✅ Yes | ❌ No |
| **<100ms latency** | ✅ ~50-100ms | ⚠️ ~200-500ms | ✅ ~10-50ms |
| **Pretrained** | ✅ Yes | ✅ Yes | ❌ Requires training |
| **Multi-timeframe** | ✅ Yes | ✅ Yes | ⚠️ Need separate models |
| **Pattern discovery** | ✅ Yes | ✅ Yes | ⚠️ Needs feature engineering |
| **Markov integration** | ✅ Probabilistic outputs | ✅ Probabilistic outputs | ❌ Point forecasts only |

**Decision:** Use **Chronos-Bolt-Base** (205M params) - good balance of speed and accuracy.

---

## 🏗️ Architecture Integration

### Current System Flow

```
┌─────────────────────────────────────────────────┐
│         Single PPO RL Agent                      │
│  - Multi-timeframe state (1min, 5min, 15min)   │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│         Swarm Orchestrator                       │
│  - Market Research Agent                        │
│  - Sentiment Agent                              │
│  - Contrarian Agent (Fear/Greed)               │
│  - Elliott Wave Agent                           │
│  - Markov Regime Analyzer                       │
│  - Analyst Agent                                │
│  - Recommendation Agent                         │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│         DecisionGate (Signal Fusion)            │
│  - RL (60%) + Swarm (40%)                       │
│  - Quality filters                              │
│  - Position sizing                              │
└─────────────────────────────────────────────────┘
```

### Enhanced System Flow (With Forecasting)

```
┌─────────────────────────────────────────────────┐
│         Single PPO RL Agent                      │
│  - Multi-timeframe state (1min, 5min, 15min)   │
│  - NEW: Regime features (from Markov)           │
│  - NEW: Forecast features (from Chronos)       │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│         Swarm Orchestrator                       │
│  - Market Research Agent                        │
│  - Sentiment Agent                              │
│  - Contrarian Agent (Fear/Greed)               │
│  - Elliott Wave Agent                           │
│  - Markov Regime Analyzer                       │
│  - NEW: Forecast Agent (Chronos-Bolt)          │
│  - Analyst Agent                                │
│  - Recommendation Agent                         │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│         DecisionGate (Signal Fusion)            │
│  - RL (60%) + Swarm (40%)                       │
│  - NEW: Forecast confidence weighting           │
│  - NEW: Regime-aware position sizing            │
│  - Quality filters                              │
└─────────────────────────────────────────────────┘
```

---

## 📝 Implementation Plan

### Phase 1: Chronos-Bolt Integration (Core Forecasting)

#### Task 1.1: Install and Setup Chronos-Bolt
- [ ] Install `chronos-forecasting` package
- [ ] Test Chronos-Bolt-Base model loading
- [ ] Verify inference speed (<100ms)
- [ ] Test on sample NT8 data

**Files:**
- `requirements.txt` - Add `chronos-forecasting`
- `src/forecasting/chronos_predictor.py` - New file

**Estimated Time:** 2-3 hours

---

#### Task 1.2: Create ChronosPredictor Class
- [ ] Implement `ChronosPredictor` class
- [ ] Load pretrained `amazon/chronos-bolt-base` model
- [ ] Implement multi-timeframe prediction (1min, 5min, 15min)
- [ ] Implement probabilistic forecasts (quantiles: 0.1, 0.5, 0.9)
- [ ] Add caching for performance (<100ms requirement)
- [ ] Handle data preprocessing (OHLCV → Chronos format)

**Interface:**
```python
class ChronosPredictor:
    def predict(
        self,
        price_data: pd.DataFrame,
        timeframes: List[int] = [1, 5, 15],
        prediction_length: int = 5,  # 1-5 bars ahead
        quantile_levels: List[float] = [0.1, 0.5, 0.9]
    ) -> Dict[int, ForecastResult]:
        """
        Returns:
        {
            1: ForecastResult(mean, quantiles, confidence),
            5: ForecastResult(...),
            15: ForecastResult(...)
        }
        """
```

**Files:**
- `src/forecasting/chronos_predictor.py` - New file
- `src/forecasting/__init__.py` - New file

**Estimated Time:** 4-6 hours

---

#### Task 1.3: Create ForecastResult Data Structure
- [ ] Define `ForecastResult` dataclass
- [ ] Include: mean forecast, quantiles, confidence, direction
- [ ] Include: probability of up/down movement
- [ ] Include: expected return and volatility

**Structure:**
```python
@dataclass
class ForecastResult:
    timeframe: int
    prediction_length: int
    mean_forecast: float  # Expected price
    quantiles: Dict[float, float]  # {0.1: lower, 0.5: median, 0.9: upper}
    confidence: float  # 0-1, model confidence
    direction: str  # "bullish", "bearish", "neutral"
    probability_up: float  # Probability price goes up
    probability_down: float  # Probability price goes down
    expected_return: float  # Expected % return
    expected_volatility: float  # Forecasted volatility
    timestamp: str
```

**Files:**
- `src/forecasting/forecast_result.py` - New file

**Estimated Time:** 1 hour

---

### Phase 2: ForecastAgent Integration (Swarm Agent)

#### Task 2.1: Create ForecastAgent Class
- [ ] Inherit from `BaseSwarmAgent`
- [ ] Integrate `ChronosPredictor`
- [ ] Implement `analyze()` method
- [ ] Return forecast-based recommendation (BUY/SELL/HOLD)
- [ ] Calculate confidence from forecast probabilities
- [ ] Support multi-timeframe aggregation

**Interface:**
```python
class ForecastAgent(BaseSwarmAgent):
    def analyze(self, market_state: Dict) -> Dict:
        """
        Returns:
        {
            "action": "BUY" | "SELL" | "HOLD",
            "confidence": 0.0-1.0,
            "forecasts": {
                1: ForecastResult,
                5: ForecastResult,
                15: ForecastResult
            },
            "aggregated_direction": "bullish" | "bearish" | "neutral",
            "reasoning": "..."
        }
        """
```

**Files:**
- `src/agentic_swarm/agents/forecast_agent.py` - New file

**Estimated Time:** 4-6 hours

---

#### Task 2.2: Integrate ForecastAgent into SwarmOrchestrator
- [ ] Add ForecastAgent to agent registry
- [ ] Add to swarm execution pipeline
- [ ] Configure agent in swarm config
- [ ] Add timeout handling (forecast should be fast)
- [ ] Add error handling (fallback if Chronos fails)

**Files:**
- `src/agentic_swarm/swarm_orchestrator.py` - Modify
- `src/agentic_swarm/config_loader.py` - Modify
- `configs/train_config_full.yaml` - Add forecast_agent config

**Estimated Time:** 2-3 hours

---

#### Task 2.3: Update DecisionGate to Use Forecast Signals
- [ ] Add forecast confidence to confluence calculation
- [ ] Weight forecast signals in decision fusion
- [ ] Use forecast probabilities for position sizing
- [ ] Add forecast-based quality filters

**Files:**
- `src/decision_gate.py` - Modify `_calculate_confluence_details()`
- `src/decision_gate.py` - Modify `_make_decision_with_swarm()`

**Estimated Time:** 3-4 hours

---

### Phase 3: Markov Regime Integration (From MARL Recommendations)

#### Task 3.1: Enhance Markov Regime Analyzer with Forecasts
- [ ] Use forecast probabilities to improve regime detection
- [ ] Add forecast-based regime transition probabilities
- [ ] Combine historical regime analysis with forecast signals
- [ ] Update regime confidence based on forecast alignment

**Rationale:** Probabilistic forecasts from Chronos can help Markov Regime Analyzer:
- Detect regime transitions earlier (forecast divergence)
- Increase confidence when forecasts align with regime
- Predict regime changes (forecast volatility changes)

**Files:**
- `src/analysis/markov_regime.py` - Modify `_detect_current_regime()`
- `src/agentic_swarm/agents/markov_regime_agent.py` - New file (if doesn't exist)

**Estimated Time:** 4-6 hours

---

#### Task 3.2: Add Regime Features to RL State (MARL Recommendation)
- [ ] Extract current regime from Markov Regime Analyzer
- [ ] Add regime indicators to RL state features
- [ ] Include: regime_id, regime_confidence, regime_duration
- [ ] Add forecast-regime alignment score

**Rationale:** From MARL doc - "Regime-Aware Single Agent" approach:
- Agent learns regime-specific behavior
- No coordination complexity
- Easy to implement (just add features)

**State Features to Add:**
```python
# In trading_env.py _extract_features()
regime_features = [
    regime_id_one_hot,  # [trending, ranging, volatile] - 3 features
    regime_confidence,   # 0-1 - 1 feature
    regime_duration,    # Normalized - 1 feature
    forecast_regime_alignment  # How well forecasts match regime - 1 feature
]
# Total: +6 features to state
```

**Files:**
- `src/trading_env.py` - Modify `_extract_features()`
- `src/trading_env.py` - Modify `__init__()` to accept regime analyzer

**Estimated Time:** 3-4 hours

---

#### Task 3.3: Regime-Aware Position Sizing (Future Enhancement)
- [ ] Adjust position size based on regime + forecast alignment
- [ ] Larger positions when regime + forecast agree
- [ ] Smaller positions when regime + forecast disagree
- [ ] Use forecast probabilities for risk adjustment

**Files:**
- `src/decision_gate.py` - Modify `_apply_position_sizing()`

**Estimated Time:** 2-3 hours

---

### Phase 4: Testing & Validation

#### Task 4.1: Unit Tests
- [ ] Test ChronosPredictor on sample data
- [ ] Test ForecastAgent analyze() method
- [ ] Test forecast-regime integration
- [ ] Test inference speed (<100ms requirement)

**Files:**
- `tests/test_chronos_predictor.py` - New file
- `tests/test_forecast_agent.py` - New file
- `tests/test_forecast_regime_integration.py` - New file

**Estimated Time:** 4-6 hours

---

#### Task 4.2: Integration Tests
- [ ] Test ForecastAgent in SwarmOrchestrator
- [ ] Test DecisionGate with forecast signals
- [ ] Test end-to-end: market data → forecast → decision
- [ ] Test performance (latency, memory)

**Files:**
- `tests/test_forecast_integration.py` - New file

**Estimated Time:** 3-4 hours

---

#### Task 4.3: Backtest Validation
- [ ] Run backtest with forecast signals enabled
- [ ] Compare performance: with vs without forecasts
- [ ] Measure: win rate, PnL, Sharpe ratio
- [ ] Validate forecast accuracy (actual vs predicted)

**Files:**
- `scripts/backtest_with_forecasts.py` - New file

**Estimated Time:** 4-6 hours

---

### Phase 5: Documentation & Configuration

#### Task 5.1: Documentation
- [ ] Document ChronosPredictor usage
- [ ] Document ForecastAgent configuration
- [ ] Document forecast-regime integration
- [ ] Update architecture diagrams

**Files:**
- `docs/FORECASTING_INTEGRATION.md` - New file
- `docs/ARCHITECTURE.md` - Update
- `README.md` - Update

**Estimated Time:** 3-4 hours

---

#### Task 5.2: Configuration Updates
- [ ] Add forecast_agent config to train_config_full.yaml
- [ ] Add Chronos model selection (bolt-base, bolt-small, etc.)
- [ ] Add forecast parameters (prediction_length, quantiles)
- [ ] Add regime-feature flags (enable/disable)

**Files:**
- `configs/train_config_full.yaml` - Update
- `configs/train_config_adaptive.yaml` - Update

**Estimated Time:** 1-2 hours

---

## 🔄 Integration with MARL Recommendations

### From MARL Evaluation Document:

#### ✅ Implemented in This Plan:

1. **Regime-Aware Single Agent** (Task 3.2)
   - Add regime features to RL state
   - Agent learns regime-specific behavior
   - No coordination complexity

2. **Forecast-Enhanced Regime Detection** (Task 3.1)
   - Use probabilistic forecasts to improve regime detection
   - Earlier regime transition detection
   - Higher confidence when forecasts align

3. **Foundation for Regime-Specific Ensemble** (Future)
   - Current plan sets up infrastructure
   - Can later train regime-specific agents
   - Forecast signals can help select which agent to use

#### 🔮 Future Enhancements (Not in This Plan):

1. **Regime-Specific Ensemble** (Phase 2 from MARL doc)
   - Train 3 separate agents (trending, ranging, volatile)
   - Use forecasts + Markov to select agent
   - Requires: profitable single agent, accurate regime detection

2. **Timeframe-Specific Agents** (Alternative approach)
   - Separate agents for 1min, 5min, 15min
   - Forecasts can coordinate timeframe signals
   - Less promising than regime ensemble

---

## 📊 Expected Benefits

### Immediate Benefits:

1. **Better Entry/Exit Timing**
   - Forecasts provide forward-looking signals
   - Probabilistic outputs give confidence levels
   - Multi-timeframe forecasts improve confluence

2. **Enhanced Regime Detection**
   - Forecasts help detect regime transitions earlier
   - Probabilistic outputs improve regime confidence
   - Better adaptation to market conditions

3. **Improved Decision Quality**
   - More signals = better confluence
   - Forecast-regime alignment = higher confidence
   - Probabilistic forecasts = better risk assessment

### Long-Term Benefits:

1. **Foundation for Regime Ensemble**
   - Forecasts can help select regime-specific agents
   - Better regime detection enables ensemble
   - Infrastructure ready for future enhancement

2. **Better Risk Management**
   - Probabilistic forecasts enable scenario analysis
   - Forecast volatility helps position sizing
   - Regime-aware sizing improves risk-adjusted returns

---

## ⚠️ Potential Challenges & Mitigations

### Challenge 1: Latency (<100ms)
**Risk:** Chronos-Bolt may be slower than 100ms  
**Mitigation:**
- Use Chronos-Bolt-Base (faster than Chronos-2)
- Implement caching (only predict when new bar arrives)
- Use GPU if available (faster inference)
- Fallback to simpler model if too slow

### Challenge 2: Forecast Accuracy
**Risk:** Forecasts may not be accurate enough  
**Mitigation:**
- Use probabilistic forecasts (quantiles, not point estimates)
- Combine with other signals (don't rely solely on forecasts)
- Validate forecast accuracy in backtests
- Adjust confidence weighting based on accuracy

### Challenge 3: Integration Complexity
**Risk:** Adding forecasts may complicate system  
**Mitigation:**
- Keep ForecastAgent separate (easy to disable)
- Use existing swarm infrastructure (no new patterns)
- Add feature flags (enable/disable forecasts)
- Comprehensive testing before deployment

### Challenge 4: Model Size & Memory
**Risk:** Chronos-Bolt-Base (205M params) may be large  
**Mitigation:**
- Use Chronos-Bolt-Small (48M) if memory constrained
- Load model once, reuse for all predictions
- Consider model quantization if needed

---

## 📈 Success Metrics

### Performance Metrics:

1. **Forecast Accuracy:**
   - Mean Absolute Error (MAE) < 0.1% of price
   - Direction accuracy > 55% (better than random)
   - Probabilistic calibration (quantiles match actuals)

2. **System Performance:**
   - Inference latency < 100ms (95th percentile)
   - Memory usage < 2GB additional
   - No degradation in existing system performance

3. **Trading Performance:**
   - Win rate improvement: +2-5% (if currently 43%)
   - Sharpe ratio improvement: +0.1-0.2
   - Drawdown reduction: -5-10%
   - Profit factor improvement: +0.05-0.1

### Integration Metrics:

1. **Signal Quality:**
   - Forecast signals contribute to 20-30% of decisions
   - Forecast-regime alignment improves confidence
   - Confluence count increases (more signals agree)

2. **System Stability:**
   - No increase in errors or crashes
   - Forecast failures handled gracefully
   - System works with forecasts disabled

---

## 🗓️ Timeline Estimate

| Phase | Tasks | Estimated Time | Priority |
|-------|-------|----------------|----------|
| **Phase 1** | Chronos Integration | 7-10 hours | High |
| **Phase 2** | ForecastAgent | 9-13 hours | High |
| **Phase 3** | Markov Integration | 9-13 hours | Medium |
| **Phase 4** | Testing | 11-16 hours | High |
| **Phase 5** | Documentation | 4-6 hours | Low |
| **Total** | | **40-58 hours** | |

**Recommendation:** Implement in 2-3 weeks:
- Week 1: Phases 1-2 (Core forecasting)
- Week 2: Phases 3-4 (Integration & Testing)
- Week 3: Phase 5 + Refinement (Documentation & Polish)

---

## 🚀 Quick Start (After Implementation)

### 1. Enable Forecasts in Config

```yaml
swarm:
  agents:
    forecast_agent:
      enabled: true
      model: "amazon/chronos-bolt-base"
      prediction_length: 5
      quantiles: [0.1, 0.5, 0.9]
      timeframes: [1, 5, 15]
      cache_enabled: true
```

### 2. Enable Regime Features in RL

```yaml
environment:
  features:
    include_regime_features: true
    include_forecast_features: true
```

### 3. Monitor Performance

- Check forecast accuracy in logs
- Monitor inference latency
- Compare trading performance (with/without forecasts)

---

## 📚 References

- **Chronos-Bolt Paper:** [Chronos-Bolt Blog Post](https://github.com/amazon-science/chronos-forecasting)
- **Chronos-2 Paper:** [arXiv:2510.15821](https://arxiv.org/abs/2510.15821)
- **MARL Evaluation:** `docs/MARL_EVALUATION_AND_RECOMMENDATION.md`
- **Current Architecture:** `src/agentic_swarm/swarm_orchestrator.py`
- **Decision Gate:** `src/decision_gate.py`

---

## ✅ Next Steps

1. **Review this plan** - Confirm approach and priorities
2. **Start Phase 1** - Install Chronos and create ChronosPredictor
3. **Iterate** - Test each phase before moving to next
4. **Validate** - Backtest and compare performance
5. **Deploy** - Enable in production when validated

---

**Status:** Ready for Implementation  
**Last Updated:** Current  
**Owner:** Development Team

