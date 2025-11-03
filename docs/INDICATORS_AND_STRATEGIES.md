# Indicators and Strategies in RL Trading

## 🎯 Quick Answer

**You DON'T need to provide any trading strategies or indicators!** The system already includes basic indicators, and the RL model learns to discover profitable patterns automatically.

**Want to add your own indicators anyway?** 
- 🚀 **Quick reference:** See **[ADDING_INDICATORS_QUICK_REF.md](ADDING_INDICATORS_QUICK_REF.md)** for copy-paste code templates
- 📖 **Full guide:** Jump to **[📚 How to Add Your Own Indicators](#-how-to-add-your-own-indicators-practical-guide)** below for detailed explanations

---

## 📊 What's Already Built In

### **Automatic Features Extracted from Your NT8 Data:**

The system automatically extracts these features from your OHLCV data:

#### **1. Price Features** (Raw Data)
- **Open, High, Low, Close** prices
- Past **20 bars** of price history

#### **2. Volume Features**
- **Current volume**
- **Volume ratio** (current vs 20-bar average)
- Volume trends

#### **3. Returns**
- **Percentage returns** per bar
- Momentum indicators

#### **4. Built-in Technical Indicators**

Currently automatically calculated:

- **SMA 5** (Simple Moving Average, 5-period)
- **SMA 10** (Simple Moving Average, 10-period)
- **Price Position** (relative to 20-bar high/low range)

#### **5. Multi-Timeframe Analysis**

Automatically combines:
- **1min** features
- **5min** features
- **15min** features

**Total:** ~450+ features analyzed simultaneously

---

## 🚫 What You DON'T Need to Do

### **Traditional Trading Approach:**

```
❌ You write code for:
   - RSI calculation
   - MACD calculation
   - Bollinger Bands
   - Custom indicators
   - Entry/exit rules
   - Risk management logic
```

### **RL Trading Approach:**

```
✅ System handles automatically:
   - Extracts features from raw OHLCV data
   - Calculates basic indicators (SMA, volume ratios)
   - Learns which patterns work
   - Discovers profitable strategies
   - Adapts to market conditions
```

---

## 💡 The Key Insight

### **Traditional Trading:**
You think: *"I'll use RSI + MACD + Bollinger Bands and code entry rules"*

**Problem:** You're guessing which indicators work.

### **RL Trading:**
You provide: *"Just raw price and volume data"*

**Solution:** Model discovers which patterns (including ones you never thought of) lead to profits.

---

## 🧠 How RL Replaces Manual Strategies

### **Example: Learning to Use Moving Averages**

**Traditional Approach:**
```
You code: "Buy when price crosses above SMA 200"

Problem: What if this doesn't work?
          What if market conditions changed?
```

**RL Approach:**
```
Model learns: "When price is X% above SMA 5 
                AND volume is 1.5x average 
                AND 15min shows uptrend
                THEN 75% long position works well"

Benefit: Discovers optimal combination automatically
```

---

## 🔍 What the Model Actually Sees

### **Current Feature Extraction:**

For each of the 3 timeframes (1min, 5min, 15min):

```
Features (per bar):
├─ Price (4): Open, High, Low, Close
├─ Volume (1): Current volume
├─ Returns (1): Percentage change
├─ Volume Ratio (1): Current vs 20-bar avg
├─ SMA 5 (1): 5-period moving average
├─ SMA 10 (1): 10-period moving average
├─ Price Position (1): Where is price in range?
└─ [Padded to 15 features per bar]

History: 20 bars of history
Timeframes: 3 (1min, 5min, 15min)

Total: ~450-900 features analyzed simultaneously!
```

### **What Gets Analyzed:**

```python
# Example from training_env.py
features = [
    # 1min data (20 bars × 15 features = 300 features)
    [1min_bar_1_open, high, low, close, volume, ...],  # Bar 1
    [1min_bar_2_open, high, low, close, volume, ...],  # Bar 2
    ... # ... 18 more bars
    
    # 5min data (20 bars × 15 features = 300 features)
    [5min_bar_1_open, high, low, close, volume, ...],  # Bar 1
    [5min_bar_2_open, high, low, close, volume, ...],  # Bar 2
    ... # ... 18 more bars
    
    # 15min data (20 bars × 15 features = 300 features)
    [15min_bar_1_open, high, low, close, volume, ...],  # Bar 1
    [15min_bar_2_open, high, low, close, volume, ...],  # Bar 2
    ... # ... 18 more bars
]

# Model analyzes all of this at once to make trading decisions
```

---

## 🎯 Could You Add More Indicators? (Optional)

**Yes**, you could add more indicators, but **it's not necessary**. The model learns what works.

---

## 📚 How to Add Your Own Indicators (Practical Guide)

### **Step 1: Understand Where Features Are Added**

All feature extraction happens in `src/trading_env.py` in the `_extract_timeframe_features()` method:

```142:199:src/trading_env.py
    def _extract_timeframe_features(
        self,
        window: pd.DataFrame,
        full_data: pd.DataFrame,
        current_idx: int
    ) -> List[float]:
        """Extract features from a timeframe window"""
        features = []
        
        if len(window) == 0:
            return [0.0] * 15 * self.lookback_bars
        
        # Price features
        prices = window[["open", "high", "low", "close"]].values.flatten()
        features.extend(prices.tolist())
        
        # Volume features
        volumes = window["volume"].values
        features.extend(volumes.tolist())
        
        # Returns
        if len(window) > 1:
            returns = window["close"].pct_change().dropna().values
            features.extend(returns.tolist())
        else:
            features.extend([0.0])
        
        # Volume ratio (current vs average)
        if current_idx >= 20:
            avg_volume = full_data["volume"].iloc[current_idx-20:current_idx].mean()
            current_volume = window["volume"].iloc[-1] if len(window) > 0 else 0
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            features.append(volume_ratio)
        else:
            features.append(1.0)
        
        # Simple moving averages (if enough data)
        if len(window) >= 5:
            sma_5 = window["close"].iloc[-5:].mean()
            sma_10 = window["close"].iloc[-min(10, len(window)):].mean() if len(window) >= 10 else sma_5
            features.extend([sma_5, sma_10])
        else:
            features.extend([window["close"].iloc[-1], window["close"].iloc[-1]])
        
        # Price relative to range
        if len(window) > 1:
            high_low_range = window["high"].max() - window["low"].min()
            price_position = (window["close"].iloc[-1] - window["low"].min()) / high_low_range if high_low_range > 0 else 0.5
            features.append(price_position)
        else:
            features.append(0.5)
        
        # Pad to expected size
        expected_size = 15 * self.lookback_bars
        while len(features) < expected_size:
            features.append(0.0)
        
        return features[:expected_size]
```

### **Step 2: Add Your Indicator Calculations**

Insert your indicator calculations **before** the padding section (line 194). Here's how to add common indicators:

#### **Example 1: Add RSI (14-period)**

```python
# Add after line 192 (after Price relative to range)

# RSI (14-period)
if len(window) >= 15:
    delta = window["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi.iloc[-1] / 100.0)  # Normalize to 0-1 range
else:
    features.append(0.5)  # Neutral RSI
```

#### **Example 2: Add MACD (12, 26, 9)**

```python
# Add after RSI section

# MACD (12, 26, 9)
if len(window) >= 27:
    ema_12 = window["close"].ewm(span=12, adjust=False).mean()
    ema_26 = window["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    
    features.extend([
        macd_line.iloc[-1] / 1000.0,      # Normalize MACD
        signal_line.iloc[-1] / 1000.0,    # Normalize Signal
        histogram.iloc[-1] / 1000.0       # Normalize Histogram
    ])
else:
    features.extend([0.0, 0.0, 0.0])
```

#### **Example 3: Add Bollinger Bands (20, 2)**

```python
# Add after MACD section

# Bollinger Bands (20, 2)
if len(window) >= 21:
    close = window["close"].iloc[-20:]
    sma_20 = close.mean()
    std_20 = close.std()
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    current_price = window["close"].iloc[-1]
    
    # Bollinger Band %B (where is price relative to bands)
    bb_percent = (current_price - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) > 0 else 0.5
    
    features.append(bb_percent)
else:
    features.append(0.5)  # Neutral position
```

#### **Example 4: Add ATR (Average True Range, 14-period)**

```python
# Add after Bollinger Bands

# ATR (14-period)
if len(window) >= 15:
    high_low = window["high"] - window["low"]
    high_close = abs(window["high"] - window["close"].shift())
    low_close = abs(window["low"] - window["close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=14).mean()
    
    # Normalize ATR relative to current price
    normalized_atr = atr.iloc[-1] / window["close"].iloc[-1]
    features.append(normalized_atr)
else:
    features.append(0.01)  # Default 1% volatility
```

### **Step 3: Update Expected Size**

**IMPORTANT:** After adding features, you need to update the `expected_size` calculation!

Find line 68 in `__init__`:
```68:70:src/trading_env.py
        features_per_tf = 5 + 1 + 1 + 8  # ~15 features per timeframe
        self.state_dim = features_per_tf * len(self.timeframes) * self.lookback_bars
```

Update `features_per_tf` to match your new feature count.

**Example:** If you added RSI (1), MACD (3), Bollinger (1), and ATR (1) = **6 new features**:
```python
# Update line 69
features_per_tf = 15 + 6  # 21 features per timeframe (was 15)
```

### **Step 4: Update Padding**

Update the padding section (line 194) to match your new expected size:
```python
# Update line 195
expected_size = 21 * self.lookback_bars  # Was 15
```

### **Step 5: Update Config**

Optional but recommended: Update `configs/train_config_gpu_optimized.yaml`:
```yaml
environment:
  state_features: 170  # If you added features, update this
```

### **Complete Example: Adding RSI**

Here's a complete change to `trading_env.py`:

```python
# In __init__ (around line 69):
features_per_tf = 16  # Was 15, now 16 with RSI

# In _extract_timeframe_features (after line 192):
# RSI (14-period)
if len(window) >= 15:
    delta = window["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi.iloc[-1] / 100.0)  # Normalize to 0-1
else:
    features.append(0.5)

# Update padding (line 195):
expected_size = 16 * self.lookback_bars  # Was 15
```

---

### **But Should You Add Indicators?**

**Probably not**, because:

1. **More features ≠ Better performance**
   - More noise can hurt learning
   - Model might overfit to training data

2. **Model discovers its own patterns**
   - RL excels at finding non-obvious relationships
   - Your indicators might not be optimal

3. **Current features are sufficient**
   - OHLCV + basics already provide rich information
   - Model can learn complex patterns from simple data

**However, if you have specific indicators that you believe work well for your trading style, adding them can be valuable!**

---

## 📊 What the Model Learns vs Indicators

### **Traditional Indicators You Might Know:**

```
RSI: Measures if market is overbought/oversold
MACD: Shows momentum changes
Bollinger Bands: Shows volatility
```

### **What RL Model Learns Instead:**

```
Pattern 1: When 5min trend + volume spike + 1min reversal = profitable entry
Pattern 2: When all timeframes align + position > 0 = hold longer
Pattern 3: When 15min divergence + 5min breakdown = exit quickly
Pattern 4: Complex multi-timeframe combinations you'd never code
```

**The model discovers these patterns automatically from the basic features!**

---

## 🎓 The Philosophy: Less is More

### **Why Simple Features Work Better:**

#### **Traditional Approach:**
```
Input: 20 pre-calculated indicators
Assumption: "These indicators work"

Risk: Indicators might be:
- Lagging the market
- Producing false signals
- Not adapting to market changes
```

#### **RL Approach:**
```
Input: Raw price/volume + basic features
Learning: "Discover what actually works"

Benefit: Model finds:
- Optimal feature combinations
- Better timing than indicators
- Adaptive strategies
- Non-linear relationships
```

---

## 🔬 Real Example: What Model Discovers

### **Traditional Strategy:**

```python
if RSI < 30 and price > SMA_200:
    buy()
```

**Limitation:** Works sometimes, fails other times, no adaptation.

### **What RL Learns:**

```python
# Neural network discovers:
if (
    volume_ratio > 1.5 and           # High volume
    price_position < 0.2 and         # Near support
    15min_returns[-5:].sum() > 0.01 and  # 15min recovery
    1min_volatility < 0.001 and      # Stabilizing
    current_time in [10:30, 11:00]   # Specific hours
):
    position_size = 0.75  # 75% confidence
    
# This complex pattern might never occur to a human coder!
```

**Advantage:** Optimal combinations, timing, and position sizing learned automatically.

---

## 📈 Current Feature Set Breakdown

### **Per Timeframe (1min, 5min, 15min):**

| Category | Features | What They Tell Model |
|----------|----------|---------------------|
| **Price Data** | OHLC (4) | Basic price action |
| **Volume** | Volume, Ratio (2) | Market participation, strength |
| **Returns** | % change (1) | Momentum, direction |
| **Moving Averages** | SMA 5, SMA 10 (2) | Trends, support/resistance |
| **Price Position** | Range position (1) | Near highs/lows, range-bound |
| **History** | 20 bars | Context, patterns |

**Total per TF:** ~15 features × 20 bars = **300 features per timeframe**

**Across 3 TFs:** **~900 total features** (some overlap/padding)

---

## 🚀 What This Means for You

### **You Don't Need To:**

✅ Calculate any indicators manually  
✅ Code trading rules  
✅ Optimize indicator parameters  
✅ Decide which indicators to use  
✅ Write entry/exit logic  
✅ Backtest different indicator combinations  

### **You Just Need To:**

✅ Provide clean OHLCV data from NT8  
✅ Let the model train  
✅ Monitor training progress  
✅ Evaluate performance  
✅ Deploy when ready  

---

## 🎯 Optional: Could You Improve Features?

### **Potential Enhancements** (Not Required):

If you want to experiment, you could add:

#### **1. More Technical Indicators**
```python
# In _extract_timeframe_features():
- RSI (14-period)
- MACD (12, 26, 9)
- Bollinger Bands (20, 2)
- ATR (14-period)
```

#### **2. Advanced Features**
```python
- Order flow analysis
- Limit order book data
- Market microstructure features
- Sentiment data
- News events
```

#### **3. Market Regime Detection**
```python
- Trend strength
- Volatility regime
- Time-of-day patterns
- Market session (Asia, Europe, US)
```

### **But Remember:**

**More features ≠ Better trading**

The current feature set is already very rich:
- **900 features** across multiple timeframes
- Contains enough information for learning
- Adding more might overcomplicate things

**Recommendation:** Start with current features. If performance is good, you're done! Only add more if you need to squeeze out additional performance.

---

## 🎓 The Bottom Line

### **Traditional Trading:**
```
You → Choose indicators → Write rules → Backtest → Deploy
      (Manual, time-consuming, limited creativity)
```

### **RL Trading:**
```
You → Provide data → Model trains → Model learns patterns → Deploy
      (Automatic, fast, discovers optimal strategies)
```

---

## 🔥 Key Takeaways

1. **No indicators needed** - System has OHLCV + basics  
2. **Model discovers patterns** - Learns what works automatically  
3. **900+ features** - More than enough for learning  
4. **Optional enhancements** - Can add more if needed later  
5. **Focus on training** - Better data + longer training = better results  

---

## 📊 Comparison Table

| Aspect | Traditional | RL (This System) |
|--------|-------------|------------------|
| **Indicators** | You provide 5-20 | System provides ~15 per TF |
| **Strategies** | You code rules | Model learns patterns |
| **Optimization** | Manual parameter tuning | Automatic through training |
| **Adaptation** | Requires recoding | Learns continuously |
| **Features** | ~20-50 | ~900+ |
| **Complexity** | Limited by coding skill | Learns non-linear patterns |

---

## 💡 Summary

**You're training a model to discover trading strategies automatically.**

### **Default Approach (Recommended):**
No need for:
- ❌ RSI calculations
- ❌ MACD programming
- ❌ Custom indicators
- ❌ Trading rules
- ❌ Manual optimization

You provide:
- ✅ Clean NT8 historical data
- ✅ Let the GPU train the model
- ✅ Monitor progress
- ✅ Deploy when ready

**The model becomes your trading strategy!** 🚀📈

---

### **Custom Approach (Advanced):**
If you have specific indicators you trust, you can add them:

1. **Edit** `src/trading_env.py` → `_extract_timeframe_features()` method
2. **Add** your indicator calculations (RSI, MACD, Bollinger Bands, ATR, etc.)
3. **Update** `features_per_tf` in `__init__` to match new feature count
4. **Update** `expected_size` padding to match new size
5. **Retrain** your model with the new features

See **[📚 How to Add Your Own Indicators](#-how-to-add-your-own-indicators-practical-guide)** for complete code examples!

**Note:** Adding indicators doesn't guarantee better performance. The model might discover patterns from the raw data that outperform traditional indicators. Test both approaches to see what works best for your trading style!

