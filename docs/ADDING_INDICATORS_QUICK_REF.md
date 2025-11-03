# Quick Reference: Adding Indicators to RL Model

## 🎯 TL;DR

**Want to add your favorite indicators?** Here's the quick version:

1. Edit `src/trading_env.py`
2. Add indicator code to `_extract_timeframe_features()` (around line 142)
3. Update `features_per_tf` in `__init__` (line 69)
4. Update `expected_size` padding (line 195)
5. Retrain your model

---

## 📝 Code Templates (Copy-Paste Ready)

### **RSI (14-period)**

```python
# Add after line 192 (Price relative to range section)

# RSI (14-period)
if len(window) >= 15:
    delta = window["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi.iloc[-1] / 100.0)  # Normalize to 0-1
else:
    features.append(0.5)  # Neutral

# UPDATE: features_per_tf = 16 (was 15)
# UPDATE: expected_size = 16 * self.lookback_bars
```

---

### **MACD (12, 26, 9)**

```python
# Add after RSI

# MACD (12, 26, 9)
if len(window) >= 27:
    ema_12 = window["close"].ewm(span=12, adjust=False).mean()
    ema_26 = window["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    
    features.extend([
        macd_line.iloc[-1] / 1000.0,      # Normalize
        signal_line.iloc[-1] / 1000.0,
        histogram.iloc[-1] / 1000.0
    ])
else:
    features.extend([0.0, 0.0, 0.0])

# UPDATE: features_per_tf = 19 (add 3)
# UPDATE: expected_size = 19 * self.lookback_bars
```

---

### **Bollinger Bands (20, 2)**

```python
# Add after previous indicators

# Bollinger Bands (20, 2)
if len(window) >= 21:
    close = window["close"].iloc[-20:]
    sma_20 = close.mean()
    std_20 = close.std()
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    current_price = window["close"].iloc[-1]
    
    # %B: where is price relative to bands
    bb_percent = (current_price - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) > 0 else 0.5
    features.append(bb_percent)
else:
    features.append(0.5)

# UPDATE: features_per_tf += 1
```

---

### **ATR (Average True Range, 14-period)**

```python
# Add after previous indicators

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
    features.append(0.01)  # 1% default

# UPDATE: features_per_tf += 1
```

---

### **Stochastic Oscillator (14, 3, 3)**

```python
# Add after previous indicators

# Stochastic %K and %D
if len(window) >= 17:
    low_14 = window["low"].rolling(window=14).min()
    high_14 = window["high"].rolling(window=14).max()
    
    # %K
    stoch_k = 100 * (window["close"] - low_14) / (high_14 - low_14)
    
    # %D (3-period SMA of %K)
    stoch_d = stoch_k.rolling(window=3).mean()
    
    features.extend([
        stoch_k.iloc[-1] / 100.0,  # Normalize to 0-1
        stoch_d.iloc[-1] / 100.0
    ])
else:
    features.extend([0.5, 0.5])

# UPDATE: features_per_tf += 2
```

---

### **ADX (Average Directional Index, 14)**

```python
# Add after previous indicators

# ADX (14-period)
if len(window) >= 15:
    # Calculate True Range
    high_low = window["high"] - window["low"]
    high_close = abs(window["high"] - window["close"].shift())
    low_close = abs(window["low"] - window["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    plus_dm = window["high"].diff()
    plus_dm[plus_dm < 0] = 0
    
    minus_dm = -window["low"].diff()
    minus_dm[minus_dm < 0] = 0
    
    # Smooth TR and DM
    atr_14 = tr.rolling(window=14).mean()
    plus_di = 100 * plus_dm.rolling(window=14).mean() / atr_14
    minus_di = 100 * minus_dm.rolling(window=14).mean() / atr_14
    
    # Calculate ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=14).mean()
    
    features.append(adx.iloc[-1] / 100.0)  # Normalize to 0-1
else:
    features.append(0.25)  # Default moderate trend

# UPDATE: features_per_tf += 1
```

---

## 🔧 Where to Make Changes

### **File: `src/trading_env.py`**

#### **Line 69 - Update feature count:**

```python
# Find this line:
features_per_tf = 5 + 1 + 1 + 8  # ~15 features per timeframe

# Update with your new count:
features_per_tf = 15 + num_new_features  # Example: 15 + 6 = 21
```

#### **Lines 142-199 - Add indicators here:**

```python
def _extract_timeframe_features(self, window, full_data, current_idx):
    features = []
    
    # ... existing code (Price, Volume, Returns, etc.) ...
    
    # ↓ ADD YOUR INDICATORS HERE ↓
    
    # Your RSI, MACD, etc. code goes here
    
    # ↑ ADD YOUR INDICATORS ABOVE ↑
    
    # ... padding section ...
    
    return features
```

#### **Line 195 - Update padding size:**

```python
# Find this line:
expected_size = 15 * self.lookback_bars

# Update with your new count:
expected_size = (15 + num_new_features) * self.lookback_bars
```

---

## ✅ Checklist

After adding indicators:

- [ ] Added indicator calculation code to `_extract_timeframe_features()`
- [ ] Updated `features_per_tf` in `__init__` method
- [ ] Updated `expected_size` padding calculation
- [ ] Tested code runs without errors
- [ ] Retrained model from scratch (or from checkpoint)

---

## 📊 Feature Count Reference

| Indicator | Features Added | Example Update |
|-----------|---------------|----------------|
| RSI (14) | 1 | `features_per_tf = 16` |
| MACD (12,26,9) | 3 | `features_per_tf = 19` |
| Bollinger Bands | 1 | `features_per_tf = 17` |
| ATR (14) | 1 | `features_per_tf = 16` |
| Stochastic (14,3,3) | 2 | `features_per_tf = 17` |
| ADX (14) | 1 | `features_per_tf = 16` |
| All of above | 9 | `features_per_tf = 24` |

---

## ⚠️ Important Notes

1. **Normalize all indicators** to 0-1 range or small values (÷ by 100 or 1000)
2. **Handle insufficient data** gracefully (use `if len(window) >= X:`)
3. **Add fallback values** for when data isn't enough
4. **Update both `features_per_tf` AND `expected_size`** - they must match!
5. **Retrain from checkpoint or scratch** after adding features

---

## 🚀 Next Steps

1. Add your indicators to `trading_env.py`
2. **Resume training** or start fresh:
   ```bash
   python resume_training.py --checkpoint models/checkpoint_30000.pt
   ```
3. **Monitor performance** to see if indicators help
4. **Compare results** with baseline (no indicators)

---

## 📚 Full Documentation

For complete step-by-step instructions and philosophy, see:
- **[INDICATORS_AND_STRATEGIES.md](INDICATORS_AND_STRATEGIES.md)** - Full guide with explanations

---

## 🎯 Quick Decision Guide

**Should you add indicators?**

- ✅ **Yes if:** You have proven indicators that work for your trading style
- ✅ **Yes if:** You want to experiment and compare performance
- ❌ **Maybe not if:** Current model is already profitable
- ❌ **Maybe not if:** You're new to RL trading (start simple)

**The model discovers patterns from raw data, so indicators are optional!**

---

**Questions?** Check the full guide or the code examples in `src/trading_env.py`

