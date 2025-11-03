# NinjaTrader 8 Parameters Location Guide

## 🎯 Where to Find Strategy Parameters

When you add the **RLTradingStrategy** to a chart, the parameters will appear in NT8's **Parameters** section. Here's where to find them:

### **Step-by-Step Location:**

1. **Right-click on a chart** → Select **"Strategies"** → Choose **"RL Trading Strategy"**

2. **Strategy Configuration Window Opens**

3. **Look for "Parameters" section** (usually in the middle/right panel of the window)

4. **You'll see these parameters:**
   ```
   Parameters
   ├── Server Host: localhost
   ├── Server Port: 8888
   ├── Enable Auto Trading: [checkbox]
   ├── Max Position Size: 1.0
   └── Update Frequency: 1
   ```

### **Visual Layout:**

```
┌─────────────────────────────────────────────────┐
│  Strategy Configuration - RL Trading Strategy  │
├──────────────────┬──────────────────────────────┤
│                  │  Parameters:                 │
│  General         │  ┌────────────────────────┐  │
│  Data            │  │ Server Host: localhost │  │
│  Order Management│  │ Server Port: 8888      │  │
│                  │  │ Enable Auto Trading ☐  │  │
│  (Scroll down    │  │ Max Position: 1.0      │  │
│   for more...)   │  │ Update Frequency: 1    │  │
│                  │  └────────────────────────┘  │
├──────────────────┴──────────────────────────────┤
│  [OK]  [Cancel]  [Apply]                        │
└─────────────────────────────────────────────────┘
```

---

## 📋 Parameter Details

### **Server Host**
- **Default**: `localhost`
- **Purpose**: Python bridge server hostname
- **Change if**: Bridge is on another machine (not recommended)

### **Server Port**
- **Default**: `8888`
- **Purpose**: Python bridge server port
- **Change if**: Port conflict (update Python config too!)

### **Enable Auto Trading**
- **Default**: `False` (unchecked)
- **Purpose**: Allow strategy to execute orders
- **⚠️ WARNING**: Only enable if you want live trading!

### **Max Position Size**
- **Default**: `1.0`
- **Purpose**: Maximum contracts to hold
- **Range**: 0.01 to 10.0
- **Note**: Normalized by RL agent (-1.0 to 1.0) then multiplied by this

### **Update Frequency**
- **Default**: `1`
- **Purpose**: Send data every N bars
- **Range**: 1 to 10
- **Note**: 1 = every bar (most data), 5 = every 5 bars (less data)

---

## 🔍 Troubleshooting

### **Can't Find Parameters?**

**Problem**: Parameters don't show up  
**Solution**:
1. **Recompile** in NT8: Tools → Compile
2. **Restart** NinjaTrader 8
3. **Remove and re-add** strategy to chart

### **Parameters Look Wrong?**

**Problem**: Wrong defaults or missing parameters  
**Solution**:
1. Check `nt8_strategy/RLTradingStrategy.cs` has `[NinjaScriptProperty]` attributes
2. Verify file is in: `Documents\NinjaTrader 8\bin\Custom\Strategies\`
3. Recompile: Tools → Compile
4. Check for compilation errors

### **Parameters Buttons Greyed Out?**

**Problem**: Can't edit parameters  
**Solution**:
1. Strategy may be enabled/running
2. Disable strategy first: Right-click → Disable
3. Then configure parameters
4. Re-enable after configuring

---

## 🎓 How It Works

**In Code** (`RLTradingStrategy.cs`):
```csharp
[NinjaScriptProperty]
[Display(Name = "Server Host", Description = "Python server hostname", GroupName = "Connection")]
public string ServerHost { get; set; } = "localhost";
```

**In NT8 UI**:
- NT8 reads the `[NinjaScriptProperty]` attribute
- Creates a UI input field
- Uses `Name` as the label
- Uses `Description` as tooltip
- May or may not use `GroupName` (sometimes all under "Parameters")

---

## ✅ Quick Setup Checklist

- [ ] File copied to NT8 Strategies folder
- [ ] Compiled successfully (Tools → Compile)
- [ ] Parameters visible in strategy config
- [ ] Server Host: `localhost`
- [ ] Server Port: `8888`
- [ ] Enable Auto Trading: **UNCHECKED** (unless live trading)
- [ ] Click **OK** to apply

---

## 📚 Related Docs

- **[NT8_CONNECTION_GUIDE.md](NT8_CONNECTION_GUIDE.md)** - Full setup guide
- **[BRIDGE_SERVER_EXPLAINED.md](BRIDGE_SERVER_EXPLAINED.md)** - How bridge works
- **[nt8_strategy/RLTradingStrategy.cs](../nt8_strategy/RLTradingStrategy.cs)** - Source code

