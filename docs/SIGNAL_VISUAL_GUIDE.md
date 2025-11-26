# Signal Visual Guide - What Gets Drawn on the Chart

## Overview

The `RLSignalIndicator` draws **visual markers directly on the price chart** to show trading signals. These markers appear as colored arrows above or below price bars.

## Visual Markers by Signal Value

### Signal_Trade Values (Most Specific - Takes Priority)

#### **Signal_Trade = 3** (Uptrend Strengthening)
- **Marker**: 🟢 **Lime Green Up Arrow** (↑)
- **Position**: Below the bar (at Low - 15% of bar range)
- **Label**: "STRONG BUY"
- **Meaning**: Strong uptrend is getting stronger - best time to enter long
- **Color**: Bright green (LimeGreen)

#### **Signal_Trade = 2** (Uptrend Pullback)
- **Marker**: 🟢 **Green Up Arrow** (↑)
- **Position**: Below the bar (at Low - 10% of bar range)
- **Label**: "BUY PULLBACK"
- **Meaning**: Temporary pullback in uptrend - good entry point
- **Color**: Green

#### **Signal_Trade = 1** (Uptrend Start)
- **Marker**: 🟢 **Light Green Up Arrow** (↑)
- **Position**: Below the bar (at Low - 10% of bar range)
- **Label**: "BUY"
- **Meaning**: New uptrend beginning - enter long
- **Color**: Light green

#### **Signal_Trade = -1** (Downtrend Start)
- **Marker**: 🔴 **Light Coral Down Arrow** (↓)
- **Position**: Above the bar (at High + 10% of bar range)
- **Label**: "SELL"
- **Meaning**: New downtrend beginning - enter short
- **Color**: Light coral

#### **Signal_Trade = -2** (Downtrend Pullback)
- **Marker**: 🔴 **Red Down Arrow** (↓)
- **Position**: Above the bar (at High + 10% of bar range)
- **Label**: "SELL PULLBACK"
- **Meaning**: Temporary pullback in downtrend - good entry point
- **Color**: Red

#### **Signal_Trade = -3** (Downtrend Strengthening)
- **Marker**: 🔴 **Dark Red Down Arrow** (↓)
- **Position**: Above the bar (at High + 15% of bar range)
- **Label**: "STRONG SELL"
- **Meaning**: Strong downtrend is getting stronger - best time to enter short
- **Color**: Dark red

### Signal_Trend Values (Used when Signal_Trade = 0)

#### **Signal_Trend = 2** (Uptrend Strong)
- **Marker**: 🔵 **Blue Up Arrow** (↑)
- **Position**: Below the bar (at Low - 10% of bar range)
- **Label**: "UP STRONG"
- **Meaning**: Strong uptrend detected
- **Color**: Blue

#### **Signal_Trend = 1** (Uptrend Weak)
- **Marker**: 🔵 **Light Blue Up Arrow** (↑)
- **Position**: Below the bar (at Low - 10% of bar range)
- **Label**: "UP"
- **Meaning**: Weak uptrend detected
- **Color**: Light blue

#### **Signal_Trend = -1** (Downtrend Weak)
- **Marker**: 🟠 **Orange Down Arrow** (↓)
- **Position**: Above the bar (at High + 10% of bar range)
- **Label**: "DOWN"
- **Meaning**: Weak downtrend detected
- **Color**: Orange

#### **Signal_Trend = -2** (Downtrend Strong)
- **Marker**: 🟠 **Dark Orange Down Arrow** (↓)
- **Position**: Above the bar (at High + 10% of bar range)
- **Label**: "DOWN STRONG"
- **Meaning**: Strong downtrend detected
- **Color**: Dark orange

## Visual Examples

### Example 1: Strong Buy Signal
```
Price Chart:
    |
    |  [BAR]
    |    |
    |    |  ← 🔴 Dark Red Arrow (Strong Sell)
    |
    |  [BAR]
    |    |
    |    |  ← 🟢 Lime Green Arrow (Strong Buy) - Signal_Trade = 3
    |
```

### Example 2: Pullback Entry
```
Price Chart:
    |
    |  [BAR]  ← 🟢 Green Arrow (Buy Pullback) - Signal_Trade = 2
    |    |
    |  [BAR]
    |    |
    |  [BAR]
```

### Example 3: Trend Detection
```
Price Chart:
    |
    |  [BAR]  ← 🔵 Blue Arrow (Up Strong) - Signal_Trend = 2
    |    |
    |  [BAR]
    |    |
```

## Marker Positioning

- **Up Arrows**: Always drawn **below** the bar (at Low price minus 10-15% of bar range)
- **Down Arrows**: Always drawn **above** the bar (at High price plus 10-15% of bar range)
- **Distance**: Markers are offset from the bar to avoid overlapping with price action

## Color Coding Summary

| Signal Type | Color | Intensity |
|------------|-------|-----------|
| Strong Buy | Lime Green | Brightest |
| Buy | Green | Medium |
| Weak Buy | Light Green | Light |
| Strong Sell | Dark Red | Brightest |
| Sell | Red | Medium |
| Weak Sell | Light Coral | Light |
| Strong Up Trend | Blue | Medium |
| Weak Up Trend | Light Blue | Light |
| Strong Down Trend | Dark Orange | Medium |
| Weak Down Trend | Orange | Light |

## When Markers Appear

1. **Only on Significant Changes**: Markers update only when action changes by >= threshold (default 0.15)
2. **On Bar Close**: Markers are drawn when the bar closes (Calculate.OnBarClose)
3. **Non-Zero Signals Only**: No markers drawn when both Signal_Trend and Signal_Trade are 0

## Integration with Trade Management Tools

Your trade management tool can:
1. **Read Signal Values**: Access `Signal_Trend` and `Signal_Trade` properties programmatically
2. **See Visual Confirmation**: View arrows on chart to confirm signals
3. **Use for Entry/Exit**: Execute trades based on signal values

## Example Trade Management Logic

```csharp
// In your trade management tool
RLSignalIndicator indicator = ...; // Get indicator instance

if (indicator.Signal_Trade == 3)
{
    // Strong buy signal - enter long with full size
    EnterLong();
}
else if (indicator.Signal_Trade == 2)
{
    // Pullback buy - enter long with reduced size
    EnterLong(0.5); // Half size
}
else if (indicator.Signal_Trade == -3)
{
    // Strong sell signal - enter short with full size
    EnterShort();
}
```

## Notes

- **Overlay Mode**: Indicator draws directly on the price chart (IsOverlay = true)
- **No Separate Panel**: No separate indicator panel - all markers are on price chart
- **Automatic Cleanup**: Previous markers are removed when new signals arrive
- **Real-time Updates**: Markers update as new signals are received from Python server

