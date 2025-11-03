# Reward Function Documentation

## Overview

The reward function is a critical component that guides the RL agent's learning. It balances profitability (PnL) with risk management (drawdowns, transaction costs) to train an agent that can make profitable trades while managing risk.

## Current Implementation (Updated 2024)

### Location
`src/trading_env.py` - `TradingEnvironment._calculate_reward()`

### Formula

```python
def _calculate_reward(self, prev_pnl: float, current_pnl: float) -> float:
    """Calculate reward based on PnL and risk"""
    
    # 1. Calculate normalized PnL change
    pnl_change = (current_pnl - prev_pnl) / self.initial_capital
    
    # 2. Calculate drawdown
    current_equity = self.initial_capital + current_pnl
    if current_equity > self.max_equity:
        self.max_equity = current_equity
    
    drawdown = (self.max_equity - current_equity) / self.max_equity if self.max_equity > 0 else 0.0
    if drawdown > self.max_drawdown:
        self.max_drawdown = drawdown
    
    # 3. Base reward components
    reward = (
        self.reward_config["pnl_weight"] * pnl_change                    # Primary: PnL change
        - self.reward_config["risk_penalty"] * 0.1 * drawdown           # Reduced: Risk penalty (10%)
        - self.reward_config["drawdown_penalty"] * 0.1 * max(0, self.max_drawdown - 0.15)  # Only if DD > 15%
    )
    
    # 4. Transaction/holding costs
    if self.state and abs(self.state.position) > 0.01:
        holding_cost = self.transaction_cost * 0.001  # 0.1% of transaction cost per step
        reward -= holding_cost
    
    # 5. Profit bonus
    if pnl_change > 0:
        reward += abs(pnl_change) * 0.1  # Small bonus multiplier for profits
    
    # 6. Scale for learning stability
    reward *= 10.0
    
    return reward
```

## Design Philosophy

### 1. **PnL-Focused Primary Signal**
- The primary reward component is PnL change normalized by initial capital
- This directly rewards profitable moves and penalizes losses
- Normalization ensures rewards are consistent regardless of capital size

### 2. **Balanced Risk Penalties**
- Risk penalties are **reduced by 90%** (multiplied by 0.1)
- Drawdown penalty only activates if drawdown > 15% (was 10%)
- This prevents penalties from overwhelming positive PnL signals
- Allows the agent to learn profitable strategies before perfect risk management

### 3. **Minimal Transaction Costs**
- Holding cost is **0.1% of full transaction cost** per step
- Applied only when position is open (not every trade)
- Prevents costs from dominating rewards during learning phase
- Encourages agent to hold positions when profitable

### 4. **Profit Encouragement**
- Small bonus multiplier (10%) added to positive PnL changes
- Reinforces profitable behavior
- Helps agent discover and maintain profitable strategies

### 5. **Learning-Friendly Scaling**
- Final reward scaled by 10x (was 100x)
- Provides stable gradients for PPO training
- Prevents exploding/vanishing gradients
- Moderate scaling balances signal strength without overwhelming

## Configuration Parameters

From `configs/train_config.yaml`:

```yaml
reward:
  pnl_weight: 1.0              # Weight for PnL contribution
  transaction_cost: 0.0001     # Transaction cost per trade
  risk_penalty: 0.5            # Risk penalty coefficient (applied at 10%)
  drawdown_penalty: 0.3        # Drawdown penalty (applied at 10%, only if DD > 15%)
```

**Effective Weights:**
- `pnl_weight`: 1.0 (full weight)
- `risk_penalty`: 0.05 (0.5 * 0.1)
- `drawdown_penalty`: 0.03 (0.3 * 0.1, only if DD > 15%)
- `holding_cost`: 0.0000001 (0.0001 * 0.001)

## Expected Reward Ranges

### Typical Episode Scenarios

**Profitable Episode (Positive PnL):**
- PnL change: +0.01 (1% gain)
- Base reward: +0.01 * 1.0 = +0.01
- Profit bonus: +0.01 * 0.1 = +0.001
- Holding cost: -0.0000001 * 10,000 steps = -0.001
- Final (scaled): (+0.011 - 0.001) * 10 = **+0.10**

**Losing Episode (Negative PnL):**
- PnL change: -0.01 (1% loss)
- Base reward: -0.01 * 1.0 = -0.01
- Drawdown penalty: -0.05 * 0.1 = -0.005 (if DD occurs)
- Holding cost: -0.001
- Final (scaled): (-0.01 - 0.005 - 0.001) * 10 = **-0.16**

**Neutral Episode (No PnL Change):**
- PnL change: 0.0
- Base reward: 0.0
- Holding cost: -0.001
- Final (scaled): -0.001 * 10 = **-0.01**

## Evolution of the Reward Function

### Previous Issues
1. **Transaction cost penalty was too high**: Applied full transaction cost every step when position open
2. **Penalties dominated rewards**: Risk/drawdown penalties overwhelmed PnL signals
3. **Scaling too aggressive**: 100x multiplier amplified negative penalties too much
4. **Always negative**: Agent couldn't achieve positive rewards even with profitable trades

### Current Solution
1. ✅ Minimal holding cost (0.1% of transaction cost per step)
2. ✅ Reduced penalty weights (90% reduction)
3. ✅ Moderate scaling (10x instead of 100x)
4. ✅ Profit bonus encourages positive rewards
5. ✅ Only penalize significant drawdowns (>15%)

## Tuning Guidelines

### If Rewards Are Too Negative
- Increase `pnl_weight` (try 1.5 or 2.0)
- Further reduce penalty weights (use 0.05 instead of 0.1)
- Increase profit bonus multiplier (try 0.2 or 0.3)

### If Agent Takes Too Much Risk
- Increase `risk_penalty` in config (agent still gets 10%)
- Decrease drawdown threshold (try 0.12 instead of 0.15)
- Add volatility penalty component

### If Agent Is Too Conservative
- Decrease `risk_penalty` in config
- Increase drawdown threshold (try 0.20)
- Add bonus for trade execution

## Monitoring Reward Health

### Healthy Training Signs
- ✅ Reward trend improving over time (less negative → positive)
- ✅ Episodes with positive rewards appearing after ~500K timesteps
- ✅ Mean reward (last 10 episodes) gradually increasing
- ✅ PnL positively correlated with reward

### Warning Signs
- ⚠️ All rewards consistently very negative (< -50)
- ⚠️ No positive rewards after 1M timesteps
- ⚠️ Rewards getting more negative over time
- ⚠️ PnL positive but rewards still negative

## Related Files

- `src/trading_env.py` - Implementation
- `configs/train_config.yaml` - Configuration
- `src/train.py` - Training loop (logs rewards)
- `docs/HOW_RL_TRADING_WORKS.md` - High-level explanation

