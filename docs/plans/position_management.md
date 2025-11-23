Core Approach
RL stays primary. RL action sets the initial direction and base size (still clipped to [-1, 1]).
Dynamic sizing from confluences. Swarm/Elliott-wave signals, DecisionGate confidence, and other agent scores become “confluence inputs” that scale size around the RL base.
Strict risk ceiling. After sizing adjustments, clamp exposure to max(1.0, capital × 2%) so we respect both normalized and hard-dollar risk.
Position Scaling
Scale-in (market favorable). When DecisionGate confirms RL direction and multiple confluences fire (e.g. swarm bias + Elliott wave + contrarian agreement, or high combined confidence), incrementally add to the position. Think sliding multiplier such as:
1 confluence ⇒ baseline size
2 confluences ⇒ baseline × 1.25
3+ confluences ⇒ baseline × 1.5 (subject to the risk cap)
Scale-out (confluences drop). If the confluence score falls (Confidence dips, agents flip neutral), trim proportionally: e.g., drop to baseline when one signal remains, to 0 when no supporting confluences left.
Break-even / Free-Trade Logic
Auto move to break-even. Once unrealized PnL reaches risk_multiple_threshold (e.g. +0.75R or +1R), move stop to entry price. For partial scaling, adjust break-even level using weighted average entry after scale-ins.
Let profits run. Keep trailing stop loose (e.g. ATR or structure-based) once break-even is in place; don’t exit until RL or confluence collapse triggers scale-out/exit.
Implementation Hooks
Extend DecisionGate or a new PositionSizer component:
Take RL action, combined confidence, agent breakdown.
Compute confluence_score (0–1) from number and strength of agreeing agents.
Apply scale_factor = 1 + alpha × (confluence_score - baseline) to RL size.
Enforce exposure ceiling min(max_normalized, max_by_balance).
Add RiskManager update to institute break-even: track per-trade cost basis; when PnL ≥ threshold, publish stop adjustment (or just return updated risk plan to NT8).
Make sure scale adjustments are symmetric for long/short and operate only when RL’s direction is backed by current confluence.
If that sounds aligned, I can draft the sizing function + break-even logic next (config knobs, adjusted decision flow). Let me know if you want to control how confluences are counted or what R-multiple should trigger the break-even.
