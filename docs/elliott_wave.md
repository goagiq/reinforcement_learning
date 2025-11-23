# Elliott Wave Agent Implementation Plan

## Goals
- Deliver a real-time Elliott Wave analyst that operates alongside existing swarm agents.
- Focus on impulsive Wave 3 and Wave 5 identification across the 1, 5, and 15 minute streams.
- Produce actionable trade signals and confidence scores for the DecisionGate workflow.

## Tasks
1. **Market Structure Utilities**
   - Create reusable swing detection helpers (zig-zag / ATR thresholds).
   - Implement wave state machine for counts (Tracks Wave 0→5 with validation rules).
   - Expose Fibonacci/Momentum scoring utilities for impulse confirmation.
2. **Agent Implementation**
   - Add `ElliottWaveAgent` (under `src/agentic_swarm/agents/`) inheriting `BaseSwarmAgent`.
   - Consume multi-timeframe bar windows from `SharedContext`.
   - Output trade recommendations (`buy`/`sell`/`hold`) plus metadata (wave labels, key levels, confidence).
   - Log reasoning trace for transparency/monitoring.
3. **Swarm Integration**
   - Register agent in `SwarmOrchestrator` configuration/registry.
   - Wire agent results into DecisionGate weighting, mirroring existing contrarian agent integration.
   - Update configuration defaults / YAML to toggle agent on/off.
4. **Live/Training Support**
   - Ensure live bridge populates shared context with latest resampled bars.
   - Verify training/backtest code can optionally invoke the agent for offline evaluation.
5. **Documentation & Validation**
   - Document tuning knobs (swing sensitivity, Fibonacci tolerances, confidence thresholds).
   - Add smoke test or diagnostic script to replay historical bars and print detected waves.
   - Update README / swarm docs with usage instructions.

## Deliverables
- New Elliott Wave agent module + utilities.
- Configuration updates enabling the agent.
- Unit/integration validation artifacts.
- Enhanced documentation covering workflow and tuning tips.


