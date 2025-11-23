"""
Capability registry for AI-generated analyses and explanatory tooltips.

Each capability maps to a specific UI surface (tab + section) so the frontend
can request contextual explanations from the backend AI service.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class CapabilityMetadata:
    """
    Metadata describing a single UI capability that needs AI analysis.

    Attributes:
        id: Stable identifier (dot separated).
        tab: High-level dashboard tab (e.g., training, trading).
        section: UI section or card name.
        title: Human readable title for reference/logging.
        prompt_template: Template passed to the AI model. May include replacement
            variables such as {locale}, {user_context}, {system_metrics}, etc.
        tooltip_template: Separate template for the tooltip-specific summary.
        locale_key: Optional base i18n key if static translations exist.
        tags: Optional list of semantic tags to help model routing and analytics.
    """

    id: str
    tab: str
    section: str
    title: str
    prompt_template: str
    tooltip_template: str
    locale_key: Optional[str] = None
    tags: List[str] = field(default_factory=list)


CAPABILITY_REGISTRY: Dict[str, CapabilityMetadata] = {
    "overview.quick_start": CapabilityMetadata(
        id="overview.quick_start",
        tab="overview",
        section="quick_start",
        title="Quick Start Callouts",
        prompt_template=(
            "You are an expert trading mentor. Explain the four onboarding steps "
            "for the NT8 RL Trading System: training, backtesting, paper trading, "
            "and going live. Include why each step matters, common pitfalls, and "
            "what success looks like. Tailor the depth of guidance using the "
            "provided user context: {user_context}. Use concise numbered sections."
        ),
        tooltip_template=(
            "Provide a one-sentence summary of why this step matters for the user "
            "given their current status: {user_context}."
        ),
        locale_key="overview.quick_start",
        tags=["overview", "onboarding", "guidance"],
    ),
    "training.device": CapabilityMetadata(
        id="training.device",
        tab="training",
        section="device_selection",
        title="Training Device Selection",
        prompt_template=(
            "Analyze the hardware configuration options for RL training. The user "
            "can choose CPU or CUDA GPU. Explain trade-offs using the latest GPU "
            "status: {gpu_status}, training configuration: {training_config}, and "
            "recent training metrics: {training_metrics}. Recommend which option "
            "to use and call out warnings if CUDA is unavailable."
        ),
        tooltip_template=(
            "Summarize in <=20 words whether the user should stay on CUDA or CPU "
            "based on {gpu_status}."
        ),
        locale_key="training.device",
        tags=["training", "hardware", "cuda"],
    ),
    "training.progress": CapabilityMetadata(
        id="training.progress",
        tab="training",
        section="progress_metrics",
        title="Training Progress Metrics",
        prompt_template=(
            "Provide a detailed analysis of the PPO training progress using the "
            "latest metrics: {training_metrics}. Highlight convergence signals, "
            "instability risks, and recommended next actions. Assume the reader is "
            "familiar with RL but wants actionable insight."
        ),
        tooltip_template=(
            "Give a short status headline (<=12 words) about PPO training health "
            "using {training_metrics}."
        ),
        locale_key="training.progress",
        tags=["training", "metrics", "analysis"],
    ),
    "training.checkpoints": CapabilityMetadata(
        id="training.checkpoints",
        tab="training",
        section="checkpoint_management",
        title="Checkpoint Promotion Guidance",
        prompt_template=(
            "Review the available checkpoints: {checkpoint_inventory} and the "
            "current configuration: {training_config}. Explain when the user "
            "should promote a checkpoint to best_model.pt, how transfer learning "
            "strategies affect outcomes, and any architectural mismatches."
        ),
        tooltip_template=(
            "In one sentence, state if the selected checkpoint should be promoted "
            "based on {checkpoint_inventory}."
        ),
        locale_key="training.checkpoints",
        tags=["training", "model-management"],
    ),
    "backtest.results": CapabilityMetadata(
        id="backtest.results",
        tab="backtest",
        section="results",
        title="Backtest Result Interpretation",
        prompt_template=(
            "Interpret the latest backtest output: {backtest_results}. Discuss "
            "risk-adjusted performance, drawdowns, sample size, and whether results "
            "justify moving to paper trading. Include at least one cautionary note."
        ),
        tooltip_template=(
            "Summarize backtest verdict in <=15 words using {backtest_results}."
        ),
        locale_key="backtest.results",
        tags=["backtest", "performance"],
    ),
    "scenarios.guidance": CapabilityMetadata(
        id="scenarios.guidance",
        tab="scenarios",
        section="scenario_runner",
        title="Scenario Simulation Guidance",
        prompt_template=(
            "Describe how each configured market scenario should be used: "
            "{scenario_config}. Provide analysis of stress levels, expected agent "
            "behavior, and how the user should interpret outcomes."
        ),
        tooltip_template=(
            "Highlight the most critical scenario to run next using {scenario_config}."
        ),
        locale_key="scenarios.guidance",
        tags=["scenarios", "risk"],
    ),
    "analytics.markov": CapabilityMetadata(
        id="analytics.markov",
        tab="analytics",
        section="markov_regime",
        title="Markov Regime Analysis",
        prompt_template=(
            "Explain the Markov regime analysis results contained in "
            "{markov_report}. Translate findings into actionable trading "
            "guidance and discuss confidence levels."
        ),
        tooltip_template=(
            "Share the current dominant regime in <=10 words using {markov_report}."
        ),
        locale_key="analytics.markov",
        tags=["analytics", "markov"],
    ),
    "trading.live": CapabilityMetadata(
        id="trading.live",
        tab="trading",
        section="live_controls",
        title="Live Trading Readiness",
        prompt_template=(
            "Evaluate whether the system is ready for live trading. Consider "
            "backtest stats: {backtest_results}, drift detection: {drift_status}, "
            "and risk settings: {risk_profile}. Provide go/no-go guidance."
        ),
        tooltip_template=(
            "State go/no-go readiness in <=12 words referencing {drift_status} and "
            "{risk_profile}."
        ),
        locale_key="trading.live",
        tags=["trading", "risk", "readiness"],
    ),
    "monitoring.performance": CapabilityMetadata(
        id="monitoring.performance",
        tab="monitoring",
        section="system_health",
        title="System Monitoring Overview",
        prompt_template=(
            "Summarize current system health using metrics: {monitoring_metrics} "
            "and alerts: {active_alerts}. Call out urgent issues and remediation "
            "steps."
        ),
        tooltip_template=(
            "Provide a color-coded severity word (e.g., GREEN, YELLOW) and brief "
            "note using {monitoring_metrics}."
        ),
        locale_key="monitoring.performance",
        tags=["monitoring", "ops"],
    ),
}


def list_capabilities_by_tab(tab: Optional[str] = None) -> List[CapabilityMetadata]:
    """Return capabilities, optionally filtered by dashboard tab."""
    capabilities = list(CAPABILITY_REGISTRY.values())
    if tab:
        capabilities = [cap for cap in capabilities if cap.tab == tab]
    return capabilities


def get_capability(capability_id: str) -> Optional[CapabilityMetadata]:
    """Lookup capability metadata by identifier."""
    return CAPABILITY_REGISTRY.get(capability_id)


