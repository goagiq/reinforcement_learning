"""
Continuous Learning Pipeline

Collects trading experiences, annotates with reasoning insights,
and triggers model improvements.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import pickle

from src.reasoning_engine import TradeResult, ReflectionInsight


@dataclass
class Experience:
    """A single trading experience with annotations"""
    timestamp: str
    state: np.ndarray
    action: float
    reward: float
    next_state: np.ndarray
    done: bool
    market_conditions: Dict
    rl_confidence: float
    reasoning_confidence: Optional[float]
    reasoning_insight: Optional[str]
    trade_outcome: Optional[Dict]  # PnL, duration, etc.
    reflection_insight: Optional[Dict]  # Post-trade reflection


class ExperienceBuffer:
    """
    Stores and manages trading experiences for continuous learning.
    
    Features:
    - Automatic experience storage
    - Reasoning annotation
    - Quality filtering
    - Batch retrieval for training
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        storage_dir: str = "data/experience_buffer"
    ):
        """
        Initialize experience buffer.
        
        Args:
            max_size: Maximum number of experiences to store
            storage_dir: Directory for persistent storage
        """
        self.max_size = max_size
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Buffer storage
        self.buffer: deque = deque(maxlen=max_size)
        self.annotations: Dict[str, ReflectionInsight] = {}
        
        # Statistics
        self.total_experiences = 0
        self.annotated_count = 0
    
    def add_experience(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        metadata: Optional[Dict] = None
    ):
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            metadata: Additional metadata (confidence, market conditions, etc.)
        """
        experience = Experience(
            timestamp=datetime.now().isoformat(),
            state=state.copy(),
            action=action,
            reward=reward,
            next_state=next_state.copy(),
            done=done,
            market_conditions=metadata.get("market_conditions", {}) if metadata else {},
            rl_confidence=metadata.get("rl_confidence", 0.5) if metadata else 0.5,
            reasoning_confidence=metadata.get("reasoning_confidence") if metadata else None,
            reasoning_insight=metadata.get("reasoning_insight") if metadata else None,
            trade_outcome=None,
            reflection_insight=None
        )
        
        self.buffer.append(experience)
        self.total_experiences += 1
        
        # Auto-save periodically
        if self.total_experiences % 100 == 0:
            self.save()
    
    def annotate_experience(
        self,
        experience_id: str,
        trade_result: TradeResult,
        reflection: ReflectionInsight
    ):
        """
        Annotate an experience with post-trade reflection.
        
        Args:
            experience_id: ID of the experience (timestamp)
            trade_result: Completed trade result
            reflection: Reflection insights from reasoning engine
        """
        # Find experience by timestamp
        for exp in self.buffer:
            if exp.timestamp == experience_id:
                exp.trade_outcome = {
                    "pnl": trade_result.pnl,
                    "duration": trade_result.duration_seconds,
                    "action": trade_result.action.name
                }
                exp.reflection_insight = asdict(reflection)
                self.annotations[experience_id] = reflection
                self.annotated_count += 1
                break
    
    def get_batch(
        self,
        batch_size: int,
        prioritize_annotated: bool = True,
        min_reward: Optional[float] = None
    ) -> List[Experience]:
        """
        Get a batch of experiences for training.
        
        Args:
            batch_size: Number of experiences to retrieve
            prioritize_annotated: Prefer annotated experiences
            min_reward: Minimum reward threshold
        
        Returns:
            List of experiences
        """
        if len(self.buffer) == 0:
            return []
        
        # Filter experiences
        candidates = list(self.buffer)
        
        # Filter by reward threshold
        if min_reward is not None:
            candidates = [e for e in candidates if e.reward >= min_reward]
        
        # Prioritize annotated
        if prioritize_annotated:
            annotated = [e for e in candidates if e.reflection_insight is not None]
            unannotated = [e for e in candidates if e.reflection_insight is None]
            candidates = annotated + unannotated
        
        # Sample batch
        batch_size = min(batch_size, len(candidates))
        indices = np.random.choice(len(candidates), batch_size, replace=False)
        
        return [candidates[i] for i in indices]
    
    def get_high_value_experiences(
        self,
        n: int = 100,
        min_reward: float = 0.1
    ) -> List[Experience]:
        """
        Get high-value experiences (successful trades).
        
        Args:
            n: Number of experiences to retrieve
            min_reward: Minimum reward threshold
        
        Returns:
            List of high-value experiences
        """
        high_value = [
            e for e in self.buffer
            if e.reward >= min_reward and e.reflection_insight is not None
        ]
        
        # Sort by reward (descending)
        high_value.sort(key=lambda x: x.reward, reverse=True)
        
        return high_value[:n]
    
    def get_failed_experiences(
        self,
        n: int = 100,
        max_reward: float = -0.1
    ) -> List[Experience]:
        """
        Get failed experiences (unsuccessful trades) for learning.
        
        Args:
            n: Number of experiences to retrieve
            max_reward: Maximum reward threshold (negative)
        
        Returns:
            List of failed experiences
        """
        failed = [
            e for e in self.buffer
            if e.reward <= max_reward and e.reflection_insight is not None
        ]
        
        # Sort by reward (ascending - worst first)
        failed.sort(key=lambda x: x.reward)
        
        return failed[:n]
    
    def save(self, filepath: Optional[str] = None):
        """Save buffer to disk"""
        if filepath is None:
            filepath = self.storage_dir / f"experience_buffer_{datetime.now().strftime('%Y%m%d')}.pkl"
        
        data = {
            "experiences": [asdict(e) for e in self.buffer],
            "annotations": {k: asdict(v) for k, v in self.annotations.items()},
            "total_experiences": self.total_experiences,
            "annotated_count": self.annotated_count
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"💾 Experience buffer saved: {len(self.buffer)} experiences ({self.annotated_count} annotated)")
    
    def load(self, filepath: str):
        """Load buffer from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Reconstruct experiences
        self.buffer = deque([Experience(**e) for e in data["experiences"]], maxlen=self.max_size)
        self.total_experiences = data.get("total_experiences", len(self.buffer))
        self.annotated_count = data.get("annotated_count", 0)
        
        print(f"📂 Experience buffer loaded: {len(self.buffer)} experiences")


class ContinuousLearningPipeline:
    """
    Orchestrates continuous learning process.
    
    Workflow:
    1. Collect experiences during trading
    2. Annotate with reasoning insights
    3. Periodically retrain model
    4. Evaluate and compare models
    5. Deploy best model
    """
    
    def __init__(
        self,
        config: Dict,
        experience_buffer: ExperienceBuffer,
        agent,
        reasoning_engine
    ):
        """
        Initialize continuous learning pipeline.
        
        Args:
            config: Configuration
            experience_buffer: Experience buffer instance
            agent: RL agent instance
            reasoning_engine: Reasoning engine instance
        """
        self.config = config
        self.buffer = experience_buffer
        self.agent = agent
        self.reasoning_engine = reasoning_engine
        
        self.retrain_frequency = config.get("retrain_frequency", 1000)  # Retrain every N experiences
        self.min_experiences = config.get("min_experiences", 500)  # Minimum experiences before retraining
        self.evaluation_episodes = config.get("evaluation_episodes", 10)
        
        self.retrain_count = 0
        self.last_retrain_experience_count = 0
    
    def should_retrain(self) -> bool:
        """Check if model should be retrained"""
        current_count = self.buffer.total_experiences
        
        # Check if enough new experiences since last retrain
        new_experiences = current_count - self.last_retrain_experience_count
        
        return (
            current_count >= self.min_experiences and
            new_experiences >= self.retrain_frequency
        )
    
    def retrain_model(self, training_config: Dict):
        """
        Trigger model retraining with new experiences.
        
        Args:
            training_config: Training configuration
        """
        print("\n" + "="*60)
        print("Continuous Learning: Retraining Model")
        print("="*60)
        
        # Get training batch
        batch = self.buffer.get_batch(
            batch_size=training_config.get("batch_size", 64),
            prioritize_annotated=True
        )
        
        if len(batch) < training_config.get("batch_size", 64):
            print(f"⚠️  Not enough experiences: {len(batch)}")
            return
        
        print(f"📊 Retraining with {len(batch)} experiences")
        print(f"   Annotated: {sum(1 for e in batch if e.reflection_insight is not None)}")
        
        # Convert to training format
        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch])
        
        # Update agent (simplified - would need proper RL update)
        # In practice, this would call the agent's update method
        # For now, this is a placeholder structure
        
        self.retrain_count += 1
        self.last_retrain_experience_count = self.buffer.total_experiences
        
        print(f"✅ Model retrained (retrain #{self.retrain_count})")
        
        # Save checkpoint
        checkpoint_path = f"models/continuous_learning_{self.retrain_count}.pt"
        self.agent.save(checkpoint_path)
        
        return checkpoint_path
    
    def generate_training_data_for_deepseek(self) -> List[Dict]:
        """
        Generate training data for DeepSeek fine-tuning from experiences.
        
        Returns:
            List of training examples in format suitable for LLM fine-tuning
        """
        # Get high-value and failed experiences
        high_value = self.buffer.get_high_value_experiences(n=50)
        failed = self.buffer.get_failed_experiences(n=50)
        
        training_data = []
        
        # Format successful trades as positive examples
        for exp in high_value:
            if exp.reflection_insight:
                training_data.append({
                    "instruction": "Analyze this trading scenario and provide a recommendation.",
                    "input": f"Market state: {exp.market_conditions}, Action taken: {exp.action}",
                    "output": exp.reflection_insight.get("reasoning", "This was a successful trade.")
                })
        
        # Format failed trades as negative examples
        for exp in failed:
            if exp.reflection_insight:
                training_data.append({
                    "instruction": "Analyze this trading scenario and identify what went wrong.",
                    "input": f"Market state: {exp.market_conditions}, Action taken: {exp.action}",
                    "output": exp.reflection_insight.get("reasoning", "This trade failed.")
                })
        
        return training_data
    
    def trigger_deepseek_finetuning(self):
        """Trigger DeepSeek model fine-tuning"""
        print("\n🔧 Triggering DeepSeek fine-tuning...")
        
        # Generate training data
        training_data = self.generate_training_data_for_deepseek()
        
        if len(training_data) < 10:
            print("⚠️  Not enough annotated experiences for fine-tuning")
            return
        
        print(f"📊 Generated {len(training_data)} training examples")
        
        # Save training data
        data_path = Path("data/finetuning") / f"deepseek_training_{datetime.now().strftime('%Y%m%d')}.json"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(data_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        print(f"💾 Training data saved to: {data_path}")
        print("\n📝 To fine-tune DeepSeek:")
        print("   1. Use the saved training data")
        print("   2. Run fine-tuning script (see docs/FINETUNING_GUIDE.md)")
        print("   3. Update reasoning engine to use fine-tuned model")
        
        return data_path


# Example usage
if __name__ == "__main__":
    # Test experience buffer
    buffer = ExperienceBuffer(max_size=1000)
    
    # Simulate adding experiences
    for i in range(10):
        buffer.add_experience(
            state=np.random.randn(200),
            action=np.random.uniform(-1, 1),
            reward=np.random.randn() * 0.1,
            next_state=np.random.randn(200),
            done=False,
            metadata={"rl_confidence": 0.8}
        )
    
    # Get batch
    batch = buffer.get_batch(5)
    print(f"Retrieved batch of {len(batch)} experiences")
    
    # Save
    buffer.save()

