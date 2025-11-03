"""
Automated Learning Orchestrator

Coordinates continuous learning, model evaluation, and deployment.
"""

import yaml
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict

from src.continuous_learning import ContinuousLearningPipeline, ExperienceBuffer
from src.model_evaluation import ModelEvaluator
from src.model_versioning import ModelVersionManager
from src.rl_agent import PPOAgent
from src.reasoning_engine import ReasoningEngine


class AutomatedLearningOrchestrator:
    """
    Orchestrates the entire continuous learning pipeline.
    
    Workflow:
    1. Monitor experience collection
    2. Trigger retraining when threshold reached
    3. Evaluate new models
    4. Compare with existing models
    5. Deploy best model if improved
    6. Trigger DeepSeek fine-tuning if needed
    """
    
    def __init__(self, config: Dict):
        """
        Initialize orchestrator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize components
        self.buffer = ExperienceBuffer(
            max_size=config.get("experience_buffer_size", 10000),
            storage_dir=config.get("experience_storage", "data/experience_buffer")
        )
        
        # Load existing buffer if available
        self._load_existing_buffer()
        
        self.version_manager = ModelVersionManager(
            models_dir=config.get("models_dir", "models")
        )
        
        self.evaluator = ModelEvaluator(config)
        
        # Learning configuration
        self.learning_config = config.get("continuous_learning", {})
    
    def _load_existing_buffer(self):
        """Load existing experience buffer"""
        buffer_dir = Path("data/experience_buffer")
        if buffer_dir.exists():
            # Find most recent buffer file
            buffer_files = sorted(buffer_dir.glob("experience_buffer_*.pkl"), reverse=True)
            if buffer_files:
                try:
                    self.buffer.load(str(buffer_files[0]))
                    print(f"📂 Loaded existing buffer: {len(self.buffer.buffer)} experiences")
                except Exception as e:
                    print(f"⚠️  Could not load buffer: {e}")
    
    def check_and_retrain(self, agent: PPOAgent):
        """
        Check if retraining is needed and trigger if so.
        
        Args:
            agent: Current RL agent
        """
        pipeline = ContinuousLearningPipeline(
            self.learning_config,
            self.buffer,
            agent,
            ReasoningEngine()
        )
        
        if pipeline.should_retrain():
            print("\n🔄 Triggering model retraining...")
            
            # Retrain
            new_model_path = pipeline.retrain_model(
                training_config=self.config.get("model", {})
            )
            
            if new_model_path:
                # Evaluate new model
                new_metrics = self.evaluator.evaluate_model(
                    new_model_path,
                    n_episodes=self.config.get("evaluation_episodes", 10)
                )
                
                # Get current production model
                current_prod = self.version_manager.get_production_version()
                
                # Compare
                if current_prod:
                    current_metrics = current_prod.performance_metrics
                    improvement = new_metrics.sharpe_ratio - current_metrics.get("sharpe_ratio", 0)
                    
                    if improvement > 0.1:  # 10% improvement threshold
                        # Create version and deploy
                        version = self.version_manager.create_version(
                            model_path=new_model_path,
                            performance_metrics={
                                "sharpe_ratio": new_metrics.sharpe_ratio,
                                "total_return": new_metrics.total_return,
                                "win_rate": new_metrics.win_rate,
                                "max_drawdown": new_metrics.max_drawdown
                            },
                            training_config=self.config.get("model", {}),
                            description=f"Auto-retrained after {self.buffer.total_experiences} experiences"
                        )
                        
                        self.version_manager.set_production(version)
                        print(f"✅ New model deployed: {version}")
                    else:
                        print(f"⚠️  New model not significantly better ({improvement:.2f} Sharpe)")
                else:
                    # No production model yet, deploy this one
                    version = self.version_manager.create_version(
                        model_path=new_model_path,
                        performance_metrics={
                            "sharpe_ratio": new_metrics.sharpe_ratio,
                            "total_return": new_metrics.total_return,
                            "win_rate": new_metrics.win_rate,
                            "max_drawdown": new_metrics.max_drawdown
                        },
                        training_config=self.config.get("model", {}),
                        description="Initial production model"
                    )
                    self.version_manager.set_production(version)
                    print(f"✅ Initial model deployed: {version}")
    
    def trigger_deepseek_finetuning(self):
        """Check if DeepSeek fine-tuning is needed"""
        pipeline = ContinuousLearningPipeline(
            self.learning_config,
            self.buffer,
            None,  # Not needed for DeepSeek
            ReasoningEngine()
        )
        
        # Check if enough annotated experiences
        if self.buffer.annotated_count >= self.learning_config.get("min_annotated_for_finetune", 100):
            print("\n🤖 Triggering DeepSeek fine-tuning...")
            data_path = pipeline.trigger_deepseek_finetuning()
            
            if data_path:
                print(f"\n📝 Next steps:")
                print(f"   1. Review training data: {data_path}")
                print(f"   2. Run fine-tuning script (see docs/FINETUNING_GUIDE.md)")
                print(f"   3. Update reasoning engine to use fine-tuned model")
    
    def run_maintenance(self):
        """Run maintenance tasks"""
        print("\n" + "="*60)
        print("Automated Learning Maintenance")
        print("="*60)
        
        # Save buffer
        self.buffer.save()
        
        # Print statistics
        print(f"\n📊 Experience Buffer:")
        print(f"   Total experiences: {self.buffer.total_experiences}")
        print(f"   Annotated: {self.buffer.annotated_count}")
        print(f"   Buffer size: {len(self.buffer.buffer)}")
        
        # Print version status
        self.version_manager.print_status()
        
        # Check for model cleanup (delete old non-production versions)
        versions = self.version_manager.list_versions()
        non_prod_versions = [v for v in versions if not v.is_production]
        if len(non_prod_versions) > 10:  # Keep last 10 versions
            for v in non_prod_versions[10:]:
                print(f"🗑️  Cleaning up old version: {v.version}")
                self.version_manager.delete_version(v.version)


def load_config(config_path: str) -> dict:
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Automated Learning Orchestrator")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Configuration file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["maintenance", "retrain", "finetune", "all"],
        default="maintenance",
        help="Operation mode"
    )
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Add continuous learning config if not present
    if "continuous_learning" not in config:
        config["continuous_learning"] = {
            "retrain_frequency": 1000,
            "min_experiences": 500,
            "evaluation_episodes": 10,
            "min_annotated_for_finetune": 100
        }
    
    orchestrator = AutomatedLearningOrchestrator(config)
    
    if args.mode == "maintenance" or args.mode == "all":
        orchestrator.run_maintenance()
    
    if args.mode == "retrain" or args.mode == "all":
        # Load agent for retraining
        prod_version = orchestrator.version_manager.get_production_version()
        if prod_version:
            agent = PPOAgent(
                state_dim=config["environment"]["state_features"],
                action_range=tuple(config["environment"]["action_range"]),
                device="cpu"
            )
            agent.load(prod_version.model_path)
            orchestrator.check_and_retrain(agent)
    
    if args.mode == "finetune" or args.mode == "all":
        orchestrator.trigger_deepseek_finetuning()


if __name__ == "__main__":
    main()

