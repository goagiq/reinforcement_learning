"""
Model Versioning and Rollback System

Manages model versions, tracks performance, and enables rollback.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class ModelVersion:
    """Model version metadata"""
    version: str
    model_path: str
    created_at: str
    performance_metrics: Dict
    training_config: Dict
    description: str
    is_production: bool = False


class ModelVersionManager:
    """
    Manages model versions, performance tracking, and rollback.
    """
    
    def __init__(self, models_dir: str = "models", versions_file: str = "models/versions.json"):
        """
        Initialize version manager.
        
        Args:
            models_dir: Directory containing model files
            versions_file: Path to versions metadata file
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.versions_file = Path(versions_file)
        self.versions_file.parent.mkdir(exist_ok=True)
        
        # Load existing versions
        self.versions: Dict[str, ModelVersion] = {}
        self._load_versions()
    
    def _load_versions(self):
        """Load version metadata from file"""
        if self.versions_file.exists():
            with open(self.versions_file, 'r') as f:
                data = json.load(f)
                self.versions = {
                    v["version"]: ModelVersion(**v)
                    for v in data.get("versions", [])
                }
        else:
            self.versions = {}
    
    def _save_versions(self):
        """Save version metadata to file"""
        data = {
            "last_updated": datetime.now().isoformat(),
            "versions": [asdict(v) for v in self.versions.values()]
        }
        
        with open(self.versions_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_version(
        self,
        model_path: str,
        performance_metrics: Dict,
        training_config: Dict,
        description: str = "",
        version_name: Optional[str] = None
    ) -> str:
        """
        Create a new model version.
        
        Args:
            model_path: Path to model file
            performance_metrics: Performance metrics
            training_config: Training configuration
            description: Version description
            version_name: Optional version name (auto-generated if None)
        
        Returns:
            Version identifier
        """
        if version_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version_name = f"v{len(self.versions) + 1}_{timestamp}"
        
        # Copy model to versioned location
        versioned_path = self.models_dir / f"{version_name}.pt"
        shutil.copy(model_path, versioned_path)
        
        # Create version metadata
        version = ModelVersion(
            version=version_name,
            model_path=str(versioned_path),
            created_at=datetime.now().isoformat(),
            performance_metrics=performance_metrics,
            training_config=training_config,
            description=description,
            is_production=False
        )
        
        self.versions[version_name] = version
        self._save_versions()
        
        print(f"✅ Created model version: {version_name}")
        return version_name
    
    def set_production(self, version: str):
        """
        Set a version as production.
        
        Args:
            version: Version identifier
        """
        if version not in self.versions:
            print(f"⚠️  Version {version} not found")
            return
        
        # Unset other production versions
        for v in self.versions.values():
            v.is_production = False
        
        # Set this as production
        self.versions[version].is_production = True
        self._save_versions()
        
        # Create symlink/copy to best_model.pt
        best_model_path = self.models_dir / "best_model.pt"
        if best_model_path.exists():
            best_model_path.unlink()
        
        shutil.copy(self.versions[version].model_path, best_model_path)
        
        print(f"✅ Set {version} as production model")
    
    def get_production_version(self) -> Optional[ModelVersion]:
        """Get current production version"""
        for version in self.versions.values():
            if version.is_production:
                return version
        return None
    
    def list_versions(self) -> List[ModelVersion]:
        """List all versions"""
        return sorted(self.versions.values(), key=lambda v: v.created_at, reverse=True)
    
    def get_version(self, version: str) -> Optional[ModelVersion]:
        """Get version by identifier"""
        return self.versions.get(version)
    
    def rollback(self, version: str) -> bool:
        """
        Rollback to a previous version.
        
        Args:
            version: Version identifier to rollback to
        
        Returns:
            True if successful
        """
        if version not in self.versions:
            print(f"⚠️  Version {version} not found")
            return False
        
        # Set as production
        self.set_production(version)
        
        print(f"✅ Rolled back to version: {version}")
        return True
    
    def delete_version(self, version: str):
        """
        Delete a version (only if not production).
        
        Args:
            version: Version identifier
        """
        if version not in self.versions:
            print(f"⚠️  Version {version} not found")
            return
        
        if self.versions[version].is_production:
            print(f"⚠️  Cannot delete production version: {version}")
            return
        
        # Delete model file
        model_path = Path(self.versions[version].model_path)
        if model_path.exists():
            model_path.unlink()
        
        # Remove from versions
        del self.versions[version]
        self._save_versions()
        
        print(f"✅ Deleted version: {version}")
    
    def print_status(self):
        """Print version status"""
        print("\n" + "="*60)
        print("MODEL VERSIONS")
        print("="*60)
        
        versions = self.list_versions()
        
        for v in versions:
            status = "🟢 PRODUCTION" if v.is_production else "⚪"
            print(f"\n{status} {v.version}")
            print(f"  Created: {v.created_at}")
            print(f"  Description: {v.description}")
            if v.performance_metrics:
                sharpe = v.performance_metrics.get("sharpe_ratio", 0)
                return_pct = v.performance_metrics.get("total_return", 0) * 100
                print(f"  Sharpe: {sharpe:.2f}, Return: {return_pct:.2f}%")
        
        print("\n" + "="*60)


# Example usage
if __name__ == "__main__":
    manager = ModelVersionManager()
    
    # Create a test version
    test_metrics = {
        "sharpe_ratio": 1.5,
        "total_return": 0.15,
        "win_rate": 0.60
    }
    
    test_config = {"learning_rate": 0.0003, "batch_size": 64}
    
    version = manager.create_version(
        model_path="models/test_model.pt",
        performance_metrics=test_metrics,
        training_config=test_config,
        description="Test version"
    )
    
    manager.print_status()

