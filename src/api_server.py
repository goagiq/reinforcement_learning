"""
FastAPI Backend Server for NT8 RL Trading System UI

Provides REST API and WebSocket endpoints to control all system operations.
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import yaml
import threading
import time
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime
from dataclasses import asdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import psutil

# Import project modules
from src.data_extraction import DataExtractor
from src.train import Trainer
from src.backtest import Backtester
from src.live_trading import LiveTradingSystem
from src.nt8_bridge_server import NT8BridgeServer
from src.automated_learning import AutomatedLearningOrchestrator
from src.monitoring import PerformanceMonitor
from src.model_versioning import ModelVersionManager
from src.auto_retrain_monitor import AutoRetrainMonitor
from src.analysis.markov_regime import (
    MarkovRegimeAnalyzer,
    RegimeConfig,
    run_and_save_report,
)
from src.ai_analysis_service import generate_analysis, record_feedback
from src.capability_registry import list_capabilities_by_tab, get_capability
from src.utils.colors import error, warn

# Deep Research System
try:
    from src.deep_research.deep_research_system import DeepResearchSystem
    from src.deep_research.preferences.feedback_tracker import FeedbackType
    DEEP_RESEARCH_AVAILABLE = True
except ImportError:
    DEEP_RESEARCH_AVAILABLE = False
    DeepResearchSystem = None
    FeedbackType = None

# Monte Carlo risk assessment
try:
    from src.monte_carlo_risk import (
        MonteCarloRiskAnalyzer,
        MonteCarloResult,
        assess_position_risk
    )
    MONTE_CARLO_AVAILABLE = True
except ImportError:
    MONTE_CARLO_AVAILABLE = False
    MonteCarloRiskAnalyzer = None

# Volatility prediction
try:
    from src.volatility_predictor import (
        VolatilityPredictor,
        VolatilityForecast,
        predict_volatility
    )
    VOLATILITY_PREDICTOR_AVAILABLE = True
except ImportError:
    VOLATILITY_PREDICTOR_AVAILABLE = False
    VolatilityPredictor = None
    VolatilityForecast = None

# Scenario simulation
try:
    from src.scenario_simulator import (
        ScenarioSimulator,
        ScenarioResult,
        StressTestResult,
        ParameterSensitivityResult,
        MarketRegime,
        run_robustness_test
    )
    SCENARIO_SIMULATOR_AVAILABLE = True
except ImportError:
    SCENARIO_SIMULATOR_AVAILABLE = False
    ScenarioSimulator = None
    ScenarioResult = None
    MarketRegime = None


app = FastAPI(title="NT8 RL Trading System API")

# CORS middleware
# Note: If using Kong Gateway, CORS can be handled by Kong's CORS plugin
# This middleware remains for backward compatibility (direct access to FastAPI)
# Set DISABLE_FASTAPI_CORS=true to disable CORS in FastAPI when using Kong
DISABLE_FASTAPI_CORS = os.getenv("DISABLE_FASTAPI_CORS", "false").lower() == "true"

if not DISABLE_FASTAPI_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, restrict to your frontend domain
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Global state
active_processes: Dict[str, subprocess.Popen] = {}
active_systems: Dict[str, any] = {}
websocket_connections: List[WebSocket] = []
main_event_loop: Optional[asyncio.AbstractEventLoop] = None
auto_retrain_monitor: Optional[AutoRetrainMonitor] = None

# Deep Research System (lazy initialization)
deep_research_system: Optional[DeepResearchSystem] = None

def get_deep_research_system() -> Optional[DeepResearchSystem]:
    """Get or initialize Deep Research System"""
    global deep_research_system
    if not DEEP_RESEARCH_AVAILABLE:
        return None
    if deep_research_system is None:
        try:
            deep_research_system = DeepResearchSystem()
        except Exception as e:
            print(f"Warning: Failed to initialize Deep Research System: {e}")
            return None
    return deep_research_system


# Pydantic models
class SetupCheckResponse(BaseModel):
    venv_exists: bool
    venv_type: Optional[str] = None  # "uv", "venv", "virtualenv", "conda", etc.
    dependencies_installed: bool
    data_directory_exists: bool
    config_exists: bool
    ready: bool
    issues: List[str]
    venv_message: Optional[str] = None  # Suggested command to create venv


class TrainingRequest(BaseModel):
    device: str = "cpu"
    total_timesteps: Optional[int] = None
    config_path: str = "configs/train_config_full.yaml"
    reasoning_model: Optional[str] = None
    checkpoint_path: Optional[str] = None  # Resume from checkpoint
    transfer_strategy: Optional[str] = None  # Transfer learning strategy: "copy_and_extend", "interpolate", "zero_pad"
    flush_old_data: bool = False  # Clear trading journal and caches when starting fresh training


class BacktestRequest(BaseModel):
    model_path: str
    episodes: int = 20
    config_path: str = "configs/train_config_full.yaml"


class ResearchSearchRequest(BaseModel):
    query: str
    sources: Optional[List[str]] = None
    max_results: int = 20


class ResearchFetchRequest(BaseModel):
    source_type: str
    source_id: str
    store: bool = True
    generate_plan: bool = False
    generate_doc: bool = False


class ResearchFeedbackRequest(BaseModel):
    item_id: str
    feedback_type: str
    metadata: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


class LiveTradingRequest(BaseModel):
    model_path: str
    paper_trading: bool = True
    config_path: str = "configs/train_config_full.yaml"


class StatusResponse(BaseModel):
    status: str
    message: str
    details: Optional[Dict] = None


class SettingsRequest(BaseModel):
    nt8_data_path: Optional[str] = None
    performance_mode: Optional[str] = None  # "quiet" or "performance"
    turbo_training_mode: Optional[bool] = None  # Enable turbo mode (max GPU usage)
    auto_retrain_enabled: Optional[bool] = None  # Enable/disable auto-retrain
    contrarian_enabled: Optional[bool] = None  # Enable/disable contrarian agent in swarm
    ai_insights_enabled: Optional[bool] = None  # Enable/disable AI detailed analyses
    ai_tooltips_enabled: Optional[bool] = None  # Enable/disable AI tooltips


class PromoteModelRequest(BaseModel):
    source_path: str
    target_name: Optional[str] = "best_model.pt"
    overwrite: bool = True


class MonteCarloRiskRequest(BaseModel):
    """Request for Monte Carlo risk assessment"""
    current_price: float
    proposed_position: float  # Position size (-1.0 to 1.0)
    current_position: float = 0.0  # Current position size
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    time_horizon: int = 1  # Number of periods to simulate
    n_simulations: int = 1000
    simulate_overnight: bool = True


class VolatilityPredictionRequest(BaseModel):
    """Request for volatility prediction"""
    method: str = "adaptive"  # "adaptive", "ewma", "historical_mean"
    lookback_periods: int = 252
    prediction_horizon: int = 1


class ScenarioSimulationRequest(BaseModel):
    """Request for scenario simulation"""
    scenarios: List[str] = ["normal", "trending_up", "trending_down", "ranging", "high_volatility", "low_volatility"]
    intensity: float = 1.0  # Intensity multiplier for scenarios
    use_rl_agent: bool = False  # Whether to use RL agent for backtesting
    model_path: Optional[str] = None  # Optional model path for backtesting


class StressTestRequest(BaseModel):
    """Request for stress testing"""
    scenarios: List[str] = ["crash", "flash_crash", "high_volatility", "gap_event"]
    intensity: float = 2.0  # Higher intensity for stress tests
    model_path: Optional[str] = None


class ParameterSensitivityRequest(BaseModel):
    """Request for parameter sensitivity analysis"""
    parameter_name: str
    parameter_values: List[float]
    base_parameters: Dict[str, Any] = {}
    regime: str = "normal"
    model_path: Optional[str] = None


class MarkovAnalysisRequest(BaseModel):
    """Request payload for offline Markov regime analysis."""

    instrument: Optional[str] = None
    timeframes: Optional[List[int]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    num_regimes: int = 3
    rolling_vol_window: int = 50
    volume_zscore_window: int = 100
    min_samples: int = 500
    save_report: bool = True
    output_path: Optional[str] = None
    config_path: Optional[str] = "configs/train_config_full.yaml"


class CapabilityGenerationRequest(BaseModel):
    """Request payload for capability analysis generation."""

    capability_id: str
    locale: Optional[str] = None
    user_id: Optional[str] = None
    context: Dict[str, Any] = {}
    force_refresh: bool = False
    provider_hint: Optional[str] = None
    model: Optional[str] = None


class CapabilityFeedbackRequest(BaseModel):
    """Request payload for user feedback on generated analyses."""

    capability_id: str
    locale: Optional[str] = None
    user_id: Optional[str] = None
    rating: Optional[int] = None
    comment: Optional[str] = None
    source: Optional[str] = "frontend"


# Helper function to send WebSocket messages
async def broadcast_message(message: Dict):
    """Broadcast message to all connected WebSocket clients"""
    disconnected = []
    for ws in websocket_connections:
        try:
            await ws.send_json(message)
        except:
            disconnected.append(ws)
    
    # Safely remove disconnected websockets
    for ws in disconnected:
        try:
            if ws in websocket_connections:
                websocket_connections.remove(ws)
        except (ValueError, RuntimeError):
            # Already removed or list modified - ignore
            pass


# Routes
@app.get("/")
async def root():
    return {"message": "NT8 RL Trading System API", "version": "1.0.0"}


@app.get("/api/monitoring/metrics")
async def get_kong_metrics():
    """Get Kong Gateway metrics summary"""
    try:
        import requests
        
        # Fetch Prometheus metrics from Kong
        metrics_url = "http://localhost:8301/metrics"
        response = requests.get(metrics_url, timeout=5)
        
        if response.status_code != 200:
            return {
                "status": "error",
                "error": f"Failed to fetch metrics: HTTP {response.status_code}",
                "metrics_url": metrics_url
            }
        
        metrics_text = response.text
        
        # Parse key metrics
        metrics = {
            "raw_metrics_url": metrics_url,
            "summary": {}
        }
        
        # Extract request counts
        for line in metrics_text.split('\n'):
            if line.startswith('kong_http_requests_total') and '{' in line and not line.startswith('#'):
                # Parse: kong_http_requests_total{service="anthropic-service"} 123
                try:
                    if 'service=' in line:
                        service = line.split('service="')[1].split('"')[0]
                        count = line.split()[-1]
                        if service not in metrics["summary"]:
                            metrics["summary"][service] = {}
                        metrics["summary"][service]["total_requests"] = int(count)
                except:
                    pass
        
        return {
            "status": "ok",
            "metrics": metrics,
            "note": "Full metrics available at http://localhost:8301/metrics"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "note": "Kong Gateway may not be running or accessible"
        }


@app.get("/api/monitoring/health")
async def get_kong_health():
    """Get Kong Gateway health status"""
    try:
        import requests
        
        # Check Kong admin API
        response = requests.get("http://localhost:8301/", timeout=2)
        kong_healthy = response.status_code == 200
        
        # Check metrics endpoint
        metrics_response = requests.get("http://localhost:8301/metrics", timeout=2)
        metrics_available = metrics_response.status_code == 200
        
        return {
            "kong_status": "healthy" if kong_healthy else "unhealthy",
            "metrics_available": metrics_available,
            "kong_admin_url": "http://localhost:8301",
            "metrics_url": "http://localhost:8301/metrics"
        }
    except Exception as e:
        return {
            "kong_status": "unreachable",
            "error": str(e),
            "note": "Kong Gateway may not be running"
        }


@app.get("/api/monitoring/services")
async def get_kong_services():
    """Get Kong Gateway services status"""
    try:
        import requests
        
        # Get all services
        response = requests.get("http://localhost:8301/services", timeout=5)
        
        if response.status_code != 200:
            return {
                "status": "error",
                "error": f"Failed to fetch services: HTTP {response.status_code}"
            }
        
        services_data = response.json()
        services = []
        
        for service in services_data.get("data", []):
            service_info = {
                "name": service.get("name"),
                "url": service.get("url"),
                "enabled": service.get("enabled", True),
                "port": service.get("port"),
                "host": service.get("host")
            }
            services.append(service_info)
        
        return {
            "status": "ok",
            "services": services,
            "count": len(services)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "note": "Kong Gateway may not be running"
        }


def detect_venv_type_backend():
    """Detect the type of virtual environment being used (backend version)"""
    in_venv = sys.prefix != sys.base_prefix
    venv_type = None
    venv_message = None
    
    if in_venv:
        # Check for uv virtual environment
        venv_path = Path(sys.prefix)
        pyvenv_cfg = venv_path / "pyvenv.cfg"
        
        if pyvenv_cfg.exists():
            try:
                with open(pyvenv_cfg, 'r') as f:
                    content = f.read()
                    if 'uv' in content.lower() or 'AppData\\Roaming\\uv' in content or '/uv/python' in content:
                        venv_type = "uv"
                        venv_message = "Using uv virtual environment"
                    else:
                        venv_type = "venv"
                        venv_message = "Using standard virtual environment"
            except:
                venv_type = "venv"
                venv_message = "Using virtual environment"
        else:
            # Could be conda or other
            if 'conda' in sys.prefix.lower():
                venv_type = "conda"
                venv_message = "Using conda environment"
            else:
                venv_type = "unknown"
                venv_message = "Using virtual environment"
    
    # Check if uv is available
    uv_available = False
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, timeout=2)
        uv_available = result.returncode == 0
    except:
        pass
    
    venv_dir_exists = Path(".venv").exists() or Path("venv").exists()
    venv_exists = in_venv or venv_dir_exists
    
    # Set default message if venv doesn't exist
    if not venv_exists:
        if uv_available:
            venv_message = "uv venv"
        else:
            venv_message = "python -m venv .venv"
    
    return {
        "venv_exists": venv_exists,
        "venv_type": venv_type,
        "venv_message": venv_message,
        "uv_available": uv_available
    }

@app.get("/api/setup/check", response_model=SetupCheckResponse)
async def check_setup():
    """Check if environment is properly set up"""
    issues = []
    
    # Detect virtual environment
    venv_info = detect_venv_type_backend()
    venv_exists = venv_info["venv_exists"]
    venv_type = venv_info["venv_type"]
    venv_message = venv_info["venv_message"]
    
    # Check dependencies - check all critical packages
    dependencies_installed = True
    missing_deps = []
    
    critical_packages = {
        "torch": "PyTorch",
        "gymnasium": "Gymnasium",
        "stable_baselines3": "Stable-Baselines3",
        "fastapi": "FastAPI",
        "uvicorn": "Uvicorn",
        "anthropic": "Anthropic",
        "ollama": "Ollama"
    }
    
    for module, name in critical_packages.items():
        try:
            __import__(module)
        except ImportError as e:
            missing_deps.append(name)
            dependencies_installed = False
            # Log for debugging
            print(f"âš  Missing dependency: {name} ({module}) - {str(e)}")
    
    if not dependencies_installed:
        if venv_info["uv_available"] or venv_type == "uv":
            issues.append(f"Missing dependencies: {', '.join(missing_deps)}. Run: uv pip install -r requirements.txt or uv sync")
        else:
            issues.append(f"Missing dependencies: {', '.join(missing_deps)}. Run: pip install -r requirements.txt")
    
    # Only warn about venv if dependencies aren't installed
    # (if dependencies work, environment is fine regardless of venv detection)
    if not venv_exists and not dependencies_installed:
        if venv_info["uv_available"]:
            issues.append(f"Virtual environment not detected. Recommended: {venv_message}")
        else:
            issues.append(f"Virtual environment not detected. Recommended: {venv_message}")
    
    # Check data directory
    data_dir = Path("data/raw")
    data_directory_exists = data_dir.exists()
    if not data_directory_exists:
        # Don't make this a blocking issue - data can be uploaded later
        # issues.append(f"Data directory not found: {data_dir}")
        pass
    
    # Check config
    config_path = Path("configs/train_config_full.yaml")
    config_exists = config_path.exists()
    if not config_exists:
        issues.append(f"Config file not found: {config_path}. Create it from configs/train_config.yaml.example if needed.")
    
    ready = len(issues) == 0
    
    return SetupCheckResponse(
        venv_exists=venv_exists,
        venv_type=venv_type,
        dependencies_installed=dependencies_installed,
        data_directory_exists=data_directory_exists,
        config_exists=config_exists,
        ready=ready,
        issues=issues,
        venv_message=venv_message
    )


@app.post("/api/setup/install-dependencies")
async def install_dependencies(background_tasks: BackgroundTasks):
    """Install Python dependencies"""
    async def _install():
        await broadcast_message({
            "type": "setup",
            "message": "Installing dependencies...",
            "progress": 0
        })
        
        venv_info = detect_venv_type_backend()
        use_uv = venv_info["uv_available"] or venv_info["venv_type"] == "uv"
        
        # Find Python executable
        venv_python = Path(".venv/Scripts/python.exe") if os.name == "nt" else Path(".venv/bin/python")
        if not venv_python.exists():
            venv_python = Path("venv/Scripts/python.exe") if os.name == "nt" else Path("venv/bin/python")
        if not venv_python.exists():
            venv_python = sys.executable
        
        try:
            if use_uv:
                # Use uv pip install
                await broadcast_message({
                    "type": "setup",
                    "message": "Using uv to install dependencies...",
                    "progress": 10
                })
                proc = subprocess.Popen(
                    ["uv", "pip", "install", "-r", "requirements.txt"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=str(project_root)
                )
            else:
                # Use standard pip
                await broadcast_message({
                    "type": "setup",
                    "message": "Using pip to install dependencies...",
                    "progress": 10
                })
                proc = subprocess.Popen(
                    [str(venv_python), "-m", "pip", "install", "-r", "requirements.txt"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=str(project_root),
                    bufsize=1,
                    universal_newlines=True
                )
            
            # Read output asynchronously
            import asyncio
            output_lines = []
            
            async def read_output():
                while True:
                    line = proc.stdout.readline()
                    if not line and proc.poll() is not None:
                        break
                    if line:
                        line = line.strip()
                        if line:
                            output_lines.append(line)
                            await broadcast_message({
                                "type": "setup",
                                "message": line,
                                "progress": None
                            })
                    await asyncio.sleep(0.1)  # Small delay to prevent tight loop
            
            # Start reading output
            read_task = asyncio.create_task(read_output())
            
            # Wait for process to complete
            proc.wait()
            await read_task
            
            if proc.returncode == 0:
                await broadcast_message({
                    "type": "setup",
                    "status": "success",
                    "message": "Dependencies installed successfully",
                    "progress": 100
                })
            else:
                error_output = '\n'.join(output_lines[-10:]) if output_lines else "No output captured"
                await broadcast_message({
                    "type": "setup",
                    "status": "error",
                    "message": f"Installation failed (return code: {proc.returncode}): {error_output}",
                    "progress": 0
                })
        except Exception as e:
            await broadcast_message({
                "type": "setup",
                "status": "error",
                "message": f"Error: {str(e)}",
                "progress": 0
            })
    
    background_tasks.add_task(_install)
    return {"status": "started", "message": "Dependency installation started"}


@app.post("/api/data/upload")
async def upload_data(files: List[UploadFile] = File(...)):
    """Upload historical data files"""
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    uploaded_files = []
    
    for file in files:
        file_path = data_dir / file.filename
        content = await file.read()
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        uploaded_files.append(file.filename)
        
        await broadcast_message({
            "type": "data",
            "message": f"Uploaded {file.filename}",
            "files": uploaded_files
        })
    
    return {
        "status": "success",
        "message": f"Uploaded {len(uploaded_files)} file(s)",
        "files": uploaded_files
    }


@app.get("/api/data/list")
async def list_data():
    """List available data files"""
    data_dir = Path("data/raw")
    if not data_dir.exists():
        return {"files": []}
    
    files = [f.name for f in data_dir.iterdir() if f.is_file()]
    return {"files": files}


@app.get("/api/data/statistics")
async def get_data_statistics():
    """
    Get statistics about available training data.
    Returns number of bars available per timeframe to help calculate total_timesteps.
    """
    try:
        import pickle
        
        processed_dir = Path("data/processed")
        instrument = "ES"  # Could be made configurable
        timeframes = [1, 5, 15]
        
        statistics = {
            "instrument": instrument,
            "timeframes": {},
            "total_bars": 0,
            "suggested_timesteps": None,
            "cache_available": False
        }
        
        # Check for cached files first
        total_bars = 0
        cache_found = False
        
        for timeframe in timeframes:
            cache_file_parquet = processed_dir / f"{instrument}_{timeframe}min_processed.parquet"
            cache_file_pkl = processed_dir / f"{instrument}_{timeframe}min_processed.pkl"
            
            cache_file = None
            cache_format = None
            
            if cache_file_parquet.exists():
                cache_file = cache_file_parquet
                cache_format = "parquet"
                statistics["cache_available"] = True
                cache_found = True
            elif cache_file_pkl.exists():
                cache_file = cache_file_pkl
                cache_format = "pickle"
                statistics["cache_available"] = True
                cache_found = True
            
            if cache_file:
                try:
                    if cache_format == "parquet":
                        df = pd.read_parquet(cache_file)
                    else:
                        with open(cache_file, 'rb') as f:
                            df = pickle.load(f)
                    
                    num_bars = len(df)
                    # Count 1min bars only for total calculation
                    if timeframe == min(timeframes):
                        total_bars = num_bars
                    
                    # Get date range
                    date_range = {}
                    if 'timestamp' in df.columns:
                        date_range = {
                            "start": str(df['timestamp'].min()),
                            "end": str(df['timestamp'].max())
                        }
                    
                    statistics["timeframes"][f"{timeframe}min"] = {
                        "bars": num_bars,
                        "cache_file": str(cache_file.name),
                        "date_range": date_range,
                        "format": cache_format
                    }
                    
                except Exception as e:
                    statistics["timeframes"][f"{timeframe}min"] = {
                        "error": str(e),
                        "bars": 0
                    }
            else:
                statistics["timeframes"][f"{timeframe}min"] = {
                    "bars": 0,
                    "cache_file": None,
                    "cache_available": False
                }
        
        # If no cached files, estimate from raw files (like old architecture)
        if not cache_found and total_bars == 0:
            raw_dir = Path("data/raw")
            if raw_dir.exists():
                # Count .Last.txt files (these contain 1-minute bars)
                last_files = list(raw_dir.glob(f"{instrument}*.Last.txt"))
                statistics["raw_files_count"] = len(last_files)
                
                if last_files:
                    # Estimate bars: Each .Last.txt file typically contains ~390 bars per trading day
                    # For 61 files, estimate based on file count
                    # Conservative estimate: ~200,000 bars per file (varies by file size)
                    estimated_bars_per_file = 200_000  # Conservative estimate
                    total_bars = len(last_files) * estimated_bars_per_file
                    
                    # More accurate: Try to count lines in a sample file
                    try:
                        sample_file = last_files[0]
                        with open(sample_file, 'r', encoding='utf-8', errors='ignore') as f:
                            # Count non-empty, non-comment lines
                            line_count = sum(1 for line in f if line.strip() and not line.strip().startswith('#'))
                            if line_count > 0:
                                # Use average from sample, but cap at reasonable max
                                avg_bars_per_file = min(line_count, 500_000)  # Cap at 500k per file
                                total_bars = len(last_files) * avg_bars_per_file
                                statistics["calculation_method"] = "estimated_from_raw_files"
                                statistics["sample_file"] = sample_file.name
                                statistics["sample_file_bars"] = line_count
                    except Exception as e:
                        # Fallback to conservative estimate
                        statistics["calculation_method"] = "estimated_from_file_count"
                        statistics["estimation_note"] = f"Could not read sample file: {e}"
        
        # Calculate suggested total_timesteps based on available data
        # Formula: For large datasets, suggest 10-20M timesteps to allow multiple passes through data
        if total_bars > 0:
            # For large datasets (5M+ bars), suggest moderate to extensive training
            if total_bars > 5_000_000:
                suggested_timesteps = 20_000_000  # 20M for extensive dataset
            elif total_bars > 1_000_000:
                suggested_timesteps = 10_000_000  # 10M for medium dataset
            elif total_bars > 100_000:
                suggested_timesteps = 5_000_000   # 5M for smaller dataset
            else:
                suggested_timesteps = 1_000_000   # 1M for small dataset
            
            statistics["total_bars"] = total_bars
            statistics["suggested_timesteps"] = suggested_timesteps
            statistics["calculation"] = {
                "description": f"Based on {total_bars:,} bars available",
                "reasoning": "Suggests 10-20M timesteps for large datasets to allow multiple passes through data"
            }
        
        return {
            "status": "success",
            "statistics": statistics
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "statistics": {}
        }


@app.get("/api/config/list")
async def list_configs():
    """List available training config files"""
    try:
        config_dir = Path("configs")
        if not config_dir.exists():
            return {"configs": []}
        
        # Get current working directory
        cwd = Path.cwd()
        
        # Find all YAML files in configs directory
        config_files = []
        for config_file in config_dir.glob("*.yaml"):
            try:
                relative_path = str(config_file.relative_to(cwd))
            except ValueError:
                # If relative path fails, just use the path as-is
                relative_path = str(config_file)
            
            config_files.append({
                "path": str(config_file.absolute()),
                "name": config_file.name,
                "relative": relative_path
            })
        
        # Also check for .yml files
        for config_file in config_dir.glob("*.yml"):
            try:
                relative_path = str(config_file.relative_to(cwd))
            except ValueError:
                relative_path = str(config_file)
            
            config_files.append({
                "path": str(config_file.absolute()),
                "name": config_file.name,
                "relative": relative_path
            })
        
        # Sort by name
        config_files.sort(key=lambda x: x["name"])
        
        return {"configs": config_files}
    except Exception as e:
        print(error(f"Error listing configs: {e}"))
        import traceback
        traceback.print_exc()
        return {"configs": [], "error": str(e)}

@app.get("/api/config/read")
async def read_config(path: str):
    """Read a config file and return its contents with calculated state_dim"""
    try:
        from pathlib import Path
        import yaml
        
        # Resolve config path
        config_file = Path(str(path).replace('\\', '/'))
        if not config_file.exists():
            # Try relative to project root
            project_root = Path(__file__).parent.parent
            config_file = project_root / str(path).replace('\\', '/').lstrip('/')
        
        if not config_file.exists():
            return {
                "error": "Config file not found",
                "path": path
            }
        
        # Read and parse YAML
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Calculate actual state_dim from config (matching trading_env.py logic)
        calculated_state_dim = None
        if config and "environment" in config:
            env_config = config["environment"]
            reward_config = env_config.get("reward", {}) if isinstance(env_config.get("reward"), dict) else {}
            
            # Get base state_dim from state_features or calculate from timeframes/lookback
            base_state_dim = env_config.get("state_features")
            if base_state_dim is None:
                # Calculate from timeframes and lookback_bars (15 features per timeframe)
                timeframes = env_config.get("timeframes", [1, 5, 15])
                lookback_bars = env_config.get("lookback_bars", 20)
                features_per_tf = 15  # OHLCV (5) + volume_ratio (1) + returns (1) + indicators (8)
                if isinstance(timeframes, list) and lookback_bars:
                    base_state_dim = features_per_tf * len(timeframes) * lookback_bars
                else:
                    base_state_dim = 900  # Default fallback
                
                # If state_features was not explicitly set, add additional features based on reward config
                regime_features_dim = 5 if reward_config.get("include_regime_features", False) else 0
                forecast_features_dim = 3 if reward_config.get("include_forecast_features", False) else 0
                strategy_validator_config = reward_config.get("strategy_validator", {})
                strategy_enabled = strategy_validator_config.get("enabled", False) if isinstance(strategy_validator_config, dict) else False
                strategy_features_dim = 8 if strategy_enabled else 0
                
                calculated_state_dim = base_state_dim + regime_features_dim + forecast_features_dim + strategy_features_dim
            else:
                # state_features is explicitly set - use it as-is (it already includes all features)
                calculated_state_dim = base_state_dim
        
        return {
            "path": str(config_file),
            "exists": True,
            "config": config,
            "calculated_state_dim": calculated_state_dim
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "path": path
        }

@app.get("/api/system/cuda-status")
async def get_cuda_status():
    """Check CUDA/GPU availability"""
    try:
        import torch
        import sys
        
        # Debug logging
        print("="*60)
        print("CUDA Status Check")
        print("="*60)
        print(f"Python executable: {sys.executable}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch location: {torch.__file__}")
        
        # Check if PyTorch has CUDA support compiled in
        has_cuda_build = hasattr(torch.version, 'cuda') and torch.version.cuda is not None
        print(f"PyTorch CUDA build: {has_cuda_build}")
        if has_cuda_build:
            print(f"PyTorch CUDA version: {torch.version.cuda}")
        
        # Check if CUDA runtime is available
        cuda_available = torch.cuda.is_available()
        print(f"CUDA runtime available: {cuda_available}")
        
        result = {
            "cuda_available": bool(cuda_available),  # Ensure boolean
            "device": "cpu",
            "pytorch_version": torch.__version__,
            "has_cuda_build": has_cuda_build
        }
        
        if cuda_available:
            try:
                result["device"] = "cuda"
                result["gpu_name"] = torch.cuda.get_device_name(0)
                result["cuda_version"] = torch.version.cuda if hasattr(torch.version, 'cuda') else "Unknown"
                result["device_count"] = torch.cuda.device_count()
                result["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
                print(f"[OK] GPU detected: {result['gpu_name']} (CUDA {result['cuda_version']})")
                print(f"   Device count: {result['device_count']}")
                print(f"   GPU Memory: {result['gpu_memory_gb']} GB")
            except Exception as gpu_error:
                print(warn(f"[WARN] Error getting GPU info: {gpu_error}"))
                import traceback
                traceback.print_exc()
                result["gpu_name"] = "Unknown"
                result["cuda_version"] = None
                result["device_count"] = 0
                result["error"] = str(gpu_error)
        else:
            result["gpu_name"] = None
            result["cuda_version"] = None
            result["device_count"] = 0
            if not has_cuda_build:
                result["error"] = "PyTorch was not compiled with CUDA support. Install CUDA-enabled PyTorch."
                print(error("[ERROR] CUDA not available - PyTorch is CPU-only build"))
            else:
                result["error"] = "CUDA runtime not available. Check NVIDIA drivers."
                print(error("[ERROR] CUDA not available - CUDA runtime not detected"))
        
        print(f"Returning result: {result}")
        print("="*60)
        return result
    except ImportError as e:
        print(error(f"PyTorch import error: {e}"))
        return {
            "cuda_available": False,
            "device": "cpu",
            "gpu_name": None,
            "cuda_version": None,
            "device_count": 0,
            "error": "PyTorch not installed"
        }
    except Exception as e:
        print(error(f"CUDA check error: {e}"))
        import traceback
        traceback.print_exc()
        return {
            "cuda_available": False,
            "device": "cpu",
            "error": str(e)
        }


def _on_auto_retrain_triggered(files):
    """
    Callback when new data files detected.
    
    This automatically triggers retraining when new data is available.
    """
    global active_processes, active_systems, main_event_loop
    
    print(f"\nðŸ“ New data detected: {len(files)} file(s)")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for f in files:
        file_path = Path(f)
        file_size = file_path.stat().st_size if file_path.exists() else 0
        print(f"  - {file_path.name} ({file_size:,} bytes)")
    
    # Check if training is already running
    if "training" in active_systems:
        system = active_systems["training"]
        thread_alive = system.get("thread") and system["thread"].is_alive()
        if thread_alive:
            print(warn("[WARN]  Training already in progress. New data detected but will not retrain yet."))
            print("   Retraining will be queued after current training completes.")
            # TODO: Implement queue system
            # For now, just notify
            if main_event_loop:
                try:
                    asyncio.run_coroutine_threadsafe(
                        broadcast_message({
                            "type": "auto_retrain",
                            "status": "queued",
                            "message": f"New data detected but training in progress. Will retrain after current training completes.",
                            "files": [str(f) for f in files]
                        }),
                        main_event_loop
                    )
                except:
                    pass
            return
    
    print("ðŸš€ Auto-triggering retraining with new data...")
    
    # Find latest checkpoint for resume
    checkpoint_path = None
    models_dir = Path("models")
    if models_dir.exists():
        checkpoints = sorted(
            [f for f in models_dir.glob("checkpoint_*.pt")],
            key=lambda x: int(x.stem.split('_')[1]) if x.stem.split('_')[1].isdigit() else 0,
            reverse=True
        )
        if checkpoints:
            checkpoint_path = str(checkpoints[0])
            print(f"   Resuming from: {checkpoint_path}")
    
    # Load default config
    config_path = "configs/train_config_full.yaml"
    if not Path(config_path).exists():
        config_path = "configs/train_config_gpu.yaml"
    
    # Create training request
    training_request = TrainingRequest(
        device="cuda",  # Try GPU first, will fallback to CPU if needed
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        total_timesteps=None  # Use config default
    )
    
    # Trigger training asynchronously via the training endpoint logic
    if main_event_loop:
        try:
            async def _start_auto_training():
                from fastapi import BackgroundTasks
                background_tasks = BackgroundTasks()
                
                await broadcast_message({
                    "type": "auto_retrain",
                    "status": "triggering",
                    "message": f"Starting automatic retraining with {len(files)} new file(s)..."
                })
                
                # Call start_training directly (it's an async function)
                await start_training(training_request, background_tasks)
            
            # Schedule training start in the event loop
            future = asyncio.run_coroutine_threadsafe(
                _start_auto_training(),
                main_event_loop
            )
            
            print("[OK] Auto-retraining triggered successfully")
            print("   Training should start shortly...")
        except Exception as e:
            print(error(f"[ERROR] Error triggering auto-retraining: {e}"))
            import traceback
            traceback.print_exc()
            
            # Fallback: just notify
            try:
                asyncio.run_coroutine_threadsafe(
                    broadcast_message({
                        "type": "auto_retrain",
                        "status": "error",
                        "message": f"Failed to trigger auto-retraining: {str(e)}",
                        "files": [str(f) for f in files]
                    }),
                    main_event_loop
                )
            except:
                pass
    else:
        print(warn("[WARN]  Main event loop not available, cannot trigger training automatically"))
        print("   Please start training manually from the UI or use the API endpoint")


@app.on_event("startup")
async def startup_event():
    """Store main event loop for use in threads"""
    global main_event_loop, auto_retrain_monitor
    
    main_event_loop = asyncio.get_event_loop()
    
    # Initialize trading journal database to ensure tables exist
    try:
        from src.trading_journal import TradingJournal
        journal = TradingJournal()
        print("[OK] Trading journal database initialized")
    except Exception as e:
        print(warn(f"[WARN] Could not initialize trading journal database: {e}"))
    
    # Initialize auto-retrain monitor if settings are configured
    settings_file = project_root / "settings.json"
    if settings_file.exists():
        try:
            with open(settings_file, 'r') as f:
                settings = json.load(f)
                
                # Load NT8 data path and auto-retrain setting
                nt8_path = settings.get("nt8_data_path")
                auto_retrain_enabled = settings.get("auto_retrain_enabled", True)
                
                if nt8_path and auto_retrain_enabled:
                    auto_retrain_monitor = AutoRetrainMonitor(
                        nt8_export_path=nt8_path,
                        auto_retrain_callback=_on_auto_retrain_triggered,
                        enabled=auto_retrain_enabled
                    )
                    auto_retrain_monitor.start()
                    print(f"[OK] Auto-retrain monitoring started on: {nt8_path}")
        except Exception as e:
            print(warn(f"[WARN] Could not initialize auto-retrain monitor: {e}"))
    
    print("[OK] API server initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop auto-retrain monitor on shutdown"""
    global auto_retrain_monitor
    if auto_retrain_monitor:
        auto_retrain_monitor.stop()


@app.post("/api/training/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start model training"""
    # CRITICAL: Log immediately when endpoint is called (before any processing)
    import sys
    sys.stdout.flush()  # Force immediate output
    print(f"\n{'='*80}")
    print(f"[API] /api/training/start ENDPOINT CALLED")
    print(f"[API] Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[API] Checkpoint path from request: {request.checkpoint_path}")
    print(f"{'='*80}\n")
    sys.stdout.flush()  # Force immediate output
    
    if "training" in active_systems:
        system = active_systems["training"]
        thread_alive = system.get("thread") and system["thread"].is_alive()
        print(warn(f"[WARN]  Training start requested but training already in active_systems"))
        print(f"   Thread alive: {thread_alive}")
        print(f"   Completed flag: {system.get('completed', False)}")
        if not thread_alive and system.get("completed", False):
            # Training completed or stopped, clean it up
            print(f"   Cleaning up stale training entry")
            active_systems.pop("training", None)
        else:
            raise HTTPException(status_code=400, detail="Training already in progress")
    
    # Log the incoming request for debugging
    print(f"\n{'='*60}")
    print(f"📥 Training Start Request Received")
    print(f"   Device: {request.device}")
    print(f"   Config: {request.config_path}")
    print(f"   Total timesteps: {request.total_timesteps}")
    print(f"   Checkpoint path: {request.checkpoint_path if request.checkpoint_path else 'None (fresh start)'}")
    print(f"   Flush old data: {request.flush_old_data}")
    print(f"{'='*60}\n")
    
    # Clear old training data if requested (fresh start only)
    if request.flush_old_data and not request.checkpoint_path:
        print(f"[INFO] Flushing old training data (fresh start requested)...")
        try:
            from src.clear_training_data import clear_all_training_data
            clear_result = clear_all_training_data(
                archive_db=True,  # Archive old database before clearing
                clear_caches_flag=True,  # Clear cache files
                clear_processed=False,  # Keep processed data (can be regenerated)
                archive_checkpoints_flag=True  # Archive and remove all checkpoints
            )
            if clear_result.get("success"):
                print(f"[OK] Old training data flushed successfully")
                if clear_result.get("checkpoints"):
                    checkpoint_info = clear_result['checkpoints']
                    print(f"   Checkpoints: {checkpoint_info.get('message', 'N/A')}")
                    if checkpoint_info.get("backup_path"):
                        print(f"   Checkpoint Archive: {checkpoint_info['backup_path']}")
                if clear_result.get("journal"):
                    print(f"   Journal: {clear_result['journal'].get('message', 'N/A')}")
                    if clear_result['journal'].get("backup_path"):
                        print(f"   Journal Backup: {clear_result['journal']['backup_path']}")
                if clear_result.get("caches"):
                    print(f"   Caches: {clear_result['caches'].get('message', 'N/A')}")
            else:
                print(warn(f"[WARN] Some data clearing operations failed: {clear_result.get('message', 'Unknown error')}"))
        except Exception as e:
            print(error(f"[ERROR] Failed to flush old training data: {e}"))
            import traceback
            traceback.print_exc()
            # Continue with training even if flush failed
    
    async def _train():
        # Force immediate output to verify function is called
        print(f"\n{'='*80}")
        print(f"[_train] [OK][OK][OK] ASYNC TRAINING FUNCTION CALLED [OK][OK][OK]")
        print(f"[_train] Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[_train] Process ID: {os.getpid()}")
        print(f"[_train] Thread ID: {threading.current_thread().ident}")
        print(f"{'='*80}\n")
        
        # Force flush output immediately
        import sys
        sys.stdout.flush()
        
        try:
            print(f"[_train] Attempting to send broadcast message...")
            await broadcast_message({
                "type": "training",
                "status": "starting",
                "message": "Initializing training..."
            })
            print(f"[_train] [OK] Broadcast message sent successfully")
            sys.stdout.flush()
        except Exception as e:
            print(error(f"[_train] [ERROR] ERROR sending broadcast: {e}"))
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
        
        # Load config
        print(f"[_train] ðŸ“„ Loading config from: {request.config_path}")
        try:
            with open(request.config_path, "r") as f:
                config = yaml.safe_load(f)
            print(f"[_train] [OK] Config loaded successfully")
            print(f"[_train]   Instrument: {config.get('environment', {}).get('instrument', 'N/A')}")
            print(f"[_train]   Timeframes: {config.get('environment', {}).get('timeframes', 'N/A')}")
        except Exception as e:
            print(error(f"[_train] [ERROR] ERROR loading config: {e}"))
            import traceback
            traceback.print_exc()
            raise
        
        if request.total_timesteps:
            config["training"]["total_timesteps"] = request.total_timesteps
        
        # Validate device selection before creating trainer
        requested_device = request.device
        if requested_device == "cuda":
            import torch
            if not torch.cuda.is_available():
                # Provide helpful diagnostic message
                cuda_available = torch.cuda.is_available()
                cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else None
                
                error_msg = (
                    f"CUDA not available. PyTorch was not compiled with CUDA support.\n"
                    f"To use GPU, you need to install PyTorch with CUDA support:\n"
                    f"  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n"
                    f"Or for CUDA 11.8: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118\n"
                    f"Check your CUDA version with: nvidia-smi\n"
                    f"Using CPU instead."
                )
                
                await broadcast_message({
                    "type": "training",
                    "status": "warning",
                    "message": error_msg
                })
                requested_device = "cpu"
            else:
                # CUDA is available - log GPU info
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
                await broadcast_message({
                    "type": "training",
                    "status": "info",
                    "message": f"Using GPU: {gpu_name} (CUDA {torch.version.cuda})"
                })
        
        config["training"]["device"] = requested_device
        
        # Update reasoning model if provided
        if request.reasoning_model:
            config["reasoning"]["model"] = request.reasoning_model
        
        # Update transfer strategy if provided (for transfer learning)
        if request.transfer_strategy:
            if "training" not in config:
                config["training"] = {}
            config["training"]["transfer_strategy"] = request.transfer_strategy
            print(f"[_train] Transfer learning strategy: {request.transfer_strategy}")
        
        try:
            # Log checkpoint path being used
            checkpoint_path_to_use = None
            if request.checkpoint_path:
                print(f"ðŸ” Attempting to resume from checkpoint: {request.checkpoint_path}")
                from pathlib import Path
                normalized_str = str(request.checkpoint_path).replace('\\', '/')
                checkpoint_test = Path(normalized_str)
                print(f"[CHECKPOINT DEBUG] Original: {repr(request.checkpoint_path)}")
                print(f"[CHECKPOINT DEBUG] Normalized: {normalized_str}")
                print(f"   Normalized path: {checkpoint_test}")
                print(f"   Path exists: {checkpoint_test.exists()}")
                if checkpoint_test.exists():
                    print(f"[CHECKPOINT DEBUG] Absolute: {checkpoint_test.resolve()}")
                    checkpoint_path_to_use = str(checkpoint_test.resolve())
                    print(f"   [OK] Using checkpoint: {checkpoint_path_to_use}")
                else:
                    # Try relative to project root
                    project_root = Path(__file__).parent.parent
                    relative_checkpoint = project_root / str(request.checkpoint_path).replace('\\', '/').lstrip('/')
                    print(f"   Trying relative path: {relative_checkpoint}")
                    if relative_checkpoint.exists():
                        checkpoint_path_to_use = str(relative_checkpoint.resolve())
                        print(f"   [OK] Using checkpoint: {checkpoint_path_to_use}")
                    else:
                        print(warn(f"   [WARN]  WARNING: Checkpoint not found! Will start fresh training."))
                        print(f"      Tried: {checkpoint_test}")
                        print(f"      Tried: {relative_checkpoint}")
                        print(warn(f"   [WARN]  NOTE: Since checkpoint not found, supervised pretraining WILL run if enabled in config"))
                        await broadcast_message({
                            "type": "training",
                            "status": "warning",
                            "message": f"Checkpoint not found: {request.checkpoint_path}. Starting fresh training (pretraining may run)."
                        })
                        # CRITICAL: Set to None explicitly to ensure pretraining logic works correctly
                        checkpoint_path_to_use = None
            
            # Create trainer and train (with optional checkpoint for resume)
            checkpoint_status = checkpoint_path_to_use if checkpoint_path_to_use else 'None (fresh start)'
            print(f"[_train] 🚀 Creating Trainer with checkpoint: {checkpoint_status}")
            if checkpoint_path_to_use:
                print(f"[_train]   ✅ Resuming from checkpoint - supervised pretraining will be SKIPPED")
            else:
                print(f"[_train]   ⚠️  Fresh start - supervised pretraining WILL run if enabled in config")
            print(f"[_train]   This may take a moment (loading data, creating environment...)")
            print(f"[_train]   Step 1/4: Starting Trainer initialization...")
            
            import time
            init_start_time = time.time()
            
            try:
                # Send progress update
                await broadcast_message({
                    "type": "training",
                    "status": "initializing",
                    "message": "Loading data files...",
                    "progress": {"step": "loading_data", "elapsed": 0}
                })
                
                # CRITICAL DEBUG: Log checkpoint path before creating Trainer
                print(f"\n{'='*70}")
                print(f"[API] About to create Trainer with checkpoint_path: {checkpoint_path_to_use}")
                print(f"[API] checkpoint_path_to_use type: {type(checkpoint_path_to_use)}")
                print(f"[API] checkpoint_path_to_use is None: {checkpoint_path_to_use is None}")
                if checkpoint_path_to_use:
                    from pathlib import Path
                    checkpoint_file = Path(checkpoint_path_to_use)
                    print(f"[API] Checkpoint file exists: {checkpoint_file.exists()}")
                    if checkpoint_file.exists():
                        print(f"[API] Checkpoint file absolute path: {checkpoint_file.resolve()}")
                print(f"{'='*70}\n")
                
                trainer = Trainer(config, checkpoint_path=checkpoint_path_to_use, config_path=request.config_path)
                init_elapsed = time.time() - init_start_time
                print(f"[_train] [OK] Trainer created successfully (took {init_elapsed:.1f}s)")
                
                # CRITICAL FIX: Update training_start_episode when trainer is created
                if "training" in active_systems and hasattr(trainer, 'episode'):
                    active_systems["training"]["training_start_episode"] = trainer.episode
                    print(f"[INFO] Training started at episode: {trainer.episode}")
                
                # Send success update
                await broadcast_message({
                    "type": "training",
                    "status": "initializing",
                    "message": f"Trainer initialized successfully ({init_elapsed:.1f}s)",
                    "progress": {"step": "trainer_ready", "elapsed": init_elapsed}
                })
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                init_elapsed = time.time() - init_start_time
                print(f"\n{'='*80}")
                print(error(f"[_train] [ERROR][ERROR][ERROR] ERROR CREATING TRAINER AFTER {init_elapsed:.1f}s [ERROR][ERROR][ERROR]"))
                print(error(f"[_train] Error: {e}"))
                print(f"[_train] Full traceback:")
                print(error(error_trace))
                print(f"{'='*80}\n")
                sys.stdout.flush()
                
                # Update active_systems to reflect error
                if "training" in active_systems:
                    active_systems["training"]["error"] = str(e)
                    active_systems["training"]["completed"] = True
                
                # Send error update
                try:
                    await broadcast_message({
                        "type": "training",
                        "status": "error",
                        "message": f"Failed to initialize trainer: {str(e)}",
                        "error": str(e),
                        "initialization_time": init_elapsed,
                        "traceback": error_trace
                    })
                except:
                    pass
                
                # Re-raise to be caught by outer try/except
                raise
            
            resume_msg = f"Resuming from checkpoint: {request.checkpoint_path}" if request.checkpoint_path else "Training started"
            await broadcast_message({
                "type": "training",
                "status": "running",
                "message": resume_msg
            })
            
            # Train in a separate thread to avoid blocking
            def train_worker():
                global main_event_loop  # Declare global at the start
                try:
                    print(f"ðŸ‹ï¸ Training worker thread started (ID: {threading.current_thread().ident})")
                    print(f"   About to call trainer.train()...")
                    trainer.train()
                    print(f"[OK] Training completed successfully in worker thread")
                    # Use the main event loop to send async messages from thread
                    if main_event_loop and main_event_loop.is_running():
                        future = asyncio.run_coroutine_threadsafe(
                            broadcast_message({
                                "type": "training",
                                "status": "completed",
                                "message": "Training completed successfully"
                            }),
                            main_event_loop
                        )
                        future.result(timeout=5)  # Wait for message to be sent
                    else:
                        # Fallback: create new event loop
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(broadcast_message({
                            "type": "training",
                            "status": "completed",
                            "message": "Training completed successfully"
                        }))
                        loop.close()
                    
                    # Mark training as completed
                    if "training" in active_systems:
                        active_systems["training"]["completed"] = True
                except Exception as e:
                    import traceback
                    error_trace = traceback.format_exc()
                    print(error(f"[ERROR] ERROR in training worker thread: {str(e)}"))
                    print(error(f"   Traceback:\n{error_trace}"))
                    # Use the main event loop to send async messages from thread
                    try:
                        if main_event_loop and main_event_loop.is_running():
                            future = asyncio.run_coroutine_threadsafe(
                                broadcast_message({
                                    "type": "training",
                                    "status": "error",
                                    "message": f"Training failed: {str(e)}"
                                }),
                                main_event_loop
                            )
                            future.result(timeout=5)
                        else:
                            # Fallback: create new event loop
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            loop.run_until_complete(broadcast_message({
                                "type": "training",
                                "status": "error",
                                "message": f"Training failed: {str(e)}"
                            }))
                            loop.close()
                    except Exception as broadcast_error:
                        print(error(f"[ERROR] Failed to broadcast training error: {broadcast_error}"))
                        print(error(f"   Original error: {str(e)}"))
                    
                    # Mark training as completed with error
                    if "training" in active_systems:
                        active_systems["training"]["completed"] = True
                        active_systems["training"]["error"] = str(e)
            
            thread = threading.Thread(target=train_worker)
            thread.daemon = True
            thread.start()
            
            # Store checkpoint resume timestamp - ALWAYS set when training starts
            # This filters out old trades from previous training sessions
            # Set to current time so only NEW trades from this training session are shown
            checkpoint_resume_timestamp = datetime.now().isoformat()
            if checkpoint_path_to_use:
                print(f"[INFO] Resuming from checkpoint: {checkpoint_path_to_use}")
            print(f"[INFO] Training start timestamp: {checkpoint_resume_timestamp} (filters trades before this time)")
            
            # Update the placeholder entry with actual trainer and thread
            # CRITICAL FIX: Store training_start_episode for filtering adjustments
            training_start_episode = trainer.episode if hasattr(trainer, 'episode') else 0
            active_systems["training"] = {
                "trainer": trainer,
                "thread": thread,
                "completed": False,
                "training_start_episode": training_start_episode,
                "status": "running",  # Update status to running
                "checkpoint_resume_timestamp": checkpoint_resume_timestamp,
                "total_timesteps": config.get("training", {}).get("total_timesteps", 20000000)  # Cache total_timesteps
            }
            print(f"[OK] Training thread started, trainer created successfully")
            print(f"   Thread ID: {thread.ident}")
            print(f"   Thread alive: {thread.is_alive()}")
            # Verify trainer is actually stored
            if "training" in active_systems and active_systems["training"].get("trainer") is not None:
                pass  # Trainer stored successfully
            else:
                print(error(f"[ERROR] WARNING: Trainer was NOT stored in active_systems!"))
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"\n{'='*80}")
            print(error(f"[ERROR][ERROR][ERROR] FAILED TO START TRAINING [ERROR][ERROR][ERROR]"))
            print(error(f"Error: {str(e)}"))
            print(f"Full traceback:")
            print(error(error_trace))
            print(f"{'='*80}\n")
            sys.stdout.flush()
            
            # Update active_systems to reflect error
            if "training" in active_systems:
                active_systems["training"]["error"] = str(e)
                active_systems["training"]["completed"] = True
            
            try:
                await broadcast_message({
                    "type": "training",
                    "status": "error",
                    "message": f"Failed to start training: {str(e)}",
                    "error": str(e),
                    "traceback": error_trace
                })
            except:
                pass
    
    # Add placeholder entry immediately so status endpoint knows training is starting
    # This prevents race condition where status is checked before _train() completes
    import time
    
    # ALWAYS set training start timestamp when training begins
    # This ensures the monitoring panel filters out old trades from previous sessions
    # Whether resuming from checkpoint or starting fresh, we want to see only NEW trades
    training_start_timestamp = datetime.now().isoformat()
    if request.checkpoint_path:
        print(f"[INFO] Resuming from checkpoint: {request.checkpoint_path}")
    print(f"[INFO] Training start timestamp: {training_start_timestamp} (will filter trades before this time in monitoring)")
    
    active_systems["training"] = {
        "trainer": None,  # Will be set when trainer is created
        "thread": None,   # Will be set when actual training thread starts
        "completed": False,
        "status": "starting",  # Mark as starting
        "start_time": time.time(),  # Track when training started to detect hangs
        "checkpoint_resume_timestamp": training_start_timestamp,
        "training_start_timestamp": training_start_timestamp,
        "training_start_episode": 0  # Will be updated when trainer is created
    }
    
    print(f"\n{'='*60}")
    print(f"ðŸ“¤ TRAINING START REQUEST RECEIVED")
    print(f"   Device: {request.device}")
    print(f"   Config: {request.config_path}")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Process ID: {os.getpid()}")
    print(f"   Thread ID: {threading.current_thread().ident}")
    print(f"{'='*60}\n")
    
    # Force stdout flush to ensure logs appear
    import sys
    sys.stdout.flush()
    
    # Start the training task immediately using the event loop
    # BackgroundTasks can be unreliable, so we'll schedule it directly
    import asyncio
    try:
        # Get the current event loop
        loop = asyncio.get_event_loop()
        print(f"[OK] Got event loop: {loop}")
        sys.stdout.flush()
        
        # Schedule _train() to run immediately
        # This ensures it actually executes
        task = loop.create_task(_train())
        print(f"[OK] Created and scheduled asyncio task for _train()")
        print(f"   Task object: {task}")
        print(f"   Task done: {task.done()}")
        print(f"   Task cancelled: {task.cancelled()}")
        print(f"   Event loop is running: {loop.is_running()}")
        print(f"   Placeholder entry added to active_systems['training']")
        sys.stdout.flush()
        
        # Give it a moment to start executing
        await asyncio.sleep(0.1)
        print(f"   After 0.1s wait - Task done: {task.done()}, cancelled: {task.cancelled()}")
        if task.done():
            try:
                result = task.result()
                print(f"   Task result: {result}")
            except Exception as task_error:
                print(error(f"   [ERROR] Task error: {task_error}"))
                import traceback
                traceback.print_exc()
        sys.stdout.flush()
        
        # Log that we're returning response (task will continue in background)
        print(f"ðŸ“¤ Returning response - _train() should execute in background")
        sys.stdout.flush()
        
    except Exception as e:
        print(error(f"[ERROR] ERROR creating/scheduling asyncio task: {e}"))
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        # Fallback: try background_tasks
        print(f"   Falling back to BackgroundTasks...")
        background_tasks.add_task(_train)
        sys.stdout.flush()
    
    # Also add to background_tasks as backup
    background_tasks.add_task(_train)
    print(f"[OK] Also added to background_tasks as backup")
    sys.stdout.flush()
    
    return {"status": "started", "message": "Training started"}


@app.get("/api/training/status")
async def training_status():
    """Get training status with detailed metrics - optimized for fast idle returns"""
    # CRITICAL: Try to access active_systems with timeout protection
    # If even dictionary access is blocking, return immediately
    import time
    start_time = time.time()
    MAX_ENDPOINT_TIME = 0.3  # Ultra-aggressive: 300ms max
    
    # ULTRA-FAST PATH: Try to get status with minimal blocking
    # Use try-except to catch any blocking operations
    try:
        # Check if training entry exists (this should be instant)
        # Use 'in' check which is O(1) and should never block
        has_training = "training" in active_systems
        if not has_training:
            return {"status": "idle", "message": "No training in progress"}
        
        # Get system entry (should be instant - just a dict lookup)
        system = active_systems.get("training")
        if system is None:
            return {"status": "idle", "message": "No training in progress"}
        
        # Get status (should be instant - just a dict.get())
        status = system.get("status", "idle")
        
        # CRITICAL: For ANY status except "idle", return immediately
        # This is the fastest possible path - no trainer access, no database queries
        if status != "idle":
            # Return immediately without any checks - use cached values only
            # All these operations are O(1) dict lookups - should never block
            last_timestep = system.get("last_timestep", 0)
            last_episode = system.get("last_episode", 0)
            
            # Return minimal response immediately for active training (no trainer access)
            # Include all cached metrics that might be available
            metrics = {
                "timestep": last_timestep,
                "episode": last_episode
            }
            
            # Include total_timesteps if cached (stored when training starts)
            total_timesteps = system.get("total_timesteps")
            if total_timesteps is not None:
                metrics["total_timesteps"] = total_timesteps
            
            # Include latest reward if cached
            last_reward = system.get("last_reward")
            if last_reward is not None:
                metrics["latest_reward"] = last_reward
            
            # Include mean_reward_10 if cached
            last_mean_reward_10 = system.get("last_mean_reward_10")
            if last_mean_reward_10 is not None:
                metrics["mean_reward_10"] = last_mean_reward_10
            
            # For "starting" status, add pretraining info if available
            if status == "starting":
                # Calculate elapsed time (should be instant)
                start_time_val = system.get("start_time", 0)
                elapsed = time.time() - start_time_val if start_time_val > 0 else 0
                metrics["pretraining"] = True
                metrics["elapsed_seconds"] = int(elapsed)
                message = f"Supervised pretraining in progress... ({elapsed:.0f}s) - This may take 5-10 minutes"
            else:
                message = "Training in progress" if status == "running" else f"Training {status}"
            
            # Return immediately - this should take <1ms
            return {
                "status": status,
                "message": message,
                "metrics": metrics
            }
    except Exception as e:
        # If even dictionary access fails, return a safe default
        # This should never happen, but protects against deadlocks
        import traceback
        print(f"[WARN] Training status endpoint error (returning safe default): {e}")
        print(f"[WARN] Traceback: {traceback.format_exc()}")
        return {
            "status": "starting",
            "message": "Training initializing... (endpoint protection active)",
            "metrics": {"pretraining": True}
        }
    
    # If we reach here, status is "idle" - handle idle case
    # For idle status, we can do minimal checks
    trainer = system.get("trainer")
    thread = system.get("thread")
    
    # Quick check for stale entries (no trainer, no thread, old status)
    if trainer is None and (thread is None or (hasattr(thread, 'is_alive') and not thread.is_alive())):
        import time
        start_time = system.get("start_time", 0)
        elapsed = time.time() - start_time if start_time > 0 else 0
        
        # If it's been more than 5 minutes with no trainer/thread, it's definitely stale
        if elapsed > 300:
            print(f"[INFO] Cleaning up stale training entry (no trainer/thread, {elapsed:.0f}s old)")
            active_systems.pop("training", None)
            return {"status": "idle", "message": "No training in progress"}
        
        # If status is "idle" or "completed" and no trainer/thread, clean up immediately
        if system.get("status") in ["idle", "completed", "error"]:
            active_systems.pop("training", None)
            return {"status": "idle", "message": "No training in progress"}
        
        # If elapsed is less than 5 minutes but no trainer/thread, still return idle quickly
        # Don't wait around - just return idle
        if elapsed > 10:  # Give it 10 seconds, then return idle
            active_systems.pop("training", None)
            return {"status": "idle", "message": "No training in progress"}
    
    # Check if training is still initializing (trainer not created yet)
    # BUT: If trainer exists, we should show metrics even if status is "starting"
    
    # Check if trainer is actively training (timestep is increasing)
    # This is more reliable than checking thread.is_alive() which can be False even if training is active
    # We'll calculate this later after we have the trainer object
    trainer_active = False
    
    # CRITICAL: Never access trainer attributes if status is "starting" - trainer is locked during pretraining
    # We should have already returned above for "starting" status, but add safety check here
    if system.get("status") == "starting":
        # This should never be reached (we return above), but if it is, return immediately
        import time
        start_time = system.get("start_time", 0)
        elapsed = time.time() - start_time if start_time > 0 else 0
        return {
            "status": "starting",
            "message": f"Supervised pretraining in progress... ({elapsed:.0f}s) - This may take 5-10 minutes",
            "metrics": {
                "pretraining": True,
                "elapsed_seconds": int(elapsed)
            },
            "pretraining": True,
            "info": "Pretraining teaches the agent basic market patterns before RL training starts."
        }
    
    # Check if pretraining is in progress (trainer exists but timestep is 0)
    is_pretraining = False
    if trainer is not None:
        try:
            trainer_timestep = getattr(trainer, 'timestep', 0)
            is_pretrained = getattr(trainer, 'pretrained', False)
            # If timestep is 0 and pretrained is False, likely still in pretraining
            if trainer_timestep == 0 and not is_pretrained:
                is_pretraining = True
        except:
            pass
    
    # CRITICAL: For "starting" status, return immediately without accessing trainer
    # During pretraining, trainer might be locked/busy, so never access it
    if system.get("status") == "starting":
        import time
        start_time = system.get("start_time", 0)
        elapsed = time.time() - start_time if start_time > 0 else 0
        
        # Return immediately with pretraining info - no trainer access
        metrics = {
            "pretraining": True,
            "elapsed_seconds": int(elapsed)
        }
        
        # If initialization takes more than 60 seconds, likely doing supervised pretraining
        if elapsed > 60:
            if elapsed < 600:  # Less than 10 minutes - likely pretraining
                return {
                    "status": "starting",
                    "message": f"Supervised pretraining in progress... ({elapsed:.0f}s) - This may take 5-10 minutes",
                    "metrics": metrics,
                    "pretraining": True,
                    "info": "Pretraining teaches the agent basic market patterns before RL training starts."
                }
            
            # After 10 minutes, something might be wrong
            if elapsed > 600:
                return {
                    "status": "starting",
                    "message": f"Initializing training... (taking longer than expected: {elapsed:.0f}s) - May be doing supervised pretraining",
                    "metrics": metrics,
                    "warning": "Initialization taking longer than expected. Check backend console for errors."
                }
            
            # After 15 minutes, suggest stopping and checking logs
            if elapsed > 900:
                return {
                    "status": "error",
                    "message": f"Training initialization timeout ({elapsed:.0f}s). Likely stuck during data loading or pretraining. Please stop and check backend console logs.",
                    "metrics": metrics,
                    "error": "Initialization timeout - check backend logs",
                    "suggestion": "Stop training and check backend console for errors. Large number of files or pretraining issues may be causing problems."
                }
        
        return {
            "status": "starting",
            "message": "Initializing training...",
            "metrics": metrics
        }
    
    # CRITICAL: If status is "starting", return immediately without accessing trainer
    # During pretraining, trainer is busy and accessing it will block
    # We already handled "starting" status above, but if we reach here, return immediately
    if system.get("status") == "starting":
        import time
        start_time = system.get("start_time", 0)
        elapsed = time.time() - start_time if start_time > 0 else 0
        return {
            "status": "starting",
            "message": f"Supervised pretraining in progress... ({elapsed:.0f}s) - This may take 5-10 minutes",
            "metrics": {
                "pretraining": True,
                "elapsed_seconds": int(elapsed)
            },
            "pretraining": True,
            "info": "Pretraining teaches the agent basic market patterns before RL training starts."
        }
    
    # If trainer exists and is active, we should show training as running
    # even if thread check fails (thread might have died but trainer is still processing)
    if trainer is not None and trainer_active:
        # Trainer is active - show as running even if thread appears dead
        # This handles cases where thread died but trainer is still processing
        if system.get("status") != "running":
            system["status"] = "running"
            print(f"[INFO] Training status set to 'running' (trainer active, timestep={getattr(trainer, 'timestep', 'N/A')})")
        # Continue to return metrics below
    
    # Check if thread exists and is alive (but don't fail if trainer is active)
    thread_alive = thread is not None and hasattr(thread, 'is_alive') and thread.is_alive()
    
    # If trainer doesn't exist and thread is dead, training is not running
    if (thread is None or not thread_alive) and trainer is None:
        # Thread doesn't exist or is dead, and trainer also doesn't exist
        if system.get("status") == "starting":
            import time
            start_time = system.get("start_time", 0)
            elapsed = time.time() - start_time if start_time > 0 else 0
            
            # If initialization takes more than 60 seconds, something is wrong
            if elapsed > 60:
                print(f"[WARN]  WARNING: Training initialization timeout ({elapsed:.1f}s)")
                print(f"   Thread never started. Check backend logs for Trainer creation errors.")
                
                # After 5 minutes, clean up stale entry
                if elapsed > 300:
                    print(f"   Cleaning up stale training entry after {elapsed:.0f}s timeout")
                    active_systems.pop("training", None)
                    return {
                        "status": "error",
                        "message": f"Training initialization failed (timeout after {elapsed:.0f}s). Likely stuck during data loading. Please check backend console logs and stop/restart if needed.",
                        "metrics": {},
                        "error": "Initialization timeout",
                        "suggestion": "With 70+ files, data loading may be very slow. Consider using fewer files or check backend console for errors."
                    }
                
                return {
                    "status": "starting",
                    "message": f"Initializing training... (timeout: {elapsed:.0f}s)",
                    "metrics": {},
                    "warning": f"Initialization timeout ({elapsed:.0f}s). Check backend console."
                }
            
            return {
                "status": "starting",
                "message": "Initializing training...",
                "metrics": {}
            }
        # Otherwise, training is dead - clean up
        active_systems.pop("training", None)
        return {"status": "idle", "message": "Training stopped"}
    
    # Check if completed flag is set - but verify trainer is actually done
    if system.get("completed", False):
        # Check if trainer is actually finished (timestep >= total_timesteps)
        trainer_finished = False
        if trainer is not None:
            try:
                trainer_timestep = getattr(trainer, 'timestep', 0)
                trainer_total_timesteps = getattr(trainer, 'total_timesteps', 0)
                trainer_finished = trainer_timestep >= trainer_total_timesteps
            except:
                trainer_finished = True  # If we can't check, assume finished
        
        # Only clean up if thread is dead AND trainer is finished
        thread_alive = system.get("thread") and hasattr(system["thread"], 'is_alive') and system["thread"].is_alive()
        if not thread_alive and (trainer_finished or trainer is None):
            # Thread is dead and training is completed - safe to remove
            print(f"[INFO] Cleaning up completed training entry (thread dead, completed=True, trainer_finished={trainer_finished})")
            if system.get("error"):
                status_msg = f"Training failed: {system['error']}"
                active_systems.pop("training", None)
                return {"status": "error", "message": status_msg}
            else:
                # Get final metrics before removing
                final_metrics = {}
                trainer = system.get("trainer")
                if trainer:
                    # Calculate mean without numpy import
                    mean_reward = sum(trainer.episode_rewards) / len(trainer.episode_rewards) if trainer.episode_rewards else 0.0
                    mean_length = sum(trainer.episode_lengths) / len(trainer.episode_lengths) if trainer.episode_lengths else 0.0
                
                final_metrics = {
                    "total_episodes": trainer.episode,
                    "total_timesteps": trainer.timestep,
                    "mean_reward": float(mean_reward),
                    "best_reward": float(max(trainer.episode_rewards)) if trainer.episode_rewards else 0.0,
                    "mean_episode_length": float(mean_length),
                }
            active_systems.pop("training", None)
            return {
                "status": "completed", 
                "message": "Training completed successfully",
                "metrics": final_metrics
            }
    
    # Check thread status - but prioritize trainer activity
    # Only check thread if we haven't already determined trainer is active
    if not trainer_active and thread and hasattr(thread, 'is_alive') and not thread.is_alive():
        # Thread finished but completed flag not set - might have crashed or completed
        # BUT: If trainer exists and is active, training is still running (handled above)
        if trainer is None:
            # No trainer and thread is dead - training is definitely done
            active_systems.pop("training", None)
            return {"status": "completed", "message": "Training finished"}
        elif not trainer_active:
            # Trainer exists but is not active (finished) - training is done
            if not system.get("completed", False):
                # Set completed flag if not already set
                system["completed"] = True
                print(f"[INFO] Thread dead and trainer inactive - marking training as completed")
            # Still return metrics below to show final state
        # If trainer exists and is active, continue to return metrics (training is running)
        else:
            # Trainer is active even though thread is dead - this shouldn't happen normally
            # but we'll show training as running based on trainer activity
            print(f"[WARN] Thread is dead but trainer appears active (timestep={getattr(trainer, 'timestep', 'N/A')})")
            if system.get("status") != "running":
                system["status"] = "running"
    
    # Training is running - get current metrics
    # Note: trainer was already retrieved earlier, but get it again to be sure
    if trainer is None:
        trainer = system.get("trainer")
    
    # Debug: Log if trainer is None when we expect it
    if trainer is None and system.get("status") != "starting":
        print(f"[WARN] Training status is '{system.get('status')}' but trainer is None")
        # If trainer is None and status is not starting, it's likely stale - clean up
        if system.get("status") in ["running", "idle"]:
            active_systems.pop("training", None)
            return {"status": "idle", "message": "No training in progress"}
    
    # CRITICAL: Final safety check - if status is "starting", return immediately
    # This prevents any trainer access during pretraining (trainer is locked)
    if system.get("status") == "starting":
        import time
        start_time = system.get("start_time", 0)
        elapsed = time.time() - start_time if start_time > 0 else 0
        return {
            "status": "starting",
            "message": f"Supervised pretraining in progress... ({elapsed:.0f}s) - This may take 5-10 minutes",
            "metrics": {
                "pretraining": True,
                "elapsed_seconds": int(elapsed)
            },
            "pretraining": True,
            "info": "Pretraining teaches the agent basic market patterns before RL training starts."
        }
    
    metrics = {}
    if trainer:
        # CRITICAL: Check for pretraining FIRST with minimal trainer access to avoid blocking
        # Use getattr with safe defaults and wrap in try-except
        try:
            # Try to get pretraining status with timeout protection
            # Use getattr which is non-blocking
            trainer_timestep = getattr(trainer, 'timestep', None)
            is_pretrained = getattr(trainer, 'pretrained', None)
            
            # If timestep is 0 or None and not pretrained, likely in pretraining
            # BUT: Only return early if we're actually in pretraining (check pretraining_progress first)
            # If pretraining is done but timestep is still 0, training might have just started
            pretraining_progress = getattr(trainer, 'pretraining_progress', {})
            is_in_pretraining = False
            if isinstance(pretraining_progress, dict):
                phase = pretraining_progress.get("phase", "")
                progress = pretraining_progress.get("progress", 0.0)
                # Only consider it pretraining if phase indicates it's active
                is_in_pretraining = phase in ["preparing", "training"] or (progress < 1.0 and phase != "complete")
            
            # Return early only if actually in pretraining (not just timestep == 0)
            # CRITICAL: If timestep > 0, training has started - don't return early, continue to full metrics
            if is_in_pretraining and (trainer_timestep is None or trainer_timestep == 0) and (is_pretrained is None or not is_pretrained):
                # Try to get progress info (non-blocking getattr)
                # Note: pretraining_progress was already retrieved above
                if not isinstance(pretraining_progress, dict):
                    pretraining_progress = {}
                
                import time
                start_time = system.get("start_time", 0)
                elapsed = time.time() - start_time if start_time > 0 else 0
                
                # Get progress from trainer if available (safe access)
                phase = pretraining_progress.get("phase", "initializing") if isinstance(pretraining_progress, dict) else "initializing"
                progress = pretraining_progress.get("progress", 0.0) if isinstance(pretraining_progress, dict) else 0.0
                progress_message = pretraining_progress.get("message", "Initializing pretraining...") if isinstance(pretraining_progress, dict) else "Initializing pretraining..."
                
                # Return immediately - don't access any other trainer attributes
                return {
                    "status": "starting",
                    "message": progress_message or f"Supervised pretraining in progress... ({elapsed:.0f}s)",
                    "metrics": {
                        "pretraining": True,
                        "elapsed_seconds": int(elapsed),
                        "pretraining_phase": phase,
                        "pretraining_progress": progress,
                        "pretraining_message": progress_message
                    },
                    "pretraining": True,
                    "pretraining_phase": phase,
                    "pretraining_progress": progress,
                    "pretraining_message": progress_message,
                    "info": "Pretraining teaches the agent basic market patterns before RL training starts."
                }
            # If timestep > 0, training has started - continue to full metrics below (don't return early)
        except Exception as e:
            # If accessing trainer attributes fails, return minimal response immediately
            # Don't try to continue - just return a safe response
            import time
            start_time = system.get("start_time", 0)
            elapsed = time.time() - start_time if start_time > 0 else 0
            return {
                "status": "starting",
                "message": f"Training initializing... ({elapsed:.0f}s) - Pretraining may be in progress",
                "metrics": {
                    "pretraining": True,
                    "elapsed_seconds": int(elapsed)
                },
                "pretraining": True,
                "pretraining_phase": "initializing",
                "pretraining_progress": 0.0,
                "pretraining_message": "Initializing pretraining..."
            }
    
    # If we get here and trainer exists, check if we should skip blocking operations
    # During pretraining, trainer might be locked, so avoid accessing attributes
    if trainer:
        # Double-check: if we're still in pretraining phase, return immediately
        # This prevents falling through to blocking code below
        # BUT: Only return early if actually in pretraining (not just timestep == 0)
        try:
            trainer_timestep = getattr(trainer, 'timestep', None)
            is_pretrained = getattr(trainer, 'pretrained', None)
            pretraining_progress = getattr(trainer, 'pretraining_progress', {})
            if not isinstance(pretraining_progress, dict):
                pretraining_progress = {}
            
            # Check if actually in pretraining (not just timestep == 0)
            phase = pretraining_progress.get("phase", "")
            progress = pretraining_progress.get("progress", 0.0)
            is_in_pretraining = phase in ["preparing", "training"] or (progress < 1.0 and phase != "complete")
            
            # Only return early if actually in pretraining
            if is_in_pretraining and (trainer_timestep is None or trainer_timestep == 0) and (is_pretrained is None or not is_pretrained):
                # Still in pretraining - return immediately without accessing other attributes
                import time
                start_time = system.get("start_time", 0)
                elapsed = time.time() - start_time if start_time > 0 else 0
                progress_message = pretraining_progress.get("message", "Initializing pretraining...")
                return {
                    "status": "starting",
                    "message": progress_message or f"Supervised pretraining in progress... ({elapsed:.0f}s)",
                    "metrics": {
                        "pretraining": True,
                        "elapsed_seconds": int(elapsed),
                        "pretraining_phase": phase,
                        "pretraining_progress": progress,
                        "pretraining_message": progress_message
                    },
                    "pretraining": True,
                    "pretraining_phase": phase,
                    "pretraining_progress": progress,
                    "pretraining_message": progress_message
                }
        except:
            # If check fails, assume pretraining and return safe response
            import time
            start_time = system.get("start_time", 0)
            elapsed = time.time() - start_time if start_time > 0 else 0
            return {
                "status": "starting",
                "message": f"Training initializing... ({elapsed:.0f}s)",
                "metrics": {"pretraining": True, "elapsed_seconds": int(elapsed)},
                "pretraining": True,
                "pretraining_phase": "initializing",
                "pretraining_progress": 0.0
            }
    
    # Only access trainer attributes if we're past pretraining
    if trainer:
        # Wrap all remaining trainer access in the main try block
        try:
            # CRITICAL: Check timeout BEFORE accessing trainer at all
            # If we're already taking too long, return immediately with cached values
            elapsed = time.time() - start_time
            if elapsed > 0.5:  # If we're past 0.5 seconds, return immediately (very aggressive)
                print(f"[WARN] Training status endpoint taking too long ({elapsed:.2f}s) - returning cached values")
                return {
                    "status": system.get("status", "running"),
                    "message": "Training in progress",
                    "metrics": {
                        "timestep": system.get("last_timestep", 0),
                        "episode": system.get("last_episode", 0)
                    }
                }
            
            # Check timeout before expensive operations
            timeout_response = check_timeout()
            if timeout_response:
                return timeout_response
            
            # Check timeout before accessing trainer attributes
            timeout_response = check_timeout()
            if timeout_response:
                return timeout_response
            
            # Calculate recent metrics without numpy
            # Use getattr with safe defaults to avoid blocking
            episode_rewards = getattr(trainer, 'episode_rewards', [])
            recent_rewards = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
            mean_reward_10 = float(sum(recent_rewards) / len(recent_rewards)) if recent_rewards else 0.0
            
            # Check timeout again before more operations
            timeout_response = check_timeout()
            if timeout_response:
                return timeout_response
            
            # Get completed episode metrics
            latest_reward = float(trainer.episode_rewards[-1]) if trainer.episode_rewards else 0.0
            latest_length = int(trainer.episode_lengths[-1]) if trainer.episode_lengths else 0
            
            # Get current in-progress episode metrics (even if not completed)
            current_episode_reward = float(getattr(trainer, 'current_episode_reward', 0.0))
            current_episode_length = int(getattr(trainer, 'current_episode_length', 0))
            
            # Detect if episode is stuck at max_steps (likely hit episode limit)
            # If length is very high (near max_steps) and no episodes completed, episode may be stuck
            max_steps_estimate = 10000  # Common max_steps value
            is_stuck = (current_episode_length >= max_steps_estimate - 100 and 
                       trainer.episode == 0 and 
                       len(trainer.episode_rewards) == 0)
            
            # Check timeout before accessing trainer.timestep
            timeout_response = check_timeout()
            if timeout_response:
                return timeout_response
            
            # Check if training is active
            training_is_active = trainer.timestep > 0  # Training has started (timestep is advancing)
            
            # Use current episode metrics if we have an active episode (length > 0 and not stuck)
            has_active_episode = current_episode_length > 0 and not is_stuck
            display_reward = current_episode_reward if has_active_episode else latest_reward
            display_length = current_episode_length if has_active_episode else latest_length
            
            # Check timeout before calculating mean
            timeout_response = check_timeout()
            if timeout_response:
                return timeout_response
            
            # Calculate mean episode length without numpy
            mean_episode_length = float(sum(trainer.episode_lengths) / len(trainer.episode_lengths)) if trainer.episode_lengths else 0.0
            
            # Current episode number (completed episodes + 1 if there's an active episode)
            # If stuck, still show episode 1 to indicate training is happening
            current_episode_number = trainer.episode + (1 if (has_active_episode or is_stuck) else 0)
            
            # Calculate trading metrics - ALWAYS read from trainer first (updated every step)
            # Priority 1: Trainer's current episode tracking (updated from step_info)
            display_trades = getattr(trainer, 'current_episode_trades', 0)
            display_pnl = getattr(trainer, 'current_episode_pnl', 0.0)
            display_equity = getattr(trainer, 'current_episode_equity', 0.0)
            display_win_rate = getattr(trainer, 'current_episode_win_rate', 0.0)
            display_max_drawdown = getattr(trainer, 'current_episode_max_drawdown', 0.0)
            
            # Check timeout before accessing environment
            timeout_response = check_timeout()
            if timeout_response:
                return timeout_response
            
            # Priority 2: If values are 0 but training is active, get directly from environment
            if training_is_active and (display_trades == 0 and display_pnl == 0.0):
                if hasattr(trainer, 'env') and trainer.env:
                    if hasattr(trainer.env, 'episode_trades'):
                        display_trades = trainer.env.episode_trades
                    if hasattr(trainer.env, 'state') and trainer.env.state:
                        # CRITICAL FIX: Use database for current session PnL to match Performance Monitoring
                        # Environment state resets each episode, database accumulates across episodes
                        current_episode_pnl = float(trainer.env.state.total_pnl)  # Current episode only
                        display_pnl = current_episode_pnl  # Will be replaced with session total if available
                        
                        # Get current session PnL from database (matches Performance Monitoring)
                        # Use timeout to avoid blocking during pretraining
                        training_start_ts = system.get("checkpoint_resume_timestamp") or system.get("training_start_timestamp")
                        if training_start_ts:
                            try:
                                import sqlite3
                                import signal
                                db_path = project_root / "logs/trading_journal.db"
                                if db_path.exists():
                                    # Use timeout for database connection to avoid blocking
                                    conn = sqlite3.connect(str(db_path), timeout=2.0)  # 2 second timeout
                                    cursor = conn.cursor()
                                    cursor.execute("PRAGMA journal_mode=WAL")  # Ensure WAL mode for faster reads
                                    cursor.execute("""
                                        SELECT SUM(net_pnl) as session_pnl
                                        FROM trades
                                        WHERE timestamp >= ?
                                    """, (training_start_ts,))
                                    result = cursor.fetchone()
                                    if result and result[0] is not None:
                                        display_pnl = float(result[0])  # Use session total instead of episode
                                    conn.close()
                            except sqlite3.OperationalError as e:
                                # Database locked or timeout - skip and use episode PnL
                                pass
                            except Exception as e:
                                # Other errors - skip and use episode PnL
                                pass
                                
                        if hasattr(trainer.env, 'initial_capital'):
                            display_equity = float(trainer.env.initial_capital + display_pnl)
                        if trainer.env.state.trades_count > 0:
                            display_win_rate = float(trainer.env.state.winning_trades / trainer.env.state.trades_count)
                    if hasattr(trainer.env, 'max_drawdown'):
                        display_max_drawdown = float(trainer.env.max_drawdown)
            
            # Priority 3: If still no values, check for latest completed episode
            if display_trades == 0 and display_pnl == 0.0 and hasattr(trainer, 'episode_trades') and trainer.episode_trades and len(trainer.episode_trades) > 0:
                display_trades = trainer.episode_trades[-1]
                display_pnl = trainer.episode_pnls[-1] if hasattr(trainer, 'episode_pnls') and trainer.episode_pnls else 0.0
                display_equity = trainer.episode_equities[-1] if hasattr(trainer, 'episode_equities') and trainer.episode_equities else 0.0
                display_win_rate = trainer.episode_win_rates[-1] if hasattr(trainer, 'episode_win_rates') and trainer.episode_win_rates else 0.0
                display_max_drawdown = trainer.episode_max_drawdowns[-1] if hasattr(trainer, 'episode_max_drawdowns') and trainer.episode_max_drawdowns else 0.0
            
            # Aggregate trading metrics across all episodes
            # Priority 1: Get cumulative counts from trainer (accumulates across all episodes)
            session_trades = getattr(trainer, 'total_trades', 0)
            session_winning = getattr(trainer, 'total_winning_trades', 0)
            session_losing = getattr(trainer, 'total_losing_trades', 0)
            
            # Priority 2: Also get current episode trade counts from environment state
            # NOTE: Environment state resets each episode, so this is only for current episode
            env_trades = 0
            env_winning = 0
            env_losing = 0
            if hasattr(trainer, 'env') and trainer.env:
                if hasattr(trainer.env, 'state') and trainer.env.state:
                    env_trades = getattr(trainer.env.state, 'trades_count', 0)
                    env_winning = getattr(trainer.env.state, 'winning_trades', 0)
                    env_losing = getattr(trainer.env.state, 'losing_trades', 0)
            
            # CRITICAL FIX: If trainer's cumulative counts are 0 but we have episode trades,
            # calculate cumulative from episode_trades list (sum all completed episodes)
            if session_trades == 0 and hasattr(trainer, 'episode_trades') and trainer.episode_trades:
                # Sum all completed episodes' trades
                session_trades = sum(trainer.episode_trades)
                # Estimate winning/losing from episode win rates if available
                if hasattr(trainer, 'episode_win_rates') and trainer.episode_win_rates:
                    for i, episode_trades in enumerate(trainer.episode_trades):
                        if i < len(trainer.episode_win_rates):
                            win_rate = trainer.episode_win_rates[i]
                            estimated_wins = int(episode_trades * win_rate / 100.0) if episode_trades > 0 else 0
                            estimated_losses = episode_trades - estimated_wins
                            session_winning += estimated_wins
                            session_losing += estimated_losses
                
                # Debug log when calculating from episode list
                if session_trades > 0:
                    print(f"[DEBUG] Calculated cumulative trades from episode list: {session_trades} trades across {len(trainer.episode_trades)} episodes")
            
            # Debug log trade count sources
            if not hasattr(trainer, '_last_logged_trade_counts') or trainer._last_logged_trade_counts != (session_trades, env_trades):
                print(f"[DEBUG] Trade count sources - Trainer cumulative: {session_trades}, Environment current: {env_trades}, Episode list length: {len(getattr(trainer, 'episode_trades', []))}")
                trainer._last_logged_trade_counts = (session_trades, env_trades)
            
            # Initialize db_total_trades for debug logging
            db_total_trades = None
            db_winning_trades = None
            db_losing_trades = None
            
            # Get trade counts from trading journal
            # For fresh training, filter by training_start_timestamp to show only current session
            # For resumed training, show all-time totals
            # Use timeout to avoid blocking during pretraining
            # SKIP database queries if trainer is not active (idle/pretraining state) to avoid blocking
            # ALSO SKIP if we're in a timeout situation (to prevent cascading timeouts)
            trainer_timestep_check = getattr(trainer, 'timestep', 0)
            skip_db_queries = (trainer_timestep_check == 0 and not training_is_active)
            
            # CRITICAL: Skip database queries entirely if training just started (timestep < 100)
            # This prevents timeouts during initialization
            if trainer_timestep_check < 100:
                skip_db_queries = True
            
            # Check timeout before doing database queries
            timeout_response = check_timeout()
            if timeout_response:
                # Skip database queries if we're already taking too long
                skip_db_queries = True
            
            if not skip_db_queries:
                try:
                    import sqlite3
                    from src.trading_journal import TradingJournal
                    # Ensure database is initialized
                    journal = TradingJournal()
                    db_path = project_root / "logs/trading_journal.db"
                    if db_path.exists():
                        # Use timeout for database connection to avoid blocking
                        conn = sqlite3.connect(str(db_path), timeout=2.0)  # 2 second timeout
                        cursor = conn.cursor()
                        cursor.execute("PRAGMA journal_mode=WAL")  # Ensure WAL mode for faster reads
                        
                        # Get training start timestamp from current system object (already loaded above)
                        # Use system variable from line 1446, don't re-fetch
                        training_start_ts = system.get("checkpoint_resume_timestamp") or system.get("training_start_timestamp")
                        
                        # Filter by training start timestamp if available (fresh training)
                        if training_start_ts:
                            # Fresh training - only show trades from current session
                            cursor.execute("SELECT COUNT(*) FROM trades WHERE timestamp >= ?", (training_start_ts,))
                            db_total_trades = cursor.fetchone()[0] or 0
                            
                            cursor.execute("SELECT COUNT(*) FROM trades WHERE is_win = 1 AND timestamp >= ?", (training_start_ts,))
                            db_winning_trades = cursor.fetchone()[0] or 0
                            
                            cursor.execute("SELECT COUNT(*) FROM trades WHERE is_win = 0 AND timestamp >= ?", (training_start_ts,))
                            db_losing_trades = cursor.fetchone()[0] or 0
                        else:
                            # No timestamp filter - show all-time totals (fallback)
                            cursor.execute("SELECT COUNT(*) FROM trades")
                            db_total_trades = cursor.fetchone()[0] or 0
                            
                            cursor.execute("SELECT COUNT(*) FROM trades WHERE is_win = 1")
                            db_winning_trades = cursor.fetchone()[0] or 0
                            
                            cursor.execute("SELECT COUNT(*) FROM trades WHERE is_win = 0")
                            db_losing_trades = cursor.fetchone()[0] or 0
                        
                        conn.close()
                except sqlite3.OperationalError as e:
                    # Database locked or timeout - skip database queries, use trainer metrics
                    # This is expected during pretraining when database might be busy
                    pass
                except Exception as e:
                    # Other database errors - skip and use trainer metrics
                    pass
            
            # Use database counts (filtered by session if fresh training)
            # Priority: Database > Trainer cumulative > Environment (current episode only)
            if db_total_trades is not None and db_total_trades > 0:
                # Database has trades - use it (most reliable)
                total_trades = db_total_trades
                total_winning_trades = db_winning_trades if db_winning_trades is not None else 0
                total_losing_trades = db_losing_trades if db_losing_trades is not None else 0
            elif session_trades > 0:
                # Database empty but trainer has cumulative counts - use trainer
                total_trades = session_trades
                total_winning_trades = session_winning
                total_losing_trades = session_losing
            elif env_trades > 0:
                # Fallback: Only environment has trades (current episode only)
                # This is less ideal but better than showing 0
                total_trades = env_trades
                total_winning_trades = env_winning
                total_losing_trades = env_losing
            else:
                # All sources are 0
                total_trades = 0
                total_winning_trades = 0
                total_losing_trades = 0
            
            overall_win_rate = float(total_winning_trades / total_trades * 100) if total_trades > 0 else 0.0
            
            # DEBUG: Log trade counts being returned to frontend (every 10 API calls to reduce noise)
            if not hasattr(trainer, '_trade_counts_log_counter'):
                trainer._trade_counts_log_counter = 0
            trainer._trade_counts_log_counter += 1
            
            # Store db_total_trades for debug logging (will be set in try block above)
            db_trades_debug = locals().get('db_total_trades', 'N/A')
            
            if trainer._trade_counts_log_counter % 10 == 0 or (total_trades > 0 and trainer._trade_counts_log_counter <= 5):
                print(f"[DEBUG API] Returning trade counts - total_trades={total_trades}, winning={total_winning_trades}, losing={total_losing_trades}, win_rate={overall_win_rate:.1f}%")
                print(f"[DEBUG API] Sources - DB={db_trades_debug}, Trainer cumulative={session_trades}, Env current={env_trades}")
            
            # Calculate mean PnL and equity across recent episodes
            # Use last 10 episodes if available from CURRENT SESSION
            # Fallback to database if trainer lists are empty (resumed from checkpoint)
            mean_pnl_10 = 0.0
            mean_equity_10 = 0.0
            mean_win_rate_10 = 0.0
            
            # Try to get from trainer's episode lists first
            if hasattr(trainer, 'episode_pnls') and trainer.episode_pnls and len(trainer.episode_pnls) > 0:
                recent_pnls = trainer.episode_pnls[-10:] if len(trainer.episode_pnls) >= 10 else trainer.episode_pnls
                mean_pnl_10 = float(sum(recent_pnls) / len(recent_pnls)) if recent_pnls else 0.0
            
            if hasattr(trainer, 'episode_equities') and trainer.episode_equities and len(trainer.episode_equities) > 0:
                recent_equities = trainer.episode_equities[-10:] if len(trainer.episode_equities) >= 10 else trainer.episode_equities
                mean_equity_10 = float(sum(recent_equities) / len(recent_equities)) if recent_equities else 0.0
            
            if hasattr(trainer, 'episode_win_rates') and trainer.episode_win_rates and len(trainer.episode_win_rates) > 0:
                recent_win_rates = trainer.episode_win_rates[-10:] if len(trainer.episode_win_rates) >= 10 else trainer.episode_win_rates
                mean_win_rate_10 = float(sum(recent_win_rates) / len(recent_win_rates)) if recent_win_rates else 0.0
            
            # FALLBACK: If trainer lists are empty (resumed from checkpoint), calculate from database
            # Group trades by episode and calculate per-episode metrics
            # CRITICAL: Skip this expensive query if timestep is low (early training) to prevent timeouts
            if mean_pnl_10 == 0.0 and mean_equity_10 == 0.0 and training_is_active and trainer_timestep_check >= 1000:
                try:
                    import sqlite3
                    from src.trading_journal import TradingJournal
                    # Ensure database is initialized
                    journal = TradingJournal()
                    db_path = project_root / "logs/trading_journal.db"
                    if db_path.exists():
                        # Use timeout to prevent blocking
                        conn = sqlite3.connect(str(db_path), timeout=1.0)
                        cursor = conn.cursor()
                        
                        # Get recent trades grouped by episode (assuming episode is in trades table)
                        # Calculate per-episode PnL, equity, win rate from last N trades
                        cursor.execute("""
                            SELECT episode, 
                                   SUM(net_pnl) as episode_pnl,
                                   COUNT(*) as trades,
                                   SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as wins
                            FROM trades
                            WHERE episode IS NOT NULL
                            GROUP BY episode
                            ORDER BY episode DESC
                            LIMIT 10
                        """)
                        recent_episodes = cursor.fetchall()
                        
                        if recent_episodes:
                            episode_pnls_db = [row[1] for row in recent_episodes if row[1] is not None]
                            episode_win_rates_db = []
                            episode_equities_db = []
                            
                            for row in recent_episodes:
                                episode, ep_pnl, trades, wins = row
                                if trades and trades > 0:
                                    win_rate = (wins / trades) * 100
                                    episode_win_rates_db.append(win_rate)
                                
                                # Calculate equity (need initial capital - assume 100k)
                                # This is approximate - actual equity would need to track capital changes
                                if ep_pnl:
                                    # For mean calculation, we can use final equity estimates
                                    # This is a simplified calculation
                                    episode_equities_db.append(100000.0 + ep_pnl)
                            
                            if episode_pnls_db:
                                mean_pnl_10 = float(sum(episode_pnls_db) / len(episode_pnls_db))
                            if episode_win_rates_db:
                                mean_win_rate_10 = float(sum(episode_win_rates_db) / len(episode_win_rates_db))
                            if episode_equities_db:
                                mean_equity_10 = float(sum(episode_equities_db) / len(episode_equities_db))
                        
                        conn.close()
                except Exception as e:
                    # Database fallback failed - use zeros
                    print(f"[WARN] Failed to calculate mean metrics from database: {e}")
                    pass
            
            # DEBUG: Add diagnostic information for timestep issue
            debug_info = {}
            if trainer:
                debug_info = {
                    "timestep_raw": trainer.timestep,
                    "episode_raw": trainer.episode,
                    "current_episode_length": getattr(trainer, 'current_episode_length', 0),
                    "episode_lengths_count": len(trainer.episode_lengths) if hasattr(trainer, 'episode_lengths') else 0,
                    "last_episode_length": trainer.episode_lengths[-1] if hasattr(trainer, 'episode_lengths') and trainer.episode_lengths else 0,
                    "training_loop_active": hasattr(trainer, 'timestep') and trainer.timestep >= 0,
                    "total_timesteps": trainer.total_timesteps,
                }
                # Check if timestep is stuck
                if trainer.episode > 0 and trainer.timestep == 0:
                    debug_info["warning"] = "Timestep is 0 but episodes are progressing - training loop may not be executing properly"
            
            metrics = {
                "episode": current_episode_number,  # Show current episode (completed + in-progress)
                "completed_episodes": trainer.episode,  # Number of fully completed episodes
                "timestep": trainer.timestep,
                "total_timesteps": trainer.total_timesteps,
                "progress_percent": float(trainer.timestep / trainer.total_timesteps * 100) if trainer.total_timesteps > 0 else 0.0,
                "debug": debug_info,  # Add debug info to API response
                "latest_reward": display_reward,  # Show current or latest completed reward
                "current_episode_reward": current_episode_reward,  # Current in-progress reward
                "mean_reward_10": mean_reward_10,
                "latest_episode_length": display_length,  # Show current or latest completed length
                "current_episode_length": current_episode_length,  # Current in-progress length
                "mean_episode_length": mean_episode_length,
                "total_episodes": len(trainer.episode_rewards),
                # Trading metrics
                "current_episode_trades": display_trades,
                "current_episode_pnl": display_pnl,
                "current_episode_equity": display_equity,
                "current_episode_win_rate": display_win_rate * 100,  # Convert to percentage
                "current_episode_max_drawdown": display_max_drawdown * 100,  # Convert to percentage
                "total_trades": total_trades,  # All-time from database
                "total_winning_trades": total_winning_trades,  # All-time from database
                "total_losing_trades": total_losing_trades,  # All-time from database
                "overall_win_rate": overall_win_rate,  # All-time from database
                # Also include session-only counts for comparison
                "session_trades": session_trades,
                "session_winning_trades": session_winning,
                "session_losing_trades": session_losing,
                "mean_pnl_10": mean_pnl_10,
                "mean_equity_10": mean_equity_10,
                "mean_win_rate_10": mean_win_rate_10 * 100,  # Convert to percentage
                # Risk/reward metrics for profitability monitoring
                "avg_win": float(getattr(trainer, 'current_avg_win', 0.0)),
                "avg_loss": float(getattr(trainer, 'current_avg_loss', 0.0)),
                "risk_reward_ratio": float(getattr(trainer, 'current_risk_reward_ratio', 0.0)),
            }
            
            # Get latest training metrics if available (from last update)
            if hasattr(trainer, 'last_update_metrics') and trainer.last_update_metrics:
                metrics["training_metrics"] = trainer.last_update_metrics
            
            # Determine status based on trainer activity
            # If trainer is active (timestep > 0 and timestep < total_timesteps), show as running
            # Use training_is_active (calculated above) instead of trainer_active (never updated)
            if training_is_active:
                status = "running"
                message = "Training in progress"
            elif system.get("completed", False):
                status = "completed" if not system.get("error") else "error"
                message = system.get("error") or "Training completed successfully"
            else:
                # Trainer exists but might be in transition state
                status = system.get("status", "running")
                message = "Training in progress"
            
            # Include actual training mode being used by the trainer
            performance_mode = getattr(trainer, 'performance_mode', 'performance')
            
            # Check settings.json for turbo_training_mode to ensure accuracy
            # Turbo mode overrides performance_mode in settings
            import json
            from pathlib import Path
            settings_file = Path("settings.json")
            if settings_file.exists():
                try:
                    with open(settings_file, 'r') as f:
                        settings = json.load(f)
                        if settings.get("turbo_training_mode", False):
                            performance_mode = "turbo"
                except:
                    pass
            
            training_mode_info = {
                "performance_mode": performance_mode,
            }
            
            # Debug: Log what mode we're reporting
            if not hasattr(trainer, '_last_logged_mode') or trainer._last_logged_mode != performance_mode:
                print(f"[INFO] Training status API: Reporting mode = {performance_mode}, status = {status}")
                trainer._last_logged_mode = performance_mode
            
            # Include checkpoint resume timestamp if available
            checkpoint_resume_timestamp = system.get("checkpoint_resume_timestamp")
            
            return {
                "status": status,
                "message": message,
                "metrics": metrics,
                "training_mode": training_mode_info,
                "checkpoint_resume_timestamp": checkpoint_resume_timestamp
            }
        except Exception as e:
            # If accessing trainer attributes fails, return minimal response immediately
            # Don't wait - return cached values from system
            print(f"[WARN] Error accessing trainer metrics (returning cached): {e}")
            # Return cached values instead of error
            return {
                "status": system.get("status", "running"),
                "message": "Training in progress",
                "metrics": {
                    "timestep": system.get("last_timestep", 0),
                    "episode": system.get("last_episode", 0)
                }
            }
    
    # Fallback if trainer not available yet - check system status
    fallback_status = system.get("status", "idle")
    fallback_message = system.get("message", "Training status unknown")
    checkpoint_resume_timestamp = system.get("checkpoint_resume_timestamp")
    
    return {
        "status": fallback_status,
        "message": fallback_message,
        "metrics": metrics,
        "checkpoint_resume_timestamp": checkpoint_resume_timestamp
    }


@app.get("/api/training/analyze-performance")
async def analyze_trading_performance():
    """
    Analyze trading performance from journal database.
    Returns diagnostic metrics to identify issues with win rate, direction, strategy compliance, etc.
    """
    import sqlite3
    import pandas as pd
    from pathlib import Path
    
    db_path = Path("logs/trading_journal.db")
    
    if not db_path.exists():
        return {
            "status": "error",
            "message": "Trading journal database not found",
            "data": None
        }
    
    try:
        # Add timeout to prevent hanging
        conn = sqlite3.connect(str(db_path), timeout=10.0)
        conn.execute("PRAGMA busy_timeout = 10000")  # 10 second timeout
        
        result = {}
        
        # 1. Overall Statistics
        query = """
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN is_win = 0 THEN 1 ELSE 0 END) as losses,
                AVG(CASE WHEN is_win = 1 THEN 1.0 ELSE 0.0 END) * 100 as win_rate_pct,
                AVG(net_pnl) as avg_pnl,
                AVG(CASE WHEN is_win = 1 THEN net_pnl ELSE NULL END) as avg_win,
                AVG(CASE WHEN is_win = 0 THEN ABS(net_pnl) ELSE NULL END) as avg_loss,
                SUM(net_pnl) as total_pnl
            FROM trades
        """
        df = pd.read_sql_query(query, conn)
        if len(df) > 0:
            row = df.iloc[0]
            # Safely extract values with None checks
            avg_win = float(row['avg_win']) if pd.notna(row['avg_win']) else 0.0
            avg_loss = float(row['avg_loss']) if pd.notna(row['avg_loss']) else 0.0
            total_pnl = float(row['total_pnl']) if pd.notna(row['total_pnl']) else 0.0
            avg_pnl = float(row['avg_pnl']) if pd.notna(row['avg_pnl']) else 0.0
            win_rate_pct = float(row['win_rate_pct']) if pd.notna(row['win_rate_pct']) else 0.0
            
            # Calculate risk_reward_ratio safely
            risk_reward_ratio = 0.0
            if pd.notna(row['avg_win']) and pd.notna(row['avg_loss']) and row['avg_loss'] > 0:
                risk_reward_ratio = float(row['avg_win'] / row['avg_loss'])
            
            overall = {
                "total_trades": int(row['total_trades']) if pd.notna(row['total_trades']) else 0,
                "wins": int(row['wins']) if pd.notna(row['wins']) else 0,
                "losses": int(row['losses']) if pd.notna(row['losses']) else 0,
                "win_rate": win_rate_pct,
                "avg_pnl": avg_pnl,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "total_pnl": total_pnl,
                "risk_reward_ratio": risk_reward_ratio
            }
            result["overall"] = overall
        
        # 2. Exit Reason Breakdown
        query = """
            SELECT 
                exit_reason,
                COUNT(*) as count,
                AVG(CASE WHEN is_win = 1 THEN 1.0 ELSE 0.0 END) * 100 as win_rate_pct,
                AVG(net_pnl) as avg_pnl,
                SUM(net_pnl) as total_pnl
            FROM trades
            WHERE exit_reason IS NOT NULL
            GROUP BY exit_reason
            ORDER BY count DESC
        """
        df = pd.read_sql_query(query, conn)
        exit_reasons = []
        total_exits = df['count'].sum() if len(df) > 0 else 1
        for _, row in df.iterrows():
            count_val = int(row['count']) if pd.notna(row['count']) else 0
            exit_reasons.append({
                "exit_reason": str(row['exit_reason']),
                "count": count_val,
                "percentage": float(count_val / total_exits * 100) if total_exits > 0 else 0.0,
                "win_rate": float(row['win_rate_pct']) if pd.notna(row['win_rate_pct']) else 0.0,
                "avg_pnl": float(row['avg_pnl']) if pd.notna(row['avg_pnl']) else 0.0,
                "total_pnl": float(row['total_pnl']) if pd.notna(row['total_pnl']) else 0.0
            })
        result["exit_reasons"] = exit_reasons
        
        # 3. Direction Breakdown
        query = """
            SELECT 
                direction,
                COUNT(*) as total,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as wins,
                AVG(CASE WHEN is_win = 1 THEN 1.0 ELSE 0.0 END) * 100 as win_rate_pct,
                AVG(CASE WHEN is_win = 1 THEN net_pnl ELSE NULL END) as avg_win,
                AVG(CASE WHEN is_win = 0 THEN ABS(net_pnl) ELSE NULL END) as avg_loss,
                SUM(net_pnl) as total_pnl
            FROM trades
            WHERE direction IS NOT NULL
            GROUP BY direction
            ORDER BY total DESC
        """
        df = pd.read_sql_query(query, conn)
        directions = []
        for _, row in df.iterrows():
            directions.append({
                "direction": str(row['direction']),
                "total": int(row['total']) if pd.notna(row['total']) else 0,
                "wins": int(row['wins']) if pd.notna(row['wins']) else 0,
                "win_rate": float(row['win_rate_pct']) if pd.notna(row['win_rate_pct']) else 0.0,
                "avg_win": float(row['avg_win']) if pd.notna(row['avg_win']) else 0.0,
                "avg_loss": float(row['avg_loss']) if pd.notna(row['avg_loss']) else 0.0,
                "total_pnl": float(row['total_pnl']) if pd.notna(row['total_pnl']) else 0.0
            })
        result["directions"] = directions
        
        # 4. Strategy Compliance Breakdown
        query = """
            SELECT 
                CASE WHEN strategy_compliant = 1 THEN 'Compliant' ELSE 'Non-Compliant' END as compliance,
                COUNT(*) as total,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as wins,
                AVG(CASE WHEN is_win = 1 THEN 1.0 ELSE 0.0 END) * 100 as win_rate_pct,
                AVG(net_pnl) as avg_pnl
            FROM trades
            WHERE strategy_compliant IS NOT NULL
            GROUP BY strategy_compliant
            ORDER BY total DESC
        """
        df = pd.read_sql_query(query, conn)
        compliance = []
        for _, row in df.iterrows():
            compliance.append({
                "compliance": str(row['compliance']),
                "total": int(row['total']) if pd.notna(row['total']) else 0,
                "wins": int(row['wins']) if pd.notna(row['wins']) else 0,
                "win_rate": float(row['win_rate_pct']) if pd.notna(row['win_rate_pct']) else 0.0,
                "avg_pnl": float(row['avg_pnl']) if pd.notna(row['avg_pnl']) else 0.0
            })
        result["strategy_compliance"] = compliance
        
        # 5. Recent Trades (Last 20)
        query = """
            SELECT 
                trade_id,
                episode,
                direction,
                entry_price,
                exit_price,
                position_size,
                net_pnl,
                is_win,
                exit_reason,
                strategy_compliant
            FROM trades
            ORDER BY timestamp DESC
            LIMIT 20
        """
        df = pd.read_sql_query(query, conn)
        recent_trades = []
        for _, row in df.iterrows():
            recent_trades.append({
                "trade_id": int(row['trade_id']) if pd.notna(row['trade_id']) else 0,
                "episode": int(row['episode']) if pd.notna(row['episode']) else 0,
                "direction": str(row['direction']) if pd.notna(row['direction']) else "UNKNOWN",
                "entry_price": float(row['entry_price']) if pd.notna(row['entry_price']) else 0.0,
                "exit_price": float(row['exit_price']) if pd.notna(row['exit_price']) else 0.0,
                "position_size": float(row['position_size']) if pd.notna(row['position_size']) else 0.0,
                "net_pnl": float(row['net_pnl']) if pd.notna(row['net_pnl']) else 0.0,
                "is_win": bool(row['is_win'] == 1) if pd.notna(row['is_win']) else False,
                "exit_reason": str(row['exit_reason']) if pd.notna(row['exit_reason']) else "UNKNOWN",
                "strategy_compliant": bool(row['strategy_compliant'] == 1) if pd.notna(row['strategy_compliant']) else None
            })
        result["recent_trades"] = recent_trades
        
        # 6. Commission Impact
        query = """
            SELECT 
                AVG(commission) as avg_commission,
                AVG(CASE WHEN is_win = 1 THEN commission ELSE NULL END) as avg_commission_win,
                AVG(CASE WHEN is_win = 0 THEN commission ELSE NULL END) as avg_commission_loss,
                SUM(commission) as total_commission,
                COUNT(*) as total_trades
            FROM trades
        """
        df = pd.read_sql_query(query, conn)
        if len(df) > 0:
            row = df.iloc[0]
            total_commission = float(row['total_commission']) if pd.notna(row['total_commission']) else 0.0
            total_trades = int(row['total_trades']) if pd.notna(row['total_trades']) else 0
            
            result["commission"] = {
                "avg_commission": float(row['avg_commission']) if pd.notna(row['avg_commission']) else 0.0,
                "avg_commission_win": float(row['avg_commission_win']) if pd.notna(row['avg_commission_win']) else 0.0,
                "avg_commission_loss": float(row['avg_commission_loss']) if pd.notna(row['avg_commission_loss']) else 0.0,
                "total_commission": total_commission,
                "commission_per_trade": float(total_commission / total_trades) if total_trades > 0 else 0.0
            }
        
        # 7. Price Movement Analysis
        query = """
            SELECT 
                direction,
                AVG((exit_price - entry_price) / entry_price * 100) as avg_price_move_pct,
                AVG(CASE WHEN is_win = 1 THEN (exit_price - entry_price) / entry_price * 100 ELSE NULL END) as avg_win_move_pct,
                AVG(CASE WHEN is_win = 0 THEN (exit_price - entry_price) / entry_price * 100 ELSE NULL END) as avg_loss_move_pct
            FROM trades
            WHERE direction IS NOT NULL
            GROUP BY direction
        """
        df = pd.read_sql_query(query, conn)
        price_movements = []
        for _, row in df.iterrows():
            direction = str(row['direction']) if pd.notna(row['direction']) else "UNKNOWN"
            avg_move = float(row['avg_price_move_pct']) if pd.notna(row['avg_price_move_pct']) else 0.0
            avg_win_move = float(row['avg_win_move_pct']) if pd.notna(row['avg_win_move_pct']) else 0.0
            avg_loss_move = float(row['avg_loss_move_pct']) if pd.notna(row['avg_loss_move_pct']) else 0.0
            
            warnings = []
            if direction == 'LONG' and avg_move < 0:
                warnings.append("LONG trades showing negative price moves (price going down)")
            elif direction == 'SHORT' and avg_move > 0:
                warnings.append("SHORT trades showing positive price moves (price going up)")
            
            price_movements.append({
                "direction": direction,
                "avg_price_move_pct": avg_move,
                "avg_win_move_pct": avg_win_move,
                "avg_loss_move_pct": avg_loss_move,
                "warnings": warnings
            })
        result["price_movements"] = price_movements
        
        conn.close()
        
        # Validate result has minimum data
        if not result.get("overall") or result["overall"].get("total_trades", 0) == 0:
            return {
                "status": "warning",
                "message": "No trades found in journal. Analysis requires at least 1 trade.",
                "data": {
                    "overall": {
                        "total_trades": 0,
                        "wins": 0,
                        "losses": 0,
                        "win_rate": 0.0,
                        "avg_pnl": 0.0,
                        "avg_win": 0.0,
                        "avg_loss": 0.0,
                        "total_pnl": 0.0,
                        "risk_reward_ratio": 0.0
                    },
                    "exit_reasons": [],
                    "directions": [],
                    "strategy_compliance": [],
                    "recent_trades": [],
                    "commission": {
                        "avg_commission": 0.0,
                        "total_commission": 0.0
                    },
                    "price_movements": []
                }
            }
        
        return {
            "status": "success",
            "message": "Analysis complete",
            "data": result
        }
        
    except sqlite3.OperationalError as e:
        # Database locked or timeout
        error_msg = str(e)
        if "locked" in error_msg.lower() or "timeout" in error_msg.lower():
            return {
                "status": "error",
                "message": "Database is busy. Please try again in a moment.",
                "data": None
            }
        return {
            "status": "error",
            "message": f"Database error: {error_msg}",
            "data": None
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] Trading Performance Diagnostic failed: {error_trace}")
        return {
            "status": "error",
            "message": f"Analysis failed: {str(e)}",
            "data": None
        }

@app.post("/api/training/stop")
async def stop_training():
    """Stop training"""
    if "training" not in active_systems:
        raise HTTPException(status_code=400, detail="No training in progress")
    
    system = active_systems["training"]
    trainer = system.get("trainer")
    thread = system.get("thread")
    
    # Set stop flag on trainer if it has one
    if trainer:
        try:
            # Try to set a stop flag on the trainer
            if hasattr(trainer, 'stop_training'):
                trainer.stop_training = True
                print("[INFO] Stop flag set on trainer")
            # Also try to set on environment if available
            if hasattr(trainer, 'env') and trainer.env:
                if hasattr(trainer.env, 'stop_training'):
                    trainer.env.stop_training = True
        except Exception as e:
            print(f"[WARN] Could not set stop flag on trainer: {e}")
    
    # Mark as stopped in active_systems
    if "training" in active_systems:
        active_systems["training"]["status"] = "stopped"
        active_systems["training"]["completed"] = True
        active_systems["training"]["stopped"] = True
    
    # Remove from active_systems after a short delay to allow cleanup
    # This allows the status endpoint to see it was stopped
    import asyncio
    async def cleanup_after_stop():
        await asyncio.sleep(1)  # Give time for status to update
        if "training" in active_systems:
            active_systems.pop("training", None)
    
    # Schedule cleanup (don't await - let it run in background)
    asyncio.create_task(cleanup_after_stop())
    
    await broadcast_message({
        "type": "training",
        "status": "stopped",
        "message": "Training stop requested"
    })
    
    return {"status": "stopped", "message": "Training stop requested. Training will stop at next checkpoint."}


@app.post("/api/backtest/run")
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """Run backtest"""
    if not Path(request.model_path).exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {request.model_path}")
    
    async def _backtest():
        await broadcast_message({
            "type": "backtest",
            "status": "starting",
            "message": "Starting backtest..."
        })
        
        try:
            with open(request.config_path, "r") as f:
                config = yaml.safe_load(f)
            
            backtester = Backtester(config, request.model_path)
            
            await broadcast_message({
                "type": "backtest",
                "status": "running",
                "message": "Backtest running..."
            })
            
            results = backtester.run_backtest(n_episodes=request.episodes)
            
            await broadcast_message({
                "type": "backtest",
                "status": "completed",
                "message": "Backtest completed",
                "results": results
            })
            
        except Exception as e:
            await broadcast_message({
                "type": "backtest",
                "status": "error",
                "message": f"Backtest failed: {str(e)}"
            })
    
    background_tasks.add_task(_backtest)
    return {"status": "started", "message": "Backtest started"}


@app.get("/api/models/list")
async def list_models():
    """List available trained models and Ollama models"""
    models_dir = Path("models")
    archive_dir = models_dir / "Archive"
    
    # Get trained RL models
    trained_models = []
    checkpoints = []
    if models_dir.exists():
        # First, check main models directory
        for file in models_dir.glob("*.pt"):
            stat = file.stat()
            model_info = {
                "name": file.name,
                "path": str(file),
                "type": "trained",
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "archived": False
            }
            
            # Classify as checkpoint or regular model
            if "checkpoint" in file.name:
                model_info["is_checkpoint"] = True
                # Try to extract timestep from checkpoint filename
                import re
                match = re.search(r'checkpoint_(\d+)\.pt', file.name)
                if match:
                    model_info["timestep"] = int(match.group(1))
                checkpoints.append(model_info)
            else:
                model_info["is_checkpoint"] = False
                trained_models.append(model_info)
        
        # Also check Archive subdirectories for checkpoints
        if archive_dir.exists():
            import re
            for archive_folder in archive_dir.iterdir():
                if archive_folder.is_dir():
                    for file in archive_folder.glob("checkpoint*.pt"):
                        try:
                            stat = file.stat()
                            model_info = {
                                "name": file.name,
                                "path": str(file),
                                "type": "trained",
                                "size": stat.st_size,
                                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                "archived": True,
                                "archive_folder": archive_folder.name,
                                "is_checkpoint": True
                            }
                            # Try to extract timestep from checkpoint filename
                            # Handle multiple formats: checkpoint_1000.pt, checkpoint_episode_10_t10.pt
                            match = re.search(r'checkpoint_(?:episode_\d+_t)?(\d+)\.pt', file.name)
                            if match:
                                model_info["timestep"] = int(match.group(1))
                            else:
                                # Fallback: use file modification time as timestep
                                model_info["timestep"] = int(stat.st_mtime)
                            checkpoints.append(model_info)
                        except Exception as e:
                            # Skip files that can't be read
                            continue
        
        # CRITICAL FIX: Also check pretraining subdirectories for checkpoints
        # Pretraining checkpoints are saved to models/pretraining/supervised/ and models/pretraining/unsupervised/
        pretraining_dirs = [
            models_dir / "pretraining" / "supervised",
            models_dir / "pretraining" / "unsupervised"
        ]
        for pretraining_dir in pretraining_dirs:
            if pretraining_dir.exists():
                # Find all checkpoint files (checkpoint_epoch_X.pt, checkpoint_emergency_epoch_X.pt, checkpoint_final.pt)
                for file in pretraining_dir.glob("checkpoint*.pt"):
                    try:
                        stat = file.stat()
                        pretraining_type = "supervised" if "supervised" in str(pretraining_dir) else "unsupervised"
                        model_info = {
                            "name": file.name,
                            "path": str(file),
                            "type": "pretraining",
                            "pretraining_type": pretraining_type,
                            "size": stat.st_size,
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            "archived": False,
                            "is_checkpoint": True
                        }
                        # Extract epoch from checkpoint filename
                        # Formats: checkpoint_epoch_5.pt, checkpoint_emergency_epoch_10.pt, checkpoint_final.pt
                        match = re.search(r'checkpoint_(?:emergency_)?epoch_(\d+)\.pt', file.name)
                        if match:
                            model_info["epoch"] = int(match.group(1))
                            model_info["timestep"] = int(match.group(1))  # Use epoch as timestep for sorting
                        elif "final" in file.name:
                            model_info["epoch"] = 9999  # Final checkpoint gets highest epoch number
                            model_info["timestep"] = 9999
                        else:
                            # Fallback: use file modification time
                            model_info["timestep"] = int(stat.st_mtime)
                        checkpoints.append(model_info)
                    except Exception as e:
                        # Skip files that can't be read
                        continue
    
    # Get Ollama models
    ollama_models = []
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            data = response.json()
            for model in data.get("models", []):
                ollama_models.append({
                    "name": model.get("name", ""),
                    "path": model.get("name", ""),  # Use name as path for Ollama models
                    "type": "ollama",
                    "size": model.get("size", 0),
                    "modified": model.get("modified_at", "")
                })
    except Exception as e:
        # Ollama not available or connection failed - that's okay
        pass
    
    all_models = trained_models + ollama_models
    checkpoints_sorted = sorted(checkpoints, key=lambda x: x.get("timestep", 0), reverse=True)
    
    return {
        "models": sorted(all_models, key=lambda x: x.get("modified", ""), reverse=True),
        "checkpoints": checkpoints_sorted,
        "latest_checkpoint": checkpoints_sorted[0] if checkpoints_sorted else None,
        "trained_count": len(trained_models),
        "ollama_count": len(ollama_models),
        "checkpoint_count": len(checkpoints)
    }


@app.post("/api/models/checkpoint/restore-latest")
async def restore_latest_checkpoint():
    """Restore the latest checkpoint from Archive to models/ directory"""
    try:
        # Import the restore function
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        from restore_latest_checkpoint import restore_latest_checkpoint
        
        # Call the restore function with return_dict=True for API response
        result = restore_latest_checkpoint(return_dict=True)
        
        if result.get("success"):
            # Refresh models list after restore
            # The frontend should call /api/models/list to refresh
            
            return {
                "success": True,
                "message": result.get("message", "Checkpoint restored successfully"),
                "path": result.get("path"),
                "timestep": result.get("timestep"),
                "archived": result.get("archived", False),
                "modified": result.get("modified")
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "Failed to restore checkpoint"),
                "message": result.get("error", "Failed to restore checkpoint")
            }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "message": f"Error restoring checkpoint: {str(e)}"
        }


@app.post("/api/models/checkpoint/restore-latest")
async def restore_latest_checkpoint_endpoint():
    """Restore the latest checkpoint from Archive to models/ directory"""
    try:
        # Import the restore function
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        from restore_latest_checkpoint import restore_latest_checkpoint
        
        # Call the restore function with return_dict=True for API response
        result = restore_latest_checkpoint(return_dict=True)
        
        if result.get("success"):
            return {
                "success": True,
                "message": result.get("message", "Checkpoint restored successfully"),
                "path": result.get("path"),
                "timestep": result.get("timestep"),
                "archived": result.get("archived", False),
                "modified": result.get("modified")
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "Failed to restore checkpoint"),
                "message": result.get("error", "Failed to restore checkpoint")
            }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "message": f"Error restoring checkpoint: {str(e)}"
        }


@app.get("/api/models/checkpoint/info")
async def get_checkpoint_info(checkpoint_path: str):
    """Get architecture and training info for a checkpoint"""
    try:
        from pathlib import Path
        import torch
        
        # Validate input
        if not checkpoint_path or checkpoint_path in ['none', 'latest']:
            return {
                "error": "Invalid checkpoint path",
                "path": checkpoint_path,
                "message": "Checkpoint path cannot be 'none' or 'latest'"
            }
        
        # Resolve checkpoint path
        checkpoint_file = Path(str(checkpoint_path).replace('\\', '/'))
        if not checkpoint_file.exists():
            # Try relative to project root
            project_root = Path(__file__).parent.parent
            checkpoint_file = project_root / str(checkpoint_path).replace('\\', '/').lstrip('/')
        
        if not checkpoint_file.exists():
            # Return 200 with error JSON instead of 404
            return {
                "error": "Checkpoint not found",
                "path": checkpoint_path,
                "exists": False,
                "message": f"Checkpoint file not found: {checkpoint_path}"
            }
        
        # Load checkpoint
        checkpoint = torch.load(str(checkpoint_file), map_location='cpu', weights_only=False)
        
        # Extract architecture info
        hidden_dims = checkpoint.get("hidden_dims", None)
        state_dim = checkpoint.get("state_dim", None)
        timestep = checkpoint.get("timestep", 0)
        episode = checkpoint.get("episode", 0)
        
        # If not in checkpoint metadata, try to infer from state dict
        # Check both RL checkpoints (actor_state_dict) and pretraining checkpoints (model_state_dict)
        state_dict_key = None
        if "actor_state_dict" in checkpoint:
            state_dict_key = "actor_state_dict"
        elif "model_state_dict" in checkpoint:
            state_dict_key = "model_state_dict"
        
        if hidden_dims is None and state_dict_key:
            actor_state = checkpoint[state_dict_key]
            inferred_dims = []
            feature_prefix = None
            
            # Try multiple possible prefixes for feature layers
            for candidate in ["feature_layers", "encoder_based_ac.shared_layers", "shared_layers"]:
                if f"{candidate}.0.weight" in actor_state:
                    feature_prefix = candidate
                    break
            
            if feature_prefix:
                layer_idx = 0
                # Try different step sizes (3 for Linear+ReLU+Dropout, or 1 if no ReLU/Dropout)
                step_sizes = [3, 2, 1]
                for step_size in step_sizes:
                    inferred_dims = []
                    layer_idx = 0
                    while f"{feature_prefix}.{layer_idx}.weight" in actor_state:
                        layer_shape = actor_state[f"{feature_prefix}.{layer_idx}.weight"].shape
                        inferred_dims.append(int(layer_shape[0]))
                        layer_idx += step_size
                    if inferred_dims:
                        break
                
                if inferred_dims:
                    hidden_dims = inferred_dims
                    if state_dim is None and feature_prefix in ["feature_layers", "shared_layers"]:
                        first_layer_shape = actor_state[f"{feature_prefix}.0.weight"].shape
                        if len(first_layer_shape) >= 2:
                            state_dim = int(first_layer_shape[1])
            else:
                # Try to infer from any Linear layer weights
                # Look for patterns like "*.0.weight", "*.1.weight", etc.
                layer_keys = [k for k in actor_state.keys() if ".weight" in k and "feature" in k.lower()]
                if layer_keys:
                    # Try to extract layer dimensions
                    dims = []
                    for key in sorted(layer_keys):
                        if ".weight" in key:
                            shape = actor_state[key].shape
                            if len(shape) >= 2:
                                dims.append(int(shape[0]))
                    if dims:
                        hidden_dims = dims[:3]  # Take first 3 layers
        
        return {
            "path": str(checkpoint_file),
            "exists": True,
            "architecture": {
                "hidden_dims": hidden_dims,
                "state_dim": state_dim
            },
            "training": {
                "timestep": timestep,
                "episode": episode
            },
            "has_architecture": hidden_dims is not None and state_dim is not None
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[WARN]  Error loading checkpoint info for '{checkpoint_path}': {e}")
        print(error(error_trace))
        return {
            "error": str(e),
            "path": checkpoint_path,
            "exists": False,
            "message": f"Error loading checkpoint: {str(e)}"
        }


@app.post("/api/models/promote")
async def promote_model(request: PromoteModelRequest):
    """Promote a checkpoint to a named model file (default: best_model.pt)"""
    try:
        if not request.source_path or request.source_path in {"none"}:
            raise HTTPException(status_code=400, detail="Invalid source checkpoint path")

        # Resolve source path (handle absolute or relative)
        source = Path(str(request.source_path).replace("\\", "/"))
        if not source.exists():
            source = project_root / str(request.source_path).replace("\\", "/").lstrip("/")

        if not source.exists():
            raise HTTPException(status_code=404, detail=f"Checkpoint not found: {request.source_path}")

        if not source.is_file():
            raise HTTPException(status_code=400, detail=f"Source is not a file: {source}")

        models_dir = project_root / "models"
        models_dir.mkdir(exist_ok=True)

        target_name = request.target_name or "best_model.pt"
        target = models_dir / target_name

        if target.exists() and not request.overwrite:
            raise HTTPException(status_code=409, detail=f"Target already exists: {target}")

        shutil.copy2(source, target)

        return {
            "status": "success",
            "message": f"Promoted {source.name} to {target.name}",
            "source": str(source),
            "target": str(target)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to promote checkpoint: {e}")


@app.get("/api/models/compare")
async def compare_models():
    """Get detailed comparison between best_model and final_model"""
    models_dir = Path("models")
    logs_dir = Path("logs")
    
    def get_model_info(model_name: str) -> Dict:
        """Extract comprehensive information about a model"""
        model_path = models_dir / model_name
        info = {
            "exists": False,
            "name": model_name,
            "path": str(model_path),
            "file_size": 0,
            "file_size_mb": 0,
            "created_at": None,
            "modified_at": None,
            "timestep": None,
            "episode": None,
            "mean_reward": None,
            "episode_count": 0,
            "backtest_results": None,
            "evaluation_needed": True
        }
        
        if not model_path.exists():
            return info
        
        info["exists"] = True
        
        # File metadata
        stat = model_path.stat()
        info["file_size"] = stat.st_size
        info["file_size_mb"] = round(stat.st_size / (1024 * 1024), 2)
        info["created_at"] = datetime.fromtimestamp(stat.st_ctime).isoformat()
        info["modified_at"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
        
        # Try to load checkpoint metadata
        try:
            import torch
            checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)
            
            info["timestep"] = checkpoint.get("timestep", None)
            info["episode"] = checkpoint.get("episode", None)
            episode_rewards = checkpoint.get("episode_rewards", [])
            episode_lengths = checkpoint.get("episode_lengths", [])
            
            if episode_rewards:
                info["mean_reward"] = round(float(np.mean(episode_rewards)), 4)
                info["episode_count"] = len(episode_rewards)
            
        except Exception as e:
            # Model exists but can't read metadata
            pass
        
        # Check for backtest/evaluation results
        # Look for JSON files in logs directory that might contain results
        if logs_dir.exists():
            # Check for files matching pattern like "backtest_best_model_*.json" or "evaluation_*.json"
            for result_file in logs_dir.glob("*backtest*.json"):
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                        # Check if this result is for our model
                        if isinstance(data, dict) and model_name.replace(".pt", "") in str(data):
                            info["backtest_results"] = data
                            info["evaluation_needed"] = False
                            break
                except:
                    pass
            
            # Also check evaluation files
            if info["backtest_results"] is None:
                for result_file in logs_dir.glob("*evaluation*.json"):
                    try:
                        with open(result_file, 'r') as f:
                            data = json.load(f)
                            if isinstance(data, dict) and model_name.replace(".pt", "") in str(data):
                                info["backtest_results"] = data
                                info["evaluation_needed"] = False
                                break
                    except:
                        pass
        
        return info
    
    # Get info for both models
    best_model_info = get_model_info("best_model.pt")
    final_model_info = get_model_info("final_model.pt")
    
    # Determine recommendation
    recommendation = None
    recommendation_reason = ""
    
    if best_model_info["exists"] and final_model_info["exists"]:
        # Compare timesteps (newer is usually better)
        if best_model_info["timestep"] and final_model_info["timestep"]:
            if final_model_info["timestep"] > best_model_info["timestep"]:
                recommendation = "final_model"
                recommendation_reason = f"Final model is newer (trained for {final_model_info['timestep']:,} timesteps vs {best_model_info['timestep']:,})"
            else:
                recommendation = "best_model"
                recommendation_reason = f"Best model has the highest training reward during training"
        elif best_model_info["mean_reward"] is not None:
            recommendation = "best_model"
            recommendation_reason = f"Best model had better mean reward ({best_model_info['mean_reward']:.4f}) during training"
        else:
            recommendation = "best_model"
            recommendation_reason = "Best model was selected for performance during training"
    elif best_model_info["exists"]:
        recommendation = "best_model"
        recommendation_reason = "Only best_model is available"
    elif final_model_info["exists"]:
        recommendation = "final_model"
        recommendation_reason = "Only final_model is available"
    
    # Use case guidance
    use_case_guidance = {
        "best_model": {
            "recommended_for": ["Live trading", "Production use", "When you want best performance"],
            "description": "This model achieved the highest mean reward during training. It represents the best-performing checkpoint saved during the training process."
        },
        "final_model": {
            "recommended_for": ["Continuing training", "Latest training state", "When training just completed"],
            "description": "This model represents the final state after all training is complete. It may not have the best performance, but contains the latest learned weights."
        }
    }
    
    return {
        "best_model": best_model_info,
        "final_model": final_model_info,
        "recommendation": recommendation,
        "recommendation_reason": recommendation_reason,
        "use_case_guidance": use_case_guidance
    }


@app.get("/api/trading/bridge-status")
async def bridge_status():
    """Get bridge server status"""
    if "bridge" not in active_processes:
        return {"status": "stopped", "running": False, "message": "Bridge server not running"}
    
    proc = active_processes["bridge"]
    
    # Check if process is still alive
    if proc.poll() is not None:
        # Process has terminated, clean up
        active_processes.pop("bridge", None)
        return {"status": "stopped", "running": False, "message": "Bridge server stopped"}
    
    return {"status": "running", "running": True, "message": "Bridge server is running", "pid": proc.pid}


@app.post("/api/trading/start-bridge")
async def start_bridge(background_tasks: BackgroundTasks):
    """Start NT8 bridge server"""
    # Check if bridge is already running and clean up dead processes
    if "bridge" in active_processes:
        proc = active_processes["bridge"]
        # Check if process is still alive
        if proc.poll() is None:
            # Process is still running
            raise HTTPException(status_code=400, detail="Bridge server already running")
        else:
            # Process died, clean it up
            active_processes.pop("bridge", None)
    
    async def _start_bridge():
        await broadcast_message({
            "type": "bridge",
            "status": "starting",
            "message": "Starting NT8 bridge server..."
        })
        
        try:
            venv_python = Path(".venv/Scripts/python.exe") if os.name == "nt" else Path(".venv/bin/python")
            if not venv_python.exists():
                venv_python = Path("venv/Scripts/python.exe") if os.name == "nt" else Path("venv/bin/python")
            if not venv_python.exists():
                venv_python = Path(sys.executable)
            
            # Run as a module
            proc = subprocess.Popen(
                [str(venv_python), "-m", "src.nt8_bridge_server"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(project_root)
            )
            
            # Wait a moment to check if it started successfully
            await asyncio.sleep(0.5)
            
            # Check if process died immediately (startup error)
            if proc.poll() is not None:
                # Process died, get error message
                stdout, stderr = proc.communicate()
                error_msg = stderr.decode('utf-8', errors='ignore') if stderr else stdout.decode('utf-8', errors='ignore')
                error_msg = error_msg.strip() or "Unknown error during startup"
                
                await broadcast_message({
                    "type": "bridge",
                    "status": "error",
                    "message": f"Failed to start bridge: {error_msg}"
                })
                return
            
            active_processes["bridge"] = proc
            
            await broadcast_message({
                "type": "bridge",
                "status": "running",
                "message": f"Bridge server started (PID: {proc.pid})"
            })
            
        except Exception as e:
            await broadcast_message({
                "type": "bridge",
                "status": "error",
                "message": f"Failed to start bridge: {str(e)}"
            })
    
    background_tasks.add_task(_start_bridge)
    return {"status": "started", "message": "Bridge server starting"}


@app.post("/api/trading/stop-bridge")
async def stop_bridge():
    """Stop bridge server"""
    if "bridge" not in active_processes:
        raise HTTPException(status_code=400, detail="Bridge server not running")
    
    proc = active_processes["bridge"]
    
    try:
        # Terminate the process
        proc.terminate()
        # Wait a bit for graceful shutdown
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            # Force kill if it doesn't terminate
            proc.kill()
            proc.wait()
        
        active_processes.pop("bridge", None)
        
        await broadcast_message({
            "type": "bridge",
            "status": "stopped",
            "message": "Bridge server stopped"
        })
        
        return {"status": "stopped", "message": "Bridge server stopped"}
    except Exception as e:
        # Clean up even if there's an error
        active_processes.pop("bridge", None)
        raise HTTPException(status_code=500, detail=f"Error stopping bridge: {str(e)}")


@app.post("/api/trading/start")
async def start_trading(request: LiveTradingRequest, background_tasks: BackgroundTasks):
    """Start live/paper trading"""
    # Check if trading is already running
    if "trading" in active_systems:
        raise HTTPException(status_code=400, detail="Trading already running")
    
    # Check if training is running (trading should be blocked during training)
    if "training" in active_systems:
        system = active_systems["training"]
        thread_alive = system.get("thread") and system["thread"].is_alive()
        if thread_alive:
            raise HTTPException(
                status_code=400, 
                detail="Cannot start trading while training is in progress. Please stop training first."
            )
    
    if not Path(request.model_path).exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {request.model_path}")
    
    async def _start_trading():
        await broadcast_message({
            "type": "trading",
            "status": "starting",
            "message": "Starting trading system..."
        })
        
        try:
            with open(request.config_path, "r") as f:
                config = yaml.safe_load(f)
            
            config["live_trading"]["paper_trading"] = request.paper_trading
            
            trading_system = LiveTradingSystem(config, request.model_path)
            
            def trading_worker():
                try:
                    trading_system.run()
                except Exception as e:
                    asyncio.create_task(broadcast_message({
                        "type": "trading",
                        "status": "error",
                        "message": f"Trading failed: {str(e)}"
                    }))
            
            thread = threading.Thread(target=trading_worker)
            thread.daemon = True
            thread.start()
            
            active_systems["trading"] = {"system": trading_system, "thread": thread}
            
            await broadcast_message({
                "type": "trading",
                "status": "running",
                "message": f"Trading started ({'paper' if request.paper_trading else 'live'} mode)"
            })
            
        except Exception as e:
            await broadcast_message({
                "type": "trading",
                "status": "error",
                "message": f"Failed to start trading: {str(e)}"
            })
    
    background_tasks.add_task(_start_trading)
    return {"status": "started", "message": "Trading system starting"}


@app.post("/api/trading/stop")
async def stop_trading():
    """Stop trading"""
    if "trading" not in active_systems:
        raise HTTPException(status_code=400, detail="Trading not running")
    
    system = active_systems["trading"]
    system["system"].running = False
    
    active_systems.pop("trading", None)
    
    await broadcast_message({
        "type": "trading",
        "status": "stopped",
        "message": "Trading stopped"
    })
    
    return {"status": "stopped", "message": "Trading stopped"}


@app.get("/api/trading/status")
async def trading_status():
    """Get trading status"""
    if "trading" not in active_systems:
        return {"status": "stopped", "message": "Trading not running"}
    
    system = active_systems["trading"]
    if not system["thread"].is_alive():
        active_systems.pop("trading", None)
        return {"status": "stopped", "message": "Trading stopped"}
    
    return {"status": "running", "message": "Trading in progress"}


@app.post("/api/learning/start")
async def start_continuous_learning(background_tasks: BackgroundTasks):
    """Start continuous learning"""
    async def _start_learning():
        await broadcast_message({
            "type": "learning",
            "status": "starting",
            "message": "Starting continuous learning..."
        })
        
        try:
            config_path = Path("configs/train_config_full.yaml")
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            
            automated_learning = AutomatedLearningOrchestrator(config)
            
            def learning_worker():
                try:
                    # Run maintenance and check for retraining
                    automated_learning.run_maintenance()
                    # Note: For full retraining, we'd need an agent instance
                    # This is a background monitoring task
                    while True:
                        time.sleep(60)  # Check every minute
                        # In a real implementation, this would check experience buffer
                        # and trigger retraining when thresholds are met
                except Exception as e:
                    asyncio.create_task(broadcast_message({
                        "type": "learning",
                        "status": "error",
                        "message": f"Continuous learning failed: {str(e)}"
                    }))
            
            thread = threading.Thread(target=learning_worker)
            thread.daemon = True
            thread.start()
            
            active_systems["learning"] = {"system": automated_learning, "thread": thread}
            
            await broadcast_message({
                "type": "learning",
                "status": "running",
                "message": "Continuous learning started"
            })
            
        except Exception as e:
            await broadcast_message({
                "type": "learning",
                "status": "error",
                "message": f"Failed to start continuous learning: {str(e)}"
            })
    
    background_tasks.add_task(_start_learning)
    return {"status": "started", "message": "Continuous learning starting"}


@app.get("/api/learning/status")
async def learning_status():
    """Get continuous learning status"""
    if "learning" not in active_systems:
        return {"status": "stopped", "message": "Continuous learning not running"}
    
    return {"status": "running", "message": "Continuous learning in progress"}


@app.get("/api/systems/status")
async def systems_status():
    """Get comprehensive system component health status"""
    components = {}
    
    # 1. Adaptive Learning Component
    try:
        adaptive_config_path = project_root / "logs/adaptive_training/current_reward_config.json"
        config_adjustments_path = project_root / "logs/adaptive_training/config_adjustments.jsonl"
        
        adaptive_status = "unknown"
        adaptive_message = "Adaptive learning not active"
        adaptive_params = {}
        last_adjustment = None
        adjustment_count = 0
        recent_adaptive_logs = []  # PHASE 6: Initialize for log database integration
        
        # Check if adaptive training is being used
        if "training" in active_systems:
            system = active_systems["training"]
            trainer = system.get("trainer")
            # Use try-except to prevent blocking on trainer attribute access
            try:
                has_adaptive = trainer and hasattr(trainer, 'adaptive_trainer') and trainer.adaptive_trainer
            except:
                has_adaptive = False
            
            if has_adaptive:
                adaptive_status = "active"
                adaptive_message = "Adaptive learning is active"
                
                # PHASE 6: Also get recent adaptive learning activity from log database
                recent_adaptive_logs = []
                try:
                    from src.log_database import LogDatabase
                    log_db = LogDatabase()
                    # Get recent adaptive adjustment messages from log database
                    adaptive_logs = log_db.query_logs(
                        limit=10,
                        categories=["adaptive_adjustment", "adaptive_stop_loss", "adaptive_enabled"]
                    )
                    recent_adaptive_logs = [
                        {
                            "timestamp": log.get("timestamp", ""),
                            "message": log.get("message", "")[:200],  # Truncate long messages
                            "category": log.get("category", "")
                        }
                        for log in adaptive_logs
                    ]
                except Exception as log_err:
                    # Log database not available or error - continue without it
                    pass
                
                # Get current parameters
                if adaptive_config_path.exists():
                    with open(adaptive_config_path, 'r') as f:
                        adaptive_params = json.load(f)
                    
                    # Get last adjustment from history
                    # CRITICAL FIX: Filter to show only adjustments from current training session
                    if config_adjustments_path.exists():
                        try:
                            with open(config_adjustments_path, 'r') as f:
                                lines = f.readlines()
                                adjustment_count = len(lines)
                                
                                # CRITICAL FIX: Filter adjustments to current training session
                                # Use episode number to filter (more reliable than timestamp when timestep is stuck)
                                try:
                                    current_episode = trainer.episode if trainer else 0
                                except:
                                    current_episode = 0
                                session_adjustments = []
                                
                                # Get training start episode (if available from system)
                                training_start_episode = system.get("training_start_episode", 0)
                                
                                for line in lines:
                                    if line.strip():
                                        try:
                                            adj_data = json.loads(line.strip())
                                            adj_episode = adj_data.get("episode")
                                            adj_timestamp = adj_data.get("timestamp", "")
                                            
                                            # Filter by episode if available (more reliable)
                                            if adj_episode is not None:
                                                # Include adjustments from current session (episode >= training_start_episode)
                                                if adj_episode >= training_start_episode:
                                                    session_adjustments.append(adj_data)
                                            else:
                                                # Fallback to timestamp filtering if episode not available
                                                training_start_ts = system.get("checkpoint_resume_timestamp") or system.get("training_start_timestamp")
                                                if training_start_ts:
                                                    if adj_timestamp >= training_start_ts:
                                                        session_adjustments.append(adj_data)
                                                else:
                                                    # No filter available - include recent adjustments (last 100)
                                                    if len(lines) <= 100:
                                                        session_adjustments.append(adj_data)
                                                    elif len(session_adjustments) == 0:
                                                        # At least include the most recent one
                                                        session_adjustments.append(adj_data)
                                        except:
                                            continue
                                
                                # Use session adjustments count
                                adjustment_count = len(session_adjustments)
                                
                                if session_adjustments:
                                    last_adjustment_data = session_adjustments[-1]
                                    last_adjustment = {
                                        "timestep": last_adjustment_data.get("timestep"),
                                        "episode": last_adjustment_data.get("episode"),  # Include episode if available
                                        "timestamp": last_adjustment_data.get("timestamp"),
                                        "adjustments": last_adjustment_data.get("adjustments", {})
                                    }
                                elif lines:
                                    # Fallback: use last line if session filtering removed everything
                                    last_line = lines[-1].strip()
                                    if last_line:
                                        last_adjustment_data = json.loads(last_line)
                                        last_adjustment = {
                                            "timestep": last_adjustment_data.get("timestep"),
                                            "episode": last_adjustment_data.get("episode"),
                                            "timestamp": last_adjustment_data.get("timestamp"),
                                            "adjustments": last_adjustment_data.get("adjustments", {})
                                        }
                                        # Mark as old data
                                        last_adjustment["_is_old_data"] = True
                        except Exception as e:
                            adaptive_message = f"Adaptive learning active but error reading adjustments: {str(e)}"
            else:
                adaptive_status = "inactive"
                adaptive_message = "Training running but adaptive learning not enabled"
        else:
            adaptive_status = "inactive"
            adaptive_message = "No training in progress"
        
        # Get enabled status from trainer if available
        enabled = False
        if "training" in active_systems:
            system = active_systems["training"]
            trainer = system.get("trainer")
            try:
                if trainer and hasattr(trainer, 'adaptive_trainer') and trainer.adaptive_trainer:
                    enabled = True
            except:
                enabled = False
        
        components["adaptive_learning"] = {
            "status": adaptive_status,
            "enabled": enabled,  # Explicitly set enabled flag
            "message": adaptive_message,
            "current_parameters": adaptive_params,
            "last_adjustment": last_adjustment,
            "total_adjustments": adjustment_count,
            "recent_logs": recent_adaptive_logs  # PHASE 6: Include recent log activity from Real-Time Log Monitoring
        }
    except Exception as e:
        components["adaptive_learning"] = {
            "status": "error",
            "enabled": False,
            "message": f"Error checking adaptive learning: {str(e)}"
        }
    
    # 2. Training System Component
    try:
        if "training" in active_systems:
            system = active_systems["training"]
            trainer = system.get("trainer")
            thread = system.get("thread")
            
            training_status = "unknown"
            training_message = ""
            gpu_status = "unknown"
            checkpoint_status = "unknown"
            
            if thread and thread.is_alive():
                if trainer:
                    # Get training status details
                    status_info = system.get("status", "unknown")
                    if status_info == "running":
                        training_status = "active"
                        training_message = "Training is running"
                    elif status_info == "starting":
                        training_status = "starting"
                        training_message = "Training is initializing"
                    else:
                        training_status = "unknown"
                        training_message = f"Training status: {status_info}"
                    
                    # Check GPU usage (with timeout protection)
                    try:
                        import torch
                        if torch.cuda.is_available():
                            gpu_status = "available"
                            try:
                                if hasattr(trainer, 'device') and 'cuda' in str(trainer.device):
                                    gpu_status = "active"
                                    training_message += " (GPU)"
                                else:
                                    gpu_status = "available_not_used"
                                    training_message += " (CPU)"
                            except:
                                gpu_status = "available_not_used"
                                training_message += " (CPU)"
                        else:
                            gpu_status = "not_available"
                            training_message += " (CPU only)"
                    except:
                        gpu_status = "unknown"
                    
                    # Check checkpoint status
                    try:
                        checkpoint_dir = project_root / "models"
                        if checkpoint_dir.exists():
                            checkpoints = list(checkpoint_dir.glob("*.pt"))
                            if checkpoints:
                                # Get most recent checkpoint
                                latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
                                checkpoint_age = time.time() - latest_checkpoint.stat().st_mtime
                                checkpoint_status = f"last_saved_{int(checkpoint_age)}s_ago"
                            else:
                                checkpoint_status = "no_checkpoints"
                        else:
                            checkpoint_status = "no_checkpoints"
                    except:
                        checkpoint_status = "unknown"
                else:
                    training_status = "starting"
                    training_message = "Trainer initializing..."
            else:
                training_status = "stopped"
                training_message = "Training not running"
        else:
            training_status = "stopped"
            training_message = "No training system active"
            gpu_status = "unknown"
            checkpoint_status = "not_applicable"
        
        components["training_system"] = {
            "status": training_status,
            "message": training_message,
            "gpu_status": gpu_status,
            "checkpoint_status": checkpoint_status
        }
    except Exception as e:
        components["training_system"] = {
            "status": "error",
            "message": f"Error checking training system: {str(e)}"
        }
    
    # 3. Trading System Component
    try:
        if "trading" in active_systems:
            system = active_systems["trading"]
            thread = system.get("thread")
            if thread and thread.is_alive():
                trading_status = "active"
                trading_message = "Trading system is running"
            else:
                trading_status = "stopped"
                trading_message = "Trading thread not alive - restart may be needed"
        else:
            # Trading not started yet - this is normal, show as inactive/ready
            # Check if bridge is available as a prerequisite
            bridge_running = "bridge" in active_processes
            if bridge_running:
                trading_status = "inactive"
                trading_message = "Ready to start (Bridge running, use Trading tab to start trading)"
            else:
                trading_status = "inactive"
                trading_message = "Not started (Start bridge server first, then start trading from Trading tab)"
        
        components["trading_system"] = {
            "status": trading_status,
            "message": trading_message
        }
    except Exception as e:
        components["trading_system"] = {
            "status": "error",
            "message": f"Error checking trading system: {str(e)}"
        }
    
    # 4. Data Pipeline Component
    try:
        data_path = project_root / "data"
        raw_data_path = data_path / "raw"
        processed_data_path = data_path / "processed"
        
        data_status = "unknown"
        data_message = ""
        data_files_count = 0
        
        if raw_data_path.exists():
            # Check for data files - include CSV, JSON, and TXT files (NT8 exports are typically .txt)
            data_files = (
                list(raw_data_path.glob("*.csv")) + 
                list(raw_data_path.glob("*.json")) + 
                list(raw_data_path.glob("*.txt"))
            )
            data_files_count = len(data_files)
            
            if data_files_count > 0:
                data_status = "available"
                data_message = f"{data_files_count} data files found"
            else:
                data_status = "no_data"
                data_message = "Data directory exists but no data files found"
        else:
            data_status = "no_directory"
            data_message = "Data directory not found"
        
        components["data_pipeline"] = {
            "status": data_status,
            "message": data_message,
            "file_count": data_files_count
        }
    except Exception as e:
        components["data_pipeline"] = {
            "status": "error",
            "message": f"Error checking data pipeline: {str(e)}"
        }
    
    # 5. Environment Component (adaptive config reading)
    try:
        adaptive_config_path = project_root / "logs/adaptive_training/current_reward_config.json"
        
        env_status = "unknown"
        env_message = ""
        env_config = {}
        
        if adaptive_config_path.exists():
            try:
                with open(adaptive_config_path, 'r') as f:
                    env_config = json.load(f)
                env_status = "reading_config"
                env_message = "Environment reading adaptive config correctly"
                
                # Check if config has expected fields
                if "min_risk_reward_ratio" in env_config:
                    env_message += f" (R:R={env_config['min_risk_reward_ratio']})"
            except Exception as e:
                env_status = "error"
                env_message = f"Error reading adaptive config: {str(e)}"
        else:
            env_status = "no_config"
            env_message = "No adaptive config file (using defaults from training config)"
        
        components["environment"] = {
            "status": env_status,
            "message": env_message,
            "config": env_config
        }
    except Exception as e:
        components["environment"] = {
            "status": "error",
            "message": f"Error checking environment: {str(e)}"
        }
    
    # 6. Decision Gate Component (quality filters)
    try:
        adaptive_config_path = project_root / "logs/adaptive_training/current_reward_config.json"
        
        decision_gate_status = "unknown"
        decision_gate_message = ""
        quality_filters = {}
        
        if adaptive_config_path.exists():
            try:
                with open(adaptive_config_path, 'r') as f:
                    config = json.load(f)
                    quality_filters = config.get("quality_filters", {})
                
                if quality_filters:
                    decision_gate_status = "active"
                    min_conf = quality_filters.get("min_action_confidence", "N/A")
                    min_quality = quality_filters.get("min_quality_score", "N/A")
                    decision_gate_message = f"Quality filters active (conf={min_conf}, quality={min_quality})"
                else:
                    decision_gate_status = "no_filters"
                    decision_gate_message = "No quality filters in config"
            except Exception as e:
                decision_gate_status = "error"
                decision_gate_message = f"Error reading quality filters: {str(e)}"
        else:
            decision_gate_status = "using_defaults"
            decision_gate_message = "Using default quality filters from training config"
        
        components["decision_gate"] = {
            "status": decision_gate_status,
            "message": decision_gate_message,
            "quality_filters": quality_filters
        }
    except Exception as e:
        components["decision_gate"] = {
            "status": "error",
            "message": f"Error checking decision gate: {str(e)}"
        }
    
    return {
        "status": "success",
        "components": components,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/monitoring/performance")
async def get_performance(since: Optional[str] = None):
    """
    Get current performance metrics - reads from trading journal for real-time updates
    
    Args:
        since: Optional ISO timestamp to filter trades (e.g., "2025-11-24T19:00:00")
               If provided, only trades after this timestamp will be included.
               Use this when resuming from checkpoint to see only new trades.
    """
    try:
        import sqlite3
        from src.trading_journal import TradingJournal
        
        # Ensure database is initialized
        journal = TradingJournal()
        
        # Always read from trading journal for real-time updates
        db_path = project_root / "logs/trading_journal.db"
        
        if not db_path.exists():
            # No journal yet - return empty metrics
            return {
                "status": "success",
                "metrics": {
                    "total_pnl": 0.0,
                    "sharpe_ratio": 0.0,
                    "sortino_ratio": 0.0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "max_drawdown": 0.0,
                    "total_trades": 0,
                    "average_trade": 0.0,
                    "source": "journal",
                    "mean_pnl_10": 0.0,
                    "risk_reward_ratio": 0.0,
                    "filtered_since": None
                }
            }
        
        # Build query with optional timestamp filter
        conn = None
        try:
            conn = sqlite3.connect(str(db_path), timeout=10.0)
            conn.execute("PRAGMA busy_timeout = 10000")  # 10 second timeout
            
            if since:
                # Filter trades since the specified timestamp
                # CRITICAL FIX: Log the filter to verify it's working
                print(f"[PERFORMANCE API] Filtering trades since: {since}", flush=True)
                query = """
                    SELECT 
                        pnl, net_pnl, is_win, entry_price, exit_price, timestamp
                    FROM trades
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                """
                trades_df = pd.read_sql_query(query, conn, params=(since,))
                print(f"[PERFORMANCE API] Found {len(trades_df)} trades after filtering (since {since})", flush=True)
            else:
                # Read all trades from journal (real-time data)
                query = """
                    SELECT 
                        pnl, net_pnl, is_win, entry_price, exit_price, timestamp
                    FROM trades
                    ORDER BY timestamp DESC
                """
                trades_df = pd.read_sql_query(query, conn)
        except Exception as db_error:
            print(f"[PERFORMANCE API] Database error: {db_error}", flush=True)
            if conn:
                try:
                    conn.close()
                except:
                    pass
            return {
                "status": "error",
                "message": f"Database connection failed: {str(db_error)}",
                "metrics": {
                    "total_pnl": 0.0,
                    "sharpe_ratio": 0.0,
                    "sortino_ratio": 0.0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "max_drawdown": 0.0,
                    "total_trades": 0,
                    "average_trade": 0.0,
                    "source": "journal",
                    "mean_pnl_10": 0.0,
                    "risk_reward_ratio": 0.0,
                    "filtered_since": None
                }
            }
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass  # Ignore errors when closing
        
        if len(trades_df) == 0:
            # No trades yet
            return {
                "status": "success",
                "metrics": {
                    "total_pnl": 0.0,
                    "sharpe_ratio": 0.0,
                    "sortino_ratio": 0.0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "max_drawdown": 0.0,
                    "total_trades": 0,
                    "average_trade": 0.0,
                    "source": "journal",
                    "mean_pnl_10": 0.0,
                    "risk_reward_ratio": 0.0
                }
            }
        
        # Calculate metrics from journal (real-time)
        total_trades = len(trades_df)
        winning_trades = trades_df[trades_df['is_win'] == 1]
        losing_trades = trades_df[trades_df['is_win'] == 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        overall_win_rate = float(win_count / total_trades) if total_trades > 0 else 0.0
        
        # PnL metrics - handle None/NaN values safely
        total_pnl = float(trades_df['net_pnl'].sum()) if pd.notna(trades_df['net_pnl'].sum()) else 0.0
        average_trade = float(trades_df['net_pnl'].mean()) if pd.notna(trades_df['net_pnl'].mean()) else 0.0
        
        # Recent 10 trades for mean PnL
        recent_trades = trades_df.head(10) if len(trades_df) >= 10 else trades_df
        mean_pnl_10 = float(recent_trades['net_pnl'].mean()) if len(recent_trades) > 0 and pd.notna(recent_trades['net_pnl'].mean()) else 0.0
        
        # Risk metrics - handle None/NaN values safely
        avg_win = float(winning_trades['net_pnl'].mean()) if len(winning_trades) > 0 and pd.notna(winning_trades['net_pnl'].mean()) else 0.0
        avg_loss = float(losing_trades['net_pnl'].mean()) if len(losing_trades) > 0 and pd.notna(losing_trades['net_pnl'].mean()) else 0.0
        
        # Profit factor - handle None/NaN values safely
        if avg_loss < 0:  # Losses are negative
            gross_profit_val = winning_trades['net_pnl'].sum() if len(winning_trades) > 0 else 0.0
            gross_loss_val = losing_trades['net_pnl'].sum() if len(losing_trades) > 0 else 0.0
            gross_profit = float(gross_profit_val) if pd.notna(gross_profit_val) else 0.0
            gross_loss = abs(float(gross_loss_val)) if pd.notna(gross_loss_val) else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)
        else:
            profit_factor = 0.0
        
        # Risk/reward ratio - handle None/NaN values safely
        risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss < 0 and pd.notna(avg_win) and pd.notna(avg_loss) else 0.0
        
        # CRITICAL FIX #5: Sharpe ratio (from percentage returns, not raw PnL)
        # Get initial capital from config (default 100000.0)
        initial_capital = 100000.0
        try:
            config_path = project_root / "configs" / "train_config_adaptive.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    initial_capital = config.get("risk_management", {}).get("initial_capital", 100000.0)
        except:
            pass  # Use default if config read fails
        
        if len(trades_df) > 1 and initial_capital > 0:
            # Convert PnL to percentage returns (standard Sharpe formula)
            pnl_values = trades_df['net_pnl'].values
            returns = pnl_values / initial_capital  # Percentage returns
            
            # Calculate mean and std of returns - handle NaN values
            mean_return_val = np.mean(returns)
            std_return_val = np.std(returns)
            mean_return = float(mean_return_val) if pd.notna(mean_return_val) else 0.0
            std_return = float(std_return_val) if pd.notna(std_return_val) else 0.0
            
            # Risk-free rate (default 0.0 for trading, can be configured)
            risk_free_rate = 0.0
            
            # Sharpe ratio = (mean_return - risk_free_rate) / std_return * sqrt(periods_per_year)
            # Using 252 trading days for annualization (standard)
            if std_return > 0:
                sharpe_ratio = (mean_return - risk_free_rate) / std_return * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        # CRITICAL FIX #5: Sortino ratio (from percentage returns, not raw PnL)
        if len(trades_df) > 1 and initial_capital > 0:
            # Convert PnL to percentage returns
            pnl_values = trades_df['net_pnl'].values
            returns = pnl_values / initial_capital  # Percentage returns
            
            mean_return_val = np.mean(returns)
            mean_return = float(mean_return_val) if pd.notna(mean_return_val) else 0.0
            downside_returns = returns[returns < 0]
            downside_std_val = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
            downside_std = float(downside_std_val) if pd.notna(downside_std_val) else 0.0
            
            # Sortino ratio = (mean_return - risk_free_rate) / downside_std * sqrt(periods_per_year)
            risk_free_rate = 0.0
            if downside_std > 0:
                sortino_ratio = (mean_return - risk_free_rate) / downside_std * np.sqrt(252)
            else:
                sortino_ratio = 0.0
        else:
            sortino_ratio = 0.0
        
        # Max drawdown (cumulative) - handle None/NaN values safely
        cumulative = trades_df['net_pnl'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        max_drawdown = float(abs(drawdown.min())) if len(drawdown) > 0 and pd.notna(drawdown.min()) else 0.0
        
        metrics = {
            "total_pnl": total_pnl,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "win_rate": overall_win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "total_trades": total_trades,
            "average_trade": average_trade,
            "source": "journal",  # Indicate this is from journal (real-time)
            "mean_pnl_10": mean_pnl_10,
            "risk_reward_ratio": risk_reward_ratio,
            "filtered_since": since  # Show if filtering was applied
        }
        
        return {"status": "success", "metrics": metrics}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.get("/api/journal/trades")
async def get_journal_trades(episode: Optional[int] = None, limit: int = 100, offset: int = 0, since: Optional[str] = None):
    """Get trades from trading journal"""
    try:
        from src.trading_journal import get_journal
        journal = get_journal()
        trades = journal.get_trades(episode=episode, limit=limit, offset=offset, since=since)
        return {"status": "success", "trades": trades, "count": len(trades), "filtered_since": since}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/journal/equity-curve")
async def get_equity_curve(episode: Optional[int] = None, limit: int = 10000, since: Optional[str] = None):
    """Get equity curve from trading journal"""
    try:
        from src.trading_journal import get_journal
        journal = get_journal()
        curve = journal.get_equity_curve(episode=episode, limit=limit, since=since)
        return {"status": "success", "equity_curve": curve, "count": len(curve), "filtered_since": since}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/journal/statistics")
async def get_journal_statistics():
    """Get trading statistics from journal"""
    try:
        from src.trading_journal import get_journal
        journal = get_journal()
        stats = journal.get_statistics()
        return {"status": "success", "statistics": stats}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ML/DL Insights API Endpoints (Phase 1)
@app.get("/api/insights/analyze")
async def get_insights_analysis():
    """Get comprehensive ML/DL insights analysis"""
    try:
        # Load settings to check if insights are enabled
        settings_path = project_root / "data" / "settings.json"
        ml_insights_enabled = False
        pattern_analysis_enabled = False
        anomaly_detection_enabled = False
        recommendations_enabled = False
        
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = json.load(f)
                ml_insights_config = settings.get('ml_insights', {})
                ml_insights_enabled = ml_insights_config.get('enabled', False)
                pattern_analysis_enabled = ml_insights_config.get('pattern_analysis', {}).get('enabled', False)
                anomaly_detection_enabled = ml_insights_config.get('anomaly_detection', {}).get('enabled', False)
                recommendations_enabled = ml_insights_config.get('recommendations', {}).get('enabled', False)
        
        # If not enabled, return disabled status
        if not ml_insights_enabled:
            return {
                "status": "disabled",
                "message": "ML/DL Insights are disabled in settings",
                "enabled": False
            }
        
        # Run insights analysis
        from src.insights.insights_analyzer import InsightsAnalyzer
        from pathlib import Path
        
        # Check if LLM explanations are enabled
        llm_explanations_enabled = False
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = json.load(f)
                ml_insights_config = settings.get('ml_insights', {})
                llm_explanations_enabled = ml_insights_config.get('llm_explanations', {}).get('enabled', False)
        
        # Get existing reasoning engine if available (optional)
        reasoning_engine = None
        try:
            from src.reasoning_engine import ReasoningEngine
            reasoning_config = settings.get('reasoning', {}) if settings_path.exists() else {}
            if reasoning_config.get('enabled', False):
                import os
                api_key = reasoning_config.get("api_key") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("GROK_API_KEY")
                reasoning_engine = ReasoningEngine(
                    provider_type=reasoning_config.get("provider", "ollama"),
                    model=reasoning_config.get("model", "deepseek-r1:8b"),
                    api_key=api_key,
                    base_url=reasoning_config.get("base_url"),
                    timeout=int(reasoning_config.get("timeout", 2.0) * 60)
                )
        except Exception as e:
            print(f"Warning: Could not initialize reasoning engine for explanations: {e}")
        
        analyzer = InsightsAnalyzer(
            db_path=Path("logs/trading_journal.db"),
            min_trades=20,
            enabled=ml_insights_enabled,
            pattern_analysis_enabled=pattern_analysis_enabled,
            anomaly_detection_enabled=anomaly_detection_enabled,
            recommendations_enabled=recommendations_enabled,
            llm_explanations_enabled=llm_explanations_enabled,
            reasoning_engine=reasoning_engine
        )
        
        findings = analyzer.analyze()
        
        if findings is None:
            return {
                "status": "insufficient_data",
                "message": "Insufficient trades for analysis (minimum 20 required)",
                "enabled": True
            }
        
        # Convert findings to dict
            findings_dict = {
                "sign_reversal_detected": findings.sign_reversal_detected,
                "sign_reversal_direction": findings.sign_reversal_direction,
                "reversal_exit_rate": findings.reversal_exit_rate,
                "direction_win_rate_gap": findings.direction_win_rate_gap,
                "commission_impact_ratio": findings.commission_impact_ratio,
                "compliance_win_rate_gap": findings.compliance_win_rate_gap,
                "overall_win_rate": findings.overall_win_rate,
                "risk_reward_ratio": findings.risk_reward_ratio,
                "total_trades": findings.total_trades,
                "critical_issues": findings.critical_issues,
                "pattern_analysis_enabled": findings.pattern_analysis_enabled,
                "pattern_analysis_available": findings.pattern_analysis_available,
                "key_patterns": findings.key_patterns,
                "feature_importance": findings.feature_importance,
                "anomaly_detection_enabled": findings.anomaly_detection_enabled,
                "anomaly_detection_available": findings.anomaly_detection_available,
                "anomaly_count": findings.anomaly_count,
                "recent_anomalies": findings.recent_anomalies,
                "recommendations_enabled": findings.recommendations_enabled,
                "recommendations_available": findings.recommendations_available,
                "recommendations": findings.recommendations,
                "root_cause_analysis_available": findings.root_cause_analysis_available,
                "failure_modes": findings.failure_modes,
                "root_causes": findings.root_causes,
                # Phase 2: Enhanced ML insights
                "ml_insights": getattr(findings, 'ml_insights', []),
                "model_metrics": getattr(findings, 'model_metrics', {}),
                "clusters": getattr(findings, 'clusters', {}),
                # Phase 6: LLM Explanations
                "llm_explanations_enabled": getattr(findings, 'llm_explanations_enabled', False),
                "recommendation_explanations": getattr(findings, 'recommendation_explanations', {}),
                "pattern_explanations": getattr(findings, 'pattern_explanations', {}),
                "anomaly_explanations": getattr(findings, 'anomaly_explanations', {}),
                "root_cause_explanations": getattr(findings, 'root_cause_explanations', {})
            }
        
        return {
            "status": "success",
            "findings": findings_dict,
            "enabled": True
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.get("/api/insights/status")
async def get_insights_status():
    """Get ML/DL Insights system status"""
    try:
        settings_path = project_root / "data" / "settings.json"
        status = {
            "enabled": False,
            "pattern_analysis_enabled": False,
            "anomaly_detection_enabled": False,
            "recommendations_enabled": False,
            "auto_apply_recommendations": False
        }
        
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = json.load(f)
                ml_insights_config = settings.get('ml_insights', {})
                status["enabled"] = ml_insights_config.get('enabled', False)
                status["pattern_analysis_enabled"] = ml_insights_config.get('pattern_analysis', {}).get('enabled', False)
                status["anomaly_detection_enabled"] = ml_insights_config.get('anomaly_detection', {}).get('enabled', False)
                status["recommendations_enabled"] = ml_insights_config.get('recommendations', {}).get('enabled', False)
                status["auto_apply_recommendations"] = ml_insights_config.get('recommendations', {}).get('auto_apply', False)
        
        return {"status": "success", "insights_status": status}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/monitoring/forecast-performance")
async def get_forecast_performance(since: Optional[str] = None):
    """Get forecast features performance analysis"""
    try:
        import sqlite3
        import numpy as np
        
        # Check config
        config_path = project_root / "configs/train_config_adaptive.yaml"
        config_info = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            env_config = config.get('environment', {})
            features_config = env_config.get('features', {})
            reward_config = env_config.get('reward', {})
            
            forecast_enabled = (
                features_config.get('include_forecast_features', False) or 
                reward_config.get('include_forecast_features', False) or
                env_config.get('include_forecast_features', False)
            )
            regime_enabled = (
                features_config.get('include_regime_features', False) or 
                reward_config.get('include_regime_features', False) or
                env_config.get('include_regime_features', False)
            )
            # Check if strategy validator is enabled (adds +8 strategy features)
            strategy_validator_config = reward_config.get('strategy_validator', {})
            strategy_enabled = strategy_validator_config.get('enabled', False) if isinstance(strategy_validator_config, dict) else False
            
            state_features = env_config.get('state_features', 900)
            # Calculate expected: base (900) + regime (+5) + forecast (+3) + strategy (+8)
            expected_state_dim = 900 + (5 if regime_enabled else 0) + (3 if forecast_enabled else 0) + (8 if strategy_enabled else 0)
            
            config_info = {
                "forecast_enabled": forecast_enabled,
                "regime_enabled": regime_enabled,
                "state_features": state_features,
                "expected_state_dim": expected_state_dim,
                "state_dimension_match": state_features == expected_state_dim
            }
        
        # Load trades from journal
        # Ensure database is initialized
        from src.trading_journal import TradingJournal
        journal = TradingJournal()
        
        db_path = project_root / "logs/trading_journal.db"
        trades_df = pd.DataFrame()
        
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path))
                if since:
                    query = """
                        SELECT 
                            timestamp, episode, strategy, entry_price, exit_price, 
                            pnl, net_pnl, strategy_confidence, is_win
                        FROM trades
                        WHERE timestamp >= ?
                        ORDER BY timestamp DESC
                        LIMIT 1000
                    """
                    trades_df = pd.read_sql_query(query, conn, params=(since,))
                else:
                    query = """
                        SELECT 
                            timestamp, episode, strategy, entry_price, exit_price, 
                            pnl, net_pnl, strategy_confidence, is_win
                        FROM trades
                        ORDER BY timestamp DESC
                        LIMIT 1000
                    """
                    trades_df = pd.read_sql_query(query, conn)
                conn.close()
            except Exception as e:
                print(f"[WARN] Failed to load trades: {e}")
        
        # Analyze performance
        performance_stats = {}
        if len(trades_df) > 0:
            total_trades = len(trades_df)
            winning_trades = trades_df[trades_df['is_win'] == 1]
            losing_trades = trades_df[trades_df['is_win'] == 0]
            
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            win_rate = win_count / total_trades if total_trades > 0 else 0.0
            
            total_pnl = float(trades_df['net_pnl'].sum())
            avg_pnl = float(trades_df['net_pnl'].mean())
            
            avg_win = float(winning_trades['net_pnl'].mean()) if len(winning_trades) > 0 else 0.0
            avg_loss = float(losing_trades['net_pnl'].mean()) if len(losing_trades) > 0 else 0.0
            
            profit_factor = abs(winning_trades['net_pnl'].sum() / losing_trades['net_pnl'].sum()) if len(losing_trades) > 0 and losing_trades['net_pnl'].sum() != 0 else 0.0
            
            # CRITICAL FIX #5: Sharpe-like ratio (from percentage returns, not raw PnL)
            initial_capital = 100000.0  # Default
            try:
                config_path = project_root / "configs" / "train_config_adaptive.yaml"
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                        initial_capital = config.get("risk_management", {}).get("initial_capital", 100000.0)
            except:
                pass
            
            if len(trades_df) > 1 and initial_capital > 0:
                # Convert PnL to percentage returns
                pnl_values = trades_df['net_pnl'].values
                returns = pnl_values / initial_capital
                
                mean_return = float(np.mean(returns))
                std_return = float(np.std(returns))
                risk_free_rate = 0.0
                
                if std_return > 0:
                    sharpe_like = (mean_return - risk_free_rate) / std_return * np.sqrt(252)
                else:
                    sharpe_like = 0.0
            else:
                sharpe_like = 0.0
            
            # Max drawdown
            cumulative = trades_df['net_pnl'].cumsum()
            running_max = cumulative.expanding().max()
            drawdown = cumulative - running_max
            max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0.0
            
            performance_stats = {
                "total_trades": total_trades,
                "win_count": win_count,
                "loss_count": loss_count,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "avg_pnl": avg_pnl,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "profit_factor": profit_factor,
                "sharpe_like": sharpe_like,
                "max_drawdown": max_drawdown,
                "label": "With Forecast Features" if config_info.get("forecast_enabled", False) else "Without Forecast Features"
            }
        else:
            performance_stats = {
                "total_trades": 0,
                "error": "No trades found"
            }
        
        return {
            "status": "success",
            "config": config_info,
            "performance": performance_stats
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.get("/api/monitoring/logs")
async def get_log_messages(limit: int = 100, last_seen: Optional[str] = None, category: Optional[str] = None):
    """
    Get recent log messages from database (PHASE 6: Database-backed logging).
    
    Replaces file parsing with database queries for better performance and scalability.
    
    Args:
        limit: Maximum number of log messages to return
        last_seen: Timestamp of last seen message (ISO format) to only return new messages
        category: Filter by specific category (optional)
        
    Returns:
        Dict with log messages categorized by type (same format as before for backward compatibility)
    """
    from collections import defaultdict
    
    try:
        # PHASE 6: Use database instead of file parsing
        try:
            from src.log_database import LogDatabase
            log_db = LogDatabase()
        except Exception as db_err:
            # Fallback to file parsing if database not available
            return {
                "status": "error",
                "message": f"Database not available: {db_err}",
                "messages": [],
                "summary": {}
            }
        
        # Build category filter
        categories = None
        if category:
            categories = [category]
        else:
            # Get all critical categories (same as before)
            categories = [
                "critical_0_percent",
                "adaptive_adjustment",
                "adaptive_stop_loss",
                "overconfident_model",
                "directional_bias",
                "rapid_drawdown",
                "reward_collapse",
                "adaptive_enabled",
                "strategy_violation"  # PHASE 6: Added strategy violation category
            ]
        
        # Query database with timeout protection
        try:
            import sqlite3
            # Query logs (timeout is handled in LogDatabase.query_logs)
            db_messages = log_db.query_logs(
                limit=limit * 2,  # Get more to filter and sort
                since_timestamp=last_seen,
                categories=categories
            )
        except sqlite3.OperationalError as db_err:
            # Database locked or timeout - return empty result instead of error
            error_msg = str(db_err)
            if "locked" in error_msg.lower() or "timeout" in error_msg.lower():
                print(f"[WARN] Log database query timeout/locked: {db_err}")
            else:
                print(f"[WARN] Log database operational error: {db_err}")
            db_messages = []
        except Exception as db_err:
            # Other database errors - return empty result
            print(f"[WARN] Log database query error: {db_err}")
            import traceback
            traceback.print_exc()
            db_messages = []
        
        # Convert to expected format (backward compatibility)
        messages_by_category = defaultdict(list)
        all_messages = []
        
        for msg in db_messages:
            msg_category = msg.get("category") or "unknown"
            msg_obj = {
                "timestamp": msg.get("timestamp", ""),
                "message": msg.get("message", "")[:500],  # Truncate very long lines
                "category": msg_category,
                "file": msg.get("source_file", "unknown")
            }
            messages_by_category[msg_category].append(msg_obj)
            all_messages.append(msg_obj)
        
        # Sort all messages by timestamp (newest first) and prioritize critical categories
        # Same priority order as before for consistency
        all_messages.sort(key=lambda x: (
            0 if x.get("category") == "rapid_drawdown" else (
                1 if x.get("category") == "critical_0_percent" else (
                    2 if x.get("category") == "reward_collapse" else (
                        3 if x.get("category") == "directional_bias" else (
                            4 if x.get("category") == "overconfident_model" else (
                                5 if x.get("category") == "strategy_violation" else 6
                            )
                        )
                    )
                )
            ),
            x.get("timestamp", "").__str__()
        ), reverse=True)
        all_messages = all_messages[:limit]
        
        # Get summary counts
        summary = {
            "total_messages": len(all_messages),
            "by_category": {cat: len(msgs) for cat, msgs in messages_by_category.items()},
            "last_timestamp": all_messages[0].get("timestamp", "") if all_messages else None
        }
        
        return {
            "status": "success",
            "messages": all_messages,
            "summary": summary,
            "source": "database"  # Indicate we're using database
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
            "messages": [],
            "summary": {}
        }


@app.post("/api/data/restore")
async def restore_data_files(source: Optional[str] = None):
    """
    Restore NT8 data files from archive directory to project data/raw folder.
    
    Args:
        source: NT8 export directory path (default: C:\\Users\\schuo\\Documents\\NinjaTrader 8\\export)
    
    Returns:
        Dict with restore results
    """
    try:
        from pathlib import Path
        from restore_nt8_data import restore_nt8_data
        
        # Default source directory
        if not source:
            source = r"C:\Users\schuo\Documents\NinjaTrader 8\export"
        
        source_dir = Path(source)
        dest_dir = project_root / "data" / "raw"
        
        # Run restore
        result = restore_nt8_data(source_dir, dest_dir, dry_run=False)
        
        return {
            "status": "success",
            "result": result
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
            "result": {}
        }


@app.post("/api/data/preprocess")
async def preprocess_data():
    """
    Pre-process all data files and create cache files for fast training initialization.
    Since preprocessing is fast, runs synchronously and returns results immediately.
    
    Returns:
        Dict with preprocessing status and results
    """
    try:
        import subprocess
        import sys
        from pathlib import Path
        
        # Run preprocess_all_data.py as a subprocess
        script_path = project_root / "preprocess_all_data.py"
        
        if not script_path.exists():
            return {
                "status": "error",
                "message": f"Script not found: {script_path}",
                "result": {}
            }
        
        print(f"[INFO] Starting data preprocessing...")
        start_time = time.time()
        
        # Run the script synchronously (it's fast, so no need for background)
        # Use the same Python executable and environment
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(project_root),
            env=os.environ.copy()  # Pass current environment
        )
        
        # Wait for completion with a reasonable timeout (5 minutes)
        try:
            stdout, stderr = process.communicate(timeout=300)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            return {
                "status": "error",
                "message": "Preprocessing timed out after 5 minutes",
                "stdout": stdout,
                "stderr": stderr,
                "result": {}
            }
        
        elapsed = time.time() - start_time
        
        # Check if processed files were created
        processed_dir = project_root / "data" / "processed"
        cache_files = []
        if processed_dir.exists():
            cache_files = [
                str(f.name) for f in processed_dir.glob("*.parquet")
            ] + [
                str(f.name) for f in processed_dir.glob("*.pkl")
            ]
        
        # Store status for status endpoint
        active_systems["preprocessing"] = {
            "status": "success" if process.returncode == 0 else "error",
            "return_code": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "cache_files": cache_files,
            "completed": True,
            "start_time": start_time,
            "elapsed_seconds": elapsed
        }
        
        if process.returncode == 0:
            print(f"[INFO] Preprocessing completed successfully in {elapsed:.1f}s")
            print(f"[INFO] Created {len(cache_files)} cache files")
            return {
                "status": "success",
                "message": f"Data preprocessing completed successfully in {elapsed:.1f}s. {len(cache_files)} cache files created.",
                "return_code": process.returncode,
                "stdout": stdout[-2000:] if len(stdout) > 2000 else stdout,  # Last 2000 chars
                "stderr": stderr[-2000:] if len(stderr) > 2000 else stderr,  # Last 2000 chars
                "cache_files": cache_files,
                "elapsed_seconds": elapsed,
                "result": {
                    "cache_files": cache_files,
                    "return_code": process.returncode
                }
            }
        else:
            print(f"[ERROR] Preprocessing failed with return code {process.returncode}")
            print(f"[ERROR] stderr: {stderr[-500:] if stderr else 'No error output'}")
            return {
                "status": "error",
                "message": f"Preprocessing failed with return code {process.returncode}",
                "return_code": process.returncode,
                "stdout": stdout[-2000:] if len(stdout) > 2000 else stdout,
                "stderr": stderr[-2000:] if len(stderr) > 2000 else stderr,
                "cache_files": cache_files,
                "elapsed_seconds": elapsed,
                "result": {}
            }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] Preprocessing exception: {e}")
        print(error_trace)
        return {
            "status": "error",
            "message": f"Preprocessing failed: {str(e)}",
            "error": str(e),
            "traceback": error_trace,
            "result": {}
        }


@app.get("/api/data/preprocess/status")
async def preprocess_data_status():
    """
    Get the current status of data preprocessing.
    
    Returns:
        Dict with preprocessing status, progress, and results if completed
    """
    if "preprocessing" not in active_systems:
        return {
            "status": "idle",
            "message": "No preprocessing in progress",
            "result": {}
        }
    
    preprocessing = active_systems["preprocessing"]
    start_time = preprocessing.get("start_time", 0)
    elapsed = time.time() - start_time if start_time > 0 else 0
    
    if preprocessing.get("completed", False):
        # Preprocessing is done
        return {
            "status": preprocessing.get("status", "unknown"),
            "message": preprocessing.get("status") == "success" and "Preprocessing completed successfully" or "Preprocessing failed",
            "elapsed_seconds": int(elapsed),
            "return_code": preprocessing.get("return_code"),
            "stdout": preprocessing.get("stdout", ""),
            "stderr": preprocessing.get("stderr", ""),
            "cache_files": preprocessing.get("cache_files", []),
            "error": preprocessing.get("error"),
            "result": {
                "cache_files": preprocessing.get("cache_files", []),
                "return_code": preprocessing.get("return_code")
            }
        }
    else:
        # Preprocessing is still running
        return {
            "status": "running",
            "message": f"Preprocessing in progress... ({int(elapsed)}s elapsed)",
            "elapsed_seconds": int(elapsed),
            "result": {}
        }


@app.get("/api/settings/get")
async def get_settings():
    """Get application settings"""
    settings_file = project_root / "settings.json"
    settings = {}
    
    if settings_file.exists():
        try:
            with open(settings_file, "r") as f:
                settings = json.load(f)
        except Exception as e:
            return {"status": "error", "message": f"Failed to load settings: {str(e)}"}
    
    # Ensure default values are present
    if "turbo_training_mode" not in settings:
        settings["turbo_training_mode"] = False
    if "contrarian_enabled" not in settings:
        settings["contrarian_enabled"] = True
    if "ai_insights_enabled" not in settings:
        settings["ai_insights_enabled"] = False
    if "ai_tooltips_enabled" not in settings:
        settings["ai_tooltips_enabled"] = False
    
    return {"status": "success", **settings}


@app.get("/api/settings/auto-retrain-status")
async def get_auto_retrain_status():
    """Get auto-retrain monitor status"""
    global auto_retrain_monitor, active_systems
    
    # Check if there's actually a training job running
    training_job_running = False
    if "training" in active_systems:
        system = active_systems["training"]
        thread = system.get("thread")
        if thread and hasattr(thread, 'is_alive') and thread.is_alive():
            training_job_running = True
    
    if auto_retrain_monitor:
        status = auto_retrain_monitor.get_status()
        
        # Count total files in directory for better visibility
        total_files = 0
        csv_count = 0
        txt_count = 0
        if auto_retrain_monitor.nt8_export_path:
            watch_path = Path(auto_retrain_monitor.nt8_export_path)
            if watch_path.exists():
                csv_files = list(watch_path.glob("*.csv"))
                txt_files = list(watch_path.glob("*.txt"))
                csv_count = len(csv_files)
                txt_count = len(txt_files)
                total_files = csv_count + txt_count
        
        # Count known files in cache
        known_files_count = 0
        if auto_retrain_monitor.event_handler:
            known_files_count = len(auto_retrain_monitor.event_handler.known_files)
        
        return {
            "status": "running" if status["running"] else "stopped",
            "enabled": status["enabled"],
            "nt8_export_path": status["nt8_export_path"],
            "files_detected": status["files_detected"],  # New files detected since monitor started
            "files_detected_description": "New files detected since monitor started (triggers retraining)",
            "total_files_in_directory": total_files,  # Total CSV/TXT files in directory
            "csv_files_count": csv_count,
            "txt_files_count": txt_count,
            "known_files_count": known_files_count,  # Files already processed (in cache)
            "last_retrain_trigger": status["last_retrain_trigger"],
            "running": status["running"],  # Monitor status (file watcher)
            "monitor_running": status["running"],  # Explicit monitor status
            "training_job_running": training_job_running,  # Actual training job status
            "message": "File monitor is active" if status["running"] else "File monitor is stopped"
        }
    else:
        # Check if it should be enabled
        settings_file = project_root / "settings.json"
        if settings_file.exists():
            try:
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                    if settings.get("auto_retrain_enabled") and settings.get("nt8_data_path"):
                        return {
                            "status": "not_started",
                            "enabled": True,
                            "nt8_export_path": settings["nt8_data_path"],
                            "monitor_running": False,
                            "training_job_running": training_job_running,
                            "message": "Auto-retrain is configured but monitor not started. Restart the API server to start monitoring."
                        }
            except:
                pass
        
        return {
            "status": "disabled",
            "enabled": False,
            "monitor_running": False,
            "training_job_running": training_job_running,
            "message": "Auto-retrain monitoring is not enabled or configured"
        }


@app.post("/api/settings/auto-retrain/clear-cache")
async def clear_auto_retrain_cache():
    """Clear the known files cache to force re-detection of files"""
    global auto_retrain_monitor
    
    cache_file = project_root / "logs" / "known_files_cache.json"
    
    try:
        if cache_file.exists():
            cache_file.unlink()
            print(f"[OK] Cleared cache file: {cache_file}")
        else:
            print(f"â„¹ï¸  Cache file does not exist: {cache_file}")
        
        # Also clear the in-memory cache if monitor is running
        if auto_retrain_monitor and auto_retrain_monitor.event_handler:
            auto_retrain_monitor.event_handler.known_files.clear()
            auto_retrain_monitor.event_handler.pending_files.clear()
            if auto_retrain_monitor.event_handler.debounce_timer:
                auto_retrain_monitor.event_handler.debounce_timer.cancel()
            print("[OK] Cleared in-memory cache")
        
        return {
            "status": "success",
            "message": "Cache cleared successfully",
            "cache_file": str(cache_file)
        }
    except Exception as e:
        print(error(f"[ERROR] Error clearing cache: {e}"))
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@app.post("/api/settings/auto-retrain/trigger-manual")
async def trigger_manual_retrain(background_tasks: BackgroundTasks):
    """Manually trigger retraining (bypasses file detection)"""
    global auto_retrain_monitor, active_systems
    
    print(f"\n{'='*80}")
    print(f"ðŸš€ MANUAL RETRAIN TRIGGER CALLED")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Process ID: {os.getpid()}")
    print(f"   Thread ID: {threading.current_thread().ident}")
    print(f"{'='*80}\n")
    import sys
    sys.stdout.flush()
    
    if not auto_retrain_monitor:
        print(error(f"[ERROR] Auto-retrain monitor not initialized"))
        raise HTTPException(status_code=400, detail="Auto-retrain monitor not initialized")
    
    if not auto_retrain_monitor.nt8_export_path:
        print(error(f"[ERROR] NT8 export path not configured"))
        raise HTTPException(status_code=400, detail="NT8 export path not configured")
    
    watch_path = Path(auto_retrain_monitor.nt8_export_path)
    if not watch_path.exists():
        print(error(f"[ERROR] NT8 export path does not exist: {watch_path}"))
        raise HTTPException(status_code=404, detail=f"NT8 export path does not exist: {watch_path}")
    
    # Find all CSV and TXT files in the directory
    csv_files = list(watch_path.glob("*.csv"))
    txt_files = list(watch_path.glob("*.txt"))
    all_files = csv_files + txt_files
    
    if not all_files:
        return {
            "status": "warning",
            "message": f"No CSV or TXT files found in {watch_path}",
            "files_found": 0
        }
    
    print(f"ðŸš€ Manual retrain triggered - found {len(all_files)} file(s) in {watch_path}")
    for f in all_files:
        print(f"  - {f.name}")
    
    # Clean up any stale training entries first
    global active_systems
    if "training" in active_systems:
        system = active_systems["training"]
        thread_alive = system.get("thread") and hasattr(system["thread"], 'is_alive') and system["thread"].is_alive()
        completed = system.get("completed", False)
        
        print(f"ðŸ” Checking existing training entry:")
        print(f"   Thread alive: {thread_alive}")
        print(f"   Completed: {completed}")
        
        if not thread_alive:
            # Thread is dead - safe to clean up regardless of completed flag
            print(f"ðŸ§¹ Cleaning up stale training entry (thread dead)")
            active_systems.pop("training", None)
        elif completed and not thread_alive:
            # Completed and thread dead - definitely safe to remove
            print(f"ðŸ§¹ Cleaning up completed training entry")
            active_systems.pop("training", None)
        elif thread_alive:
            # Thread is actually running - can't start new training
            print(f"[WARN]  Training is currently running, cannot start manual retrain")
            raise HTTPException(status_code=400, detail="Training already in progress. Please stop current training first.")
    
    # Instead of using callback (which has BackgroundTasks issues), call start_training directly
    try:
        from fastapi import BackgroundTasks as BGTasks
        from src.api_server import TrainingRequest
        
        # Load default config
        config_path = "configs/train_config_full.yaml"
        if not Path(config_path).exists():
            config_path = "configs/train_config_gpu.yaml"
        
        # Find latest checkpoint for resume
        checkpoint_path = None
        models_dir = Path("models")
        if models_dir.exists():
            checkpoints = sorted(
                [f for f in models_dir.glob("checkpoint_*.pt")],
                key=lambda x: int(x.stem.split('_')[1]) if x.stem.split('_')[1].isdigit() else 0,
                reverse=True
            )
            if checkpoints:
                checkpoint_path = str(checkpoints[0])
                print(f"   Will resume from: {checkpoint_path}")
        
        # Create training request
        training_request = TrainingRequest(
            device="cuda",  # Try GPU first
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            total_timesteps=None
        )
        
        print(f"ðŸš€ Manual retrain - calling start_training() directly")
        print(f"   Found {len(all_files)} file(s) in directory")
        print(f"   Using config: {config_path}")
        print(f"   Checkpoint: {checkpoint_path if checkpoint_path else 'None (fresh start)'}")
        
        # Call start_training directly with proper BackgroundTasks
        try:
            await start_training(training_request, background_tasks)
            
            # Wait a brief moment to catch immediate initialization errors
            import asyncio
            await asyncio.sleep(0.5)
            
            # Check if training actually started (not just scheduled)
            if "training" in active_systems:
                training_status = active_systems["training"]
                if training_status.get("error"):
                    # Training failed immediately during initialization
                    error_msg = training_status.get("error", "Unknown error")
                    print(error(f"[ERROR] Training failed immediately: {error_msg}"))
                    raise HTTPException(status_code=500, detail=f"Training failed to start: {error_msg}")
            
            return {
                "status": "success",
                "message": f"Retraining triggered manually with {len(all_files)} file(s) found in directory",
                "files_found": len(all_files),
                "files": [str(f) for f in all_files[:10]],  # Show first 10
                "note": "Training will use files based on config instrument/timeframes, not all files listed"
            }
        except HTTPException:
            # Re-raise HTTP exceptions (like "training already in progress")
            raise
        except Exception as e:
            # Catch any other errors from start_training
            error_msg = str(e)
            print(error(f"[ERROR] Error calling start_training: {error_msg}"))
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to start training: {error_msg}")
    except Exception as e:
        print(error(f"[ERROR] Error triggering manual retrain: {e}"))
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to trigger retraining: {str(e)}")


@app.get("/api/settings/auto-retrain/diagnostics")
async def get_auto_retrain_diagnostics():
    """Get detailed diagnostics about auto-retrain monitoring"""
    global auto_retrain_monitor
    
    diagnostics = {
        "monitor_initialized": auto_retrain_monitor is not None,
        "cache_file": str(project_root / "logs" / "known_files_cache.json"),
        "cache_exists": (project_root / "logs" / "known_files_cache.json").exists(),
    }
    
    if auto_retrain_monitor:
        diagnostics.update({
            "enabled": auto_retrain_monitor.enabled,
            "running": auto_retrain_monitor.running,
            "nt8_export_path": str(auto_retrain_monitor.nt8_export_path),
            "observer_running": auto_retrain_monitor.observer.is_alive() if auto_retrain_monitor.observer else False,
            "event_handler_exists": auto_retrain_monitor.event_handler is not None,
        })
        
        # Check if watch path exists
        if auto_retrain_monitor.nt8_export_path:
            watch_path = Path(auto_retrain_monitor.nt8_export_path)
            diagnostics["watch_path_exists"] = watch_path.exists()
            
            if watch_path.exists():
                # Count files in directory
                csv_files = list(watch_path.glob("*.csv"))
                txt_files = list(watch_path.glob("*.txt"))
                diagnostics.update({
                    "csv_files_count": len(csv_files),
                    "txt_files_count": len(txt_files),
                    "total_data_files": len(csv_files) + len(txt_files),
                    "recent_files": [f.name for f in sorted(csv_files + txt_files, key=lambda x: x.stat().st_mtime, reverse=True)[:10]]
                })
                
                # Check cache contents if it exists
                if diagnostics["cache_exists"]:
                    try:
                        cache_file = project_root / "logs" / "known_files_cache.json"
                        with open(cache_file, 'r') as f:
                            cache_data = json.load(f)
                            diagnostics["cached_files_count"] = len(cache_data.get("files", []))
                            diagnostics["cached_files_sample"] = list(cache_data.get("files", []))[:10]
                    except:
                        diagnostics["cache_read_error"] = True
            else:
                diagnostics["watch_path_exists"] = False
                
        # Get status from monitor
        status = auto_retrain_monitor.get_status()
        diagnostics["monitor_status"] = status
        
        # Check pending files if event handler exists
        if auto_retrain_monitor.event_handler:
            diagnostics["pending_files_count"] = len(auto_retrain_monitor.event_handler.pending_files)
            diagnostics["known_files_count"] = len(auto_retrain_monitor.event_handler.known_files)
    
    return diagnostics


@app.post("/api/settings/set")
async def set_settings(request: SettingsRequest):
    """Save application settings"""
    settings_file = project_root / "settings.json"
    
    # Load existing settings
    settings = {}
    if settings_file.exists():
        try:
            with open(settings_file, "r") as f:
                settings = json.load(f)
        except:
            pass
    
    # Update with new values
    if request.nt8_data_path is not None:
        settings["nt8_data_path"] = request.nt8_data_path
    
    if request.performance_mode is not None:
        settings["performance_mode"] = request.performance_mode
        
        # Notify active training if running
        if "training" in active_processes:
            print(f"ðŸ”„ Performance mode updated to: {request.performance_mode}")
            print("   Changes will take effect on next training update cycle")
    
    if request.turbo_training_mode is not None:
        settings["turbo_training_mode"] = request.turbo_training_mode
        
        # Notify active training if running
        if "training" in active_processes:
            print(f"ðŸ”„ Turbo training mode updated to: {request.turbo_training_mode}")
            print("   Changes will take effect on next training update cycle")
    
    if request.auto_retrain_enabled is not None:
        settings["auto_retrain_enabled"] = request.auto_retrain_enabled
        
        # Restart auto-retrain monitor if enabled state changed
        global auto_retrain_monitor
        if auto_retrain_monitor:
            auto_retrain_monitor.configure(enabled=request.auto_retrain_enabled)
        elif request.auto_retrain_enabled and settings.get("nt8_data_path"):
            # Start monitor if it wasn't running
            auto_retrain_monitor = AutoRetrainMonitor(
                nt8_export_path=settings["nt8_data_path"],
                auto_retrain_callback=_on_auto_retrain_triggered,
                enabled=True
            )
            auto_retrain_monitor.start()
    
    if request.contrarian_enabled is not None:
        settings["contrarian_enabled"] = request.contrarian_enabled
        print(f"ðŸ” Contrarian agent enabled: {request.contrarian_enabled}")

    if request.ai_insights_enabled is not None:
        settings["ai_insights_enabled"] = bool(request.ai_insights_enabled)

    if request.ai_tooltips_enabled is not None:
        settings["ai_tooltips_enabled"] = bool(request.ai_tooltips_enabled)
    
    # Save settings to file
    try:
        with open(settings_file, "w") as f:
            json.dump(settings, f, indent=2)
        
        # Log what was saved for debugging
        print(f"ðŸ’¾ Settings saved:")
        print(f"   performance_mode: {settings.get('performance_mode')}")
        print(f"   turbo_training_mode: {settings.get('turbo_training_mode')}")
        print(f"   auto_retrain_enabled: {settings.get('auto_retrain_enabled')}")
        print(f"   contrarian_enabled: {settings.get('contrarian_enabled')}")
        
        return {"status": "success", "message": "Settings saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {str(e)}")


@app.post("/api/config/update-forecast-features")
async def update_forecast_features(request: Dict[str, Any]):
    """Update forecast features setting in config file"""
    try:
        enabled = request.get("enabled", True)
        config_path = request.get("config_path", "configs/train_config_adaptive.yaml")
        
        # Resolve config path
        config_file = Path(str(config_path).replace('\\', '/'))
        if not config_file.exists():
            # Try relative to project root
            config_file = project_root / str(config_path).replace('\\', '/').lstrip('/')
        
        if not config_file.exists():
            raise HTTPException(status_code=404, detail=f"Config file not found: {config_path}")
        
        # Read existing config
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update forecast features setting
        if 'environment' not in config:
            config['environment'] = {}
        if 'features' not in config['environment']:
            config['environment']['features'] = {}
        
        config['environment']['features']['include_forecast_features'] = enabled
        
        # Update state dimension
        # Base: 900, Regime: +5, Forecast: +3
        base_dim = 900
        regime_enabled = config['environment']['features'].get('include_regime_features', False)
        forecast_enabled = enabled
        
        regime_dim = 5 if regime_enabled else 0
        forecast_dim = 3 if forecast_enabled else 0
        new_state_dim = base_dim + regime_dim + forecast_dim
        
        config['environment']['state_features'] = new_state_dim
        
        # Write back to file
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        print(f"✅ Updated forecast features in {config_file}")
        print(f"   include_forecast_features: {enabled}")
        print(f"   state_features: {new_state_dim}")
        
        return {
            "status": "success",
            "message": f"Forecast features {'enabled' if enabled else 'disabled'}",
            "config_path": str(config_file),
            "include_forecast_features": enabled,
            "state_features": new_state_dim
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")


# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive and wait for client messages
            data = await websocket.receive_text()
            # Echo back or handle commands
            await websocket.send_json({"type": "echo", "message": data})
    except WebSocketDisconnect:
        # Normal disconnection - client closed connection cleanly
        pass
    except Exception as e:
        # Log unexpected errors but don't crash
        print(f"[WARN]  WebSocket error: {e}")
    finally:
        # Always clean up connection, even if it's already removed
        try:
            if websocket in websocket_connections:
                websocket_connections.remove(websocket)
        except (ValueError, RuntimeError):
            # Already removed or list modified - ignore
            pass


# Monte Carlo Risk Assessment Endpoints
@app.post("/api/risk/monte-carlo")
async def assess_monte_carlo_risk(request: MonteCarloRiskRequest):
    """
    Assess trade risk using Monte Carlo simulation.
    
    Returns risk metrics including VaR, expected PnL, win probability, and optimal position size.
    """
    if not MONTE_CARLO_AVAILABLE or MonteCarloRiskAnalyzer is None:
        raise HTTPException(
            status_code=503,
            detail="Monte Carlo risk assessment not available. Ensure scipy is installed."
        )
    
    try:
        # Load price data for volatility estimation
        data_dir = Path("data/raw")
        price_data = None
        
        # Try to find recent price data
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            # Use most recent file
            latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
            try:
                df = pd.read_csv(latest_file)
                if 'close' in df.columns:
                    price_data = df
            except Exception as e:
                print(f"Warning: Could not load price data: {e}")
        
        if price_data is None or len(price_data) < 10:
            # Use synthetic data if no real data available
            # Note: np is already imported at module level (line 26), no need for local import
            # Use current_price if provided, otherwise default to 5000
            base_price = request.current_price if request.current_price and request.current_price > 0 else 5000.0
            dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
            synthetic_prices = base_price * (1 + np.random.randn(100).cumsum() * 0.01)
            price_data = pd.DataFrame({
                'close': synthetic_prices,
                'open': synthetic_prices,
                'high': synthetic_prices * 1.01,
                'low': synthetic_prices * 0.99,
                'volume': np.random.randint(1000, 10000, 100)
            })
        
        # Create analyzer
        analyzer = MonteCarloRiskAnalyzer(
            initial_capital=100000.0,  # Default, can be made configurable
            n_simulations=request.n_simulations,
            max_position_risk=0.02
        )
        
        # Run risk assessment
        result = analyzer.assess_trade_risk(
            current_price=request.current_price,
            proposed_position=request.proposed_position,
            current_position=request.current_position,
            price_data=price_data,
            entry_price=request.entry_price or request.current_price,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            time_horizon=request.time_horizon,
            simulate_overnight=request.simulate_overnight
        )
        
        # Return results
        return {
            "status": "success",
            "risk_metrics": {
                "expected_pnl": result.risk_metrics.expected_pnl,
                "expected_return": result.risk_metrics.expected_return,
                "var_95": result.risk_metrics.var_95,
                "var_99": result.risk_metrics.var_99,
                "cvar_95": result.risk_metrics.cvar_95,
                "max_drawdown": result.risk_metrics.max_drawdown,
                "win_probability": result.risk_metrics.win_probability,
                "tail_risk": result.risk_metrics.tail_risk,
                "optimal_position_size": result.risk_metrics.optimal_position_size
            },
            "simulation_config": result.simulation_config,
            "scenario_stats": {
                "min_pnl": float(np.min(result.scenario_pnls)),
                "max_pnl": float(np.max(result.scenario_pnls)),
                "median_pnl": float(np.median(result.scenario_pnls)),
                "std_pnl": float(np.std(result.scenario_pnls)),
                "percentile_5": float(np.percentile(result.scenario_pnls, 5)),
                "percentile_25": float(np.percentile(result.scenario_pnls, 25)),
                "percentile_75": float(np.percentile(result.scenario_pnls, 75)),
                "percentile_95": float(np.percentile(result.scenario_pnls, 95))
            },
            "recommendation": _get_risk_recommendation(result.risk_metrics)
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] Monte Carlo risk assessment failed: {e}")
        print(f"[ERROR] Traceback: {error_trace}")
        raise HTTPException(
            status_code=500,
            detail=f"Monte Carlo risk assessment failed: {str(e)}"
        )


@app.post("/api/risk/scenario-analysis")
async def scenario_analysis(request: MonteCarloRiskRequest):
    """
    Analyze trade under different market scenarios.
    
    Scenarios: normal, high_volatility, trending, ranging
    """
    if not MONTE_CARLO_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Monte Carlo risk assessment not available"
        )
    
    try:
        # Load price data
        data_dir = Path("data/raw")
        price_data = None
        
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
            try:
                df = pd.read_csv(latest_file)
                if 'close' in df.columns:
                    price_data = df
            except:
                pass
        
        if price_data is None or len(price_data) < 10:
            import numpy as np
            dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
            synthetic_prices = request.current_price * (1 + np.random.randn(100).cumsum() * 0.01)
            price_data = pd.DataFrame({
                'close': synthetic_prices,
                'open': synthetic_prices,
                'high': synthetic_prices * 1.01,
                'low': synthetic_prices * 0.99,
                'volume': np.random.randint(1000, 10000, 100)
            })
        
        analyzer = MonteCarloRiskAnalyzer(
            initial_capital=100000.0,
            n_simulations=request.n_simulations
        )
        
        scenarios = analyzer.scenario_analysis(
            current_price=request.current_price,
            position_size=request.proposed_position,
            price_data=price_data,
            scenarios=["normal", "high_volatility", "trending", "ranging"]
        )
        
        # Format results
        scenario_results = {}
        for scenario_name, scenario_result in scenarios.items():
            scenario_results[scenario_name] = {
                "expected_pnl": scenario_result.risk_metrics.expected_pnl,
                "var_95": scenario_result.risk_metrics.var_95,
                "win_probability": scenario_result.risk_metrics.win_probability,
                "tail_risk": scenario_result.risk_metrics.tail_risk,
                "max_drawdown": scenario_result.risk_metrics.max_drawdown
            }
        
        return {
            "status": "success",
            "scenarios": scenario_results,
            "recommendation": _get_scenario_recommendation(scenario_results)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Scenario analysis failed: {str(e)}"
        )


def _get_risk_recommendation(metrics) -> str:
    """Get human-readable risk recommendation"""
    if metrics.tail_risk > 0.10:
        return "HIGH_RISK - Tail risk too high, reduce position size"
    elif metrics.var_99 < -5000:  # VaR > $5000
        return "MODERATE_RISK - Significant downside risk, consider reducing position"
    elif metrics.win_probability < 0.45:
        return "POOR_EDGE - Win probability below 50%, reconsider trade"
    elif metrics.optimal_position_size < abs(metrics.optimal_position_size) * 0.7:
        return "POSITION_REDUCED - Optimal position size is smaller than proposed"
    else:
        return "ACCEPTABLE_RISK - Risk metrics within acceptable range"


def _get_scenario_recommendation(scenarios: Dict) -> str:
    """Get recommendation based on scenario analysis"""
    high_vol_risk = scenarios.get("high_volatility", {}).get("tail_risk", 0)
    normal_win_prob = scenarios.get("normal", {}).get("win_probability", 0)
    
    if high_vol_risk > 0.15:
        return "CAUTION - High volatility scenario shows significant tail risk"
    elif normal_win_prob < 0.50:
        return "WEAK_EDGE - Win probability below 50% in normal market conditions"
    else:
        return "ROBUST - Trade shows acceptable risk across scenarios"


# Volatility Prediction Endpoints
@app.post("/api/volatility/predict")
async def predict_volatility_endpoint(request: VolatilityPredictionRequest):
    """
    Predict future volatility using historical price data.
    
    Returns volatility forecasts and trading recommendations.
    """
    if not VOLATILITY_PREDICTOR_AVAILABLE or VolatilityPredictor is None:
        raise HTTPException(
            status_code=503,
            detail="Volatility prediction not available. Ensure required dependencies are installed."
        )
    
    try:
        # Load price data
        data_dir = Path("data/raw")
        price_data = None
        
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
            try:
                df = pd.read_csv(latest_file)
                if 'close' in df.columns:
                    price_data = df
            except Exception as e:
                print(f"Warning: Could not load price data: {e}")
        
        if price_data is None or len(price_data) < 10:
            # Use synthetic data if no real data available
            import numpy as np
            dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
            base_price = 5000.0
            synthetic_prices = base_price * (1 + np.random.randn(100).cumsum() * 0.01)
            price_data = pd.DataFrame({
                'close': synthetic_prices,
                'open': synthetic_prices,
                'high': synthetic_prices * 1.01,
                'low': synthetic_prices * 0.99,
                'volume': np.random.randint(1000, 10000, 100)
            })
        
        # Create predictor
        predictor = VolatilityPredictor(
            lookback_periods=request.lookback_periods,
            prediction_horizon=request.prediction_horizon,
            volatility_window=20
        )
        
        # Predict volatility
        forecast = predictor.predict_volatility(price_data, method=request.method)
        
        # Return results
        return {
            "status": "success",
            "current_volatility": forecast.current_volatility,
            "predicted_volatility": forecast.predicted_volatility,
            "predicted_volatility_5period": forecast.predicted_volatility_5period,
            "predicted_volatility_20period": forecast.predicted_volatility_20period,
            "volatility_trend": forecast.volatility_trend,
            "confidence": forecast.confidence,
            "volatility_percentile": forecast.volatility_percentile,
            "gap_risk_probability": forecast.gap_risk_probability,
            "recommendations": forecast.recommendations,
            "prediction_method": request.method
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] Volatility prediction failed: {e}")
        print(f"[ERROR] Traceback: {error_trace}")
        raise HTTPException(
            status_code=500,
            detail=f"Volatility prediction failed: {str(e)}"
        )


class AdaptiveSizingRequest(BaseModel):
    """Request for adaptive position sizing"""
    base_position: float
    current_price: Optional[float] = None


@app.post("/api/volatility/adaptive-sizing")
async def get_adaptive_position_sizing(request: AdaptiveSizingRequest):
    """
    Get adaptive position size multiplier based on predicted volatility.
    
    Args:
        base_position: Base position size from RL agent (-1.0 to 1.0)
        current_price: Current market price (optional, for context)
    """
    if not VOLATILITY_PREDICTOR_AVAILABLE or VolatilityPredictor is None:
        raise HTTPException(
            status_code=503,
            detail="Volatility prediction not available. Ensure required dependencies are installed."
        )
    
    try:
        # Load price data
        data_dir = Path("data/raw")
        price_data = None
        
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
            try:
                df = pd.read_csv(latest_file)
                if 'close' in df.columns:
                    price_data = df
            except:
                pass
        
        if price_data is None or len(price_data) < 10:
            import numpy as np
            dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
            base_price = request.current_price or 5000.0
            synthetic_prices = base_price * (1 + np.random.randn(100).cumsum() * 0.01)
            price_data = pd.DataFrame({
                'close': synthetic_prices,
                'open': synthetic_prices,
                'high': synthetic_prices * 1.01,
                'low': synthetic_prices * 0.99,
                'volume': np.random.randint(1000, 10000, 100)
            })
        
        predictor = VolatilityPredictor()
        forecast = predictor.predict_volatility(price_data, method="adaptive")
        
        multiplier = predictor.get_adaptive_position_multiplier(request.base_position, forecast)
        stop_loss_multiplier = predictor.get_adaptive_stop_loss_multiplier(1.0, forecast)
        
        adjusted_position = request.base_position * multiplier
        
        return {
            "status": "success",
            "base_position": request.base_position,
            "adjusted_position": adjusted_position,
            "position_multiplier": multiplier,
            "stop_loss_multiplier": stop_loss_multiplier,
            "volatility_percentile": forecast.volatility_percentile,
            "volatility_trend": forecast.volatility_trend,
            "recommendations": forecast.recommendations
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Adaptive sizing calculation failed: {str(e)}"
        )


# Scenario Simulation Endpoints
@app.post("/api/scenario/robustness-test")
async def run_robustness_test_endpoint(request: ScenarioSimulationRequest):
    """
    Run robustness test across multiple market scenarios.
    
    Tests strategy performance under different market conditions.
    """
    if not SCENARIO_SIMULATOR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Scenario simulator not available"
        )
    
    try:
        # Load price data
        data_dir = Path("data/raw")
        price_data = None
        
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
            try:
                df = pd.read_csv(latest_file)
                if 'close' in df.columns:
                    price_data = df
            except Exception as e:
                print(f"Warning: Could not load price data: {e}")
        
        if price_data is None or len(price_data) < 10:
            # Use synthetic data if no real data available
            import numpy as np
            dates = pd.date_range(end=datetime.now(), periods=200, freq='D')
            base_price = 5000.0
            synthetic_prices = base_price * (1 + np.random.randn(200).cumsum() * 0.01)
            price_data = pd.DataFrame({
                'close': synthetic_prices,
                'open': synthetic_prices,
                'high': synthetic_prices * 1.01,
                'low': synthetic_prices * 0.99,
                'volume': np.random.randint(1000, 10000, 200)
            })
        
        # Create simulator
        simulator = ScenarioSimulator(price_data, initial_capital=100000.0)
        
        # Check if RL agent backtesting is requested
        use_rl_agent = request.use_rl_agent
        model_path = request.model_path
        
        # Create backtest function
        if use_rl_agent:
            # Use RL agent backtesting
            rl_backtest_func = ScenarioSimulator.create_rl_agent_backtest_func(
                model_path=model_path,
                n_episodes=1
            )
            backtest_func = lambda data, **kwargs: rl_backtest_func(
                data,
                timeframes=[1, 5, 15],
                initial_capital=100000.0,
                transaction_cost=0.0001
            )
        else:
            # Use simple backtest
            backtest_func = None
        
        # Map scenario names to MarketRegime
        regime_map = {
            "normal": MarketRegime.NORMAL,
            "trending_up": MarketRegime.TRENDING_UP,
            "trending_down": MarketRegime.TRENDING_DOWN,
            "ranging": MarketRegime.RANGING,
            "high_volatility": MarketRegime.HIGH_VOLATILITY,
            "low_volatility": MarketRegime.LOW_VOLATILITY,
            "gap_event": MarketRegime.GAP_EVENT,
            "low_liquidity": MarketRegime.LOW_LIQUIDITY,
            "crash": MarketRegime.CRASH,
            "flash_crash": MarketRegime.FLASH_CRASH
        }
        
        # Run scenarios
        scenario_results = []
        for scenario_name in request.scenarios:
            if scenario_name not in regime_map:
                continue
            
            regime = regime_map[scenario_name]
            result = simulator.simulate_scenario(
                scenario_name=scenario_name,
                regime=regime,
                intensity=request.intensity,
                backtest_func=backtest_func
            )
            
            # Convert to dict
            scenario_results.append({
                "scenario_name": result.scenario_name,
                "market_regime": result.market_regime,
                "total_return": result.total_return,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
                "total_trades": result.total_trades,
                "winning_trades": result.winning_trades,
                "losing_trades": result.losing_trades,
                "avg_win": result.avg_win,
                "avg_loss": result.avg_loss,
                "largest_win": result.largest_win,
                "largest_loss": result.largest_loss,
                "total_pnl": result.total_pnl,
                "volatility": result.volatility,
                "calmar_ratio": result.calmar_ratio
            })
        
        return {
            "status": "success",
            "scenarios": scenario_results,
            "summary": {
                "total_scenarios": len(scenario_results),
                "average_return": float(np.mean([r["total_return"] for r in scenario_results])) if scenario_results else 0.0,
                "average_sharpe": float(np.mean([r["sharpe_ratio"] for r in scenario_results])) if scenario_results else 0.0,
                "worst_drawdown": float(min([r["max_drawdown"] for r in scenario_results])) if scenario_results else 0.0
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Robustness test failed: {str(e)}"
        )


@app.post("/api/scenario/stress-test")
async def run_stress_test_endpoint(request: StressTestRequest):
    """
    Run stress test under extreme market conditions.
    
    Tests strategy resilience under severe market stress.
    """
    if not SCENARIO_SIMULATOR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Scenario simulator not available"
        )
    
    try:
        # Load price data
        data_dir = Path("data/raw")
        price_data = None
        
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
            try:
                df = pd.read_csv(latest_file)
                if 'close' in df.columns:
                    price_data = df
            except:
                pass
        
        if price_data is None or len(price_data) < 10:
            import numpy as np
            dates = pd.date_range(end=datetime.now(), periods=200, freq='D')
            base_price = 5000.0
            synthetic_prices = base_price * (1 + np.random.randn(200).cumsum() * 0.01)
            price_data = pd.DataFrame({
                'close': synthetic_prices,
                'open': synthetic_prices,
                'high': synthetic_prices * 1.01,
                'low': synthetic_prices * 0.99,
                'volume': np.random.randint(1000, 10000, 200)
            })
        
        simulator = ScenarioSimulator(price_data, initial_capital=100000.0)
        
        # Map scenarios
        regime_map = {
            "crash": MarketRegime.CRASH,
            "flash_crash": MarketRegime.FLASH_CRASH,
            "high_volatility": MarketRegime.HIGH_VOLATILITY,
            "gap_event": MarketRegime.GAP_EVENT
        }
        
        # Create stress test scenarios
        stress_scenarios = [
            (name, regime_map[name], request.intensity)
            for name in request.scenarios
            if name in regime_map
        ]
        
        # Run stress tests
        stress_results = simulator.run_stress_test(stress_scenarios)
        
        # Convert to dict
        results = []
        for result in stress_results:
            results.append({
                "scenario_name": result.scenario_name,
                "max_drawdown": result.max_drawdown,
                "recovery_time": result.recovery_time,
                "worst_case_loss": result.worst_case_loss,
                "survived": result.survived,
                "equity_at_min": result.equity_at_min,
                "details": result.details
            })
        
        return {
            "status": "success",
            "stress_tests": results,
            "summary": {
                "total_tests": len(results),
                "survived_count": sum(1 for r in results if r["survived"]),
                "worst_drawdown": min([r["max_drawdown"] for r in results]) if results else 0.0,
                "avg_recovery_time": float(np.mean([r["recovery_time"] for r in results])) if results else 0.0
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Stress test failed: {str(e)}"
        )


@app.post("/api/scenario/parameter-sensitivity")
async def run_parameter_sensitivity_endpoint(request: ParameterSensitivityRequest):
    """
    Analyze parameter sensitivity.
    
    Tests how strategy performance varies with different parameter values.
    """
    if not SCENARIO_SIMULATOR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Scenario simulator not available"
        )
    
    try:
        # Load price data
        data_dir = Path("data/raw")
        price_data = None
        
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
            try:
                df = pd.read_csv(latest_file)
                if 'close' in df.columns:
                    price_data = df
            except:
                pass
        
        if price_data is None or len(price_data) < 10:
            import numpy as np
            dates = pd.date_range(end=datetime.now(), periods=200, freq='D')
            base_price = 5000.0
            synthetic_prices = base_price * (1 + np.random.randn(200).cumsum() * 0.01)
            price_data = pd.DataFrame({
                'close': synthetic_prices,
                'open': synthetic_prices,
                'high': synthetic_prices * 1.01,
                'low': synthetic_prices * 0.99,
                'volume': np.random.randint(1000, 10000, 200)
            })
        
        simulator = ScenarioSimulator(price_data, initial_capital=100000.0)
        
        # Map regime
        regime_map = {
            "normal": MarketRegime.NORMAL,
            "trending_up": MarketRegime.TRENDING_UP,
            "high_volatility": MarketRegime.HIGH_VOLATILITY
        }
        regime = regime_map.get(request.regime, MarketRegime.NORMAL)
        
        # Simple backtest function (placeholder)
        def simple_backtest(data, **params):
            returns = data['close'].pct_change().fillna(0)
            equity_curve = 100000.0 * (1 + returns).cumprod()
            trades = []
            for i in range(10, len(data), 20):
                entry = data['close'].iloc[i]
                exit = data['close'].iloc[min(i+10, len(data)-1)]
                trades.append((exit - entry) / entry)
            return {
                'equity_curve': equity_curve.tolist(),
                'trades': trades
            }
        
        # Run sensitivity analysis
        sensitivity_result = simulator.parameter_sensitivity_analysis(
            parameter_name=request.parameter_name,
            parameter_values=request.parameter_values,
            backtest_func=simple_backtest,
            base_params=request.base_parameters,
            regime=regime
        )
        
        return {
            "status": "success",
            "parameter_name": sensitivity_result.parameter_name,
            "parameter_values": sensitivity_result.parameter_values,
            "performance_metrics": sensitivity_result.performance_metrics,
            "optimal_value": sensitivity_result.optimal_value,
            "sensitivity_score": sensitivity_result.sensitivity_score,
            "recommendations": sensitivity_result.recommendations
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Parameter sensitivity analysis failed: {str(e)}"
        )


@app.post("/api/analysis/markov/run")
async def run_markov_regime_analysis(request: MarkovAnalysisRequest):
    """
    Execute offline Markov regime analysis using historical market data.

    Returns transition probabilities, stationary distribution, and regime statistics.
    """

    def resolve_instrument_and_timeframes() -> Dict[str, Any]:
        instrument = request.instrument
        timeframes = request.timeframes

        config_data: Dict[str, Any] = {}
        config_path = request.config_path
        if config_path:
            candidate_path = Path(config_path)
            if not candidate_path.is_absolute():
                candidate_path = project_root / candidate_path
            if candidate_path.exists():
                try:
                    with candidate_path.open("r", encoding="utf-8") as f:
                        config_data = yaml.safe_load(f) or {}
                except Exception:
                    config_data = {}

        env_config = config_data.get("environment", {})
        if not instrument:
            instrument = env_config.get("instrument")
        if not timeframes:
            tf = env_config.get("timeframes")
            if isinstance(tf, (list, tuple)):
                try:
                    timeframes = [int(x) for x in tf]
                except Exception:
                    timeframes = None

        instrument = instrument or "ES"
        if not timeframes:
            timeframes = [1, 5, 15]
        else:
            timeframes = [int(x) for x in timeframes]

        return {
            "instrument": instrument,
            "timeframes": timeframes,
            "config_path": str(config_path) if config_path else None,
        }

    async def execute_analysis():
        defaults = resolve_instrument_and_timeframes()
        regime_config = RegimeConfig(
            instrument=defaults["instrument"],
            timeframes=defaults["timeframes"],
            start_date=request.start_date,
            end_date=request.end_date,
            num_regimes=request.num_regimes,
            rolling_vol_window=request.rolling_vol_window,
            volume_zscore_window=request.volume_zscore_window,
            min_samples=request.min_samples,
        )

        if request.save_report:
            output_path = request.output_path or "reports/markov_regime_report.json"
            output_path_obj = Path(output_path)
            if not output_path_obj.is_absolute():
                output_path_obj = project_root / output_path_obj

            def _run_and_save():
                result = run_and_save_report(output_path_obj, config=regime_config)
                return result, output_path_obj

            result, saved_path = await asyncio.to_thread(_run_and_save)
        else:
            analyzer = MarkovRegimeAnalyzer(config=regime_config)

            def _run():
                return analyzer.run()

            result = await asyncio.to_thread(_run)
            saved_path = None

        return result, regime_config, defaults, saved_path

    start_time = time.perf_counter()
    try:
        result, regime_config, defaults, saved_path = await execute_analysis()

        transition_df = result.transition_matrix.round(6)
        transition_payload = {
            "index": [str(idx) for idx in transition_df.index],
            "columns": [str(col) for col in transition_df.columns],
            "values": transition_df.values.tolist(),
        }

        stationary_payload = [
            {"regime": str(idx), "probability": round(float(value), 6)}
            for idx, value in result.stationary_distribution.items()
        ]

        summary_df = result.regime_summary.reset_index().copy()
        if "index" in summary_df.columns and "regime_id" not in summary_df.columns:
            summary_df = summary_df.rename(columns={"index": "regime_id"})
        summary_df = summary_df.round(6)
        regime_summary = summary_df.to_dict(orient="records")

        preview_columns = [
            col
            for col in ["timestamp", "regime_id", "return", "rolling_vol", "volume_zscore"]
            if col in result.clustered_data.columns
        ]
        clustered_preview = []
        if preview_columns:
            preview_df = result.clustered_data[preview_columns].tail(10).copy()
            if "timestamp" in preview_df.columns:
                preview_df["timestamp"] = preview_df["timestamp"].astype(str)
            clustered_preview = preview_df.to_dict(orient="records")

        payload = {
            "success": True,
            "transition_matrix": transition_payload,
            "stationary_distribution": stationary_payload,
            "regime_summary": regime_summary,
            "cluster_preview": clustered_preview,
            "metadata": {
                "instrument": regime_config.instrument,
                "timeframes": list(regime_config.timeframes),
                "num_regimes": regime_config.num_regimes,
                "rolling_vol_window": regime_config.rolling_vol_window,
                "volume_zscore_window": regime_config.volume_zscore_window,
                "min_samples": regime_config.min_samples,
                "start_date": request.start_date,
                "end_date": request.end_date,
                "config_path": defaults.get("config_path"),
                "sample_count": int(result.clustered_data.shape[0]),
            },
            "saved_report": str(saved_path.as_posix()) if saved_path else None,
            "runtime_ms": round((time.perf_counter() - start_time) * 1000, 2),
        }

        return JSONResponse(content=payload)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Markov analysis failed: {str(e)}",
        )


@app.get("/api/ai/capabilities")
async def api_list_capabilities(tab: Optional[str] = None):
    """List capability metadata for AI explanations."""
    capabilities = list_capabilities_by_tab(tab)
    return {
        "count": len(capabilities),
        "capabilities": [asdict(cap) for cap in capabilities],
    }


@app.get("/api/ai/capabilities/{capability_id}")
async def api_get_capability(capability_id: str):
    """Return metadata for a specific capability."""
    capability = get_capability(capability_id)
    if not capability:
        raise HTTPException(status_code=404, detail="Capability not found")
    return asdict(capability)


@app.post("/api/ai/capabilities/generate")
async def api_generate_capability(request: CapabilityGenerationRequest):
    """Generate or fetch cached AI analysis for a capability."""
    result = generate_analysis(
        capability_id=request.capability_id,
        locale=request.locale or "en-US",
        user_id=request.user_id,
        context=request.context or {},
        force_refresh=request.force_refresh,
        provider_hint=request.provider_hint,
        model=request.model,
    )
    return {
        "status": "ok",
        "cached": result["cached"],
        "analysis": result["data"],
    }


@app.post("/api/ai/capabilities/feedback")
async def api_capability_feedback(request: CapabilityFeedbackRequest):
    """Record user feedback on generated analyses."""
    record_feedback(
        capability_id=request.capability_id,
        locale=request.locale or "en-US",
        user_id=request.user_id,
        feedback={
            "rating": request.rating,
            "comment": request.comment,
            "source": request.source,
        },
    )
    return {"status": "ok"}


# ============================================================================
# Deep Research API Endpoints
# ============================================================================

@app.get("/api/research/list")
async def list_research(
    search: Optional[str] = None,
    source: Optional[str] = None,
    category: Optional[str] = None,
    status: Optional[str] = None,
    dateRange: Optional[str] = None,
    max_results: int = 50
):
    """List research items from knowledge base"""
    system = get_deep_research_system()
    if not system:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Deep Research System not available"}
        )
    
    try:
        # Query knowledge base
        results = system.query_knowledge_base(
            query=search,
            source_type=source if source != "all" else None,
            limit=max_results
        )
        
        # Format for frontend
        research_items = []
        for item in results:
            research_items.append({
                "id": item.get("item_id", "unknown"),
                "title": item.get("title", "Unknown"),
                "source_type": item.get("source_type", "unknown"),
                "source_url": item.get("url", ""),
                "date_published": item.get("date", datetime.now()).isoformat() if item.get("date") else None,
                "date_analyzed": item.get("ingested_at", datetime.now()).isoformat() if item.get("ingested_at") else None,
                "category": item.get("categories", [""])[0] if item.get("categories") else "unknown",
                "impact_score": item.get("impact_score", {}).get("overall_score", 0.5) if isinstance(item.get("impact_score"), dict) else item.get("impact_score", 0.5),
                "relevance_score": item.get("relevance_score", 0.5),
                "status": item.get("status", "not_reviewed"),
                "authors": item.get("authors", []),
                "abstract": item.get("abstract", "")[:200] + "..." if item.get("abstract") else "",
                "executive_summary": item.get("executive_summary", ""),
                "technical_details": item.get("technical_details"),
                "recommendations": item.get("recommendations", []),
                "implementation_plan_path": item.get("implementation_plan_path")
            })
        
        return JSONResponse(content={
            "status": "success",
            "research_items": research_items,
            "total": len(research_items)
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.get("/api/research/{item_id}")
async def get_research_detail(item_id: str):
    """Get detailed research item information"""
    system = get_deep_research_system()
    if not system:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Deep Research System not available"}
        )
    
    try:
        # Get from knowledge base
        results = system.query_knowledge_base(query=None, limit=1000)
        item = next((r for r in results if r.get("item_id") == item_id), None)
        
        if not item:
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": "Research item not found"}
            )
        
        # Format for frontend
        research = {
            "id": item.get("item_id", "unknown"),
            "title": item.get("title", "Unknown"),
            "source_type": item.get("source_type", "unknown"),
            "source_url": item.get("url", ""),
            "date_published": item.get("date", datetime.now()).isoformat() if item.get("date") else None,
            "date_analyzed": item.get("ingested_at", datetime.now()).isoformat() if item.get("ingested_at") else None,
            "category": item.get("categories", [""])[0] if item.get("categories") else "unknown",
            "impact_score": item.get("impact_score", {}).get("overall_score", 0.5) if isinstance(item.get("impact_score"), dict) else item.get("impact_score", 0.5),
            "relevance_score": item.get("relevance_score", 0.5),
            "status": item.get("status", "not_reviewed"),
            "authors": item.get("authors", []),
            "abstract": item.get("abstract", ""),
            "executive_summary": item.get("executive_summary", ""),
            "technical_details": item.get("technical_details"),
            "recommendations": item.get("recommendations", []),
            "implementation_plan_path": item.get("implementation_plan_path"),
            "synthesis": item.get("synthesis"),
            "architecture_comparison": item.get("architecture_comparison")
        }
        
        return JSONResponse(content={
            "status": "success",
            "research": research
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.post("/api/research/search")
async def search_research(request: Dict[str, Any]):
    """Search for research across sources"""
    system = get_deep_research_system()
    if not system:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Deep Research System not available"}
        )
    
    try:
        query = request.get("query", "")
        sources = request.get("sources", [])
        max_results = request.get("max_results", 20)
        
        # Search across sources
        results = system.search(
            query=query,
            sources=sources if sources else None,
            max_results=max_results
        )
        
        # Format results
        research_items = []
        for item in results:
            research_items.append({
                "id": f"{item.source_type}:{item.source_id}",
                "title": item.title,
                "source_type": item.source_type,
                "source_url": item.url if hasattr(item, 'url') else "",
                "date_published": item.date.isoformat() if item.date else None,
                "abstract": (item.abstract or item.content or "")[:200] + "..." if (item.abstract or item.content) else "",
                "category": "unknown",
                "impact_score": 0.5,
                "status": "not_reviewed"
            })
        
        return JSONResponse(content={
            "status": "success",
            "research_items": research_items,
            "total": len(research_items)
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.post("/api/research/fetch")
async def fetch_research(request: Dict[str, Any]):
    """Fetch and analyze a specific research item"""
    system = get_deep_research_system()
    if not system:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Deep Research System not available"}
        )
    
    try:
        source_type = request.get("source_type")
        source_id = request.get("source_id")
        store = request.get("store", True)
        generate_plan = request.get("generate_plan", False)
        generate_doc = request.get("generate_doc", False)
        
        if not source_type or not source_id:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "source_type and source_id required"}
            )
        
        # Fetch item
        item = system.fetch(source_type, source_id, store_result=store)
        
        if not item:
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": "Research item not found"}
            )
        
        # Analyze if requested
        analysis = None
        if store:
            analysis = system.analyze_and_generate(
                item=item,
                generate_plan=generate_plan,
                generate_doc=generate_doc,
                check_alert=True,
                deep_analysis=True,
                compare_with_codebase=True
            )
        
        return JSONResponse(content={
            "status": "success",
            "item": {
                "id": f"{item.source_type}:{item.source_id}",
                "title": item.title,
                "source_type": item.source_type,
                "source_id": item.source_id,
                "abstract": item.abstract or "",
                "date": item.date.isoformat() if item.date else None
            },
            "analysis": analysis
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.post("/api/research/feedback")
async def add_research_feedback(request: Dict[str, Any]):
    """Add feedback for a research item"""
    system = get_deep_research_system()
    if not system:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Deep Research System not available"}
        )
    
    try:
        item_id = request.get("item_id")
        feedback_type = request.get("feedback_type")
        metadata = request.get("metadata", {})
        notes = request.get("notes")
        
        if not item_id or not feedback_type:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "item_id and feedback_type required"}
            )
        
        system.add_feedback(
            item_id=item_id,
            feedback_type=feedback_type,
            metadata=metadata,
            notes=notes
        )
        
        return JSONResponse(content={
            "status": "success",
            "message": "Feedback added successfully"
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.get("/api/research/stats")
async def get_research_stats():
    """Get Deep Research system statistics"""
    system = get_deep_research_system()
    if not system:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Deep Research System not available"}
        )
    
    try:
        stats = system.get_statistics()
        feedback_stats = system.get_feedback_statistics()
        
        return JSONResponse(content={
            "status": "success",
            "stats": {
                "total_items": stats.get("total_items", 0),
                "by_source": stats.get("by_source", {}),
                "vector_store_items": stats.get("vector_store_items", 0),
                "code_snippets": stats.get("code_snippets", 0),
                "citation_graph": stats.get("citation_graph", {}),
                "feedback": feedback_stats
            }
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.post("/api/research/semantic-search")
async def semantic_search_research(request: Dict[str, Any]):
    """Semantic search over research content"""
    system = get_deep_research_system()
    if not system:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Deep Research System not available"}
        )
    
    try:
        query = request.get("query", "")
        top_k = request.get("top_k", 10)
        
        if not query:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "query required"}
            )
        
        results = system.semantic_search(query=query, top_k=top_k)
        
        return JSONResponse(content={
            "status": "success",
            "results": results
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


if __name__ == "__main__":
    uvicorn.run("src.api_server:app", host="0.0.0.0", port=8200, log_level="info", reload=False)


