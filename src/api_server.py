"""
FastAPI Backend Server for NT8 RL Trading System UI

Provides REST API and WebSocket endpoints to control all system operations.
"""

import asyncio
import json
import os
import subprocess
import sys
import yaml
import threading
import time
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import uvicorn
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


app = FastAPI(title="NT8 RL Trading System API")

# CORS middleware
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


# Pydantic models
class SetupCheckResponse(BaseModel):
    venv_exists: bool
    dependencies_installed: bool
    data_directory_exists: bool
    config_exists: bool
    ready: bool
    issues: List[str]


class TrainingRequest(BaseModel):
    device: str = "cpu"
    total_timesteps: Optional[int] = None
    config_path: str = "configs/train_config.yaml"
    reasoning_model: Optional[str] = None
    checkpoint_path: Optional[str] = None  # Resume from checkpoint


class BacktestRequest(BaseModel):
    model_path: str
    episodes: int = 20
    config_path: str = "configs/train_config.yaml"


class LiveTradingRequest(BaseModel):
    model_path: str
    paper_trading: bool = True
    config_path: str = "configs/train_config.yaml"


class StatusResponse(BaseModel):
    status: str
    message: str
    details: Optional[Dict] = None


class SettingsRequest(BaseModel):
    nt8_data_path: Optional[str] = None
    performance_mode: Optional[str] = None  # "quiet" or "performance"
    auto_retrain_enabled: Optional[bool] = None  # Enable/disable auto-retrain


# Helper function to send WebSocket messages
async def broadcast_message(message: Dict):
    """Broadcast message to all connected WebSocket clients"""
    disconnected = []
    for ws in websocket_connections:
        try:
            await ws.send_json(message)
        except:
            disconnected.append(ws)
    
    for ws in disconnected:
        websocket_connections.remove(ws)


# Routes
@app.get("/")
async def root():
    return {"message": "NT8 RL Trading System API", "version": "1.0.0"}


@app.get("/api/setup/check", response_model=SetupCheckResponse)
async def check_setup():
    """Check if environment is properly set up"""
    issues = []
    
    # Check virtual environment - multiple methods:
    # 1. Check if we're already running in a venv (sys.prefix != sys.base_prefix)
    #    This works for venv, virtualenv, uv, conda, etc.
    # 2. Check for .venv or venv directories
    in_venv = sys.prefix != sys.base_prefix
    venv_dir_exists = Path(".venv").exists() or Path("venv").exists()
    venv_exists = in_venv or venv_dir_exists
    
    # Check dependencies (simplified - check if key packages are importable)
    dependencies_installed = True
    try:
        import torch
        import gymnasium
        import stable_baselines3
    except ImportError:
        dependencies_installed = False
        issues.append("Required dependencies not installed. Run: pip install -r requirements.txt OR uv pip install -r requirements.txt")
    
    # Only warn about venv if dependencies aren't installed
    # (if dependencies work, environment is fine regardless of venv detection)
    if not venv_exists and not dependencies_installed:
        issues.append("Virtual environment not detected. Recommended: python -m venv .venv OR uv venv")
    
    # Check data directory
    data_dir = Path("data/raw")
    data_directory_exists = data_dir.exists()
    if not data_directory_exists:
        issues.append(f"Data directory not found: {data_dir}")
    
    # Check config
    config_path = Path("configs/train_config.yaml")
    config_exists = config_path.exists()
    if not config_exists:
        issues.append(f"Config file not found: {config_path}")
    
    ready = len(issues) == 0
    
    return SetupCheckResponse(
        venv_exists=venv_exists,
        dependencies_installed=dependencies_installed,
        data_directory_exists=data_directory_exists,
        config_exists=config_exists,
        ready=ready,
        issues=issues
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
        
        # Find Python executable
        venv_python = Path(".venv/Scripts/python.exe") if os.name == "nt" else Path(".venv/bin/python")
        if not venv_python.exists():
            venv_python = Path("venv/Scripts/python.exe") if os.name == "nt" else Path("venv/bin/python")
        if not venv_python.exists():
            venv_python = sys.executable
        
        try:
            proc = subprocess.Popen(
                [str(venv_python), "-m", "pip", "install", "-r", "requirements.txt"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            for line in proc.stdout:
                await broadcast_message({
                    "type": "setup",
                    "message": line.strip(),
                    "progress": None
                })
            
            proc.wait()
            
            if proc.returncode == 0:
                await broadcast_message({
                    "type": "setup",
                    "status": "success",
                    "message": "Dependencies installed successfully",
                    "progress": 100
                })
            else:
                error = proc.stderr.read()
                await broadcast_message({
                    "type": "setup",
                    "status": "error",
                    "message": f"Installation failed: {error}",
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
        print(f"Error listing configs: {e}")
        import traceback
        traceback.print_exc()
        return {"configs": [], "error": str(e)}

@app.get("/api/system/cuda-status")
async def get_cuda_status():
    """Check CUDA/GPU availability"""
    try:
        import torch
        
        # Debug logging
        print("Checking CUDA availability...")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        cuda_available = torch.cuda.is_available()
        
        result = {
            "cuda_available": cuda_available,
            "device": "cpu"
        }
        
        if cuda_available:
            try:
                result["device"] = "cuda"
                result["gpu_name"] = torch.cuda.get_device_name(0)
                result["cuda_version"] = torch.version.cuda
                result["device_count"] = torch.cuda.device_count()
                print(f"GPU detected: {result['gpu_name']} (CUDA {result['cuda_version']})")
            except Exception as gpu_error:
                print(f"Error getting GPU info: {gpu_error}")
                result["gpu_name"] = "Unknown"
                result["cuda_version"] = None
                result["device_count"] = 0
        else:
            result["gpu_name"] = None
            result["cuda_version"] = None
            result["device_count"] = 0
            print("CUDA not available - PyTorch may not be compiled with CUDA support")
        
        print(f"Returning result: {result}")
        return result
    except ImportError as e:
        print(f"PyTorch import error: {e}")
        return {
            "cuda_available": False,
            "device": "cpu",
            "gpu_name": None,
            "cuda_version": None,
            "device_count": 0,
            "error": "PyTorch not installed"
        }
    except Exception as e:
        print(f"CUDA check error: {e}")
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
    
    This queues retraining to avoid interrupting existing training.
    """
    global active_processes, main_event_loop
    
    print(f"\n📁 New data detected: {len(files)} file(s)")
    
    # Check if training is already running
    if "training" in active_processes:
        print("⚠️  Training already in progress. New data detected but will not retrain yet.")
        print("   Retraining will be queued after current training completes.")
        # TODO: Implement queue system
        return
    
    # TODO: Trigger retraining here
    print("🚀 Would trigger retraining here (not implemented yet)")
    
    # Broadcast message to UI
    if main_event_loop:
        try:
            asyncio.run_coroutine_threadsafe(
                broadcast_message({
                    "type": "auto_retrain",
                    "status": "triggered",
                    "message": f"New data detected: {len(files)} file(s). Retraining recommended.",
                    "files": [str(f) for f in files]
                }),
                main_event_loop
            )
        except:
            pass


@app.on_event("startup")
async def startup_event():
    """Store main event loop for use in threads"""
    global main_event_loop, auto_retrain_monitor
    
    main_event_loop = asyncio.get_event_loop()
    
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
                    print(f"✅ Auto-retrain monitoring started on: {nt8_path}")
        except Exception as e:
            print(f"⚠️  Could not initialize auto-retrain monitor: {e}")
    
    print("✅ API server initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop auto-retrain monitor on shutdown"""
    global auto_retrain_monitor
    if auto_retrain_monitor:
        auto_retrain_monitor.stop()


@app.post("/api/training/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start model training"""
    if "training" in active_systems:
        system = active_systems["training"]
        thread_alive = system.get("thread") and system["thread"].is_alive()
        print(f"⚠️  Training start requested but training already in active_systems")
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
    print(f"{'='*60}\n")
    
    async def _train():
        print(f"[_train] Starting async training function")
        await broadcast_message({
            "type": "training",
            "status": "starting",
            "message": "Initializing training..."
        })
        print(f"[_train] Broadcast message sent")
        
        # Load config
        print(f"[_train] Loading config from: {request.config_path}")
        try:
            with open(request.config_path, "r") as f:
                config = yaml.safe_load(f)
            print(f"[_train] Config loaded successfully")
        except Exception as e:
            print(f"[_train] ❌ ERROR loading config: {e}")
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
        
        try:
            # Log checkpoint path being used
            checkpoint_path_to_use = None
            if request.checkpoint_path:
                print(f"🔍 Attempting to resume from checkpoint: {request.checkpoint_path}")
                from pathlib import Path
                checkpoint_test = Path(str(request.checkpoint_path).replace('\\', '/'))
                print(f"   Normalized path: {checkpoint_test}")
                print(f"   Path exists: {checkpoint_test.exists()}")
                if checkpoint_test.exists():
                    checkpoint_path_to_use = str(checkpoint_test.resolve())
                    print(f"   ✅ Using checkpoint: {checkpoint_path_to_use}")
                else:
                    # Try relative to project root
                    project_root = Path(__file__).parent.parent
                    relative_checkpoint = project_root / str(request.checkpoint_path).replace('\\', '/').lstrip('/')
                    print(f"   Trying relative path: {relative_checkpoint}")
                    if relative_checkpoint.exists():
                        checkpoint_path_to_use = str(relative_checkpoint.resolve())
                        print(f"   ✅ Using checkpoint: {checkpoint_path_to_use}")
                    else:
                        print(f"   ⚠️  WARNING: Checkpoint not found! Will start fresh training.")
                        print(f"      Tried: {checkpoint_test}")
                        print(f"      Tried: {relative_checkpoint}")
                        await broadcast_message({
                            "type": "training",
                            "status": "warning",
                            "message": f"Checkpoint not found: {request.checkpoint_path}. Starting fresh training."
                        })
            
            # Create trainer and train (with optional checkpoint for resume)
            print(f"[_train] 🚀 Creating Trainer with checkpoint: {checkpoint_path_to_use if checkpoint_path_to_use else 'None (fresh start)'}")
            print(f"[_train]   This may take a moment (loading data, creating environment...)")
            try:
                trainer = Trainer(config, checkpoint_path=checkpoint_path_to_use)
                print(f"[_train] ✅ Trainer created successfully")
            except Exception as e:
                import traceback
                print(f"[_train] ❌ ERROR creating Trainer: {e}")
                print(f"[_train]   Traceback:\n{traceback.format_exc()}")
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
                    print(f"🏋️ Training worker thread started (ID: {threading.current_thread().ident})")
                    print(f"   About to call trainer.train()...")
                    trainer.train()
                    print(f"✅ Training completed successfully in worker thread")
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
                    print(f"❌ ERROR in training worker thread: {str(e)}")
                    print(f"   Traceback:\n{error_trace}")
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
                        print(f"❌ Failed to broadcast training error: {broadcast_error}")
                        print(f"   Original error: {str(e)}")
                    
                    # Mark training as completed with error
                    if "training" in active_systems:
                        active_systems["training"]["completed"] = True
                        active_systems["training"]["error"] = str(e)
            
            thread = threading.Thread(target=train_worker)
            thread.daemon = True
            thread.start()
            
            # Update the placeholder entry with actual trainer and thread
            active_systems["training"] = {
                "trainer": trainer,
                "thread": thread,
                "completed": False,
                "status": "running"  # Update status to running
            }
            print(f"✅ Training thread started, trainer created successfully")
            print(f"   Thread ID: {thread.ident}")
            print(f"   Thread alive: {thread.is_alive()}")
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"❌ ERROR: Failed to start training: {str(e)}")
            print(f"   Traceback:\n{error_trace}")
            await broadcast_message({
                "type": "training",
                "status": "error",
                "message": f"Failed to start training: {str(e)}"
            })
    
    # Add placeholder entry immediately so status endpoint knows training is starting
    # This prevents race condition where status is checked before _train() completes
    import time
    active_systems["training"] = {
        "trainer": None,  # Will be set when trainer is created
        "thread": None,   # Will be set when actual training thread starts
        "completed": False,
        "status": "starting",  # Mark as starting
        "start_time": time.time()  # Track when training started to detect hangs
    }
    
    background_tasks.add_task(_train)
    print(f"📤 Training start request queued in background tasks")
    print(f"   Placeholder entry added to active_systems")
    return {"status": "started", "message": "Training started"}


@app.get("/api/training/status")
async def training_status():
    """Get training status with detailed metrics"""
    if "training" not in active_systems:
        return {"status": "idle", "message": "No training in progress"}
    
    # Check if thread is still alive and if training completed
    system = active_systems["training"]
    
    # Check if training is still initializing (trainer not created yet)
    if system.get("status") == "starting" and system.get("trainer") is None:
        import time
        start_time = system.get("start_time", 0)
        elapsed = time.time() - start_time if start_time > 0 else 0
        
        # If initialization takes more than 60 seconds, something is wrong
        if elapsed > 60:
            print(f"⚠️  WARNING: Training initialization taking too long ({elapsed:.1f}s)")
            print(f"   This may indicate an error during trainer creation")
            print(f"   Check backend logs for errors")
            return {
                "status": "starting",
                "message": f"Initializing training... (taking longer than expected: {elapsed:.0f}s)",
                "metrics": {},
                "warning": "Initialization taking longer than expected. Check backend console for errors."
            }
        
        return {
            "status": "starting",
            "message": "Initializing training...",
            "metrics": {}
        }
    
    # Check if thread exists and is alive
    thread = system.get("thread")
    if thread is None or not (hasattr(thread, 'is_alive') and thread.is_alive()):
        # Thread doesn't exist or is dead, but trainer might not be created yet
        if system.get("status") == "starting":
            import time
            start_time = system.get("start_time", 0)
            elapsed = time.time() - start_time if start_time > 0 else 0
            
            # If initialization takes more than 60 seconds, something is wrong
            if elapsed > 60:
                print(f"⚠️  WARNING: Training initialization timeout ({elapsed:.1f}s)")
                print(f"   Thread never started. Check backend logs for Trainer creation errors.")
                # Clean up stale entry
                active_systems.pop("training", None)
                return {
                    "status": "error",
                    "message": f"Training initialization failed (timeout after {elapsed:.0f}s). Check backend console for errors.",
                    "metrics": {}
                }
            
            return {
                "status": "starting",
                "message": "Initializing training...",
                "metrics": {}
            }
        # Otherwise, training is dead - clean up
        active_systems.pop("training", None)
        return {"status": "idle", "message": "Training stopped"}
    
    # Check if completed flag is set
    if system.get("completed", False):
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
    
    # Check if thread is still alive
    if not system["thread"].is_alive():
        # Thread finished but completed flag not set - might have crashed
        active_systems.pop("training", None)
        return {"status": "completed", "message": "Training finished"}
    
    # Training is running - get current metrics
    trainer = system.get("trainer")
    metrics = {}
    if trainer:
        # Calculate recent metrics without numpy
        recent_rewards = trainer.episode_rewards[-10:] if len(trainer.episode_rewards) >= 10 else trainer.episode_rewards
        mean_reward_10 = float(sum(recent_rewards) / len(recent_rewards)) if recent_rewards else 0.0
        
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
        
        # Use current episode metrics if we have an active episode (length > 0 and not stuck)
        # Otherwise use the latest completed episode metrics
        has_active_episode = current_episode_length > 0 and not is_stuck
        display_reward = current_episode_reward if has_active_episode else latest_reward
        display_length = current_episode_length if has_active_episode else latest_length
        
        # Calculate mean episode length without numpy
        mean_episode_length = float(sum(trainer.episode_lengths) / len(trainer.episode_lengths)) if trainer.episode_lengths else 0.0
        
        # Current episode number (completed episodes + 1 if there's an active episode)
        # If stuck, still show episode 1 to indicate training is happening
        current_episode_number = trainer.episode + (1 if (has_active_episode or is_stuck) else 0)
        
        metrics = {
            "episode": current_episode_number,  # Show current episode (completed + in-progress)
            "completed_episodes": trainer.episode,  # Number of fully completed episodes
            "timestep": trainer.timestep,
            "total_timesteps": trainer.total_timesteps,
            "progress_percent": float(trainer.timestep / trainer.total_timesteps * 100) if trainer.total_timesteps > 0 else 0.0,
            "latest_reward": display_reward,  # Show current or latest completed reward
            "current_episode_reward": current_episode_reward,  # Current in-progress reward
            "mean_reward_10": mean_reward_10,
            "latest_episode_length": display_length,  # Show current or latest completed length
            "current_episode_length": current_episode_length,  # Current in-progress length
            "mean_episode_length": mean_episode_length,
            "total_episodes": len(trainer.episode_rewards),
        }
        
        # Get latest training metrics if available (from last update)
        if hasattr(trainer, 'last_update_metrics') and trainer.last_update_metrics:
            metrics["training_metrics"] = trainer.last_update_metrics
    
    return {
        "status": "running", 
        "message": "Training in progress",
        "metrics": metrics
    }


@app.post("/api/training/stop")
async def stop_training():
    """Stop training"""
    if "training" not in active_systems:
        raise HTTPException(status_code=400, detail="No training in progress")
    
    # Note: Actual stopping would require more complex coordination
    active_systems.pop("training", None)
    
    await broadcast_message({
        "type": "training",
        "status": "stopped",
        "message": "Training stopped"
    })
    
    return {"status": "stopped", "message": "Training stopped"}


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
            
            results = backtester.run(episodes=request.episodes)
            
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
    
    # Get trained RL models
    trained_models = []
    checkpoints = []
    if models_dir.exists():
        for file in models_dir.glob("*.pt"):
            stat = file.stat()
            model_info = {
                "name": file.name,
                "path": str(file),
                "type": "trained",
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
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
    if "trading" in active_processes:
        raise HTTPException(status_code=400, detail="Trading already running")
    
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
            config_path = Path("configs/train_config.yaml")
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


@app.get("/api/monitoring/performance")
async def get_performance():
    """Get current performance metrics"""
    try:
        monitor = PerformanceMonitor()
        metrics = monitor.get_current_metrics()
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        return {"status": "error", "message": str(e)}


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
    
    return {"status": "success", **settings}


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
            print(f"🔄 Performance mode updated to: {request.performance_mode}")
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
    
    # Save
    try:
        with open(settings_file, "w") as f:
            json.dump(settings, f, indent=2)
        return {"status": "success", "message": "Settings saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {str(e)}")


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
        websocket_connections.remove(websocket)


if __name__ == "__main__":
    uvicorn.run("src.api_server:app", host="0.0.0.0", port=8200, log_level="info", reload=False)

