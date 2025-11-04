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
import numpy as np
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
    turbo_training_mode: Optional[bool] = None  # Enable turbo mode (max GPU usage)
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
    
    This automatically triggers retraining when new data is available.
    """
    global active_processes, active_systems, main_event_loop
    
    print(f"\n📁 New data detected: {len(files)} file(s)")
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
            print("⚠️  Training already in progress. New data detected but will not retrain yet.")
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
    
    print("🚀 Auto-triggering retraining with new data...")
    
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
    config_path = "configs/train_config.yaml"
    if not Path(config_path).exists():
        config_path = "configs/train_config_gpu_optimized.yaml"
    
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
            
            print("✅ Auto-retraining triggered successfully")
            print("   Training should start shortly...")
        except Exception as e:
            print(f"❌ Error triggering auto-retraining: {e}")
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
        print("⚠️  Main event loop not available, cannot trigger training automatically")
        print("   Please start training manually from the UI or use the API endpoint")


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
        # Force immediate output to verify function is called
        print(f"\n{'='*80}")
        print(f"[_train] ✅✅✅ ASYNC TRAINING FUNCTION CALLED ✅✅✅")
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
            print(f"[_train] ✅ Broadcast message sent successfully")
            sys.stdout.flush()
        except Exception as e:
            print(f"[_train] ❌ ERROR sending broadcast: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
        
        # Load config
        print(f"[_train] 📄 Loading config from: {request.config_path}")
        try:
            with open(request.config_path, "r") as f:
                config = yaml.safe_load(f)
            print(f"[_train] ✅ Config loaded successfully")
            print(f"[_train]   Instrument: {config.get('environment', {}).get('instrument', 'N/A')}")
            print(f"[_train]   Timeframes: {config.get('environment', {}).get('timeframes', 'N/A')}")
        except Exception as e:
            print(f"[_train] ❌ ERROR loading config: {e}")
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
                
                trainer = Trainer(config, checkpoint_path=checkpoint_path_to_use)
                init_elapsed = time.time() - init_start_time
                print(f"[_train] ✅ Trainer created successfully (took {init_elapsed:.1f}s)")
                
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
                print(f"[_train] ❌❌❌ ERROR CREATING TRAINER AFTER {init_elapsed:.1f}s ❌❌❌")
                print(f"[_train] Error: {e}")
                print(f"[_train] Full traceback:")
                print(error_trace)
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
            print(f"\n{'='*80}")
            print(f"❌❌❌ FAILED TO START TRAINING ❌❌❌")
            print(f"Error: {str(e)}")
            print(f"Full traceback:")
            print(error_trace)
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
    active_systems["training"] = {
        "trainer": None,  # Will be set when trainer is created
        "thread": None,   # Will be set when actual training thread starts
        "completed": False,
        "status": "starting",  # Mark as starting
        "start_time": time.time()  # Track when training started to detect hangs
    }
    
    print(f"\n{'='*60}")
    print(f"📤 TRAINING START REQUEST RECEIVED")
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
        print(f"✅ Got event loop: {loop}")
        sys.stdout.flush()
        
        # Schedule _train() to run immediately
        # This ensures it actually executes
        task = loop.create_task(_train())
        print(f"✅ Created and scheduled asyncio task for _train()")
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
                print(f"   ❌ Task error: {task_error}")
                import traceback
                traceback.print_exc()
        sys.stdout.flush()
        
        # Log that we're returning response (task will continue in background)
        print(f"📤 Returning response - _train() should execute in background")
        sys.stdout.flush()
        
    except Exception as e:
        print(f"❌ ERROR creating/scheduling asyncio task: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        # Fallback: try background_tasks
        print(f"   Falling back to BackgroundTasks...")
        background_tasks.add_task(_train)
        sys.stdout.flush()
    
    # Also add to background_tasks as backup
    background_tasks.add_task(_train)
    print(f"✅ Also added to background_tasks as backup")
    sys.stdout.flush()
    
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
            
            # After 5 minutes, suggest stopping and checking logs
            if elapsed > 300:
                return {
                    "status": "error",
                    "message": f"Training initialization timeout ({elapsed:.0f}s). Likely stuck during data loading. Please stop and check backend console logs.",
                    "metrics": {},
                    "error": "Initialization timeout - check backend logs",
                    "suggestion": "Stop training and check backend console for data loading errors. Large number of files may be causing issues."
                }
            
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
    
    # Check if completed flag is set - clean up immediately
    if system.get("completed", False):
        # Training is done - clean it up immediately
        thread_alive = system.get("thread") and hasattr(system["thread"], 'is_alive') and system["thread"].is_alive()
        if not thread_alive:
            # Thread is dead and training is completed - safe to remove
            print(f"🧹 Cleaning up completed training entry (thread dead, completed=True)")
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
        
        # Include actual training mode being used by the trainer
        performance_mode = getattr(trainer, 'performance_mode', 'quiet')
        
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
            print(f"📊 Training status API: Reporting mode = {performance_mode}")
            trainer._last_logged_mode = performance_mode
        
        return {
            "status": "running", 
            "message": "Training in progress",
            "metrics": metrics,
            "training_mode": training_mode_info
        }
    
    # Fallback if trainer not available yet
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
    
    # Ensure default values are present
    if "turbo_training_mode" not in settings:
        settings["turbo_training_mode"] = False
    
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
            print(f"✅ Cleared cache file: {cache_file}")
        else:
            print(f"ℹ️  Cache file does not exist: {cache_file}")
        
        # Also clear the in-memory cache if monitor is running
        if auto_retrain_monitor and auto_retrain_monitor.event_handler:
            auto_retrain_monitor.event_handler.known_files.clear()
            auto_retrain_monitor.event_handler.pending_files.clear()
            if auto_retrain_monitor.event_handler.debounce_timer:
                auto_retrain_monitor.event_handler.debounce_timer.cancel()
            print("✅ Cleared in-memory cache")
        
        return {
            "status": "success",
            "message": "Cache cleared successfully",
            "cache_file": str(cache_file)
        }
    except Exception as e:
        print(f"❌ Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@app.post("/api/settings/auto-retrain/trigger-manual")
async def trigger_manual_retrain(background_tasks: BackgroundTasks):
    """Manually trigger retraining (bypasses file detection)"""
    global auto_retrain_monitor, active_systems
    
    print(f"\n{'='*80}")
    print(f"🚀 MANUAL RETRAIN TRIGGER CALLED")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Process ID: {os.getpid()}")
    print(f"   Thread ID: {threading.current_thread().ident}")
    print(f"{'='*80}\n")
    import sys
    sys.stdout.flush()
    
    if not auto_retrain_monitor:
        print(f"❌ Auto-retrain monitor not initialized")
        raise HTTPException(status_code=400, detail="Auto-retrain monitor not initialized")
    
    if not auto_retrain_monitor.nt8_export_path:
        print(f"❌ NT8 export path not configured")
        raise HTTPException(status_code=400, detail="NT8 export path not configured")
    
    watch_path = Path(auto_retrain_monitor.nt8_export_path)
    if not watch_path.exists():
        print(f"❌ NT8 export path does not exist: {watch_path}")
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
    
    print(f"🚀 Manual retrain triggered - found {len(all_files)} file(s) in {watch_path}")
    for f in all_files:
        print(f"  - {f.name}")
    
    # Clean up any stale training entries first
    global active_systems
    if "training" in active_systems:
        system = active_systems["training"]
        thread_alive = system.get("thread") and hasattr(system["thread"], 'is_alive') and system["thread"].is_alive()
        completed = system.get("completed", False)
        
        print(f"🔍 Checking existing training entry:")
        print(f"   Thread alive: {thread_alive}")
        print(f"   Completed: {completed}")
        
        if not thread_alive:
            # Thread is dead - safe to clean up regardless of completed flag
            print(f"🧹 Cleaning up stale training entry (thread dead)")
            active_systems.pop("training", None)
        elif completed and not thread_alive:
            # Completed and thread dead - definitely safe to remove
            print(f"🧹 Cleaning up completed training entry")
            active_systems.pop("training", None)
        elif thread_alive:
            # Thread is actually running - can't start new training
            print(f"⚠️  Training is currently running, cannot start manual retrain")
            raise HTTPException(status_code=400, detail="Training already in progress. Please stop current training first.")
    
    # Instead of using callback (which has BackgroundTasks issues), call start_training directly
    try:
        from fastapi import BackgroundTasks as BGTasks
        from src.api_server import TrainingRequest
        
        # Load default config
        config_path = "configs/train_config.yaml"
        if not Path(config_path).exists():
            config_path = "configs/train_config_gpu_optimized.yaml"
        
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
        
        print(f"🚀 Manual retrain - calling start_training() directly")
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
                    print(f"❌ Training failed immediately: {error_msg}")
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
            print(f"❌ Error calling start_training: {error_msg}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to start training: {error_msg}")
    except Exception as e:
        print(f"❌ Error triggering manual retrain: {e}")
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
            print(f"🔄 Performance mode updated to: {request.performance_mode}")
            print("   Changes will take effect on next training update cycle")
    
    if request.turbo_training_mode is not None:
        settings["turbo_training_mode"] = request.turbo_training_mode
        
        # Notify active training if running
        if "training" in active_processes:
            print(f"🔄 Turbo training mode updated to: {request.turbo_training_mode}")
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
    
    # Save settings to file
    try:
        with open(settings_file, "w") as f:
            json.dump(settings, f, indent=2)
        
        # Log what was saved for debugging
        print(f"💾 Settings saved:")
        print(f"   performance_mode: {settings.get('performance_mode')}")
        print(f"   turbo_training_mode: {settings.get('turbo_training_mode')}")
        print(f"   auto_retrain_enabled: {settings.get('auto_retrain_enabled')}")
        
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
        # Normal disconnection - client closed connection cleanly
        pass
    except Exception as e:
        # Log unexpected errors but don't crash
        print(f"⚠️  WebSocket error: {e}")
    finally:
        # Always clean up connection, even if it's already removed
        try:
            if websocket in websocket_connections:
                websocket_connections.remove(websocket)
        except (ValueError, RuntimeError):
            # Already removed or list modified - ignore
            pass


if __name__ == "__main__":
    uvicorn.run("src.api_server:app", host="0.0.0.0", port=8200, log_level="info", reload=False)

