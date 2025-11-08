"""
Startup script for NT8 RL Trading System Production UI

This script starts:
1. Kong Gateway (via Docker Compose)
2. Backend API server (FastAPI)
3. Frontend development server (Vite/React)
"""

import subprocess
import sys
import os
import time
import requests
from pathlib import Path

def detect_venv_type():
    """Detect the type of virtual environment being used"""
    # Check if we're in a virtual environment
    in_venv = sys.prefix != sys.base_prefix
    
    # Check for uv virtual environment
    is_uv_venv = False
    venv_path = Path(sys.prefix)
    pyvenv_cfg = venv_path / "pyvenv.cfg"
    
    if pyvenv_cfg.exists():
        try:
            with open(pyvenv_cfg, 'r') as f:
                content = f.read()
                # uv stores Python in AppData/Roaming/uv/python or uses uv-managed paths
                if 'uv' in content.lower() or 'AppData\\Roaming\\uv' in content or '/uv/python' in content:
                    is_uv_venv = True
        except:
            pass
    
    # Check if uv is available
    uv_available = False
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, timeout=2)
        uv_available = result.returncode == 0
    except:
        pass
    
    return {
        "in_venv": in_venv,
        "is_uv_venv": is_uv_venv,
        "uv_available": uv_available,
        "venv_path": sys.prefix
    }

def check_cuda():
    """Check CUDA availability and display GPU information"""
    try:
        import torch
        
        # Check PyTorch version info
        torch_version = torch.__version__
        has_cuda_build = "+cu" in torch_version or torch.version.cuda is not None
        
        # Check if CUDA runtime is available
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            print(f"✓ CUDA Available: {cuda_version}")
            print(f"  GPU: {gpu_name}")
            print(f"  GPU Count: {device_count}")
            return True
        else:
            # PyTorch is installed but CUDA is not available
            # Check if it's CPU-only build
            if not has_cuda_build:
                print("⚠ CUDA not available - PyTorch is CPU-only build")
                print(f"  Current PyTorch version: {torch_version}")
                print()
                print("  To enable GPU support:")
                venv_info = detect_venv_type()
                if venv_info["is_uv_venv"] or venv_info["uv_available"]:
                    print("  1. Uninstall current PyTorch: uv pip uninstall torch torchvision torchaudio")
                    print("  2. Install CUDA-enabled PyTorch:")
                else:
                    print("  1. Uninstall current PyTorch: pip uninstall torch torchvision torchaudio")
                    print("  2. Install CUDA-enabled PyTorch:")
                
                # Try to detect system CUDA version
                system_cuda_version = None
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        # Check for CUDA toolkit version (approximate)
                        # This is a heuristic - nvidia-smi shows driver version, not CUDA toolkit
                        venv_info = detect_venv_type()
                        if venv_info["is_uv_venv"] or venv_info["uv_available"]:
                            print("     For CUDA 11.8: uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
                            print("     For CUDA 12.1: uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                            print("     For CUDA 12.4+ (recommended for Python 3.13): uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
                        else:
                            print("     For CUDA 11.8: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
                            print("     For CUDA 12.1: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                            print("     For CUDA 12.4+ (recommended for Python 3.13): pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
                except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                    # nvidia-smi not available or can't determine
                    venv_info = detect_venv_type()
                    if venv_info["is_uv_venv"] or venv_info["uv_available"]:
                        print("     For CUDA 11.8: uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
                        print("     For CUDA 12.1: uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                        print("     For CUDA 12.4+ (recommended for Python 3.13): uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
                    else:
                        print("     For CUDA 11.8: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
                        print("     For CUDA 12.1: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                        print("     For CUDA 12.4+ (recommended for Python 3.13): pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
                    print()
                    print("  Check your CUDA version with: nvidia-smi")
            else:
                print("⚠ CUDA not available - PyTorch has CUDA support but CUDA runtime not detected")
                print(f"  PyTorch version: {torch_version}")
                print("  Make sure:")
                print("    - NVIDIA drivers are installed")
                print("    - CUDA toolkit is installed")
                print("    - GPU is properly connected")
                print("  Check with: nvidia-smi")
            return False
    except ImportError:
        print("⚠ PyTorch not found - Install with: pip install torch")
        return False
    except Exception as e:
        print(f"⚠ Error checking CUDA: {e}")
        return False

def check_dependencies():
    """Check if key dependencies are installed"""
    venv_info = detect_venv_type()
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import fastapi
    except ImportError:
        missing.append("fastapi")
    
    try:
        import uvicorn
    except ImportError:
        missing.append("uvicorn")
    
    try:
        import anthropic
    except ImportError:
        missing.append("anthropic")
    
    if missing:
        print(f"⚠ Missing dependencies: {', '.join(missing)}")
        if venv_info["is_uv_venv"] or venv_info["uv_available"]:
            print(f"  Install with uv: uv pip install {' '.join(missing)}")
            print(f"  Or install all: uv pip install -r requirements.txt")
            print(f"  Or sync from pyproject.toml: uv sync")
        else:
            print(f"  Install with: pip install {' '.join(missing)}")
            print(f"  Or install all: pip install -r requirements.txt")
        return False
    return True

def check_docker():
    """Check if Docker is available"""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            timeout=2,
            text=True
        )
        if result.returncode == 0:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False

def check_kong_running():
    """Check if Kong Gateway is already running"""
    try:
        response = requests.get("http://localhost:8301/", timeout=2)
        return response.status_code == 200
    except:
        return False

def start_kong():
    """Start Kong Gateway using Docker Compose"""
    kong_dir = Path("kong")
    docker_compose_file = kong_dir / "docker-compose.yml"
    
    if not docker_compose_file.exists():
        print("⚠ Kong configuration not found at kong/docker-compose.yml")
        print("  Kong Gateway will not be started")
        return False
    
    # Check if Kong is already running
    if check_kong_running():
        print("✓ Kong Gateway is already running")
        return True
    
    print("Starting Kong Gateway...")
    print("  This may take a minute on first start...")
    
    try:
        # Change to kong directory and run docker-compose
        result = subprocess.run(
            ["docker-compose", "up", "-d"],
            cwd=str(kong_dir),
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            print(f"⚠ WARNING: Failed to start Kong Gateway")
            print(f"  Error: {result.stderr}")
            print("  You can start Kong manually with: cd kong && docker-compose up -d")
            return False
        
        print("✓ Kong Gateway containers started")
        print("  Waiting for Kong to be ready...")
        
        # Wait for Kong to be ready (up to 30 seconds)
        max_attempts = 30
        for i in range(max_attempts):
            if check_kong_running():
                print("✓ Kong Gateway is ready")
                return True
            time.sleep(1)
            if i % 5 == 0:
                print(f"  Still waiting... ({i}/{max_attempts})")
        
        print("⚠ WARNING: Kong Gateway started but not responding yet")
        print("  It may still be initializing. Continuing anyway...")
        return True
        
    except subprocess.TimeoutExpired:
        print("⚠ WARNING: Timeout starting Kong Gateway")
        print("  You can start Kong manually with: cd kong && docker-compose up -d")
        return False
    except FileNotFoundError:
        print("⚠ WARNING: docker-compose command not found")
        print("  Make sure Docker and Docker Compose are installed")
        print("  You can start Kong manually with: cd kong && docker-compose up -d")
        return False
    except Exception as e:
        print(f"⚠ WARNING: Error starting Kong Gateway: {e}")
        print("  You can start Kong manually with: cd kong && docker-compose up -d")
        return False

def check_prometheus_running():
    """Check if Prometheus is running"""
    try:
        response = requests.get("http://localhost:9090/-/ready", timeout=2)
        return response.status_code == 200
    except:
        return False

def check_grafana_running():
    """Check if Grafana is running"""
    try:
        response = requests.get("http://localhost:3000/api/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def start_monitoring(enable_monitoring=False):
    """Start Prometheus and Grafana monitoring services (optional)"""
    if not enable_monitoring:
        return False
    
    kong_dir = Path("kong")
    docker_compose_file = kong_dir / "docker-compose-prometheus.yml"
    
    if not docker_compose_file.exists():
        print("⚠ Monitoring configuration not found at kong/docker-compose-prometheus.yml")
        return False
    
    # Check if already running
    prometheus_running = check_prometheus_running()
    grafana_running = check_grafana_running()
    
    if prometheus_running and grafana_running:
        print("✓ Monitoring services (Prometheus/Grafana) are already running")
        return True
    
    print("Starting monitoring services (Prometheus & Grafana)...")
    print("  This may take a minute on first start...")
    
    try:
        result = subprocess.run(
            ["docker-compose", "-f", "docker-compose-prometheus.yml", "up", "-d"],
            cwd=str(kong_dir),
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            print(f"⚠ WARNING: Failed to start monitoring services")
            print(f"  Error: {result.stderr}")
            print("  You can start monitoring manually with: cd kong && docker-compose -f docker-compose-prometheus.yml up -d")
            return False
        
        print("✓ Monitoring services containers started")
        print("  Waiting for services to be ready...")
        
        # Wait for services to be ready (up to 30 seconds)
        max_attempts = 30
        for i in range(max_attempts):
            if check_prometheus_running() and check_grafana_running():
                print("✓ Monitoring services are ready")
                print("  Prometheus: http://localhost:9090")
                print("  Grafana: http://localhost:3000 (admin/admin)")
                return True
            time.sleep(1)
            if i % 5 == 0:
                print(f"  Still waiting... ({i}/{max_attempts})")
        
        print("⚠ WARNING: Monitoring services started but not fully ready yet")
        print("  They may still be initializing. Continuing anyway...")
        return True
        
    except subprocess.TimeoutExpired:
        print("⚠ WARNING: Timeout starting monitoring services")
        print("  You can start monitoring manually with: cd kong && docker-compose -f docker-compose-prometheus.yml up -d")
        return False
    except FileNotFoundError:
        print("⚠ WARNING: docker-compose command not found")
        return False
    except Exception as e:
        print(f"⚠ WARNING: Error starting monitoring services: {e}")
        print("  You can start monitoring manually with: cd kong && docker-compose -f docker-compose-prometheus.yml up -d")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Start NT8 RL Trading System UI")
    parser.add_argument(
        '--monitoring',
        action='store_true',
        help='Also start Prometheus and Grafana monitoring services'
    )
    parser.add_argument(
        '--no-kong',
        action='store_true',
        help='Skip starting Kong Gateway'
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("NT8 RL Trading System - Production UI")
    print("=" * 60)
    print()
    
    # Detect virtual environment type
    venv_info = detect_venv_type()
    if venv_info["in_venv"]:
        if venv_info["is_uv_venv"]:
            print("✓ Using uv virtual environment")
        else:
            print("✓ Using virtual environment")
    else:
        print("⚠ Not in a virtual environment - recommended to use one")
        if venv_info["uv_available"]:
            print("  Create with uv: uv venv")
        else:
            print("  Create with: python -m venv .venv")
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        print("  Some dependencies are missing. Install them and try again.")
        print()
    else:
        print("✓ Key dependencies found")
        print()
    
    # Check CUDA/GPU
    print("Checking GPU/CUDA support...")
    cuda_available = check_cuda()
    print()
    
    # Start Kong Gateway
    kong_started = False
    if not args.no_kong:
        if check_docker():
            kong_started = start_kong()
            print()
        else:
            print("⚠ Docker not found - Skipping Kong Gateway startup")
            print("  Install Docker to use Kong Gateway")
            print("  You can start Kong manually with: cd kong && docker-compose up -d")
            print()
    
    # Start monitoring services (optional)
    monitoring_started = False
    if args.monitoring:
        if check_docker():
            monitoring_started = start_monitoring(enable_monitoring=True)
            print()
        else:
            print("⚠ Docker not found - Skipping monitoring services startup")
            print()
    
    # Use the current Python executable (works with uv, venv, etc.)
    venv_python = sys.executable
    
    # Start backend API server
    print("Starting backend API server...")
    try:
        # Don't redirect stdout/stderr so we can see errors
        backend_process = subprocess.Popen(
            [venv_python, "-m", "uvicorn", "src.api_server:app", "--host", "0.0.0.0", "--port", "8200"],
            stdout=None,  # Show output in console
            stderr=subprocess.STDOUT  # Merge stderr into stdout
        )
        
        print(f"✓ Backend API server process started (PID: {backend_process.pid})")
        print("  API will be available at: http://localhost:8200")
        print("  Waiting for server to initialize...")
        print()
        
        # Wait a bit for backend to start and check if it's still running
        time.sleep(3)
        
        # Check if process is still alive
        if backend_process.poll() is not None:
            print("⚠ ERROR: Backend API server exited immediately!")
            print("  Return code:", backend_process.returncode)
            print("  This usually indicates:")
            venv_info = detect_venv_type()
            if venv_info["is_uv_venv"] or venv_info["uv_available"]:
                print("    - Missing dependencies (run: uv pip install -r requirements.txt or uv sync)")
            else:
                print("    - Missing dependencies (run: pip install -r requirements.txt)")
            print("    - Import errors in src/api_server.py or imported modules")
            print("    - Missing type imports (e.g., Tuple from typing)")
            print("    - Configuration issues")
            print()
            print("  Please check the error messages above and fix any issues.")
            sys.exit(1)
        else:
            print("✓ Backend API server is running")
            print()
    except Exception as e:
        print(f"✗ ERROR: Failed to start backend API server: {e}")
        venv_info = detect_venv_type()
        if venv_info["is_uv_venv"] or venv_info["uv_available"]:
            print("  Make sure you're in a virtual environment with all dependencies installed.")
            print("  Install dependencies with: uv pip install -r requirements.txt or uv sync")
        else:
            print("  Make sure you're in a virtual environment with all dependencies installed.")
            print("  Install dependencies with: pip install -r requirements.txt")
        sys.exit(1)
    
    # Start frontend (if package.json exists)
    frontend_dir = Path("frontend")
    frontend_process = None
    if (frontend_dir / "package.json").exists():
        print("Starting frontend development server...")
        print("  (This will open in a new terminal window)")
        print()
        
        try:
            # Use start command for Windows to open new window
            if os.name == "nt":
                frontend_process = subprocess.Popen(
                    ["npm", "run", "dev"],
                    cwd=str(frontend_dir),
                    shell=True,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                frontend_process = subprocess.Popen(
                    ["npm", "run", "dev"],
                    cwd=str(frontend_dir)
                )
            
            print(f"✓ Frontend server starting (PID: {frontend_process.pid})")
            print("  UI will be available at: http://localhost:3200")
            print()
        except Exception as e:
            print(f"⚠ WARNING: Failed to start frontend: {e}")
            print("  You can start it manually with: cd frontend && npm run dev")
            print()
            frontend_process = None
    else:
        print("⚠ Frontend directory not found. Run 'cd frontend && npm install' first.")
        print()
    
    print("=" * 60)
    print("Servers are starting...")
    print()
    if kong_started:
        print("Kong Gateway: http://localhost:8300")
        print("  - Proxy endpoint: http://localhost:8300")
        print("  - Admin API: http://localhost:8301")
        print("  - Metrics: http://localhost:8301/metrics")
    if monitoring_started:
        print("Monitoring:")
        print("  - Prometheus: http://localhost:9090")
        print("  - Grafana: http://localhost:3000 (admin/admin)")
    print("Backend API: http://localhost:8200")
    print("  - Direct API: http://localhost:8200/api/...")
    if kong_started:
        print("  - Via Kong: http://localhost:8300/api/... (requires API key)")
        print("  - Monitoring API: http://localhost:8200/api/monitoring/...")
    print("  - WebSocket: ws://localhost:8200/ws")
    if (frontend_dir / "package.json").exists():
        print("Frontend UI: http://localhost:3200")
        print("  - Web interface will open automatically")
    print()
    if cuda_available:
        print("✓ GPU training is available - Select 'CUDA (GPU)' in Training panel")
    else:
        print("⚠ Training will use CPU - Install PyTorch with CUDA for GPU support")
    if not args.monitoring and check_docker():
        print("ℹ  Tip: Use --monitoring flag to start Prometheus & Grafana")
    print()
    print("Press Ctrl+C to stop all servers")
    print("=" * 60)
    print()
    
    try:
        # Monitor processes and keep script running
        import signal
        
        def signal_handler(sig, frame):
            print()
            print()
            print("Stopping servers...")
            
            # Stop backend
            try:
                backend_process.terminate()
                backend_process.wait(timeout=5)
                print("✓ Backend API server stopped")
            except:
                try:
                    backend_process.kill()
                    print("✓ Backend API server killed")
                except:
                    pass
            
            # Stop frontend
            if frontend_process is not None:
                try:
                    frontend_process.terminate()
                    frontend_process.wait(timeout=5)
                    print("✓ Frontend server stopped")
                except:
                    try:
                        frontend_process.kill()
                        print("✓ Frontend server killed")
                    except:
                        pass
            
            # Note: We don't stop Kong/Monitoring here - let the user manage it with stop_ui.py
            # This is because these services might be used by other applications
            if kong_started:
                print("⚠ Kong Gateway is still running")
                print("  Stop it with: python stop_ui.py --kong")
                print("  Or manually: cd kong && docker-compose down")
            if monitoring_started:
                print("⚠ Monitoring services (Prometheus/Grafana) are still running")
                print("  Stop them with: python stop_ui.py --monitoring")
                print("  Or manually: cd kong && docker-compose -f docker-compose-prometheus.yml down")
            
            print("✓ Servers stopped")
            sys.exit(0)
        
        # Register signal handler for Ctrl+C
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Keep script running and monitor backend process
        while True:
            # Check if backend process is still alive
            if backend_process.poll() is not None:
                print()
                print("⚠ ERROR: Backend API server has stopped!")
                print(f"  Return code: {backend_process.returncode}")
                print("  Please check the error messages above.")
                break
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        signal_handler(None, None)
    except Exception as e:
        print(f"\n⚠ Error monitoring processes: {e}")
        signal_handler(None, None)

if __name__ == "__main__":
    main()

