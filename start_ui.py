"""
Startup script for NT8 RL Trading System Production UI

This script starts both the backend API server and frontend development server.
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def check_cuda():
    """Check CUDA availability and display GPU information"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            print(f"✓ CUDA Available: {cuda_version}")
            print(f"  GPU: {gpu_name}")
            print(f"  GPU Count: {device_count}")
            return True
        else:
            print("⚠ CUDA not available - Training will use CPU")
            print("  To enable GPU support, install PyTorch with CUDA:")
            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            return False
    except ImportError:
        print("⚠ PyTorch not found - Install with: pip install torch")
        return False
    except Exception as e:
        print(f"⚠ Error checking CUDA: {e}")
        return False

def check_dependencies():
    """Check if key dependencies are installed"""
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
    
    if missing:
        print(f"⚠ Missing dependencies: {', '.join(missing)}")
        print(f"  Install with: pip install {' '.join(missing)}")
        return False
    return True

def main():
    print("=" * 60)
    print("NT8 RL Trading System - Production UI")
    print("=" * 60)
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
    
    # Check if we're in a virtual environment
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
            print("    - Missing dependencies (run: pip install -r requirements.txt)")
            print("    - Import errors in src/api_server.py")
            print("    - Configuration issues")
            print()
            print("  Please check the error messages above and fix any issues.")
            sys.exit(1)
        else:
            print("✓ Backend API server is running")
            print()
    except Exception as e:
        print(f"✗ ERROR: Failed to start backend API server: {e}")
        print("  Make sure you're in a virtual environment with all dependencies installed.")
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
    print("Backend API: http://localhost:8200")
    print("  - API endpoints: http://localhost:8200/api/...")
    print("  - WebSocket: ws://localhost:8200/ws")
    if (frontend_dir / "package.json").exists():
        print("Frontend UI: http://localhost:3200")
        print("  - Web interface will open automatically")
    print()
    if cuda_available:
        print("✓ GPU training is available - Select 'CUDA (GPU)' in Training panel")
    else:
        print("⚠ Training will use CPU - Install PyTorch with CUDA for GPU support")
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
            try:
                backend_process.terminate()
                backend_process.wait(timeout=5)
            except:
                backend_process.kill()
            
            if frontend_process is not None:
                try:
                    frontend_process.terminate()
                    frontend_process.wait(timeout=5)
                except:
                    try:
                        frontend_process.kill()
                    except:
                        pass
            
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

