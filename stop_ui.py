"""
Stop script for NT8 RL Trading System Production UI

Stops:
1. Kong Gateway (via Docker Compose, optional)
2. Processes on ports 3200 (frontend) and 8200 (backend)

Usage:
    python stop_ui.py              # Stop FastAPI and Frontend only
    python stop_ui.py --kong       # Also stop Kong Gateway
    python stop_ui.py --all        # Stop everything including Kong

Or on Linux/Mac:
    ./stop_ui.py  (if executable permissions are set)
"""

import subprocess
import sys
import os
import time
import argparse
import requests
from pathlib import Path


def find_processes_on_port_windows(port):
    """Find process IDs listening on a specific port on Windows"""
    try:
        # Run netstat to find processes using the port
        result = subprocess.run(
            ["netstat", "-ano"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True  # Use shell for better Windows compatibility
        )
        
        pids = []
        for line in result.stdout.split('\n'):
            # Check for LISTENING state on the target port
            if f':{port}' in line and 'LISTENING' in line:
                # Extract PID from the last column
                parts = line.split()
                if len(parts) > 0:
                    pid = parts[-1]
                    if pid.isdigit():
                        pids.append(int(pid))
        
        return list(set(pids))  # Remove duplicates
    except Exception as e:
        print(f"  ⚠ Error finding processes on port {port}: {e}")
        return []


def find_processes_on_port_linux(port):
    """Find process IDs listening on a specific port on Linux/Mac"""
    try:
        # Try lsof first (Linux/Mac)
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            pids = [int(pid) for pid in result.stdout.strip().split('\n') if pid.isdigit()]
            return pids
        
        # Fallback to ss (Linux)
        result = subprocess.run(
            ["ss", "-ltnp"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        pids = []
        for line in result.stdout.split('\n'):
            if f':{port}' in line:
                # Extract PID from pid=XXXX
                if 'pid=' in line:
                    try:
                        pid_part = line.split('pid=')[1].split(',')[0]
                        pid = int(pid_part)
                        pids.append(pid)
                    except:
                        pass
        
        return list(set(pids))
    except Exception as e:
        print(f"Error finding processes on port {port}: {e}")
        return []


def find_processes_on_port(port):
    """Find process IDs listening on a specific port"""
    if os.name == "nt":  # Windows
        return find_processes_on_port_windows(port)
    else:  # Linux/Mac
        return find_processes_on_port_linux(port)


def kill_process(pid):
    """Kill a process by PID"""
    try:
        if os.name == "nt":  # Windows
            result = subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True  # Use shell for better Windows compatibility
            )
            # Check if kill was successful
            if result.returncode == 0 or "SUCCESS" in result.stdout:
                return True
            else:
                # Process might not exist, which is fine
                if "not found" in result.stderr.lower() or "does not exist" in result.stderr.lower():
                    return True  # Process already gone
                return False
        else:  # Linux/Mac
            result = subprocess.run(
                ["kill", "-9", str(pid)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            return result.returncode == 0
    except Exception as e:
        print(f"  ⚠ Error killing process {pid}: {e}")
        return False


def wait_for_port_free(port, timeout=10):
    """Wait for a port to become free"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        pids = find_processes_on_port(port)
        if len(pids) == 0:
            return True
        time.sleep(0.5)
    return False


def get_process_name(pid):
    """Get process name for a given PID"""
    try:
        if os.name == "nt":  # Windows
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True
            )
            if result.returncode == 0 and result.stdout.strip():
                # CSV format: "process name","PID","Session Name","Session#","Mem Usage"
                parts = result.stdout.strip().split(',')
                if len(parts) > 0:
                    return parts[0].strip('"')
        else:  # Linux/Mac
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "comm="],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
    except:
        pass
    return "Unknown"


def check_kong_running():
    """Check if Kong Gateway is running"""
    try:
        response = requests.get("http://localhost:8301/", timeout=2)
        return response.status_code == 200
    except:
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

def stop_monitoring():
    """Stop Prometheus and Grafana monitoring services"""
    kong_dir = Path("kong")
    docker_compose_file = kong_dir / "docker-compose-prometheus.yml"
    
    if not docker_compose_file.exists():
        print("⚠ Monitoring configuration not found at kong/docker-compose-prometheus.yml")
        return False
    
    if not check_prometheus_running() and not check_grafana_running():
        print("✓ Monitoring services (Prometheus/Grafana) are not running")
        return True
    
    print("Stopping monitoring services (Prometheus & Grafana)...")
    
    try:
        result = subprocess.run(
            ["docker-compose", "-f", "docker-compose-prometheus.yml", "down"],
            cwd=str(kong_dir),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print(f"⚠ WARNING: Failed to stop monitoring services")
            print(f"  Error: {result.stderr}")
            print("  You can stop monitoring manually with: cd kong && docker-compose -f docker-compose-prometheus.yml down")
            return False
        
        print("✓ Monitoring services stopped")
        return True
        
    except subprocess.TimeoutExpired:
        print("⚠ WARNING: Timeout stopping monitoring services")
        print("  You can stop monitoring manually with: cd kong && docker-compose -f docker-compose-prometheus.yml down")
        return False
    except FileNotFoundError:
        print("⚠ WARNING: docker-compose command not found")
        print("  You can stop monitoring manually with: cd kong && docker-compose -f docker-compose-prometheus.yml down")
        return False
    except Exception as e:
        print(f"⚠ WARNING: Error stopping monitoring services: {e}")
        print("  You can stop monitoring manually with: cd kong && docker-compose -f docker-compose-prometheus.yml down")
        return False

def stop_kong():
    """Stop Kong Gateway using Docker Compose"""
    kong_dir = Path("kong")
    docker_compose_file = kong_dir / "docker-compose.yml"
    
    if not docker_compose_file.exists():
        print("⚠ Kong configuration not found at kong/docker-compose.yml")
        return False
    
    if not check_kong_running():
        print("✓ Kong Gateway is not running")
        return True
    
    print("Stopping Kong Gateway...")
    
    try:
        result = subprocess.run(
            ["docker-compose", "down"],
            cwd=str(kong_dir),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print(f"⚠ WARNING: Failed to stop Kong Gateway")
            print(f"  Error: {result.stderr}")
            print("  You can stop Kong manually with: cd kong && docker-compose down")
            return False
        
        print("✓ Kong Gateway stopped")
        return True
        
    except subprocess.TimeoutExpired:
        print("⚠ WARNING: Timeout stopping Kong Gateway")
        print("  You can stop Kong manually with: cd kong && docker-compose down")
        return False
    except FileNotFoundError:
        print("⚠ WARNING: docker-compose command not found")
        print("  You can stop Kong manually with: cd kong && docker-compose down")
        return False
    except Exception as e:
        print(f"⚠ WARNING: Error stopping Kong Gateway: {e}")
        print("  You can stop Kong manually with: cd kong && docker-compose down")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Stop NT8 RL Trading System UI servers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python stop_ui.py              # Stop FastAPI and Frontend only
    python stop_ui.py --kong       # Also stop Kong Gateway
    python stop_ui.py --monitoring # Also stop Prometheus & Grafana
    python stop_ui.py --all        # Stop everything (Kong + Monitoring)
    python stop_ui.py --yes        # Non-interactive (auto-confirms)
    python stop_ui.py -y           # Short form
        """
    )
    parser.add_argument(
        '-y', '--yes',
        action='store_true',
        help='Auto-confirm without prompting'
    )
    parser.add_argument(
        '--kong',
        action='store_true',
        help='Also stop Kong Gateway'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Stop everything including Kong Gateway and monitoring services'
    )
    parser.add_argument(
        '--monitoring',
        action='store_true',
        help='Also stop Prometheus and Grafana monitoring services'
    )
    parser.add_argument(
        '--ports',
        nargs='+',
        type=int,
        default=[8200, 3200],
        help='Ports to check (default: 8200 3200)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NT8 RL Trading System - Stop UI Servers")
    print("=" * 60)
    print()
    
    # Stop Kong if requested
    stop_kong_flag = args.kong or args.all
    if stop_kong_flag:
        if not args.yes:
            try:
                response = input("Stop Kong Gateway? [Y/n]: ").strip().lower()
                if response and response != 'y' and response != 'yes':
                    print("  Skipping Kong Gateway")
                    stop_kong_flag = False
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return
        
        if stop_kong_flag:
            stop_kong()
            print()
    
    # Stop monitoring services if requested
    stop_monitoring_flag = args.monitoring or args.all
    if stop_monitoring_flag:
        if not args.yes:
            try:
                response = input("Stop monitoring services (Prometheus/Grafana)? [Y/n]: ").strip().lower()
                if response and response != 'y' and response != 'yes':
                    print("  Skipping monitoring services")
                    stop_monitoring_flag = False
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return
        
        if stop_monitoring_flag:
            stop_monitoring()
            print()
    
    ports = args.ports
    all_pids = []
    
    # Find all processes
    for port in ports:
        print(f"Checking port {port}...")
        pids = find_processes_on_port(port)
        
        if pids:
            print(f"  Found {len(pids)} process(es) on port {port}:")
            for pid in pids:
                proc_name = get_process_name(pid)
                print(f"    - PID {pid} ({proc_name})")
            all_pids.extend([(pid, port) for pid in pids])
        else:
            print(f"  ✓ No processes found on port {port}")
    
    print()
    
    if not all_pids:
        print("✓ No processes found. Servers are already stopped.")
        return
    
    # Ask for confirmation (unless --yes flag is used)
    if not args.yes:
        try:
            response = input(f"Kill {len(all_pids)} process(es)? [Y/n]: ").strip().lower()
            if response and response != 'y' and response != 'yes':
                print("Cancelled.")
                return
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return
    
    print()
    print(f"Stopping {len(all_pids)} process(es)...")
    print()
    
    killed_pids = {}
    failed_pids = []
    for pid, port in all_pids:
        proc_name = get_process_name(pid)
        print(f"  Killing PID {pid} ({proc_name}) on port {port}...", end=" ")
        if kill_process(pid):
            print("✓")
            if port not in killed_pids:
                killed_pids[port] = []
            killed_pids[port].append(pid)
        else:
            print("✗ Failed")
            failed_pids.append((pid, port))
    
    print()
    
    if failed_pids:
        print(f"⚠ Warning: Failed to kill {len(failed_pids)} process(es):")
        for pid, port in failed_pids:
            print(f"    - PID {pid} on port {port}")
        print()
    
    print("Waiting for processes to terminate...")
    time.sleep(1)  # Give processes a moment to terminate
    
    # Wait for all ports to become free
    all_free = True
    for port in ports:
        print(f"  Checking port {port}...", end=" ")
        if wait_for_port_free(port, timeout=5):
            print("✓ Port is free")
        else:
            print("⚠ Port still in use")
            all_free = False
    
    print()
    if all_free:
        print("=" * 60)
        print("✓ All servers stopped successfully")
        if stop_kong_flag:
            print("✓ Kong Gateway stopped")
        elif check_kong_running():
            print("ℹ Kong Gateway is still running (use --kong to stop it)")
        if stop_monitoring_flag:
            print("✓ Monitoring services stopped")
        elif check_prometheus_running() or check_grafana_running():
            print("ℹ Monitoring services are still running (use --monitoring to stop them)")
        print("=" * 60)
    else:
        print("⚠ Some processes may still be running")
        print("  You may need to manually kill them:")
        for port in ports:
            remaining = find_processes_on_port(port)
            if remaining:
                for pid in remaining:
                    proc_name = get_process_name(pid)
                    print(f"    Port {port}: PID {pid} ({proc_name})")
                print(f"  Manual kill command:")
                if os.name == "nt":
                    print(f"    taskkill /F /PID {' /PID '.join(map(str, remaining))}")
                else:
                    print(f"    kill -9 {' '.join(map(str, remaining))}")
        
        if not stop_kong_flag and check_kong_running():
            print()
            print("ℹ Kong Gateway is still running")
            print("  Stop it with: python stop_ui.py --kong")
            print("  Or manually: cd kong && docker-compose down")
        
        if not stop_monitoring_flag and (check_prometheus_running() or check_grafana_running()):
            print()
            print("ℹ Monitoring services are still running")
            print("  Stop them with: python stop_ui.py --monitoring")
            print("  Or manually: cd kong && docker-compose -f docker-compose-prometheus.yml down")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("\nStopped by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)

