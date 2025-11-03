"""
Automatic Retraining Monitor

Monitors NT8 export directory for new data files and triggers retraining.
Designed to work gracefully without interrupting existing training jobs.
"""

import time
import json
import hashlib
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Set, Callable
import logging

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NT8DataMonitor(FileSystemEventHandler):
    """
    Monitors NT8 export directory for new data files.
    
    Detects when new CSV/TXT files are added and triggers retraining.
    """
    
    def __init__(
        self,
        watch_path: str,
        callback: Optional[Callable] = None,
        debounce_seconds: int = 30
    ):
        """
        Initialize monitor.
        
        Args:
            watch_path: Path to NT8 export directory
            callback: Function to call when new data detected
            debounce_seconds: Wait time before triggering (prevents multiple triggers)
        """
        self.watch_path = Path(watch_path)
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        
        # Track known files to avoid duplicate triggers
        self.known_files: Set[str] = set()
        self.last_event_time = {}
        
        # Debounce timer
        self.debounce_timer = None
        self.pending_files: Set[Path] = set()
        
        # Load known files cache
        self._load_cache()
    
    def _load_cache(self):
        """Load cache of known files"""
        cache_file = Path("logs") / "known_files_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    self.known_files = set(data.get("files", []))
                    logger.info(f"Loaded {len(self.known_files)} known files from cache")
            except:
                pass
    
    def _save_cache(self):
        """Save cache of known files"""
        cache_file = Path("logs") / "known_files_cache.json"
        cache_file.parent.mkdir(exist_ok=True)
        
        try:
            with open(cache_file, 'w') as f:
                json.dump({"files": list(self.known_files)}, f, indent=2)
        except:
            pass
    
    def _should_trigger(self, file_path: Path) -> bool:
        """Check if file should trigger retraining"""
        # Only watch CSV and TXT files
        if file_path.suffix.lower() not in ['.csv', '.txt']:
            return False
        
        # Check if we already processed this file
        file_key = f"{file_path.stem}_{file_path.stat().st_size}"
        if file_key in self.known_files:
            return False
        
        # Check if file is complete (not being written)
        try:
            stat = file_path.stat()
            # Wait a bit and check if file size changed
            time.sleep(1)
            new_stat = file_path.stat()
            if stat.st_size != new_stat.st_size:
                logger.info(f"File still being written: {file_path.name}")
                return False
        except:
            return False
        
        return True
    
    def on_created(self, event: FileCreatedEvent):
        """Handle file creation event"""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Check if this is a data file we care about
        if not self._should_trigger(file_path):
            return
        
        logger.info(f"📁 New file detected: {file_path.name}")
        
        # Mark as known to avoid duplicate processing
        file_key = f"{file_path.stem}_{file_path.stat().st_size}"
        self.known_files.add(file_key)
        self._save_cache()
        
        # Add to pending files (debounce)
        self.pending_files.add(file_path)
        self.last_event_time[file_path] = time.time()
        
        # Schedule callback with debounce
        self._schedule_callback()
    
    def on_modified(self, event: FileModifiedEvent):
        """Handle file modification event (in case file is replaced)"""
        # Similar to on_created, but check if size changed significantly
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Only process if file size increased significantly (file update, not just access)
        if file_path.stat().st_size < 1000:  # Ignore small files
            return
        
        if not self._should_trigger(file_path):
            return
        
        logger.info(f"📝 File updated: {file_path.name}")
        
        # Mark as known
        file_key = f"{file_path.stem}_{file_path.stat().st_size}"
        self.known_files.add(file_key)
        self._save_cache()
        
        # Add to pending files
        self.pending_files.add(file_path)
        self.last_event_time[file_path] = time.time()
        
        # Schedule callback
        self._schedule_callback()
    
    def _schedule_callback(self):
        """Schedule callback with debounce"""
        # Cancel existing timer if any
        if self.debounce_timer is not None:
            self.debounce_timer.cancel()
        
        # Schedule new callback
        def debounced_callback():
            if self.pending_files and self.callback:
                files = list(self.pending_files)
                self.pending_files.clear()
                
                logger.info(f"🚀 Triggering retrain for {len(files)} new file(s)")
                self.callback(files)
        
        self.debounce_timer = threading.Timer(self.debounce_seconds, debounced_callback)
        self.debounce_timer.start()


class AutoRetrainMonitor:
    """
    Automatic retraining monitor that watches NT8 export directory.
    
    Main class that coordinates file watching and retraining triggers.
    """
    
    def __init__(
        self,
        nt8_export_path: Optional[str] = None,
        auto_retrain_callback: Optional[Callable] = None,
        enabled: bool = True
    ):
        """
        Initialize auto-retrain monitor.
        
        Args:
            nt8_export_path: Path to NT8 export directory
            auto_retrain_callback: Function to call when retraining needed
            enabled: Whether monitoring is enabled
        """
        self.nt8_export_path = nt8_export_path
        self.auto_retrain_callback = auto_retrain_callback
        self.enabled = enabled
        
        # File watcher
        self.observer: Optional[Observer] = None
        self.event_handler: Optional[NT8DataMonitor] = None
        
        # Statistics
        self.files_detected = 0
        self.last_retrain_trigger = None
        self.running = False
    
    def start(self):
        """Start monitoring NT8 export directory"""
        if not self.enabled:
            logger.info("Auto-retrain monitoring disabled")
            return
        
        if not self.nt8_export_path:
            logger.warning("No NT8 export path configured")
            return
        
        watch_path = Path(self.nt8_export_path)
        
        if not watch_path.exists():
            logger.warning(f"NT8 export path does not exist: {watch_path}")
            logger.info("Will start monitoring when path becomes available")
            # Create a poller thread to check periodically
            threading.Thread(target=self._poll_for_path, daemon=True, args=(watch_path,)).start()
            return
        
        try:
            # Create event handler
            self.event_handler = NT8DataMonitor(
                watch_path=str(watch_path),
                callback=self._on_new_data_detected,
                debounce_seconds=30  # Wait 30 seconds after last file change
            )
            
            # Create observer
            self.observer = Observer()
            self.observer.schedule(self.event_handler, str(watch_path), recursive=False)
            self.observer.start()
            
            self.running = True
            logger.info(f"✅ Auto-retrain monitoring started on: {watch_path}")
            
        except Exception as e:
            logger.error(f"Failed to start file watcher: {e}")
    
    def _poll_for_path(self, watch_path: Path):
        """Poll for path existence (fallback if path doesn't exist yet)"""
        while not self.running:
            time.sleep(60)  # Check every minute
            
            if watch_path.exists():
                logger.info(f"Path now available: {watch_path}")
                self.start()
                break
    
    def _on_new_data_detected(self, files: list):
        """
        Callback when new data files detected.
        
        Args:
            files: List of new file paths
        """
        self.files_detected += len(files)
        self.last_retrain_trigger = datetime.now().isoformat()
        
        logger.info(f"📊 New data detected: {len(files)} file(s)")
        for f in files:
            logger.info(f"  - {Path(f).name}")
        
        # Call the callback if provided
        if self.auto_retrain_callback:
            try:
                self.auto_retrain_callback(files)
            except Exception as e:
                logger.error(f"Error in retrain callback: {e}")
    
    def stop(self):
        """Stop monitoring"""
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=5)
            self.running = False
            logger.info("Auto-retrain monitoring stopped")
    
    def get_status(self) -> Dict:
        """Get monitoring status"""
        return {
            "enabled": self.enabled,
            "running": self.running,
            "nt8_export_path": str(self.nt8_export_path),
            "files_detected": self.files_detected,
            "last_retrain_trigger": self.last_retrain_trigger
        }
    
    def configure(self, nt8_export_path: Optional[str] = None, enabled: bool = None):
        """Update configuration dynamically"""
        if nt8_export_path is not None:
            # Restart observer if path changed
            was_running = self.running
            if was_running:
                self.stop()
            
            self.nt8_export_path = nt8_export_path
            
            if was_running:
                self.start()
        
        if enabled is not None:
            self.enabled = enabled
            if not enabled and self.running:
                self.stop()


# Example usage
if __name__ == "__main__":
    # Example callback
    def retrain_callback(files):
        print(f"\n🚀 Retraining triggered by {len(files)} new file(s):")
        for f in files:
            print(f"  - {Path(f).name}")
        print("\n(Would start retraining here)")
    
    # Create monitor
    monitor = AutoRetrainMonitor(
        nt8_export_path=r"C:\Users\sovan\Documents\NinjaTrader 8\export",
        auto_retrain_callback=retrain_callback,
        enabled=True
    )
    
    monitor.start()
    
    try:
        print("\nMonitoring for new data files...")
        print("Press Ctrl+C to stop\n")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        monitor.stop()

