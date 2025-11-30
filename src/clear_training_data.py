"""
Clear Training Data Utility

Provides functions to clear old training data when starting fresh training.
This includes:
- Trading journal database (archive or clear)
- Caches (file cache, data cache)
- Processed data (optional)
"""

import sqlite3
from pathlib import Path
from datetime import datetime
import shutil
import json
from typing import Optional


def clear_trading_journal(db_path: Path, archive: bool = True) -> dict:
    """
    Clear or archive the trading journal database.
    
    Args:
        db_path: Path to trading journal database
        archive: If True, backup old database before clearing. If False, just clear.
    
    Returns:
        Dictionary with status and details
    """
    result = {
        "cleared": False,
        "archived": False,
        "backup_path": None,
        "trades_backed_up": 0,
        "message": ""
    }
    
    if not db_path.exists():
        result["message"] = "Trading journal database does not exist - nothing to clear"
        return result
    
    try:
        # Count existing trades before clearing
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trades")
        trade_count = cursor.fetchone()[0] or 0
        conn.close()
        result["trades_backed_up"] = trade_count
        
        if archive and trade_count > 0:
            # Backup database before clearing
            archive_dir = db_path.parent / "trading_journal_archive"
            archive_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = archive_dir / f"trading_journal_backup_{timestamp}.db"
            shutil.copy2(db_path, backup_path)
            result["backup_path"] = str(backup_path)
            result["archived"] = True
            result["message"] = f"Archived {trade_count} trades to {backup_path.name}"
        
        # Clear the database by recreating it
        db_path.unlink()
        result["cleared"] = True
        
        if not result["message"]:
            result["message"] = f"Cleared trading journal database ({trade_count} trades removed)"
            
    except Exception as e:
        result["message"] = f"Error clearing trading journal: {str(e)}"
        result["error"] = str(e)
    
    return result


def clear_caches(cache_paths: Optional[list] = None) -> dict:
    """
    Clear cache files.
    
    Args:
        cache_paths: List of cache file paths to clear. If None, uses default caches.
    
    Returns:
        Dictionary with status and details
    """
    if cache_paths is None:
        # Default cache paths
        project_root = Path(__file__).parent.parent
        cache_paths = [
            project_root / "logs" / "known_files_cache.json",
            project_root / "data" / "processed",  # Processed data cache
        ]
    
    result = {
        "cleared_files": [],
        "failed_files": [],
        "message": ""
    }
    
    for cache_path in cache_paths:
        cache_path = Path(cache_path)
        try:
            if cache_path.is_file():
                cache_path.unlink()
                result["cleared_files"].append(str(cache_path))
            elif cache_path.is_dir():
                # Clear directory contents but keep the directory
                for item in cache_path.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                result["cleared_files"].append(f"{str(cache_path)}/*")
        except Exception as e:
            result["failed_files"].append((str(cache_path), str(e)))
    
    if result["cleared_files"]:
        result["message"] = f"Cleared {len(result['cleared_files'])} cache locations"
    if result["failed_files"]:
        result["message"] += f", {len(result['failed_files'])} failed"
    
    return result


def archive_checkpoints(models_dir: Path, archive: bool = True) -> dict:
    """
    Archive and remove all checkpoints and model files.
    Uses CheckpointManager for main models, then handles pretraining checkpoints separately.
    
    Args:
        models_dir: Path to models directory
        archive: If True, archive checkpoints before removing. If False, just remove.
    
    Returns:
        Dictionary with status and details
    """
    result = {
        "archived": False,
        "removed": [],
        "failed": [],
        "backup_path": None,
        "checkpoint_count": 0,
        "message": ""
    }
    
    if not models_dir.exists():
        result["message"] = "Models directory does not exist - nothing to archive"
        return result
    
    try:
        # Use existing CheckpointManager for main models directory
        from src.training.checkpoint_manager import CheckpointManager
        checkpoint_manager = CheckpointManager(model_dir=models_dir)
        
        # Archive main models using existing functionality
        archived_count = checkpoint_manager.archive_existing_models()
        result["checkpoint_count"] = archived_count
        
        # Handle pretraining checkpoints separately (not handled by archive_existing_models)
        pretraining_dirs = [
            models_dir / "pretraining" / "supervised",
            models_dir / "pretraining" / "unsupervised"
        ]
        
        pretraining_files = []
        for pretraining_dir in pretraining_dirs:
            if pretraining_dir.exists():
                pretraining_files.extend(list(pretraining_dir.glob("checkpoint*.pt")))
        
        if pretraining_files:
            # Get the archive directory that was just created by CheckpointManager
            # It creates archive_YYYYMMDD_HHMMSS, find the most recent one
            archive_base = models_dir / "Archive"
            if archive_base.exists():
                archive_subdirs = sorted(archive_base.glob("archive_*"), key=lambda x: x.stat().st_mtime, reverse=True)
                if archive_subdirs:
                    archive_dir = archive_subdirs[0]  # Use the most recent archive
                    result["backup_path"] = str(archive_dir)
                    
                    # Archive pretraining checkpoints, preserving directory structure
                    for checkpoint_file in pretraining_files:
                        try:
                            # Preserve directory structure: pretraining/supervised/checkpoint_epoch_X.pt
                            relative_path = checkpoint_file.relative_to(models_dir)
                            archive_path = archive_dir / relative_path
                            archive_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            if archive:
                                shutil.move(str(checkpoint_file), str(archive_path))  # Move to archive (consistent with CheckpointManager)
                            else:
                                checkpoint_file.unlink()  # Just remove
                            
                            result["removed"].append(str(checkpoint_file))
                        except Exception as e:
                            result["failed"].append((str(checkpoint_file), str(e)))
        
        result["archived"] = True
        total_archived = archived_count + len(result["removed"])
        
        if total_archived > 0:
            result["message"] = f"Archived {total_archived} checkpoint(s) and model file(s)"
            if result["failed"]:
                result["message"] += f", {len(result['failed'])} failed"
        else:
            result["message"] = "No checkpoints found to archive"
            
    except Exception as e:
        result["message"] = f"Error archiving checkpoints: {str(e)}"
        result["error"] = str(e)
    
    return result


def clear_all_training_data(archive_db: bool = True, clear_caches_flag: bool = True, clear_processed: bool = False, archive_checkpoints_flag: bool = True) -> dict:
    """
    Clear all training data for fresh start.
    
    Args:
        archive_db: If True, archive trading journal before clearing
        clear_caches_flag: If True, clear cache files
        clear_processed: If True, clear processed data directory
        archive_checkpoints_flag: If True, archive and remove all checkpoints
    
    Returns:
        Dictionary with comprehensive status
    """
    project_root = Path(__file__).parent.parent
    db_path = project_root / "logs" / "trading_journal.db"
    models_dir = project_root / "models"
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "journal": None,
        "caches": None,
        "processed_data": None,
        "checkpoints": None,
        "success": False,
        "message": ""
    }
    
    # Clear trading journal
    journal_result = clear_trading_journal(db_path, archive=archive_db)
    result["journal"] = journal_result
    
    # Archive and remove checkpoints
    if archive_checkpoints_flag:
        checkpoint_result = archive_checkpoints(models_dir, archive=True)
        result["checkpoints"] = checkpoint_result
    
    # Clear caches
    # CRITICAL FIX: Parameter renamed to 'clear_caches_flag' to avoid shadowing the function name
    if clear_caches_flag:
        cache_result = clear_caches()  # This now correctly refers to the function, not the parameter
        result["caches"] = cache_result
    
    # Clear processed data (optional)
    if clear_processed:
        processed_dir = project_root / "data" / "processed"
        processed_result = {"cleared": False, "message": ""}
        try:
            if processed_dir.exists():
                for item in processed_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                processed_result["cleared"] = True
                processed_result["message"] = "Processed data directory cleared"
            else:
                processed_result["message"] = "Processed data directory does not exist"
        except Exception as e:
            processed_result["message"] = f"Error clearing processed data: {str(e)}"
            processed_result["error"] = str(e)
        
        result["processed_data"] = processed_result
    
    # Determine overall success
    result["success"] = (
        journal_result.get("cleared", False) and
        (not clear_caches_flag or result["caches"] is None or len(result["caches"].get("cleared_files", [])) > 0 or len(result["caches"].get("failed_files", [])) == 0) and
        (not archive_checkpoints_flag or result["checkpoints"] is None or len(result["checkpoints"].get("removed", [])) > 0 or result["checkpoints"].get("checkpoint_count", 0) == 0)
    )
    
    # Build summary message
    messages = []
    if journal_result.get("message"):
        messages.append(journal_result["message"])
    if archive_checkpoints_flag and result["checkpoints"] and result["checkpoints"].get("message"):
        messages.append(result["checkpoints"]["message"])
    if clear_caches_flag and result["caches"] and result["caches"].get("message"):
        messages.append(result["caches"]["message"])
    if clear_processed and result["processed_data"].get("message"):
        messages.append(result["processed_data"]["message"])
    
    result["message"] = " | ".join(messages) if messages else "No data cleared"
    
    return result

