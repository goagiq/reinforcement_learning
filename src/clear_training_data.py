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


def clear_all_training_data(archive_db: bool = True, clear_caches_flag: bool = True, clear_processed: bool = False) -> dict:
    """
    Clear all training data for fresh start.
    
    Args:
        archive_db: If True, archive trading journal before clearing
        clear_caches_flag: If True, clear cache files
        clear_processed: If True, clear processed data directory
    
    Returns:
        Dictionary with comprehensive status
    """
    project_root = Path(__file__).parent.parent
    db_path = project_root / "logs" / "trading_journal.db"
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "journal": None,
        "caches": None,
        "processed_data": None,
        "success": False,
        "message": ""
    }
    
    # Clear trading journal
    journal_result = clear_trading_journal(db_path, archive=archive_db)
    result["journal"] = journal_result
    
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
        (not clear_caches_flag or result["caches"] is None or len(result["caches"].get("cleared_files", [])) > 0 or len(result["caches"].get("failed_files", [])) == 0)
    )
    
    # Build summary message
    messages = []
    if journal_result.get("message"):
        messages.append(journal_result["message"])
    if clear_caches_flag and result["caches"] and result["caches"].get("message"):
        messages.append(result["caches"]["message"])
    if clear_processed and result["processed_data"].get("message"):
        messages.append(result["processed_data"]["message"])
    
    result["message"] = " | ".join(messages) if messages else "No data cleared"
    
    return result

