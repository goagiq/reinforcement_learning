"""
Restore Archived Data Files

Restores all archived data files from Archive folders back to data/raw
for use in supervised pre-training.
"""

import shutil
from pathlib import Path
from src.utils.colors import success, info, warn, error

def restore_archived_data():
    """Restore all archived data files to data/raw"""
    
    project_root = Path(__file__).parent
    archive_dir = project_root / "data" / "raw" / "Archive"
    raw_dir = project_root / "data" / "raw"
    
    if not archive_dir.exists():
        print(warn(f"[RESTORE] Archive directory not found: {archive_dir}"))
        return
    
    print(info("\n" + "="*70))
    print(info("RESTORING ARCHIVED DATA FILES"))
    print(info("="*70))
    
    # Find all archived files
    archived_files = []
    for archive_subdir in archive_dir.iterdir():
        if archive_subdir.is_dir():
            print(info(f"\n[RESTORE] Checking archive: {archive_subdir.name}"))
            for file_path in archive_subdir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in ['.csv', '.txt']:
                    archived_files.append((file_path, archive_subdir.name))
                    print(info(f"  Found: {file_path.name}"))
    
    if not archived_files:
        print(warn("[RESTORE] No archived data files found"))
        return
    
    print(info(f"\n[RESTORE] Found {len(archived_files)} archived file(s) to restore"))
    
    # Restore files
    restored_count = 0
    skipped_count = 0
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path, archive_name in archived_files:
        dest_path = raw_dir / file_path.name
        
        # Check if file already exists
        if dest_path.exists():
            print(warn(f"  [SKIP] {file_path.name} already exists in data/raw, skipping"))
            skipped_count += 1
            continue
        
        try:
            # Copy file back to data/raw
            shutil.copy2(file_path, dest_path)
            print(success(f"  [RESTORED] {file_path.name} from {archive_name}"))
            restored_count += 1
        except Exception as e:
            print(error(f"  [ERROR] Failed to restore {file_path.name}: {e}"))
    
    print(info("\n" + "="*70))
    print(success(f"[RESTORE] Restore complete!"))
    print(info(f"  Restored: {restored_count} file(s)"))
    print(info(f"  Skipped: {skipped_count} file(s) (already exist)"))
    print(info("="*70 + "\n"))
    
    return restored_count

if __name__ == "__main__":
    restore_archived_data()

