"""
NT8 Data Extraction Module

Handles data extraction from NinjaTrader 8 for training and inference.
Supports both historical data export and real-time streaming.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

from src.trading_hours import TradingHoursManager
from src.utils.colors import error, warn

@dataclass
class MarketBar:
    """Single market bar (OHLCV)"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume
        }


class DataExtractor:
    """Extract and process data from NT8"""
    
    def __init__(self, data_dir: str = "data", nt8_data_path: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # NT8 data folder mapping (supports UNC paths and local paths)
        self.nt8_data_path = None
        
        # Track files used for archiving after training
        self.used_data_files = []  # List of (source_path, local_path) tuples
        
        # Load from settings.json if not provided directly
        if not nt8_data_path:
            try:
                import json
                settings_file = Path("settings.json")
                if settings_file.exists():
                    with open(settings_file, 'r') as f:
                        settings = json.load(f)
                        nt8_data_path = settings.get("nt8_data_path")
            except Exception:
                pass  # Settings file not found or invalid - that's okay
        
        if nt8_data_path:
            try:
                # Convert to Path - supports both UNC (\\server\share) and local paths
                # Handle UNC paths
                if nt8_data_path.startswith("\\\\") or nt8_data_path.startswith("//"):
                    # UNC path - use as is
                    self.nt8_data_path = Path(nt8_data_path)
                elif nt8_data_path.startswith("file://"):
                    # URI path
                    from urllib.parse import urlparse
                    parsed = urlparse(nt8_data_path)
                    self.nt8_data_path = Path(parsed.path)
                else:
                    # Regular path
                    self.nt8_data_path = Path(nt8_data_path)
                
                # Note: Don't verify path exists here - it might be a network path
                # that's not currently mounted, or might be created later
                # We'll check when actually trying to load files
            except Exception as e:
                print(f"Warning: Could not set up NT8 data path: {e}")
                self.nt8_data_path = None
    
    def load_historical_data(
        self,
        instrument: str,
        timeframe: int,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        trading_hours: Optional[TradingHoursManager] = None,
    ) -> pd.DataFrame:
        """
        Load historical data from CSV/TXT file exported from NT8.
        
        Supports multiple NT8 export formats:
        - Standard: ES_1min.csv, ES_1min.txt
        - NT8 semicolon: ES_1min.cvs.txt (single column with semicolons)
        - NT8 date-range: ES 12-14.Last.txt (contains multiple timeframes)
        """
        """
        Load historical data from CSV file exported from NT8.
        
        Args:
            instrument: Instrument symbol (e.g., "ES", "MES")
            timeframe: Timeframe in minutes (1, 5, 15, etc.)
            start_date: Start date (YYYY-MM-DD) or None for all data
            end_date: End date (YYYY-MM-DD) or None for all data
        
        Returns:
            DataFrame with OHLCV data
        """
        # Look for data file - check both .csv and .txt formats (NT8 exports .txt)
        filename_csv = f"{instrument}_{timeframe}min.csv"
        filename_txt = f"{instrument}_{timeframe}min.txt"
        base_filename = f"{instrument}_{timeframe}min"
        
        # NT8 also exports with date ranges like "ES 12-14.Last.txt" - these contain all timeframes
        # We'll need to look for files matching the instrument and extract the right timeframe
        
        # Helper to find and load file (flexible pattern matching)
        def find_and_load_file(filename_variants, search_paths):
            """Find file in search paths using flexible pattern matching"""
            for search_path in search_paths:
                if not search_path.exists():
                    continue
                
                # First try exact matches
                for filename in filename_variants:
                    filepath = search_path / filename
                    if filepath.exists():
                        return filepath
                    
                    # Also try variations like .cvs.txt (NT8 typo), .csv.txt, etc.
                    base_name = filename.rsplit('.', 1)[0]  # Remove extension
                    for ext_variant in ['.cvs.txt', '.csv.txt', '.cvs', filename.split('.')[-1]]:
                        variant_path = search_path / f"{base_name}{ext_variant}"
                        if variant_path.exists():
                            return variant_path
                
                # Then search all files in directory for pattern matches
                instrument_lower = instrument.lower()
                timeframe_str = str(timeframe)
                
                # Patterns to match: ES, 1min, .txt or .csv
                matching_files = []
                
                # Get all files once (avoid repeated iterdir calls) and filter early
                all_files = list(search_path.iterdir())
                total_files = len(all_files)
                if total_files > 10:
                    print(f"  Searching {total_files} files for {instrument} {timeframe}min data...")
                
                # Early filter: only files with instrument name and valid extensions
                relevant_files = []
                for actual_file in all_files:
                    if not actual_file.is_file():
                        continue
                    name_lower = actual_file.name.lower()
                    ext = actual_file.suffix.lower()
                    # Quick filter: must have instrument name and be .txt/.csv
                    if instrument_lower in name_lower and (ext in ['.txt', '.csv'] or name_lower.endswith(('.txt', '.csv', '.cvs.txt', '.csv.txt'))):
                        relevant_files.append(actual_file)
                
                if total_files > 10:
                    print(f"  Filtered to {len(relevant_files)} relevant files containing '{instrument}'")
                
                timeframe_str = str(timeframe)
                
                for actual_file in relevant_files:
                    name_lower = actual_file.name.lower()
                    name_no_ext = actual_file.stem.lower()  # filename without extension
                    # Handle double extensions like .cvs.txt
                    if '.' in name_no_ext:
                        # Remove last extension (if it's something like .cvs.txt)
                        name_base = name_no_ext.rsplit('.', 1)[0]
                    else:
                        name_base = name_no_ext
                    
                    # Priority 1: Exact match on expected pattern
                    if timeframe_str in name_lower and ('min' in name_lower or 'm' in name_lower):
                        # High confidence match
                        matching_files.append((actual_file, 10))
                        continue
                    
                    # Priority 2: Contains instrument and timeframe number
                    if timeframe_str in name_base:
                        # Medium confidence - might be a match
                        matching_files.append((actual_file, 5))
                        continue
                    
                    # Priority 3: Contains instrument and any number (more flexible)
                    if re.search(r'\d', name_base):
                        # Lower confidence, but could still match
                        # Check if the number could be the timeframe
                        numbers = re.findall(r'\d+', name_base)
                        if any(timeframe_str in num or num == timeframe_str for num in numbers):
                            matching_files.append((actual_file, 3))
                    
                    # Priority 4: NT8 date-range files like "ES 12-14.Last.txt" or "ES 03-25.Last.txt"
                    # These files typically contain the "Last" contract data with multiple timeframes
                    # We need to check the file contents to see if it has the requested timeframe
                    if name_lower.startswith(instrument_lower.lower()) and '.last' in name_lower:
                        # Check if file might contain our timeframe by checking filename
                        # Filename might be "ES 12-14.Last.txt" - the numbers are dates, not timeframes
                        # We'll try to load it and filter by timeframe from the data
                        matching_files.append((actual_file, 2))
                    
                    # Priority 5: Any file starting with instrument (catch-all)
                    if name_lower.startswith(instrument_lower.lower()):
                        # Very low priority - might match but need to verify contents
                        matching_files.append((actual_file, 1))
                
                # Return highest confidence match
                if matching_files:
                    # Sort by confidence (higher is better)
                    matching_files.sort(key=lambda x: x[1], reverse=True)
                    best_match = matching_files[0][0]
                    if total_files > 10:
                        print(f"  [OK] Found {instrument} {timeframe}min data: {best_match.name} (from {total_files} files)")
                    return best_match
                elif total_files > 10:
                    print(warn(f"  [WARN] No matching file found for {instrument} {timeframe}min in {total_files} files"))
            
            return None
        
        # Priority 1: Local data/raw folder (check both .csv and .txt)
        local_file = find_and_load_file([filename_csv, filename_txt], [self.raw_dir])
        if local_file:
            # Track this file for archiving
            self.used_data_files.append(("local_file", str(local_file)))
            return self._load_data_file(
                local_file,
                start_date,
                end_date,
                timeframe,
                trading_hours=trading_hours,
                instrument=instrument,
            )
        
        # Priority 2: Mapped NT8 data folder
        if self.nt8_data_path:
            nt8_file = find_and_load_file([filename_txt, filename_csv], [self.nt8_data_path])
            if nt8_file:
                print(f"Found data in NT8 folder: {nt8_file}")
                # Track this file for archiving
                self.used_data_files.append(("nt8_source", str(nt8_file)))
                
                # Copy to local folder for faster access next time
                try:
                    import shutil
                    self.raw_dir.mkdir(parents=True, exist_ok=True)
                    # Always copy as .csv to local folder for consistency
                    local_copy = self.raw_dir / filename_csv
                    shutil.copy2(nt8_file, local_copy)
                    print(f"Copied to local folder: {local_copy}")
                    # Track local copy too
                    self.used_data_files.append(("local_copy", str(local_copy)))
                    return self._load_data_file(
                        local_copy,
                        start_date,
                        end_date,
                        timeframe,
                        trading_hours=trading_hours,
                        instrument=instrument,
                    )
                except Exception as e:
                    print(f"Warning: Could not copy file, using directly: {e}")
                    return self._load_data_file(
                        nt8_file,
                        start_date,
                        end_date,
                        timeframe,
                        trading_hours=trading_hours,
                        instrument=instrument,
                    )
        
        # Priority 3: Try common NT8 export locations
        common_paths = [
            Path("C:/Users/schuo/Documents/NinjaTrader 8/export"),  # Current user path
            Path("C:/Users/sovan/Documents/NinjaTrader 8/export"),  # Previous user path
            Path.home() / "Documents" / "NinjaTrader 8" / "export",  # lowercase
            Path.home() / "Documents" / "NinjaTrader 8" / "Export",  # capitalized
            Path.home() / "Documents" / "NinjaTrader 8" / "db",
            Path("C:/Users/Public/Documents/NinjaTrader 8/Export"),
        ]
        
        for common_path in common_paths:
            if common_path.exists():
                nt8_file = find_and_load_file([filename_txt, filename_csv], [common_path])
                if nt8_file:
                    print(f"Found data in NT8 export folder: {nt8_file}")
                    # Track this file for archiving
                    self.used_data_files.append(("nt8_source", str(nt8_file)))
                    
                    # Copy to local folder
                    try:
                        import shutil
                        self.raw_dir.mkdir(parents=True, exist_ok=True)
                        local_copy = self.raw_dir / filename_csv
                        shutil.copy2(nt8_file, local_copy)
                        print(f"Copied to local folder: {local_copy}")
                        # Track local copy too
                        self.used_data_files.append(("local_copy", str(local_copy)))
                        return self._load_data_file(
                            local_copy,
                            start_date,
                            end_date,
                            timeframe,
                            trading_hours=trading_hours,
                            instrument=instrument,
                        )
                    except Exception as e:
                        print(f"Warning: Could not copy file, using directly: {e}")
                    return self._load_data_file(
                        nt8_file,
                        start_date,
                        end_date,
                        timeframe,
                        trading_hours=trading_hours,
                        instrument=instrument,
                    )
        
        # Not found anywhere - provide detailed error with file listing
        checked_locations = []
        checked_locations.append(f"  1. {self.raw_dir / filename_csv}")
        checked_locations.append(f"      {self.raw_dir / filename_txt}")
        
        # List files found in NT8 directories for debugging
        all_search_paths = []
        if self.nt8_data_path and self.nt8_data_path.exists():
            all_search_paths.append(self.nt8_data_path)
        all_search_paths.extend([p for p in common_paths if p.exists()])
        
        error_msg = f"Data file not found: {base_filename} (checked both .txt and .csv)\n\n"
        error_msg += f"Checked locations:\n"
        error_msg += f"  1. {self.raw_dir}\n"
        
        for idx, search_path in enumerate(all_search_paths, start=2):
            error_msg += f"  {idx}. {search_path}\n"
            
            # List actual files in this directory to help user
            try:
                files = [f.name for f in search_path.iterdir() if f.is_file() and f.suffix.lower() in ['.txt', '.csv']]
                if files:
                    error_msg += f"      Files found: {', '.join(files[:10])}"  # Show first 10
                    if len(files) > 10:
                        error_msg += f" ... and {len(files) - 10} more"
                    error_msg += "\n"
            except:
                pass
        
        error_msg += f"\nExpected filename pattern: *{instrument}*{timeframe}*min*.txt or *.csv\n"
        error_msg += f"\nPlease either:\n"
        error_msg += f"  - Export data from NT8 (saves as .txt) - check filename matches pattern\n"
        error_msg += f"  - Set NT8 data folder path in Settings if files are in a different location\n"
        error_msg += f"  - Or manually copy/rename files to: {self.raw_dir / filename_txt}"
        raise FileNotFoundError(error_msg)
    
    def _load_data_file(
        self,
        filepath: Path,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: int = 1,
        trading_hours: Optional[TradingHoursManager] = None,
        instrument: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load and process data file (CSV or TXT from NT8)"""
        
        # Determine file format - check if it's .txt or has .txt in the name
        is_txt = filepath.suffix.lower() == '.txt' or '.txt' in filepath.name.lower()
        
        # Try to read first line to detect format
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                second_line = f.readline().strip() if first_line else None
        except:
            first_line = None
            second_line = None
        
        # Check if this is a multi-timeframe file (date-range format like "ES 12-14.Last.txt")
        is_multi_timeframe_file = '.last' in filepath.name.lower() or 'last' in filepath.name.lower()
        
        # Detect NT8's semicolon-separated format (single column with semicolons)
        # Format: "20241215 213000;6121.5;6121.5;6121.5;6121.5;1"
        if first_line and ';' in first_line and not any(char in first_line for char in ['\t', ',']) and len(first_line.split(';')) >= 5:
            # NT8 semicolon-separated format - parse manually
            data_rows = []
            current_section_timeframe = None
            
            try:
                # For large multi-timeframe files, show progress
                file_size = filepath.stat().st_size if filepath.exists() else 0
                is_large_file = file_size > 1_000_000  # > 1MB
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines_processed = 0
                    matching_rows = 0
                    
                    for line_num, line in enumerate(f, 1):
                        # Show progress every 100k lines for large files
                        if is_large_file and line_num % 100000 == 0:
                            print(f"    Processing {filepath.name}: {line_num:,} lines read, {matching_rows:,} rows matched...")
                        
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Check if this line indicates a timeframe section
                        # NT8 formats: "15 min", "Timeframe: 15", "15 Minute", etc.
                        line_lower = line.lower()
                        timeframe_match = re.search(r'(?:timeframe|tf)[\s:=]*(\d+)|(\d+)\s*min', line_lower)
                        if timeframe_match:
                            # Found a timeframe section header
                            tf_value = timeframe_match.group(1) or timeframe_match.group(2)
                            current_section_timeframe = int(tf_value) if tf_value else None
                            continue
                        
                        # Skip header rows (Time, Open, High, Low, Close, Volume)
                        if any(word in line_lower for word in ['time', 'open', 'high', 'low', 'close', 'volume']) and ';' not in line:
                            continue
                        
                        # Skip lines that don't have semicolons or have too few parts
                        if ';' not in line or len(line.split(';')) < 5:
                            continue
                        
                        parts = line.split(';')
                        if len(parts) >= 6:  # Date/Time;Open;High;Low;Close;Volume
                            try:
                                # If we're in a multi-timeframe file, only process matching timeframe
                                if is_multi_timeframe_file:
                                    if current_section_timeframe is None or current_section_timeframe != timeframe:
                                        continue
                                
                                # Parse date/time: "20241215 213000" -> "2024-12-15 21:30:00"
                                dt_str = parts[0].strip()
                                if len(dt_str) >= 14:
                                    # Format: YYYYMMDD HHMMSS
                                    date_str = dt_str[:8]  # YYYYMMDD
                                    time_str = dt_str[9:15] if len(dt_str) > 8 else "000000"  # HHMMSS
                                    timestamp = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
                                    
                                    data_rows.append({
                                        'timestamp': timestamp,
                                        'open': float(parts[1]),
                                        'high': float(parts[2]),
                                        'low': float(parts[3]),
                                        'close': float(parts[4]),
                                        'volume': float(parts[5]) if len(parts) > 5 else 0.0
                                    })
                                    matching_rows += 1
                            except (ValueError, IndexError) as e:
                                # Skip malformed lines
                                continue
                
                if is_large_file:
                    print(f"    [OK] Finished processing {filepath.name}: {line_num:,} total lines, {matching_rows:,} rows extracted")
                
                if data_rows:
                    df = pd.DataFrame(data_rows)
                    # If we're in a multi-timeframe file but got no data, the timeframe wasn't found
                    if len(df) == 0 and is_multi_timeframe_file:
                        raise ValueError(
                            f"File '{filepath.name}' contains data but no {timeframe}min timeframe found. "
                            f"Please export {timeframe}min data separately from NT8, or ensure the file contains a section labeled '{timeframe} min'."
                        )
                else:
                    if is_multi_timeframe_file:
                        instrument_str = instrument if instrument else "ES"
                        raise ValueError(
                            f"No {timeframe}min data found in file '{filepath.name}'. "
                            f"This file may contain different timeframes. Please export {timeframe}min data from NT8 with a specific filename like '{instrument_str}_{timeframe}min.txt'."
                        )
                    else:
                        raise ValueError("No valid data rows found in semicolon-separated format")
            except ValueError:
                raise  # Re-raise ValueError as-is
            except Exception as e:
                raise ValueError(f"Failed to parse NT8 semicolon-separated format: {e}")
        
        elif is_txt:
            # NT8 text files - try different separators
            try:
                # First try tab-separated (most common NT8 format)
                df = pd.read_csv(filepath, sep='\t', skipinitialspace=True, on_bad_lines='skip')
            except:
                try:
                    # Try comma-separated
                    df = pd.read_csv(filepath, sep=',', skipinitialspace=True, on_bad_lines='skip')
                except:
                    # Try space-separated
                    df = pd.read_csv(filepath, sep=r'\s+', engine='python', on_bad_lines='skip')
        else:
            # CSV file
            try:
                df = pd.read_csv(filepath, on_bad_lines='skip')
            except:
                # Try semicolon-separated CSV
                df = pd.read_csv(filepath, sep=';', on_bad_lines='skip')
        
        # Standardize column names (NT8 exports may vary)
        # Convert to lowercase for matching
        df.columns = [col.strip().lower() for col in df.columns]
        
        # Handle NT8's common column variations
        # NT8 often has: Date, Time (separate) or DateTime (combined)
        if "date" in df.columns and "time" in df.columns:
            # Combine Date and Time columns
            df["timestamp"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
            df = df.drop(columns=["date", "time"])
        elif "datetime" in df.columns:
            df = df.rename(columns={"datetime": "timestamp"})
        elif "time" in df.columns:
            df = df.rename(columns={"time": "timestamp"})
        elif "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})
        
        # Map common NT8 column name variations
        column_mapping = {
            "o": "open",
            "h": "high", 
            "l": "low",
            "c": "close",
            "v": "volume",
            "vol": "volume"
        }
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Ensure required columns exist
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}\n"
                f"Found columns: {df.columns.tolist()}\n"
                f"File: {filepath}"
            )
        
        # Parse timestamp if it exists
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
            # Remove rows with invalid timestamps
            df = df.dropna(subset=["timestamp"])
        else:
            # If no timestamp, create index-based timestamps
            df["timestamp"] = pd.date_range(start=start_date or "2020-01-01", periods=len(df), freq=f"{timeframe}min")
        
        # Filter by date if specified
        if start_date:
            df = df[df["timestamp"] >= start_date]
        if end_date:
            df = df[df["timestamp"] <= end_date]
        
        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Validate data
        df = self._validate_data(df)

        if trading_hours:
            df = trading_hours.filter_dataframe(df)
        
        # Safety: Track this file if not already tracked (backup mechanism)
        # This ensures we catch files loaded from any path, even if tracking was missed earlier
        filepath_str = str(filepath.resolve())
        already_tracked = any(
            str(Path(tracked_path).resolve()) == filepath_str 
            for _, tracked_path in self.used_data_files
        )
        if not already_tracked:
            # Determine file type based on path
            if "NinjaTrader" in str(filepath) or "export" in str(filepath).lower():
                self.used_data_files.append(("nt8_source", filepath_str))
            else:
                self.used_data_files.append(("local_file", filepath_str))
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean data.
        
        CRITICAL FIX #4: Enhanced price validation to prevent training crashes.
        Validates:
        - Zero/negative prices (critical for division operations)
        - NaN/Inf values in price columns
        - Extreme price jumps (>50% likely data error)
        - Price consistency (high >= low, OHLC within bounds)
        """
        if df.empty:
            return df
        
        original_len = len(df)
        
        # CRITICAL FIX #4: Check for zero or negative prices BEFORE other operations
        # These cause division by zero errors in PnL calculations
        price_columns = ["open", "high", "low", "close"]
        
        # Remove rows with zero or negative prices
        for col in price_columns:
            if col in df.columns:
                df = df[df[col] > 0]  # Must be strictly positive
        
        # CRITICAL FIX #4: Check for NaN/Inf values in price columns (per column, not just rows)
        for col in price_columns:
            if col in df.columns:
                # Remove rows with NaN or Inf values
                df = df[df[col].notna()]  # Remove NaN
                df = df[np.isfinite(df[col])]  # Remove Inf
        
        # Remove NaN rows (general cleanup)
        df = df.dropna()
        
        # CRITICAL FIX #4: Detect and remove extreme price jumps (>50% likely data error)
        if len(df) > 1 and "close" in df.columns:
            # Calculate price change percentage between consecutive bars
            price_changes = df["close"].pct_change().abs()
            # Remove bars with >50% price change (likely data error)
            extreme_jumps = price_changes > 0.5
            if extreme_jumps.any():
                num_extreme = extreme_jumps.sum()
                print(warn(f"[WARN] Data validation: Removed {num_extreme} bars with >50% price jumps (likely data errors)"))
                df = df[~extreme_jumps]
        
        # Ensure high >= low
        df = df[df["high"] >= df["low"]]
        
        # Ensure OHLC values are within high/low bounds
        df.loc[df["open"] > df["high"], "open"] = df["high"]
        df.loc[df["open"] < df["low"], "open"] = df["low"]
        df.loc[df["close"] > df["high"], "close"] = df["high"]
        df.loc[df["close"] < df["low"], "close"] = df["low"]
        
        # Remove zero/negative volume (if any)
        if "volume" in df.columns:
            df = df[df["volume"] > 0]
        
        # Log validation summary
        removed_count = original_len - len(df)
        if removed_count > 0:
            print(f"[INFO] Data validation: Removed {removed_count} invalid rows ({original_len} -> {len(df)} bars)")
        
        return df.reset_index(drop=True)
    
    def _resample_timeframe(self, df: pd.DataFrame, target_timeframe: int) -> pd.DataFrame:
        """
        Resample a DataFrame to a higher timeframe.
        
        Args:
            df: DataFrame with OHLCV data and timestamp column
            target_timeframe: Target timeframe in minutes (e.g., 5 for 5-minute bars)
        
        Returns:
            Resampled DataFrame with target timeframe bars
        """
        if df.empty:
            return df
        
        # Ensure timestamp column exists and is datetime
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame must have a 'timestamp' column for resampling")
        
        # Convert timestamp to datetime if needed
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Remove rows with invalid timestamps
        df = df.dropna(subset=['timestamp'])
        
        if df.empty:
            return df
        
        # Set timestamp as index for resampling
        df_indexed = df.set_index('timestamp')
        
        # Ensure we have OHLCV columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df_indexed.columns]
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns for resampling: {missing_columns}")
        
        # Resample to target timeframe
        # Open: first value of the period
        # High: maximum high in the period
        # Low: minimum low in the period
        # Close: last close in the period
        # Volume: sum of volumes in the period
        resampled = df_indexed.resample(f'{target_timeframe}min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()
        
        return resampled
    
    def load_multi_timeframe_data(
        self,
        instrument: str,
        timeframes: List[int],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        trading_hours: Optional[TradingHoursManager] = None,
    ) -> Dict[int, pd.DataFrame]:
        """
        Load data for multiple timeframes.
        
        Args:
            instrument: Instrument symbol
            timeframes: List of timeframes in minutes (e.g., [1, 5, 15])
            start_date: Start date (optional)
            end_date: End date (optional)
        
        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        data = {}
        total_tfs = len(timeframes)
        # Load 1-minute data first if it's in the requested timeframes
        # This will be used as a fallback to resample higher timeframes
        base_timeframe = min(timeframes)
        base_data = None
        
        for idx, tf in enumerate(timeframes, 1):
            print(f"  Loading {tf}min data ({idx}/{total_tfs})...")
            import sys
            sys.stdout.flush()
            try:
                data[tf] = self.load_historical_data(
                    instrument,
                    tf,
                    start_date,
                    end_date,
                    trading_hours=trading_hours,
                )
                print(f"  [OK] {tf}min data loaded: {len(data[tf])} bars")
                sys.stdout.flush()
                
                # Store base timeframe data for resampling
                if tf == base_timeframe:
                    base_data = data[tf].copy()
            except Exception as e:
                # Fallback: Try to resample from base timeframe (usually 1-minute)
                if tf > base_timeframe and base_timeframe in timeframes:
                    try:
                        print(f"  [INFO] {tf}min data not found in file, resampling from {base_timeframe}min data...")
                        sys.stdout.flush()
                        if base_data is None:
                            # Load base timeframe data first if not already loaded
                            print(f"  Loading {base_timeframe}min data for resampling...")
                            base_data = self.load_historical_data(
                                instrument,
                                base_timeframe,
                                start_date,
                                end_date,
                                trading_hours=trading_hours,
                            )
                            print(f"  [OK] {base_timeframe}min data loaded: {len(base_data)} bars")
                        
                        # Resample base timeframe data to requested timeframe
                        data[tf] = self._resample_timeframe(base_data, tf)
                        print(f"  [OK] Resampled {tf}min data: {len(data[tf])} bars (from {len(base_data)} {base_timeframe}min bars)")
                        sys.stdout.flush()
                    except Exception as resample_error:
                        # Only show warning if resampling also fails
                        print(warn(f"  [WARN] Failed to load {tf}min data directly: {e}"))
                        print(error(f"  [ERROR] Failed to resample {tf}min data: {resample_error}"))
                        sys.stdout.flush()
                        raise ValueError(f"Failed to load {tf}min data: {e}. Resampling also failed: {resample_error}")
                else:
                    # Can't resample, show warning and re-raise
                    print(warn(f"  [WARN] Failed to load {tf}min data directly: {e}"))
                    print(warn(f"  [WARN] Cannot resample (base timeframe {base_timeframe}min not available)"))
                    sys.stdout.flush()
                    raise
        
        return data
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Save processed data to processed directory"""
        filepath = self.processed_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Saved processed data to: {filepath}")
    
    def parse_realtime_bar(self, bar_data: Dict) -> MarketBar:
        """
        Parse real-time bar data from NT8 socket message.
        
        Args:
            bar_data: Dictionary with bar data from NT8
        
        Returns:
            MarketBar object
        """
        return MarketBar(
            timestamp=datetime.fromisoformat(bar_data["timestamp"]),
            open=float(bar_data["open"]),
            high=float(bar_data["high"]),
            low=float(bar_data["low"]),
            close=float(bar_data["close"]),
            volume=float(bar_data["volume"])
        )


# Example usage
if __name__ == "__main__":
    extractor = DataExtractor()
    
    # Example: Load ES 1-minute data
    try:
        df = extractor.load_historical_data("ES", 1)
        print(f"Loaded {len(df)} bars of ES 1-minute data")
        print(df.head())
        print(f"\nDate range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo get started:")
        print("1. Open NinjaTrader 8")
        print("2. Go to Tools → Historical Data Manager")
        print("3. Export ES or MES data for 1min, 5min, 15min")
        print("4. Save CSV files as: data/raw/ES_1min.csv, ES_5min.csv, ES_15min.csv")
    
    # Example: Load multi-timeframe data
    try:
        multi_tf_data = extractor.load_multi_timeframe_data("ES", [1, 5, 15])
        print(f"\nLoaded multi-timeframe data:")
        for tf, df in multi_tf_data.items():
            print(f"  {tf}min: {len(df)} bars")
    except FileNotFoundError:
        print("\nMulti-timeframe data not available yet")

