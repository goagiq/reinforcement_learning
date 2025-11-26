"""
Trading Journal - Non-intrusive trade logging and analysis

Captures trade-level data during training without impacting performance.
Stores data in SQLite database for persistence and analysis.
"""

import sqlite3
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import deque
import time


@dataclass
class TradeEntry:
    """Single trade entry"""
    trade_id: Optional[int] = None
    timestamp: Optional[str] = None
    episode: int = 0
    step: int = 0
    entry_price: float = 0.0
    exit_price: float = 0.0
    position_size: float = 0.0
    pnl: float = 0.0
    commission: float = 0.0
    net_pnl: float = 0.0
    strategy: str = "RL"  # RL, Elliott Wave, Contrarian, etc.
    strategy_confidence: float = 0.0
    is_win: bool = False
    duration_steps: int = 0
    entry_timestamp: Optional[str] = None
    exit_timestamp: Optional[str] = None
    market_conditions: Optional[str] = None  # JSON string
    decision_metadata: Optional[str] = None  # JSON string


class TradingJournal:
    """
    Non-intrusive trading journal that captures trade data.
    
    Features:
    - SQLite database for persistence
    - Background thread for async writes (non-blocking)
    - Tracks individual trades with strategy information
    - Calculates equity curve and statistics
    - Thread-safe operations
    """
    
    def __init__(self, db_path: str = "logs/trading_journal.db"):
        """Initialize trading journal"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread-safe queue for async writes
        self.write_queue = deque()
        self.write_lock = threading.Lock()
        self.running = False
        self.background_thread: Optional[threading.Thread] = None
        
        # Initialize database
        self._init_database()
        
        # Track open positions (for calculating duration)
        self.open_positions: Dict[int, Dict] = {}  # episode -> position info
        
        # Statistics cache
        self._stats_cache: Optional[Dict] = None
        self._cache_timestamp: Optional[float] = None
        self._cache_ttl = 5.0  # Cache for 5 seconds
        
    def _init_database(self):
        """Initialize SQLite database schema"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                episode INTEGER NOT NULL,
                step INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                position_size REAL NOT NULL,
                pnl REAL NOT NULL,
                commission REAL NOT NULL,
                net_pnl REAL NOT NULL,
                strategy TEXT NOT NULL,
                strategy_confidence REAL NOT NULL,
                is_win INTEGER NOT NULL,
                duration_steps INTEGER NOT NULL,
                entry_timestamp TEXT,
                exit_timestamp TEXT,
                market_conditions TEXT,
                decision_metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Episodes table (for equity curve)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id INTEGER PRIMARY KEY,
                episode_number INTEGER NOT NULL,
                start_timestamp TEXT NOT NULL,
                end_timestamp TEXT,
                total_trades INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0.0,
                final_equity REAL,
                max_drawdown REAL DEFAULT 0.0,
                win_rate REAL DEFAULT 0.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Equity curve points (for charting)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS equity_curve (
                point_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                episode INTEGER NOT NULL,
                step INTEGER NOT NULL,
                equity REAL NOT NULL,
                cumulative_pnl REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_episode 
            ON trades(episode)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_timestamp 
            ON trades(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_equity_curve_episode 
            ON equity_curve(episode, step)
        """)
        
        conn.commit()
        
        # Verify tables were created successfully
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name IN ('trades', 'equity_curve', 'episodes')
        """)
        existing_tables = {row[0] for row in cursor.fetchall()}
        required_tables = {'trades', 'equity_curve', 'episodes'}
        missing_tables = required_tables - existing_tables
        
        if missing_tables:
            print(f"[WARN] Trading journal: Missing tables {missing_tables}, attempting to recreate...")
            # If tables are missing, something went wrong - log but don't fail
            # The CREATE TABLE IF NOT EXISTS should have created them
        
        conn.close()
        
    def start(self):
        """Start background thread for async writes"""
        if self.running:
            return
        
        self.running = True
        self.background_thread = threading.Thread(
            target=self._background_writer,
            daemon=True,
            name="TradingJournalWriter"
        )
        self.background_thread.start()
        print("[OK] Trading Journal started (background writer)")
    
    def stop(self):
        """Stop background thread and flush queue"""
        self.running = False
        if self.background_thread:
            self.background_thread.join(timeout=5.0)
        self._flush_queue()
        print("[OK] Trading Journal stopped")
    
    def _background_writer(self):
        """Background thread that writes queued entries"""
        while self.running:
            try:
                # Process queue
                batch = []
                with self.write_lock:
                    # Collect up to 100 entries or wait 1 second
                    for _ in range(100):
                        if self.write_queue:
                            batch.append(self.write_queue.popleft())
                        else:
                            break
                
                if batch:
                    self._write_batch(batch)
                
                # Sleep if queue is empty
                if not batch:
                    time.sleep(0.1)
            except Exception as e:
                print(f"[ERROR] Trading Journal background writer error: {e}")
                time.sleep(1.0)
    
    def _write_batch(self, entries: List[Dict]):
        """Write batch of entries to database"""
        if not entries:
            return
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            for entry in entries:
                if entry.get("type") == "trade":
                    self._insert_trade(cursor, entry["data"])
                elif entry.get("type") == "equity":
                    self._insert_equity_point(cursor, entry["data"])
                elif entry.get("type") == "episode":
                    self._insert_episode(cursor, entry["data"])
            
            conn.commit()
        except Exception as e:
            print(f"[ERROR] Trading Journal write error: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def _insert_trade(self, cursor, trade: TradeEntry):
        """Insert trade into database"""
        cursor.execute("""
            INSERT INTO trades (
                timestamp, episode, step, entry_price, exit_price,
                position_size, pnl, commission, net_pnl, strategy,
                strategy_confidence, is_win, duration_steps,
                entry_timestamp, exit_timestamp, market_conditions, decision_metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.timestamp or datetime.now().isoformat(),
            trade.episode,
            trade.step,
            trade.entry_price,
            trade.exit_price,
            trade.position_size,
            trade.pnl,
            trade.commission,
            trade.net_pnl,
            trade.strategy,
            trade.strategy_confidence,
            1 if trade.is_win else 0,
            trade.duration_steps,
            trade.entry_timestamp,
            trade.exit_timestamp,
            trade.market_conditions,
            trade.decision_metadata
        ))
    
    def _insert_equity_point(self, cursor, data: Dict):
        """Insert equity curve point"""
        cursor.execute("""
            INSERT INTO equity_curve (timestamp, episode, step, equity, cumulative_pnl)
            VALUES (?, ?, ?, ?, ?)
        """, (
            data["timestamp"],
            data["episode"],
            data["step"],
            data["equity"],
            data["cumulative_pnl"]
        ))
    
    def _insert_episode(self, cursor, data: Dict):
        """Insert or update episode summary"""
        cursor.execute("""
            INSERT OR REPLACE INTO episodes (
                episode_number, start_timestamp, end_timestamp,
                total_trades, total_pnl, final_equity, max_drawdown, win_rate
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data["episode_number"],
            data["start_timestamp"],
            data.get("end_timestamp"),
            data.get("total_trades", 0),
            data.get("total_pnl", 0.0),
            data.get("final_equity"),
            data.get("max_drawdown", 0.0),
            data.get("win_rate", 0.0)
        ))
    
    def log_trade(
        self,
        episode: int,
        step: int,
        entry_price: float,
        exit_price: float,
        position_size: float,
        pnl: float,
        commission: float,
        strategy: str = "RL",
        strategy_confidence: float = 0.0,
        entry_timestamp: Optional[str] = None,
        exit_timestamp: Optional[str] = None,
        market_conditions: Optional[Dict] = None,
        decision_metadata: Optional[Dict] = None
    ):
        """Log a completed trade (non-blocking, queues for async write)"""
        net_pnl = pnl - commission
        is_win = net_pnl > 0
        
        # Calculate duration if we have entry info
        duration_steps = 0
        if entry_timestamp and exit_timestamp:
            try:
                entry_dt = datetime.fromisoformat(entry_timestamp)
                exit_dt = datetime.fromisoformat(exit_timestamp)
                duration_steps = int((exit_dt - entry_dt).total_seconds() / 60)  # Approximate steps
            except:
                pass
        
        trade = TradeEntry(
            timestamp=exit_timestamp or datetime.now().isoformat(),
            episode=episode,
            step=step,
            entry_price=entry_price,
            exit_price=exit_price,
            position_size=position_size,
            pnl=pnl,
            commission=commission,
            net_pnl=net_pnl,
            strategy=strategy,
            strategy_confidence=strategy_confidence,
            is_win=is_win,
            duration_steps=duration_steps,
            entry_timestamp=entry_timestamp,
            exit_timestamp=exit_timestamp,
            market_conditions=json.dumps(market_conditions) if market_conditions else None,
            decision_metadata=json.dumps(decision_metadata) if decision_metadata else None
        )
        
        # Queue for async write
        with self.write_lock:
            self.write_queue.append({
                "type": "trade",
                "data": trade
            })
    
    def log_equity_point(
        self,
        episode: int,
        step: int,
        equity: float,
        cumulative_pnl: float
    ):
        """Log equity curve point (non-blocking)"""
        with self.write_lock:
            self.write_queue.append({
                "type": "equity",
                "data": {
                    "timestamp": datetime.now().isoformat(),
                    "episode": episode,
                    "step": step,
                    "equity": equity,
                    "cumulative_pnl": cumulative_pnl
                }
            })
    
    def log_episode_summary(
        self,
        episode_number: int,
        start_timestamp: str,
        end_timestamp: Optional[str] = None,
        total_trades: int = 0,
        total_pnl: float = 0.0,
        final_equity: Optional[float] = None,
        max_drawdown: float = 0.0,
        win_rate: float = 0.0
    ):
        """Log episode summary (non-blocking)"""
        with self.write_lock:
            self.write_queue.append({
                "type": "episode",
                "data": {
                    "episode_number": episode_number,
                    "start_timestamp": start_timestamp,
                    "end_timestamp": end_timestamp,
                    "total_trades": total_trades,
                    "total_pnl": total_pnl,
                    "final_equity": final_equity,
                    "max_drawdown": max_drawdown,
                    "win_rate": win_rate
                }
            })
    
    def _flush_queue(self):
        """Flush remaining queue items (call before shutdown)"""
        batch = []
        with self.write_lock:
            while self.write_queue:
                batch.append(self.write_queue.popleft())
        
        if batch:
            self._write_batch(batch)
    
    def get_trades(
        self,
        episode: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
        since: Optional[str] = None
    ) -> List[Dict]:
        """Get trades from database"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if episode is not None:
            if since:
                cursor.execute("""
                    SELECT * FROM trades
                    WHERE episode = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                """, (episode, since, limit, offset))
            else:
                cursor.execute("""
                    SELECT * FROM trades
                    WHERE episode = ?
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                """, (episode, limit, offset))
        else:
            if since:
                cursor.execute("""
                    SELECT * FROM trades
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                """, (since, limit, offset))
            else:
                cursor.execute("""
                    SELECT * FROM trades
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_equity_curve(
        self,
        episode: Optional[int] = None,
        limit: int = 10000,
        since: Optional[str] = None
    ) -> List[Dict]:
        """Get equity curve points"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if episode is not None:
            if since:
                cursor.execute("""
                    SELECT * FROM equity_curve
                    WHERE episode = ? AND timestamp >= ?
                    ORDER BY step ASC
                    LIMIT ?
                """, (episode, since, limit))
            else:
                cursor.execute("""
                    SELECT * FROM equity_curve
                    WHERE episode = ?
                    ORDER BY step ASC
                    LIMIT ?
                """, (episode, limit))
        else:
            if since:
                # Get points since timestamp
                cursor.execute("""
                    SELECT * FROM equity_curve
                    WHERE timestamp >= ?
                    ORDER BY episode ASC, step ASC
                    LIMIT ?
                """, (since, limit))
            else:
                # Get latest points across all episodes
                cursor.execute("""
                    SELECT * FROM equity_curve
                    ORDER BY episode ASC, step ASC
                    LIMIT ?
                """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_statistics(self, force_refresh: bool = False) -> Dict:
        """Get trading statistics (cached for performance)"""
        # Check cache
        if not force_refresh and self._stats_cache:
            if time.time() - self._cache_timestamp < self._cache_ttl:
                return self._stats_cache
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Total trades
        cursor.execute("SELECT COUNT(*) FROM trades")
        total_trades = cursor.fetchone()[0]
        
        # Winning/losing trades
        cursor.execute("SELECT COUNT(*) FROM trades WHERE is_win = 1")
        winning_trades = cursor.fetchone()[0]
        losing_trades = total_trades - winning_trades
        
        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Total PnL
        cursor.execute("SELECT SUM(net_pnl) FROM trades")
        total_pnl = cursor.fetchone()[0] or 0.0
        
        # Average trade
        avg_trade = total_pnl / total_trades if total_trades > 0 else 0.0
        
        # Average win/loss
        cursor.execute("SELECT AVG(net_pnl) FROM trades WHERE is_win = 1")
        avg_win = cursor.fetchone()[0] or 0.0
        
        cursor.execute("SELECT AVG(ABS(net_pnl)) FROM trades WHERE is_win = 0")
        avg_loss = cursor.fetchone()[0] or 0.0
        
        # Risk/reward ratio
        risk_reward = avg_win / avg_loss if avg_loss > 0 else 0.0
        
        # Profit factor
        cursor.execute("SELECT SUM(net_pnl) FROM trades WHERE is_win = 1")
        gross_profit = cursor.fetchone()[0] or 0.0
        
        cursor.execute("SELECT SUM(ABS(net_pnl)) FROM trades WHERE is_win = 0")
        gross_loss = cursor.fetchone()[0] or 0.0
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        # Best/worst trades
        cursor.execute("SELECT MAX(net_pnl) FROM trades")
        best_trade = cursor.fetchone()[0] or 0.0
        
        cursor.execute("SELECT MIN(net_pnl) FROM trades")
        worst_trade = cursor.fetchone()[0] or 0.0
        
        # Win/loss streaks
        cursor.execute("""
            SELECT is_win, COUNT(*) as streak
            FROM (
                SELECT is_win,
                       ROW_NUMBER() OVER (ORDER BY timestamp) - 
                       ROW_NUMBER() OVER (PARTITION BY is_win ORDER BY timestamp) as grp
                FROM trades
            ) t
            GROUP BY is_win, grp
            ORDER BY streak DESC
            LIMIT 1
        """)
        streak_row = cursor.fetchone()
        longest_win_streak = streak_row[1] if streak_row and streak_row[0] == 1 else 0
        longest_loss_streak = streak_row[1] if streak_row and streak_row[0] == 0 else 0
        
        # Strategy breakdown
        cursor.execute("""
            SELECT strategy, COUNT(*) as count, SUM(net_pnl) as total_pnl, AVG(net_pnl) as avg_pnl
            FROM trades
            GROUP BY strategy
        """)
        strategy_stats = {}
        for row in cursor.fetchall():
            strategy_stats[row[0]] = {
                "count": row[1],
                "total_pnl": row[2] or 0.0,
                "avg_pnl": row[3] or 0.0
            }
        
        conn.close()
        
        stats = {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "average_trade": avg_trade,
            "average_win": avg_win,
            "average_loss": avg_loss,
            "risk_reward_ratio": risk_reward,
            "profit_factor": profit_factor,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "longest_win_streak": longest_win_streak,
            "longest_loss_streak": longest_loss_streak,
            "strategy_breakdown": strategy_stats
        }
        
        # Update cache
        self._stats_cache = stats
        self._cache_timestamp = time.time()
        
        return stats


# Global journal instance (singleton)
_global_journal: Optional[TradingJournal] = None


def get_journal() -> TradingJournal:
    """Get global trading journal instance"""
    global _global_journal
    if _global_journal is None:
        _global_journal = TradingJournal()
        _global_journal.start()
    return _global_journal

