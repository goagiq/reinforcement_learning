"""
Analyze monitoring data to assess model performance
"""
import sqlite3
import pandas as pd
import requests
import json
from datetime import datetime, timedelta

def analyze_performance():
    print("=" * 80)
    print("MODEL PERFORMANCE ANALYSIS")
    print("=" * 80)
    print()
    
    # 1. Performance Metrics
    print("[1] PERFORMANCE MONITORING")
    print("-" * 80)
    try:
        r = requests.get('http://localhost:8200/api/monitoring/performance', timeout=5)
        if r.status_code == 200:
            data = r.json()
            metrics = data.get('metrics', {})
            print(f"Total Trades: {metrics.get('total_trades', 0):,}")
            print(f"Win Rate: {metrics.get('win_rate', 0)*100:.2f}%")
            print(f"Total PnL: ${metrics.get('total_pnl', 0):,.2f}")
            print(f"Average Trade: ${metrics.get('average_trade', 0):.2f}")
            print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"Max Drawdown: ${metrics.get('max_drawdown', 0):,.2f}")
            print(f"Risk/Reward Ratio: {metrics.get('risk_reward_ratio', 0):.2f}")
            print(f"Mean PnL (Last 10): ${metrics.get('mean_pnl_10', 0):.2f}")
        else:
            print(f"Error: {r.status_code}")
    except Exception as e:
        print(f"Error fetching performance: {e}")
    
    print()
    
    # 2. Trading Journal Analysis
    print("[2] TRADING JOURNAL ANALYSIS")
    print("-" * 80)
    try:
        conn = sqlite3.connect('logs/trading_journal.db')
        
        # Overall stats
        df_all = pd.read_sql('SELECT * FROM trades', conn)
        if len(df_all) > 0:
            print(f"Total Trades: {len(df_all):,}")
            print(f"Overall Win Rate: {df_all['is_win'].mean()*100:.2f}%")
            print(f"Total PnL: ${df_all['net_pnl'].sum():,.2f}")
            
            wins = df_all[df_all['is_win'] == 1]
            losses = df_all[df_all['is_win'] == 0]
            if len(wins) > 0:
                print(f"Average Win: ${wins['net_pnl'].mean():.2f}")
            if len(losses) > 0:
                print(f"Average Loss: ${abs(losses['net_pnl'].mean()):.2f}")
            
            # Recent trades
            print(f"\nRecent 50 Trades:")
            recent = df_all.tail(50)
            print(f"  Win Rate: {recent['is_win'].mean()*100:.2f}%")
            print(f"  Total PnL: ${recent['net_pnl'].sum():,.2f}")
            print(f"  Avg PnL: ${recent['net_pnl'].mean():.2f}")
            
            # Last hour
            now = datetime.now()
            one_hour_ago = (now - timedelta(hours=1)).isoformat()
            recent_hour = df_all[df_all['timestamp'] >= one_hour_ago]
            if len(recent_hour) > 0:
                print(f"\nLast Hour:")
                print(f"  Trades: {len(recent_hour)}")
                print(f"  Win Rate: {recent_hour['is_win'].mean()*100:.2f}%")
                print(f"  Total PnL: ${recent_hour['net_pnl'].sum():,.2f}")
        else:
            print("No trades found")
        
        conn.close()
    except Exception as e:
        print(f"Error analyzing journal: {e}")
    
    print()
    
    # 3. Equity Curve
    print("[3] EQUITY CURVE")
    print("-" * 80)
    try:
        conn = sqlite3.connect('logs/trading_journal.db')
        df_equity = pd.read_sql('SELECT * FROM equity_curve ORDER BY timestamp DESC LIMIT 100', conn)
        if len(df_equity) > 0:
            current_equity = df_equity.iloc[0]['equity']
            max_equity = df_equity['equity'].max()
            min_equity = df_equity['equity'].min()
            drawdown = ((max_equity - current_equity) / max_equity * 100) if max_equity > 0 else 0
            
            print(f"Current Equity: ${current_equity:,.2f}")
            print(f"Peak Equity: ${max_equity:,.2f}")
            print(f"Min Equity: ${min_equity:,.2f}")
            print(f"Current Drawdown: {drawdown:.2f}%")
            
            # Trend
            if len(df_equity) >= 10:
                recent_10 = df_equity.head(10)['equity']
                older_10 = df_equity.tail(10)['equity']
                trend = recent_10.mean() - older_10.mean()
                print(f"Trend (Last 10 vs Previous 10): ${trend:,.2f}")
        else:
            print("No equity data found")
        conn.close()
    except Exception as e:
        print(f"Error analyzing equity: {e}")
    
    print()
    
    # 4. Forecast Features Performance
    print("[4] FORECAST FEATURES PERFORMANCE")
    print("-" * 80)
    try:
        r = requests.get('http://localhost:8200/api/monitoring/forecast-performance', timeout=5)
        if r.status_code == 200:
            data = r.json()
            config = data.get('config', {})
            perf = data.get('performance', {})
            
            print("Configuration:")
            print(f"  Forecast Enabled: {config.get('forecast_enabled', False)}")
            print(f"  Regime Enabled: {config.get('regime_enabled', False)}")
            print(f"  State Dim Match: {config.get('state_dimension_match', False)}")
            
            print("\nPerformance (With Forecast Features):")
            print(f"  Total Trades: {perf.get('total_trades', 0):,}")
            print(f"  Win Rate: {perf.get('win_rate', 0)*100:.2f}%")
            print(f"  Total PnL: ${perf.get('total_pnl', 0):,.2f}")
            print(f"  Profit Factor: {perf.get('profit_factor', 0):.2f}")
            print(f"  Avg Win: ${perf.get('avg_win', 0):.2f}")
            print(f"  Avg Loss: ${perf.get('avg_loss', 0):.2f}")
        else:
            print(f"Error: {r.status_code}")
    except Exception as e:
        print(f"Error fetching forecast performance: {e}")
    
    print()
    print("=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    
    # Overall assessment
    try:
        conn = sqlite3.connect('logs/trading_journal.db')
        df = pd.read_sql('SELECT * FROM trades', conn)
        conn.close()
        
        if len(df) > 0:
            win_rate = df['is_win'].mean()
            total_pnl = df['net_pnl'].sum()
            profit_factor = metrics.get('profit_factor', 0)
            
            print(f"\nOverall Assessment:")
            print(f"  Win Rate: {win_rate*100:.1f}% ({'GOOD' if win_rate >= 0.45 else 'NEEDS IMPROVEMENT'})")
            print(f"  Profit Factor: {profit_factor:.2f} ({'GOOD' if profit_factor >= 1.0 else 'NEEDS IMPROVEMENT'})")
            print(f"  Total PnL: ${total_pnl:,.2f} ({'PROFITABLE' if total_pnl > 0 else 'LOSING'})")
            
            if win_rate >= 0.45 and profit_factor >= 1.0 and total_pnl > 0:
                print(f"\n✅ Model is performing WELL")
            elif win_rate >= 0.40 and profit_factor >= 0.8:
                print(f"\n⚠️  Model is performing MODERATELY - needs improvement")
            else:
                print(f"\n❌ Model is performing POORLY - significant issues")
    except Exception as e:
        print(f"Error in summary: {e}")

if __name__ == "__main__":
    analyze_performance()

