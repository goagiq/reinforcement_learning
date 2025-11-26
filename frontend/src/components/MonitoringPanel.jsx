import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { TrendingUp, TrendingDown, DollarSign, BarChart3, RefreshCw, BookOpen, Zap } from 'lucide-react'
import CapabilityExplainer from './CapabilityExplainer'

// Import recharts components (ES6 import - works with Vite)
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts'

const MonitoringPanel = () => {
  const [performance, setPerformance] = useState(null)
  const [loading, setLoading] = useState(true)
  const [lastUpdated, setLastUpdated] = useState(null)
  const [initialLoad, setInitialLoad] = useState(true)
  const [equityCurve, setEquityCurve] = useState([])
  const [trades, setTrades] = useState([])
  const [showJournal, setShowJournal] = useState(false)
  const [showAllPoints, setShowAllPoints] = useState(false)
  const [forecastPerformance, setForecastPerformance] = useState(null)
  const [forecastLoading, setForecastLoading] = useState(false)
  const [forecastLastUpdated, setForecastLastUpdated] = useState(null)
  const [checkpointResumeTimestamp, setCheckpointResumeTimestamp] = useState(null)
  const [filterBySession, setFilterBySession] = useState(false)  // Toggle: false = all trades, true = current session only

  // Fetch checkpoint resume timestamp from training status
  const fetchCheckpointResumeTimestamp = async () => {
    try {
      const response = await axios.get('/api/training/status')
      if (response.data.checkpoint_resume_timestamp) {
        console.log('[MonitoringPanel] Checkpoint resume timestamp:', response.data.checkpoint_resume_timestamp)
        setCheckpointResumeTimestamp(response.data.checkpoint_resume_timestamp)
      } else {
        console.log('[MonitoringPanel] No checkpoint resume timestamp found')
        setCheckpointResumeTimestamp(null)
      }
    } catch (error) {
      console.error('Failed to fetch checkpoint resume timestamp:', error)
    }
  }

  const loadPerformance = async (isInitialLoad = false) => {
    try {
      // Only show loading spinner on initial load to avoid UI flicker during auto-refresh
      if (isInitialLoad) {
        setLoading(true)
      }
      
      // Build URL - Always load all data by default (filterBySession is for user preference)
      // Only apply filter if explicitly enabled by user
      let url = '/api/monitoring/performance'
      if (filterBySession && checkpointResumeTimestamp) {
        url += `?since=${checkpointResumeTimestamp}`
        console.log('[MonitoringPanel] Loading performance with timestamp filter (current session only):', checkpointResumeTimestamp)
      } else {
        // Always load all data to ensure latest episodes are visible
        console.log('[MonitoringPanel] Loading performance without timestamp filter (showing all trades)')
      }
      
      // Force refresh to get latest data
      url += (url.includes('?') ? '&' : '?') + '_t=' + Date.now()
      
      const response = await axios.get(url)
      if (response.data.status === 'success') {
        const metrics = response.data.metrics
        setPerformance(metrics)
        setLastUpdated(new Date())
        
        // Log filtering status for verification
        if (checkpointResumeTimestamp) {
          console.log('[MonitoringPanel] Performance loaded (FILTERED):', {
            total_trades: metrics.total_trades,
            total_pnl: metrics.total_pnl,
            filtered_since: metrics.filtered_since
          })
        } else {
          console.log('[MonitoringPanel] Performance loaded (ALL TRADES):', {
            total_trades: metrics.total_trades,
            total_pnl: metrics.total_pnl
          })
        }
      }
    } catch (error) {
      console.error('Failed to load performance:', error)
    } finally {
      if (isInitialLoad) {
        setLoading(false)
      }
    }
  }

  const loadEquityCurve = async (showAll = showAllPoints) => {
    try {
      const params = { limit: 10000 }
      if (filterBySession && checkpointResumeTimestamp) {
        params.since = checkpointResumeTimestamp
        console.log('[MonitoringPanel] Loading equity curve with timestamp filter (current session only):', checkpointResumeTimestamp)
      }
      const response = await axios.get('/api/journal/equity-curve', { params })
      if (response.data.status === 'success') {
        if (checkpointResumeTimestamp) {
          console.log('[MonitoringPanel] Equity curve loaded (FILTERED):', {
            count: response.data.count,
            filtered_since: response.data.filtered_since
          })
        }
        // Format for chart: convert to cumulative equity points
        let points = response.data.equity_curve.map((point, idx) => ({
          index: idx,
          equity: point.equity,
          timestamp: point.timestamp,
          episode: point.episode,
          step: point.step
        }))
        
        // Downsample if too many points (keep chart readable)
        // Show max 300 points for clean rendering (or all if showAll is true)
        if (!showAll && points.length > 300) {
          const step = Math.ceil(points.length / 300)
          // Keep first, last, and evenly spaced middle points
          const downsampled = []
          for (let i = 0; i < points.length; i += step) {
            downsampled.push(points[i])
          }
          // Always include the last point
          if (downsampled[downsampled.length - 1] !== points[points.length - 1]) {
            downsampled.push(points[points.length - 1])
          }
          points = downsampled
        }
        
        setEquityCurve(points)
      }
    } catch (error) {
      console.error('Failed to load equity curve:', error)
    }
  }

  const loadTrades = async () => {
    try {
      const params = { limit: 50 }
      if (filterBySession && checkpointResumeTimestamp) {
        params.since = checkpointResumeTimestamp
      }
      const response = await axios.get('/api/journal/trades', { params })
      if (response.data.status === 'success') {
        setTrades(response.data.trades)
        if (checkpointResumeTimestamp) {
          console.log('[MonitoringPanel] Trades loaded (FILTERED):', {
            count: response.data.count,
            filtered_since: response.data.filtered_since
          })
        }
      }
    } catch (error) {
      console.error('Failed to load trades:', error)
    }
  }

  const loadForecastPerformance = async (isInitialLoad = false) => {
    try {
      // Only show loading spinner on initial load to avoid UI flicker during auto-refresh
      if (isInitialLoad) {
        setForecastLoading(true)
      }
      const params = {}
      if (filterBySession && checkpointResumeTimestamp) {
        params.since = checkpointResumeTimestamp
        console.log('[MonitoringPanel] Loading forecast performance with timestamp filter (current session only):', checkpointResumeTimestamp)
      }
      const response = await axios.get('/api/monitoring/forecast-performance', { params })
      if (response.data.status === 'success') {
        setForecastPerformance(response.data)
        setForecastLastUpdated(new Date())
        if (checkpointResumeTimestamp) {
          console.log('[MonitoringPanel] Forecast performance loaded (FILTERED)')
        }
      }
    } catch (error) {
      console.error('Failed to load forecast performance:', error)
    } finally {
      if (isInitialLoad) {
        setForecastLoading(false)
      }
    }
  }

  useEffect(() => {
    let intervalId = null
    let isMounted = true
    
    // Fetch checkpoint resume timestamp first
    fetchCheckpointResumeTimestamp()
    
    // Initial load with loading indicators
    const initialLoad = async () => {
      await Promise.all([
        loadPerformance(true),
        loadEquityCurve(showAllPoints),
        loadForecastPerformance(true)
      ])
    }
    initialLoad()
    
    // Set up auto-refresh interval (silent refresh, no loading indicators)
    const startInterval = () => {
      if (intervalId) {
        clearInterval(intervalId)
      }
      intervalId = setInterval(() => {
        if (!isMounted) {
          if (intervalId) clearInterval(intervalId)
          return
        }
        // Refresh checkpoint timestamp periodically (in case training just started)
        fetchCheckpointResumeTimestamp().catch(err => console.error('Checkpoint timestamp refresh error:', err))
        // Silent refresh - no loading indicators
        loadPerformance(false).catch(err => console.error('Performance refresh error:', err))
        loadEquityCurve(showAllPoints).catch(err => console.error('Equity curve refresh error:', err))
        loadForecastPerformance(false).catch(err => console.error('Forecast performance refresh error:', err))
        if (showJournal) {
          loadTrades().catch(err => console.error('Trades refresh error:', err))
        }
      }, 5000) // Refresh every 5 seconds
    }
    
    startInterval()
    
    return () => {
      isMounted = false
      if (intervalId) {
        clearInterval(intervalId)
      }
    }
  }, [showJournal, showAllPoints, checkpointResumeTimestamp, filterBySession]) // Include checkpointResumeTimestamp and filterBySession in dependencies

  const metrics = [
    { key: 'total_pnl', label: 'Total P&L', icon: DollarSign, color: null, dynamicColor: true, formatCurrency: true },
    { key: 'sharpe_ratio', label: 'Sharpe Ratio', icon: TrendingUp, color: 'text-blue-600' },
    { key: 'sortino_ratio', label: 'Sortino Ratio', icon: TrendingUp, color: 'text-blue-600' },
    { key: 'win_rate', label: 'Win Rate', icon: BarChart3, color: 'text-purple-600', formatPercentage: true },
    { key: 'profit_factor', label: 'Profit Factor', icon: TrendingUp, color: 'text-green-600' },
    { key: 'max_drawdown', label: 'Max Drawdown', icon: TrendingDown, color: 'text-red-600' },
    { key: 'total_trades', label: 'Total Trades', icon: BarChart3, color: 'text-gray-600' },
    { key: 'average_trade', label: 'Avg Trade', icon: DollarSign, color: null, dynamicColor: true, formatCurrency: true },
  ]

  if (loading && !performance) {
    return (
      <div className="bg-white rounded-lg shadow p-12 text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
        <p className="text-gray-600">Loading performance data...</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow p-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h2 className="text-2xl font-bold text-gray-800">Performance Monitoring</h2>
          {/* Filter status badge */}
          {filterBySession ? (
            <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
              Current Session Only
            </span>
          ) : (
            <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded">
              {performance?.win_rate !== undefined 
                ? `All Trades (${(performance.win_rate * 100).toFixed(1)}% win rate)`
                : 'All Trades'}
            </span>
          )}
          <CapabilityExplainer
            capabilityId="monitoring.performance"
            context={{
              metrics: performance,
              lastUpdated: lastUpdated?.toISOString(),
            }}
          />
        </div>
        <div className="flex items-center gap-4">
          {/* Toggle Switch: All Trades vs Current Session */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-600 font-medium">
              {filterBySession ? 'Current Session' : 'All Trades'}
            </span>
            <button
              type="button"
              onClick={() => {
                setFilterBySession(!filterBySession)
                // Reload data immediately when toggling
                loadPerformance(false)
                loadEquityCurve()
                loadTrades()
              }}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                filterBySession ? 'bg-blue-600' : 'bg-gray-300'
              }`}
              role="switch"
              aria-checked={filterBySession}
              aria-label="Toggle between all trades and current session only"
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  filterBySession ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
          {lastUpdated && (
            <span className="text-sm text-gray-500">
              Updated: {lastUpdated.toLocaleTimeString()}
            </span>
          )}
          <button
            onClick={() => {
              fetchCheckpointResumeTimestamp()
              loadPerformance(false)
            }}
            className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 flex items-center gap-2"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh
          </button>
        </div>
      </div>

      {/* Performance Metrics */}
      {performance ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {metrics.map(metric => {
            const Icon = metric.icon
            const value = performance[metric.key]
            
            // Determine display value with formatting
            let displayValue
            if (typeof value === 'number') {
              if (metric.formatCurrency) {
                displayValue = `$${value.toFixed(2)}`
              } else if (metric.formatPercentage) {
                displayValue = `${(value * 100).toFixed(2)}%`
              } else {
                displayValue = value.toFixed(2)
              }
            } else {
              displayValue = value || 'N/A'
            }
            
            // Determine color (dynamic or static)
            let colorClass = metric.color
            if (metric.dynamicColor && typeof value === 'number') {
              colorClass = value >= 0 ? 'text-green-600' : 'text-red-600'
            } else if (!colorClass) {
              colorClass = 'text-gray-600'
            }
            
            return (
              <div key={metric.key} className="bg-white rounded-lg shadow p-6">
                <div className="flex items-center justify-between mb-2">
                  <Icon className={`w-6 h-6 ${colorClass}`} />
                </div>
                <div className="text-sm text-gray-600 mb-1">{metric.label}</div>
                <div className={`text-3xl font-bold ${colorClass}`}>
                  {displayValue}
                </div>
              </div>
            )
          })}
        </div>
      ) : (
        <div className="bg-white rounded-lg shadow p-12 text-center">
          <p className="text-gray-600">No performance data available</p>
          <p className="text-sm text-gray-500 mt-2">
            Start trading to see performance metrics
          </p>
        </div>
      )}

      {/* Equity Curve Chart */}
      {equityCurve.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-800">Equity Curve</h3>
            <div className="flex items-center gap-4">
              <span className="text-sm text-gray-500">
                {equityCurve.length} points
              </span>
              <button
                onClick={() => {
                  const newValue = !showAllPoints
                  setShowAllPoints(newValue)
                  // Reload to apply downsampling change
                  loadEquityCurve(newValue)
                }}
                className="text-sm px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded text-gray-700 transition-colors"
              >
                {showAllPoints ? 'Show Less' : 'Show All'}
              </button>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={equityCurve} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" opacity={0.3} />
              <XAxis 
                dataKey="index" 
                label={{ value: 'Time', position: 'insideBottom', offset: -5 }}
                stroke="#6b7280"
                tick={{ fill: '#6b7280', fontSize: 11 }}
                tickCount={8}
                interval="preserveStartEnd"
              />
              <YAxis 
                label={{ value: 'Equity ($)', angle: -90, position: 'insideLeft' }}
                domain={['dataMin - 1000', 'dataMax + 1000']}
                stroke="#6b7280"
                tick={{ fill: '#6b7280', fontSize: 11 }}
                tickFormatter={(value) => {
                  if (value >= 1000000) return `$${(value / 1000000).toFixed(1)}M`
                  if (value >= 1000) return `$${(value / 1000).toFixed(0)}k`
                  return `$${value.toFixed(0)}`
                }}
                width={80}
              />
              <Tooltip 
                formatter={(value) => [`$${value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`, 'Equity']}
                labelFormatter={(label, payload) => {
                  if (payload && payload[0]) {
                    const point = equityCurve[payload[0].payload.index]
                    return `Episode ${point?.episode || 'N/A'}, Step ${point?.step || 'N/A'}`
                  }
                  return `Point ${label}`
                }}
                contentStyle={{ 
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  border: '1px solid #e5e7eb',
                  borderRadius: '6px',
                  padding: '8px'
                }}
              />
              <Line 
                type="monotone" 
                dataKey="equity" 
                stroke="#3b82f6" 
                strokeWidth={1.5}
                dot={false}
                activeDot={{ r: 5, fill: '#3b82f6', stroke: '#fff', strokeWidth: 2 }}
                name="Equity"
                isAnimationActive={false}
                connectNulls={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Trading Journal */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-800 flex items-center gap-2">
            <BookOpen className="w-5 h-5" />
            Trading Journal
          </h3>
          <button
            onClick={() => {
              setShowJournal(!showJournal)
              if (!showJournal) {
                loadTrades()
              }
            }}
            className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
          >
            {showJournal ? 'Hide' : 'Show'} Journal
          </button>
        </div>
        
        {showJournal && (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Episode</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Strategy</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Entry</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Exit</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Size</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">PnL</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Net PnL</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Result</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {trades.length > 0 ? (
                  trades.map((trade, idx) => (
                    <tr key={trade.trade_id || idx} className={trade.is_win ? 'bg-green-50' : 'bg-red-50'}>
                      <td className="px-4 py-3 text-sm text-gray-900">{trade.episode}</td>
                      <td className="px-4 py-3 text-sm text-gray-900">{trade.strategy}</td>
                      <td className="px-4 py-3 text-sm text-gray-900">${trade.entry_price.toFixed(2)}</td>
                      <td className="px-4 py-3 text-sm text-gray-900">${trade.exit_price.toFixed(2)}</td>
                      <td className="px-4 py-3 text-sm text-gray-900">{trade.position_size.toFixed(3)}</td>
                      <td className="px-4 py-3 text-sm text-gray-900">${trade.pnl.toFixed(2)}</td>
                      <td className={`px-4 py-3 text-sm font-semibold ${trade.net_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        ${trade.net_pnl.toFixed(2)}
                      </td>
                      <td className="px-4 py-3 text-sm">
                        <span className={`px-2 py-1 rounded ${trade.is_win ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                          {trade.is_win ? 'Win' : 'Loss'}
                        </span>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="8" className="px-4 py-8 text-center text-gray-500">
                      No trades recorded yet. Trades will appear here as training progresses.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Forecast Features Performance */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-800 flex items-center gap-2">
            <Zap className="w-5 h-5 text-yellow-500" />
            Forecast Features Performance
          </h3>
          <div className="flex items-center gap-4">
            {forecastLastUpdated && (
              <span className="text-sm text-gray-500">
                Updated: {forecastLastUpdated.toLocaleTimeString()}
              </span>
            )}
            <button
              onClick={() => loadForecastPerformance(false)}
              disabled={forecastLoading}
              className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <RefreshCw className={`w-4 h-4 ${forecastLoading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>
        </div>

        {forecastLoading && !forecastPerformance ? (
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto mb-2"></div>
            <p className="text-gray-600">Loading forecast performance...</p>
          </div>
        ) : forecastPerformance ? (
          <div className="space-y-6">
            {/* Configuration Status */}
            {forecastPerformance.config && (
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-semibold text-gray-800 mb-3">Configuration</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <div className="text-sm text-gray-600">Forecast Features</div>
                    <div className={`font-semibold ${forecastPerformance.config.forecast_enabled ? 'text-green-600' : 'text-gray-400'}`}>
                      {forecastPerformance.config.forecast_enabled ? 'ENABLED' : 'DISABLED'}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-600">Regime Features</div>
                    <div className={`font-semibold ${forecastPerformance.config.regime_enabled ? 'text-green-600' : 'text-gray-400'}`}>
                      {forecastPerformance.config.regime_enabled ? 'ENABLED' : 'DISABLED'}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-600">State Features</div>
                    <div className="font-semibold text-gray-800">
                      {forecastPerformance.config.state_features}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-600">State Dim Match</div>
                    <div className={`font-semibold ${forecastPerformance.config.state_dimension_match ? 'text-green-600' : 'text-red-600'}`}>
                      {forecastPerformance.config.state_dimension_match ? 'OK' : 'MISMATCH'}
                    </div>
                    {!forecastPerformance.config.state_dimension_match && (
                      <div className="text-xs text-red-600 mt-1">
                        Expected: {forecastPerformance.config.expected_state_dim}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Performance Metrics */}
            {forecastPerformance.performance && !forecastPerformance.performance.error && (
              <div>
                <h4 className="font-semibold text-gray-800 mb-3">
                  {forecastPerformance.performance.label || 'Performance Analysis'}
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="bg-blue-50 rounded-lg p-4">
                    <div className="text-sm text-gray-600 mb-1">Total Trades</div>
                    <div className="text-2xl font-bold text-blue-600">
                      {forecastPerformance.performance.total_trades.toLocaleString()}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      {forecastPerformance.performance.win_count}W / {forecastPerformance.performance.loss_count}L
                    </div>
                  </div>
                  
                  <div className="bg-purple-50 rounded-lg p-4">
                    <div className="text-sm text-gray-600 mb-1">Win Rate</div>
                    <div className={`text-2xl font-bold ${forecastPerformance.performance.win_rate >= 0.5 ? 'text-green-600' : 'text-red-600'}`}>
                      {(forecastPerformance.performance.win_rate * 100).toFixed(2)}%
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      Target: &gt;50%
                    </div>
                  </div>
                  
                  <div className="bg-green-50 rounded-lg p-4">
                    <div className="text-sm text-gray-600 mb-1">Profit Factor</div>
                    <div className={`text-2xl font-bold ${forecastPerformance.performance.profit_factor >= 1.2 ? 'text-green-600' : 'text-red-600'}`}>
                      {forecastPerformance.performance.profit_factor.toFixed(2)}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      Target: &gt;1.2
                    </div>
                  </div>
                  
                  <div className="bg-yellow-50 rounded-lg p-4">
                    <div className="text-sm text-gray-600 mb-1">Total PnL</div>
                    <div className={`text-2xl font-bold ${forecastPerformance.performance.total_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      ${forecastPerformance.performance.total_pnl.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      Avg: ${forecastPerformance.performance.avg_pnl.toFixed(2)}
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="text-sm text-gray-600 mb-1">Avg Win</div>
                    <div className="text-xl font-bold text-green-600">
                      ${forecastPerformance.performance.avg_win.toFixed(2)}
                    </div>
                  </div>
                  
                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="text-sm text-gray-600 mb-1">Avg Loss</div>
                    <div className="text-xl font-bold text-red-600">
                      ${forecastPerformance.performance.avg_loss.toFixed(2)}
                    </div>
                  </div>
                  
                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="text-sm text-gray-600 mb-1">Sharpe-like Ratio</div>
                    <div className={`text-xl font-bold ${forecastPerformance.performance.sharpe_like >= 1.0 ? 'text-green-600' : 'text-red-600'}`}>
                      {forecastPerformance.performance.sharpe_like.toFixed(2)}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      Target: &gt;1.0
                    </div>
                  </div>
                </div>

                <div className="mt-4 bg-red-50 rounded-lg p-4">
                  <div className="text-sm text-gray-600 mb-1">Max Drawdown</div>
                  <div className="text-xl font-bold text-red-600">
                    ${forecastPerformance.performance.max_drawdown.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </div>
                </div>
              </div>
            )}

            {forecastPerformance.performance && forecastPerformance.performance.error && (
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <p className="text-yellow-800">{forecastPerformance.performance.error}</p>
              </div>
            )}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <p>No forecast performance data available</p>
            <p className="text-sm mt-2">Click Refresh to load data</p>
          </div>
        )}
      </div>

      {/* Additional Information */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Performance Targets</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="font-semibold text-gray-800 mb-2">Sharpe Ratio</div>
            <div className="text-sm text-gray-600">Target: &gt; 1.5</div>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="font-semibold text-gray-800 mb-2">Win Rate</div>
            <div className="text-sm text-gray-600">Target: &gt; 55%</div>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="font-semibold text-gray-800 mb-2">Profit Factor</div>
            <div className="text-sm text-gray-600">Target: &gt; 1.5</div>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="font-semibold text-gray-800 mb-2">Max Drawdown</div>
            <div className="text-sm text-gray-600">Target: &lt; 20%</div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default MonitoringPanel

