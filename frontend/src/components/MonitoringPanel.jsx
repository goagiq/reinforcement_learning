import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { TrendingUp, TrendingDown, DollarSign, BarChart3, RefreshCw } from 'lucide-react'

const MonitoringPanel = () => {
  const [performance, setPerformance] = useState(null)
  const [loading, setLoading] = useState(true)
  const [lastUpdated, setLastUpdated] = useState(null)

  const loadPerformance = async () => {
    try {
      setLoading(true)
      const response = await axios.get('/api/monitoring/performance')
      if (response.data.status === 'success') {
        setPerformance(response.data.metrics)
        setLastUpdated(new Date())
      }
    } catch (error) {
      console.error('Failed to load performance:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadPerformance()
    const interval = setInterval(loadPerformance, 5000) // Refresh every 5 seconds
    return () => clearInterval(interval)
  }, [])

  const metrics = [
    { key: 'total_pnl', label: 'Total P&L', icon: DollarSign, color: 'text-green-600' },
    { key: 'sharpe_ratio', label: 'Sharpe Ratio', icon: TrendingUp, color: 'text-blue-600' },
    { key: 'sortino_ratio', label: 'Sortino Ratio', icon: TrendingUp, color: 'text-blue-600' },
    { key: 'win_rate', label: 'Win Rate', icon: BarChart3, color: 'text-purple-600' },
    { key: 'profit_factor', label: 'Profit Factor', icon: TrendingUp, color: 'text-green-600' },
    { key: 'max_drawdown', label: 'Max Drawdown', icon: TrendingDown, color: 'text-red-600' },
    { key: 'total_trades', label: 'Total Trades', icon: BarChart3, color: 'text-gray-600' },
    { key: 'average_trade', label: 'Avg Trade', icon: DollarSign, color: 'text-gray-600' },
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
        <h2 className="text-2xl font-bold text-gray-800">Performance Monitoring</h2>
        <div className="flex items-center gap-4">
          {lastUpdated && (
            <span className="text-sm text-gray-500">
              Updated: {lastUpdated.toLocaleTimeString()}
            </span>
          )}
          <button
            onClick={loadPerformance}
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
            const displayValue = typeof value === 'number' 
              ? (value.toFixed(2))
              : (value || 'N/A')
            
            return (
              <div key={metric.key} className="bg-white rounded-lg shadow p-6">
                <div className="flex items-center justify-between mb-2">
                  <Icon className={`w-6 h-6 ${metric.color}`} />
                </div>
                <div className="text-sm text-gray-600 mb-1">{metric.label}</div>
                <div className={`text-3xl font-bold ${metric.color}`}>
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

