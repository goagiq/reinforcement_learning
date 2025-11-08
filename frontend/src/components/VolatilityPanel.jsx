import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { TrendingUp, TrendingDown, Activity, AlertTriangle, BarChart2, Target } from 'lucide-react'

const VolatilityPanel = () => {
  const [loading, setLoading] = useState(false)
  const [volatilityData, setVolatilityData] = useState(null)
  const [adaptiveSizing, setAdaptiveSizing] = useState(null)
  const [error, setError] = useState(null)
  const [method, setMethod] = useState('adaptive')
  const [basePosition, setBasePosition] = useState(0.5)

  useEffect(() => {
    fetchVolatility()
  }, [method])

  const fetchVolatility = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await axios.post('/api/volatility/predict', {
        method: method,
        lookback_periods: 252,
        prediction_horizon: 1
      })

      setVolatilityData(response.data)

      // Also fetch adaptive sizing
      try {
        const sizingResponse = await axios.post('/api/volatility/adaptive-sizing', {
          base_position: basePosition,
          current_price: null
        })
        setAdaptiveSizing(sizingResponse.data)
      } catch (err) {
        console.warn('Adaptive sizing failed:', err)
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Volatility prediction failed')
      console.error('Volatility prediction error:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleAdaptiveSizing = async () => {
    if (!basePosition) return

    setLoading(true)
    try {
      const response = await axios.post('/api/volatility/adaptive-sizing', {
        base_position: parseFloat(basePosition),
        current_price: null
      })
      setAdaptiveSizing(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Adaptive sizing failed')
    } finally {
      setLoading(false)
    }
  }

  const formatPercent = (value) => {
    return `${(value * 100).toFixed(2)}%`
  }

  const getTrendIcon = (trend) => {
    switch (trend) {
      case 'increasing':
        return <TrendingUp className="w-5 h-5 text-red-600" />
      case 'decreasing':
        return <TrendingDown className="w-5 h-5 text-green-600" />
      default:
        return <Activity className="w-5 h-5 text-gray-600" />
    }
  }

  const getTrendColor = (trend) => {
    switch (trend) {
      case 'increasing':
        return 'text-red-600'
      case 'decreasing':
        return 'text-green-600'
      default:
        return 'text-gray-600'
    }
  }

  const getVolatilityLevel = (percentile) => {
    if (percentile > 80) return { level: 'HIGH', color: 'text-red-600', bg: 'bg-red-50' }
    if (percentile > 60) return { level: 'ELEVATED', color: 'text-orange-600', bg: 'bg-orange-50' }
    if (percentile > 40) return { level: 'NORMAL', color: 'text-gray-600', bg: 'bg-gray-50' }
    if (percentile > 20) return { level: 'LOW', color: 'text-blue-600', bg: 'bg-blue-50' }
    return { level: 'VERY LOW', color: 'text-green-600', bg: 'bg-green-50' }
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <BarChart2 className="w-5 h-5 text-purple-600" />
          Volatility Prediction & Adaptive Risk Management
        </h3>
        <div className="flex items-center gap-2">
          <select
            value={method}
            onChange={(e) => setMethod(e.target.value)}
            className="border rounded px-2 py-1 text-sm"
            disabled={loading}
          >
            <option value="adaptive">Adaptive</option>
            <option value="ewma">EWMA</option>
            <option value="historical_mean">Mean Reversion</option>
          </select>
          <button
            onClick={fetchVolatility}
            disabled={loading}
            className="px-3 py-1 bg-purple-600 text-white rounded text-sm hover:bg-purple-700 disabled:opacity-50"
          >
            {loading ? 'Loading...' : 'Refresh'}
          </button>
        </div>
      </div>

      {loading && !volatilityData && (
        <div className="text-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-600 mx-auto"></div>
          <p className="text-gray-500 mt-2">Predicting volatility...</p>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded p-4 mb-4">
          <p className="text-red-800 text-sm">{error}</p>
        </div>
      )}

      {volatilityData && !loading && (
        <>
          {/* Volatility Summary */}
          <div className="mb-6">
            {(() => {
              const volLevel = getVolatilityLevel(volatilityData.volatility_percentile)
              return (
                <div className={`rounded-lg p-4 mb-4 ${volLevel.bg} border`}>
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold">Volatility Level</span>
                    <div className="flex items-center gap-2">
                      {getTrendIcon(volatilityData.volatility_trend)}
                      <span className={`px-3 py-1 rounded-full text-sm font-semibold ${volLevel.color} bg-white`}>
                        {volLevel.level}
                      </span>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-3">
                    <div>
                      <span className="text-sm text-gray-600">Current Volatility</span>
                      <div className="text-lg font-bold">{formatPercent(volatilityData.current_volatility)}</div>
                    </div>
                    <div>
                      <span className="text-sm text-gray-600">Predicted (1 period)</span>
                      <div className="text-lg font-bold">{formatPercent(volatilityData.predicted_volatility)}</div>
                    </div>
                    <div>
                      <span className="text-sm text-gray-600">Percentile</span>
                      <div className="text-lg font-bold">{volatilityData.volatility_percentile.toFixed(1)}%</div>
                    </div>
                    <div>
                      <span className="text-sm text-gray-600">Confidence</span>
                      <div className="text-lg font-bold">{formatPercent(volatilityData.confidence)}</div>
                    </div>
                  </div>
                </div>
              )
            })()}
          </div>

          {/* Volatility Forecasts */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="border rounded p-4">
              <h4 className="font-semibold mb-2 text-sm flex items-center gap-2">
                <Target className="w-4 h-4" />
                1 Period Forecast
              </h4>
              <div className="text-2xl font-bold">{formatPercent(volatilityData.predicted_volatility)}</div>
              <div className={`text-sm mt-1 ${getTrendColor(volatilityData.volatility_trend)}`}>
                Trend: {volatilityData.volatility_trend}
              </div>
            </div>
            <div className="border rounded p-4">
              <h4 className="font-semibold mb-2 text-sm">5 Period Forecast</h4>
              <div className="text-2xl font-bold">{formatPercent(volatilityData.predicted_volatility_5period)}</div>
              <div className="text-xs text-gray-500 mt-1">Mean-reverting forecast</div>
            </div>
            <div className="border rounded p-4">
              <h4 className="font-semibold mb-2 text-sm">20 Period Forecast</h4>
              <div className="text-2xl font-bold">{formatPercent(volatilityData.predicted_volatility_20period)}</div>
              <div className="text-xs text-gray-500 mt-1">Long-term forecast</div>
            </div>
          </div>

          {/* Gap Risk */}
          <div className="border rounded p-4 mb-6">
            <h4 className="font-semibold mb-2 text-sm flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-orange-600" />
              Overnight Gap Risk
            </h4>
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Probability of significant gap</span>
              <span className={`text-lg font-bold ${
                volatilityData.gap_risk_probability > 0.10 ? 'text-red-600' :
                volatilityData.gap_risk_probability > 0.05 ? 'text-orange-600' :
                'text-green-600'
              }`}>
                {formatPercent(volatilityData.gap_risk_probability)}
              </span>
            </div>
            {volatilityData.gap_risk_probability > 0.10 && (
              <div className="mt-2 bg-red-50 border border-red-200 rounded p-2">
                <p className="text-xs text-red-800">
                  ⚠️ High gap risk detected. Consider reducing overnight exposure.
                </p>
              </div>
            )}
          </div>

          {/* Recommendations */}
          <div className="border rounded p-4 mb-6">
            <h4 className="font-semibold mb-3 text-sm">Trading Recommendations</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {Object.entries(volatilityData.recommendations).map(([key, value]) => (
                <div key={key} className="bg-gray-50 rounded p-3">
                  <div className="font-semibold text-xs mb-1 capitalize">
                    {key.replace('_', ' ')}
                  </div>
                  <div className="text-sm text-gray-700">{value}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Adaptive Position Sizing */}
          <div className="border rounded p-4">
            <h4 className="font-semibold mb-3 text-sm">Adaptive Position Sizing</h4>
            <div className="flex items-center gap-2 mb-3">
              <input
                type="number"
                step="0.1"
                min="-1"
                max="1"
                value={basePosition}
                onChange={(e) => setBasePosition(e.target.value)}
                className="border rounded px-3 py-2 w-32"
                placeholder="Base position"
              />
              <button
                onClick={handleAdaptiveSizing}
                disabled={loading}
                className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50"
              >
                Calculate
              </button>
            </div>

            {adaptiveSizing && (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-gray-50 rounded p-3">
                    <span className="text-xs text-gray-600">Base Position</span>
                    <div className="text-lg font-bold">
                      {(adaptiveSizing.base_position * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="bg-blue-50 rounded p-3">
                    <span className="text-xs text-gray-600">Adjusted Position</span>
                    <div className="text-lg font-bold text-blue-600">
                      {(adaptiveSizing.adjusted_position * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-gray-50 rounded p-3">
                    <span className="text-xs text-gray-600">Position Multiplier</span>
                    <div className={`text-lg font-bold ${
                      adaptiveSizing.position_multiplier > 1.0 ? 'text-green-600' :
                      adaptiveSizing.position_multiplier < 1.0 ? 'text-red-600' :
                      'text-gray-600'
                    }`}>
                      {adaptiveSizing.position_multiplier.toFixed(2)}x
                    </div>
                  </div>
                  <div className="bg-gray-50 rounded p-3">
                    <span className="text-xs text-gray-600">Stop Loss Multiplier</span>
                    <div className="text-lg font-bold">
                      {adaptiveSizing.stop_loss_multiplier.toFixed(2)}x
                    </div>
                  </div>
                </div>
                {Math.abs(adaptiveSizing.position_multiplier - 1.0) > 0.05 && (
                  <div className={`p-3 rounded ${
                    adaptiveSizing.position_multiplier < 1.0 ? 'bg-red-50 border border-red-200' :
                    'bg-green-50 border border-green-200'
                  }`}>
                    <p className="text-xs">
                      {adaptiveSizing.position_multiplier < 1.0 ? 
                        '⚠️ Position size reduced due to high volatility' :
                        '✅ Position size increased due to low volatility'
                      }
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}

export default VolatilityPanel

