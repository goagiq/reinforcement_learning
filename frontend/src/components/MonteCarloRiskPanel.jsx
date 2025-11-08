import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { BarChart3, AlertTriangle, TrendingUp, TrendingDown, DollarSign, Shield, Target } from 'lucide-react'

const MonteCarloRiskPanel = ({ currentPrice, proposedPosition, currentPosition = 0.0 }) => {
  const [loading, setLoading] = useState(false)
  const [riskData, setRiskData] = useState(null)
  const [scenarioData, setScenarioData] = useState(null)
  const [error, setError] = useState(null)
  const [simulations, setSimulations] = useState(1000)

  useEffect(() => {
    if (currentPrice && proposedPosition !== undefined && proposedPosition !== 0) {
      assessRisk()
    }
  }, [currentPrice, proposedPosition, currentPosition, simulations])

  const assessRisk = async () => {
    if (!currentPrice || proposedPosition === undefined) return
    
    setLoading(true)
    setError(null)

    try {
      // Monte Carlo risk assessment
      const riskResponse = await axios.post('/api/risk/monte-carlo', {
        current_price: currentPrice,
        proposed_position: proposedPosition,
        current_position: currentPosition,
        n_simulations: simulations,
        simulate_overnight: true
      })

      setRiskData(riskResponse.data)

      // Scenario analysis
      try {
        const scenarioResponse = await axios.post('/api/risk/scenario-analysis', {
          current_price: currentPrice,
          proposed_position: proposedPosition,
          current_position: currentPosition,
          n_simulations: Math.min(simulations, 500) // Use fewer simulations for scenario analysis
        })
        setScenarioData(scenarioResponse.data)
      } catch (err) {
        console.warn('Scenario analysis failed:', err)
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Risk assessment failed')
      console.error('Risk assessment error:', err)
    } finally {
      setLoading(false)
    }
  }

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value)
  }

  const formatPercent = (value) => {
    return `${(value * 100).toFixed(2)}%`
  }

  const getRiskLevel = (tailRisk, var99) => {
    if (tailRisk > 0.10 || var99 < -5000) return 'high'
    if (tailRisk > 0.05 || var99 < -2500) return 'medium'
    return 'low'
  }

  const riskLevel = riskData ? getRiskLevel(
    riskData.risk_metrics.tail_risk,
    riskData.risk_metrics.var_99
  ) : 'unknown'

  if (!currentPrice || proposedPosition === undefined || proposedPosition === 0) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Shield className="w-5 h-5 text-blue-600" />
          Monte Carlo Risk Assessment
        </h3>
        <p className="text-gray-500 text-sm">Enter a position to assess risk</p>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-blue-600" />
          Monte Carlo Risk Assessment
        </h3>
        <div className="flex items-center gap-2">
          <label className="text-sm text-gray-600">Simulations:</label>
          <select
            value={simulations}
            onChange={(e) => setSimulations(parseInt(e.target.value))}
            className="border rounded px-2 py-1 text-sm"
            disabled={loading}
          >
            <option value={500}>500</option>
            <option value={1000}>1,000</option>
            <option value={2000}>2,000</option>
            <option value={5000}>5,000</option>
          </select>
        </div>
      </div>

      {loading && (
        <div className="text-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
          <p className="text-gray-500 mt-2">Running Monte Carlo simulation...</p>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded p-4 mb-4">
          <p className="text-red-800 text-sm">{error}</p>
        </div>
      )}

      {riskData && !loading && (
        <>
          {/* Risk Summary */}
          <div className="mb-6">
            <div className={`rounded-lg p-4 mb-4 ${
              riskLevel === 'high' ? 'bg-red-50 border border-red-200' :
              riskLevel === 'medium' ? 'bg-yellow-50 border border-yellow-200' :
              'bg-green-50 border border-green-200'
            }`}>
              <div className="flex items-center justify-between mb-2">
                <span className="font-semibold">Risk Level</span>
                <span className={`px-3 py-1 rounded-full text-sm font-semibold ${
                  riskLevel === 'high' ? 'bg-red-200 text-red-800' :
                  riskLevel === 'medium' ? 'bg-yellow-200 text-yellow-800' :
                  'bg-green-200 text-green-800'
                }`}>
                  {riskLevel.toUpperCase()}
                </span>
              </div>
              <p className="text-sm text-gray-700">{riskData.recommendation}</p>
            </div>

            {/* Key Metrics Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-gray-50 rounded p-3">
                <div className="flex items-center gap-2 mb-1">
                  <TrendingUp className="w-4 h-4 text-green-600" />
                  <span className="text-xs text-gray-600">Expected P&L</span>
                </div>
                <div className={`text-lg font-bold ${
                  riskData.risk_metrics.expected_pnl >= 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  {formatCurrency(riskData.risk_metrics.expected_pnl)}
                </div>
              </div>

              <div className="bg-gray-50 rounded p-3">
                <div className="flex items-center gap-2 mb-1">
                  <AlertTriangle className="w-4 h-4 text-red-600" />
                  <span className="text-xs text-gray-600">VaR (95%)</span>
                </div>
                <div className="text-lg font-bold text-red-600">
                  {formatCurrency(riskData.risk_metrics.var_95)}
                </div>
              </div>

              <div className="bg-gray-50 rounded p-3">
                <div className="flex items-center gap-2 mb-1">
                  <Target className="w-4 h-4 text-blue-600" />
                  <span className="text-xs text-gray-600">Win Probability</span>
                </div>
                <div className="text-lg font-bold text-blue-600">
                  {formatPercent(riskData.risk_metrics.win_probability)}
                </div>
              </div>

              <div className="bg-gray-50 rounded p-3">
                <div className="flex items-center gap-2 mb-1">
                  <Shield className="w-4 h-4 text-purple-600" />
                  <span className="text-xs text-gray-600">Tail Risk</span>
                </div>
                <div className={`text-lg font-bold ${
                  riskData.risk_metrics.tail_risk > 0.10 ? 'text-red-600' : 'text-gray-700'
                }`}>
                  {formatPercent(riskData.risk_metrics.tail_risk)}
                </div>
              </div>
            </div>
          </div>

          {/* Detailed Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div className="border rounded p-4">
              <h4 className="font-semibold mb-3 text-sm">Risk Metrics</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">VaR (99%):</span>
                  <span className="font-semibold text-red-600">
                    {formatCurrency(riskData.risk_metrics.var_99)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">CVaR (95%):</span>
                  <span className="font-semibold text-red-600">
                    {formatCurrency(riskData.risk_metrics.cvar_95)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Max Drawdown:</span>
                  <span className="font-semibold">
                    {formatPercent(riskData.risk_metrics.max_drawdown)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Expected Return:</span>
                  <span className={`font-semibold ${
                    riskData.risk_metrics.expected_return >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {formatPercent(riskData.risk_metrics.expected_return)}
                  </span>
                </div>
              </div>
            </div>

            <div className="border rounded p-4">
              <h4 className="font-semibold mb-3 text-sm">Position Sizing</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Proposed Size:</span>
                  <span className="font-semibold">
                    {(proposedPosition * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Optimal Size:</span>
                  <span className="font-semibold text-blue-600">
                    {(riskData.risk_metrics.optimal_position_size * 100).toFixed(1)}%
                  </span>
                </div>
                {Math.abs(riskData.risk_metrics.optimal_position_size) < Math.abs(proposedPosition) && (
                  <div className="bg-yellow-50 border border-yellow-200 rounded p-2 mt-2">
                    <p className="text-xs text-yellow-800">
                      ⚠️ Risk analysis suggests reducing position size
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Scenario Stats */}
          {riskData.scenario_stats && (
            <div className="border rounded p-4 mb-4">
              <h4 className="font-semibold mb-3 text-sm">Scenario Statistics</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                <div>
                  <span className="text-gray-600">Min P&L:</span>
                  <div className="font-semibold text-red-600">
                    {formatCurrency(riskData.scenario_stats.min_pnl)}
                  </div>
                </div>
                <div>
                  <span className="text-gray-600">Max P&L:</span>
                  <div className="font-semibold text-green-600">
                    {formatCurrency(riskData.scenario_stats.max_pnl)}
                  </div>
                </div>
                <div>
                  <span className="text-gray-600">Median P&L:</span>
                  <div className="font-semibold">
                    {formatCurrency(riskData.scenario_stats.median_pnl)}
                  </div>
                </div>
                <div>
                  <span className="text-gray-600">Std Dev:</span>
                  <div className="font-semibold">
                    {formatCurrency(riskData.scenario_stats.std_pnl)}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Scenario Analysis */}
          {scenarioData && (
            <div className="border rounded p-4">
              <h4 className="font-semibold mb-3 text-sm">Market Scenario Analysis</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {Object.entries(scenarioData.scenarios).map(([scenario, metrics]) => (
                  <div key={scenario} className="bg-gray-50 rounded p-3">
                    <div className="font-semibold text-sm mb-2 capitalize">
                      {scenario.replace('_', ' ')}
                    </div>
                    <div className="space-y-1 text-xs">
                      <div className="flex justify-between">
                        <span>Expected P&L:</span>
                        <span className={metrics.expected_pnl >= 0 ? 'text-green-600' : 'text-red-600'}>
                          {formatCurrency(metrics.expected_pnl)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Win Prob:</span>
                        <span>{formatPercent(metrics.win_probability)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Tail Risk:</span>
                        <span className={metrics.tail_risk > 0.10 ? 'text-red-600' : ''}>
                          {formatPercent(metrics.tail_risk)}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              {scenarioData.recommendation && (
                <div className="mt-3 bg-blue-50 border border-blue-200 rounded p-2">
                  <p className="text-xs text-blue-800">{scenarioData.recommendation}</p>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default MonteCarloRiskPanel

