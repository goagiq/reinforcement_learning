import React, { useState } from 'react'
import axios from 'axios'
import { TestTube, AlertTriangle, TrendingUp, BarChart3, Settings } from 'lucide-react'
import CapabilityExplainer from './CapabilityExplainer'

const ScenarioSimulationPanel = () => {
  const [loading, setLoading] = useState(false)
  const [testType, setTestType] = useState('robustness') // 'robustness', 'stress', 'sensitivity'
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)

  // Robustness test state
  const [robustnessScenarios, setRobustnessScenarios] = useState([
    'normal', 'trending_up', 'trending_down', 'ranging', 'high_volatility', 'low_volatility'
  ])
  const [intensity, setIntensity] = useState(1.0)

  // Stress test state
  const [stressScenarios, setStressScenarios] = useState([
    'crash', 'flash_crash', 'high_volatility', 'gap_event'
  ])
  const [stressIntensity, setStressIntensity] = useState(2.0)

  // Sensitivity test state
  const [paramName, setParamName] = useState('position_size')
  const [paramValues, setParamValues] = useState('0.2,0.4,0.6,0.8,1.0')
  const [regime, setRegime] = useState('normal')

  const availableScenarios = [
    { value: 'normal', label: 'Normal' },
    { value: 'trending_up', label: 'Trending Up' },
    { value: 'trending_down', label: 'Trending Down' },
    { value: 'ranging', label: 'Ranging' },
    { value: 'high_volatility', label: 'High Volatility' },
    { value: 'low_volatility', label: 'Low Volatility' },
    { value: 'gap_event', label: 'Gap Event' },
    { value: 'low_liquidity', label: 'Low Liquidity' },
    { value: 'crash', label: 'Crash' },
    { value: 'flash_crash', label: 'Flash Crash' }
  ]

  const runRobustnessTest = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await axios.post('/api/scenario/robustness-test', {
        scenarios: robustnessScenarios,
        intensity: intensity
      })

      setResults({ type: 'robustness', data: response.data })
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Robustness test failed')
    } finally {
      setLoading(false)
    }
  }

  const runStressTest = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await axios.post('/api/scenario/stress-test', {
        scenarios: stressScenarios,
        intensity: stressIntensity
      })

      setResults({ type: 'stress', data: response.data })
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Stress test failed')
    } finally {
      setLoading(false)
    }
  }

  const runSensitivityTest = async () => {
    setLoading(true)
    setError(null)

    try {
      const values = paramValues.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v))
      
      const response = await axios.post('/api/scenario/parameter-sensitivity', {
        parameter_name: paramName,
        parameter_values: values,
        base_parameters: {},
        regime: regime
      })

      setResults({ type: 'sensitivity', data: response.data })
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Sensitivity analysis failed')
    } finally {
      setLoading(false)
    }
  }

  const formatPercent = (value) => {
    return `${(value * 100).toFixed(2)}%`
  }

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value)
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold flex items-center gap-2 text-gray-800">
          <TestTube className="w-5 h-5 text-orange-600" />
          Scenario Simulation & Robustness Testing
        </h3>
        <CapabilityExplainer
          capabilityId="scenarios.guidance"
          context={{
            testType,
            robustnessScenarios,
            stressScenarios,
            intensity,
            stressIntensity,
            paramName,
            paramValues,
            regime,
          }}
        />
      </div>

      {/* Test Type Selection */}
      <div className="flex gap-2 mb-4 border-b pb-4">
        <button
          onClick={() => setTestType('robustness')}
          className={`px-4 py-2 rounded ${testType === 'robustness' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
        >
          Robustness Test
        </button>
        <button
          onClick={() => setTestType('stress')}
          className={`px-4 py-2 rounded ${testType === 'stress' ? 'bg-red-600 text-white' : 'bg-gray-200'}`}
        >
          Stress Test
        </button>
        <button
          onClick={() => setTestType('sensitivity')}
          className={`px-4 py-2 rounded ${testType === 'sensitivity' ? 'bg-purple-600 text-white' : 'bg-gray-200'}`}
        >
          Parameter Sensitivity
        </button>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded p-4 mb-4">
          <p className="text-red-800 text-sm">{error}</p>
        </div>
      )}

      {/* Robustness Test */}
      {testType === 'robustness' && (
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">Scenarios to Test</label>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
              {availableScenarios.map(scenario => (
                <label key={scenario.value} className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={robustnessScenarios.includes(scenario.value)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setRobustnessScenarios([...robustnessScenarios, scenario.value])
                      } else {
                        setRobustnessScenarios(robustnessScenarios.filter(s => s !== scenario.value))
                      }
                    }}
                    className="w-4 h-4"
                  />
                  <span className="text-sm">{scenario.label}</span>
                </label>
              ))}
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Intensity: {intensity.toFixed(1)}x</label>
            <input
              type="range"
              min="0.5"
              max="2.0"
              step="0.1"
              value={intensity}
              onChange={(e) => setIntensity(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>
          <button
            onClick={runRobustnessTest}
            disabled={loading || robustnessScenarios.length === 0}
            className="px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? 'Running...' : 'Run Robustness Test'}
          </button>
        </div>
      )}

      {/* Stress Test */}
      {testType === 'stress' && (
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">Stress Scenarios</label>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {['crash', 'flash_crash', 'high_volatility', 'gap_event'].map(scenario => (
                <label key={scenario} className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={stressScenarios.includes(scenario)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setStressScenarios([...stressScenarios, scenario])
                      } else {
                        setStressScenarios(stressScenarios.filter(s => s !== scenario))
                      }
                    }}
                    className="w-4 h-4"
                  />
                  <span className="text-sm capitalize">{scenario.replace('_', ' ')}</span>
                </label>
              ))}
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Stress Intensity: {stressIntensity.toFixed(1)}x</label>
            <input
              type="range"
              min="1.0"
              max="3.0"
              step="0.1"
              value={stressIntensity}
              onChange={(e) => setStressIntensity(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>
          <button
            onClick={runStressTest}
            disabled={loading || stressScenarios.length === 0}
            className="px-6 py-2 bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50"
          >
            {loading ? 'Running...' : 'Run Stress Test'}
          </button>
        </div>
      )}

      {/* Parameter Sensitivity */}
      {testType === 'sensitivity' && (
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">Parameter Name</label>
            <input
              type="text"
              value={paramName}
              onChange={(e) => setParamName(e.target.value)}
              className="border rounded px-3 py-2 w-full"
              placeholder="position_size"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Parameter Values (comma-separated)</label>
            <input
              type="text"
              value={paramValues}
              onChange={(e) => setParamValues(e.target.value)}
              className="border rounded px-3 py-2 w-full"
              placeholder="0.2,0.4,0.6,0.8,1.0"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Market Regime</label>
            <select
              value={regime}
              onChange={(e) => setRegime(e.target.value)}
              className="border rounded px-3 py-2 w-full"
            >
              <option value="normal">Normal</option>
              <option value="trending_up">Trending Up</option>
              <option value="high_volatility">High Volatility</option>
            </select>
          </div>
          <button
            onClick={runSensitivityTest}
            disabled={loading}
            className="px-6 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50"
          >
            {loading ? 'Running...' : 'Run Sensitivity Analysis'}
          </button>
        </div>
      )}

      {/* Results */}
      {results && !loading && (
        <div className="mt-6 border-t pt-6">
          {results.type === 'robustness' && (
            <div>
              <h4 className="font-semibold mb-4">Robustness Test Results</h4>
              {results.data.summary && (
                <div className="bg-blue-50 border border-blue-200 rounded p-4 mb-4">
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <span className="text-sm text-gray-600">Total Scenarios</span>
                      <div className="text-lg font-bold">{results.data.summary.total_scenarios}</div>
                    </div>
                    <div>
                      <span className="text-sm text-gray-600">Avg Return</span>
                      <div className="text-lg font-bold">{formatPercent(results.data.summary.average_return)}</div>
                    </div>
                    <div>
                      <span className="text-sm text-gray-600">Worst Drawdown</span>
                      <div className="text-lg font-bold text-red-600">{formatPercent(results.data.summary.worst_drawdown)}</div>
                    </div>
                  </div>
                </div>
              )}
              <div className="space-y-2">
                {results.data.scenarios?.map((scenario, idx) => (
                  <div key={idx} className="border rounded p-4">
                    <div className="font-semibold mb-2 capitalize">{scenario.scenario_name.replace('_', ' ')}</div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
                      <div>
                        <span className="text-gray-600">Return:</span>
                        <span className={`ml-2 font-semibold ${scenario.total_return >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {formatPercent(scenario.total_return)}
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-600">Sharpe:</span>
                        <span className="ml-2 font-semibold">{scenario.sharpe_ratio.toFixed(2)}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Drawdown:</span>
                        <span className="ml-2 font-semibold text-red-600">{formatPercent(scenario.max_drawdown)}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Win Rate:</span>
                        <span className="ml-2 font-semibold">{formatPercent(scenario.win_rate)}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {results.type === 'stress' && (
            <div>
              <h4 className="font-semibold mb-4">Stress Test Results</h4>
              {results.data.summary && (
                <div className={`border rounded p-4 mb-4 ${
                  results.data.summary.survived_count === results.data.summary.total_tests ?
                    'bg-green-50 border-green-200' : 'bg-yellow-50 border-yellow-200'
                }`}>
                  <div className="grid grid-cols-4 gap-4">
                    <div>
                      <span className="text-sm text-gray-600">Total Tests</span>
                      <div className="text-lg font-bold">{results.data.summary.total_tests}</div>
                    </div>
                    <div>
                      <span className="text-sm text-gray-600">Survived</span>
                      <div className="text-lg font-bold text-green-600">{results.data.summary.survived_count}</div>
                    </div>
                    <div>
                      <span className="text-sm text-gray-600">Worst Drawdown</span>
                      <div className="text-lg font-bold text-red-600">{formatPercent(results.data.summary.worst_drawdown)}</div>
                    </div>
                    <div>
                      <span className="text-sm text-gray-600">Avg Recovery</span>
                      <div className="text-lg font-bold">{results.data.summary.avg_recovery_time.toFixed(0)} periods</div>
                    </div>
                  </div>
                </div>
              )}
              <div className="space-y-2">
                {results.data.stress_tests?.map((test, idx) => (
                  <div key={idx} className={`border rounded p-4 ${
                    test.survived ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'
                  }`}>
                    <div className="flex items-center justify-between mb-2">
                      <div className="font-semibold capitalize">{test.scenario_name.replace('_', ' ')}</div>
                      <span className={`px-3 py-1 rounded text-sm font-semibold ${
                        test.survived ? 'bg-green-200 text-green-800' : 'bg-red-200 text-red-800'
                      }`}>
                        {test.survived ? 'SURVIVED' : 'FAILED'}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
                      <div>
                        <span className="text-gray-600">Max Drawdown:</span>
                        <span className="ml-2 font-semibold text-red-600">{formatPercent(test.max_drawdown)}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Recovery Time:</span>
                        <span className="ml-2 font-semibold">{test.recovery_time} periods</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Worst Loss:</span>
                        <span className="ml-2 font-semibold text-red-600">{formatPercent(test.worst_case_loss)}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Equity at Min:</span>
                        <span className="ml-2 font-semibold">{formatCurrency(test.equity_at_min)}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {results.type === 'sensitivity' && (
            <div>
              <h4 className="font-semibold mb-4">Parameter Sensitivity Analysis</h4>
              <div className="bg-purple-50 border border-purple-200 rounded p-4 mb-4">
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <span className="text-sm text-gray-600">Parameter</span>
                    <div className="text-lg font-bold">{results.data.parameter_name}</div>
                  </div>
                  <div>
                    <span className="text-sm text-gray-600">Optimal Value</span>
                    <div className="text-lg font-bold text-purple-600">{results.data.optimal_value.toFixed(4)}</div>
                  </div>
                  <div>
                    <span className="text-sm text-gray-600">Sensitivity Score</span>
                    <div className="text-lg font-bold">
                      {(results.data.sensitivity_score * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>
              {results.data.recommendations && results.data.recommendations.length > 0 && (
                <div className="bg-yellow-50 border border-yellow-200 rounded p-3 mb-4">
                  <h5 className="font-semibold text-sm mb-2">Recommendations</h5>
                  <ul className="list-disc list-inside text-sm space-y-1">
                    {results.data.recommendations.map((rec, idx) => (
                      <li key={idx}>{rec}</li>
                    ))}
                  </ul>
                </div>
              )}
              {results.data.performance_metrics && (
                <div className="border rounded p-4">
                  <h5 className="font-semibold text-sm mb-3">Performance by Parameter Value</h5>
                  <div className="space-y-2">
                    {results.data.parameter_values.map((value, idx) => (
                      <div key={idx} className="flex items-center justify-between text-sm">
                        <span className="font-mono">{value.toFixed(4)}</span>
                        <div className="flex gap-4">
                          <span>Return: {formatPercent(results.data.performance_metrics.total_return[idx])}</span>
                          <span>Sharpe: {results.data.performance_metrics.sharpe_ratio[idx].toFixed(2)}</span>
                          <span>DD: {formatPercent(results.data.performance_metrics.max_drawdown[idx])}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default ScenarioSimulationPanel

