import React, { useState } from 'react'
import axios from 'axios'
import { BarChart3, Loader, AlertCircle, CheckCircle, Download, RefreshCw } from 'lucide-react'
import CapabilityExplainer from './CapabilityExplainer'

const defaultFormState = {
  instrument: '',
  timeframes: '1,5,15',
  numRegimes: 3,
  rollingVolWindow: 50,
  volumeZscoreWindow: 100,
  minSamples: 500,
  saveReport: true,
  configPath: 'configs/train_config_full.yaml'
}

const parseTimeframes = (value) =>
  value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => Number.parseInt(item, 10))
    .filter((num) => !Number.isNaN(num))

const formatProbability = (value) => `${(value * 100).toFixed(2)}%`

const MarkovAnalysisPanel = () => {
  const [form, setForm] = useState(defaultFormState)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)

  const handleChange = (field, value) => {
    setForm((prev) => ({
      ...prev,
      [field]: value
    }))
  }

  const runAnalysis = async () => {
    setLoading(true)
    setError(null)

    const payload = {
      instrument: form.instrument || undefined,
      timeframes: parseTimeframes(form.timeframes),
      num_regimes: Number(form.numRegimes),
      rolling_vol_window: Number(form.rollingVolWindow),
      volume_zscore_window: Number(form.volumeZscoreWindow),
      min_samples: Number(form.minSamples),
      save_report: Boolean(form.saveReport),
      config_path: form.configPath || undefined
    }

    try {
      const response = await axios.post('/api/analysis/markov/run', payload)
      setResult(response.data)
    } catch (err) {
      const message = err.response?.data?.detail || err.message || 'Failed to run Markov analysis'
      setError(message)
    } finally {
      setLoading(false)
    }
  }

  const resetForm = () => {
    setForm(defaultFormState)
    setResult(null)
    setError(null)
  }

  const transitionMatrix = result?.transition_matrix
  const stationary = result?.stationary_distribution || []
  const regimeSummary = result?.regime_summary || []
  const clusterPreview = result?.cluster_preview || []
  const metadata = result?.metadata

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
              <BarChart3 className="w-6 h-6 text-primary-600" />
              Markov Regime Analysis
            </h2>
            <p className="text-sm text-gray-600 mt-1">
              Run offline Markov regime clustering to understand market transitions and dominant regimes.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <CapabilityExplainer
              capabilityId="analytics.markov"
              context={{
                form,
                hasResult: Boolean(result),
                metadata,
              }}
            />
            <button
              type="button"
              onClick={resetForm}
              className="px-3 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg flex items-center gap-2 text-gray-700"
            >
              <RefreshCw className="w-4 h-4" />
              Reset
            </button>
            <button
              type="button"
              onClick={runAnalysis}
              disabled={loading}
              className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 flex items-center gap-2"
            >
              {loading ? (
                <>
                  <Loader className="w-4 h-4 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <BarChart3 className="w-4 h-4" />
                  Run Analysis
                </>
              )}
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-1">Instrument</label>
            <input
              value={form.instrument}
              onChange={(e) => handleChange('instrument', e.target.value)}
              placeholder="ES"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
            <p className="text-xs text-gray-500 mt-1">Leave blank to use the instrument from the selected config.</p>
          </div>

          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-1">Timeframes (comma separated)</label>
            <input
              value={form.timeframes}
              onChange={(e) => handleChange('timeframes', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
            <p className="text-xs text-gray-500 mt-1">Example: 1,5,15</p>
          </div>

          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-1">Config Path</label>
            <input
              value={form.configPath}
              onChange={(e) => handleChange('configPath', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
            <p className="text-xs text-gray-500 mt-1">Used to resolve defaults when fields are left blank.</p>
          </div>

          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-1">Number of Regimes</label>
            <input
              type="number"
              min={2}
              value={form.numRegimes}
              onChange={(e) => handleChange('numRegimes', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>

          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-1">Rolling Volatility Window</label>
            <input
              type="number"
              min={5}
              value={form.rollingVolWindow}
              onChange={(e) => handleChange('rollingVolWindow', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>

          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-1">Volume Z-Score Window</label>
            <input
              type="number"
              min={10}
              value={form.volumeZscoreWindow}
              onChange={(e) => handleChange('volumeZscoreWindow', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>

          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-1">Minimum Samples</label>
            <input
              type="number"
              min={100}
              value={form.minSamples}
              onChange={(e) => handleChange('minSamples', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
            <p className="text-xs text-gray-500 mt-1">Analysis will fail if fewer samples are available.</p>
          </div>

          <div className="flex items-center gap-2 mt-6">
            <input
              id="save-report"
              type="checkbox"
              checked={form.saveReport}
              onChange={(e) => handleChange('saveReport', e.target.checked)}
              className="w-4 h-4"
            />
            <label htmlFor="save-report" className="text-sm text-gray-700">
              Save report to <code>reports/markov_regime_report.json</code>
            </label>
          </div>
        </div>

        {error && (
          <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-center gap-3 text-red-700">
            <AlertCircle className="w-5 h-5" />
            <span>{error}</span>
          </div>
        )}

        {result && (
          <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg flex items-center gap-3 text-green-700">
            <CheckCircle className="w-5 h-5" />
            <div>
              <div className="font-semibold">Analysis completed successfully</div>
              <div className="text-sm opacity-80">
                Runtime: {result.runtime_ms.toFixed(0)} ms
                {result.saved_report && (
                  <span className="ml-2 inline-flex items-center gap-1">
                    <Download className="w-4 h-4" />
                    Saved to: <code>{result.saved_report}</code>
                  </span>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {result && (
        <>
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Analysis Summary</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="text-xs text-gray-500">Instrument</div>
                <div className="text-lg font-semibold text-gray-800">{metadata?.instrument}</div>
              </div>
              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="text-xs text-gray-500">Timeframes</div>
                <div className="text-lg font-semibold text-gray-800">
                  {metadata?.timeframes?.join(', ') || 'N/A'}
                </div>
              </div>
              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="text-xs text-gray-500">Detected Regimes</div>
                <div className="text-lg font-semibold text-gray-800">{metadata?.num_regimes}</div>
              </div>
              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="text-xs text-gray-500">Rolling Vol Window</div>
                <div className="text-lg font-semibold text-gray-800">{metadata?.rolling_vol_window}</div>
              </div>
              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="text-xs text-gray-500">Volume Z-Score Window</div>
                <div className="text-lg font-semibold text-gray-800">{metadata?.volume_zscore_window}</div>
              </div>
              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="text-xs text-gray-500">Samples Analyzed</div>
                <div className="text-lg font-semibold text-gray-800">
                  {metadata?.sample_count?.toLocaleString() || 'N/A'}
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Stationary Distribution</h3>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="text-left text-gray-600 border-b">
                    <th className="py-2 pr-4">Regime</th>
                    <th className="py-2 pr-4">Probability</th>
                  </tr>
                </thead>
                <tbody>
                  {stationary.map((row) => (
                    <tr key={row.regime} className="border-b last:border-0">
                      <td className="py-2 pr-4 font-medium text-gray-800">{row.regime}</td>
                      <td className="py-2 pr-4 text-gray-700">{formatProbability(row.probability)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {transitionMatrix && (
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Transition Matrix</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead>
                    <tr className="text-left text-gray-600 border-b">
                      <th className="py-2 pr-4">From \\ To</th>
                      {transitionMatrix.columns.map((col) => (
                        <th key={col} className="py-2 px-3 text-right">
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {transitionMatrix.values.map((row, rowIdx) => (
                      <tr key={`${transitionMatrix.index[rowIdx]}-row`} className="border-b last:border-0">
                        <td className="py-2 pr-4 font-medium text-gray-800">{transitionMatrix.index[rowIdx]}</td>
                        {row.map((value, colIdx) => (
                          <td key={`${rowIdx}-${colIdx}`} className="py-2 px-3 text-right text-gray-700">
                            {(value * 100).toFixed(2)}%
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Regime Summary</h3>
            <div className="overflow-x-auto">
              <table className="min-w-full text-xs md:text-sm">
                <thead>
                  <tr className="text-left text-gray-600 border-b">
                    <th className="py-2 pr-3">Regime</th>
                    <th className="py-2 pr-3 text-right">Mean Return</th>
                    <th className="py-2 pr-3 text-right">Median Return</th>
                    <th className="py-2 pr-3 text-right">Volatility</th>
                    <th className="py-2 pr-3 text-right">Volume Z-Score</th>
                    <th className="py-2 pr-3 text-right">Frequency</th>
                    <th className="py-2 pr-3 text-right">Frequency %</th>
                  </tr>
                </thead>
                <tbody>
                  {regimeSummary.map((row) => (
                    <tr key={row.regime_id} className="border-b last:border-0">
                      <td className="py-2 pr-3 font-medium text-gray-800">{row.regime_id}</td>
                      <td className="py-2 pr-3 text-right text-gray-700">{row.mean_return?.toFixed(6)}</td>
                      <td className="py-2 pr-3 text-right text-gray-700">{row.median_return?.toFixed(6)}</td>
                      <td className="py-2 pr-3 text-right text-gray-700">{row.volatility?.toFixed(6)}</td>
                      <td className="py-2 pr-3 text-right text-gray-700">{row.avg_volume_z?.toFixed(4)}</td>
                      <td className="py-2 pr-3 text-right text-gray-700">{row.frequency?.toLocaleString()}</td>
                      <td className="py-2 pr-3 text-right text-gray-700">{formatProbability(row.frequency_pct || 0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {clusterPreview.length > 0 && (
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Recent Regime Assignments</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full text-xs md:text-sm">
                  <thead>
                    <tr className="text-left text-gray-600 border-b">
                      {Object.keys(clusterPreview[0]).map((key) => (
                        <th key={key} className="py-2 pr-4 capitalize">
                          {key.replace('_', ' ')}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {clusterPreview.map((row, idx) => (
                      <tr key={idx} className="border-b last:border-0">
                        {Object.values(row).map((value, valueIdx) => (
                          <td key={`${idx}-${valueIdx}`} className="py-2 pr-4 text-gray-700">
                            {typeof value === 'number' ? value.toFixed(4) : value}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default MarkovAnalysisPanel


