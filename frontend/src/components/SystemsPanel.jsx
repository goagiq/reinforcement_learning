import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { 
  RefreshCw, 
  Brain, 
  TrendingUp, 
  Play, 
  Database, 
  Settings, 
  Filter,
  CheckCircle2,
  AlertCircle,
  XCircle,
  Clock
} from 'lucide-react'

const SystemsPanel = () => {
  const [systemsStatus, setSystemsStatus] = useState(null)
  const [loading, setLoading] = useState(true)
  const [lastUpdated, setLastUpdated] = useState(null)
  const [error, setError] = useState(null)

  const loadSystemsStatus = async () => {
    try {
      setError(null)
      const response = await axios.get('/api/systems/status')
      if (response.data.status === 'success') {
        setSystemsStatus(response.data.components)
        setLastUpdated(new Date())
      } else {
        setError('Failed to load systems status')
      }
    } catch (err) {
      console.error('Failed to load systems status:', err)
      setError(err.response?.data?.detail || 'Failed to load systems status')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    // Initial load
    loadSystemsStatus()
    
    // Auto-refresh every 10 seconds
    const intervalId = setInterval(() => {
      loadSystemsStatus()
    }, 10000)
    
    return () => {
      clearInterval(intervalId)
    }
  }, [])

  const getStatusIcon = (status, componentName = '') => {
    // For trading system, "stopped" is a normal state (not an error)
    const isStoppedNormal = componentName === 'trading_system' && status === 'stopped'
    
    switch (status) {
      case 'active':
      case 'reading_config':
      case 'available':
        return <CheckCircle2 className="w-6 h-6 text-green-600" />
      case 'starting':
      case 'inactive':
      case 'using_defaults':
      case 'no_config':
        return <AlertCircle className="w-6 h-6 text-yellow-600" />
      case 'stopped':
        // "stopped" is normal for trading (yellow), but error for others
        return isStoppedNormal 
          ? <AlertCircle className="w-6 h-6 text-yellow-600" />
          : <XCircle className="w-6 h-6 text-red-600" />
      case 'error':
      case 'no_data':
      case 'no_directory':
      case 'not_applicable':
        return <XCircle className="w-6 h-6 text-red-600" />
      default:
        return <Clock className="w-6 h-6 text-gray-400" />
    }
  }

  const getStatusColor = (status, componentName = '') => {
    // For trading system, "stopped" is a normal state (not an error)
    const isStoppedNormal = componentName === 'trading_system' && status === 'stopped'
    
    switch (status) {
      case 'active':
      case 'reading_config':
      case 'available':
        return 'bg-green-50 border-green-200'
      case 'starting':
      case 'inactive':
      case 'using_defaults':
      case 'no_config':
        return 'bg-yellow-50 border-yellow-200'
      case 'stopped':
        // "stopped" is normal for trading (yellow), but error for others
        return isStoppedNormal
          ? 'bg-yellow-50 border-yellow-200'
          : 'bg-red-50 border-red-200'
      case 'error':
      case 'no_data':
      case 'no_directory':
      case 'not_applicable':
        return 'bg-red-50 border-red-200'
      default:
        return 'bg-gray-50 border-gray-200'
    }
  }

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'Never'
    try {
      const date = new Date(timestamp)
      return date.toLocaleString()
    } catch {
      return timestamp
    }
  }

  const formatTimeAgo = (timestamp) => {
    if (!timestamp) return 'Never'
    try {
      const date = new Date(timestamp)
      const now = new Date()
      const diffMs = now - date
      const diffSec = Math.floor(diffMs / 1000)
      const diffMin = Math.floor(diffSec / 60)
      const diffHour = Math.floor(diffMin / 60)
      
      if (diffSec < 60) return `${diffSec}s ago`
      if (diffMin < 60) return `${diffMin}m ago`
      if (diffHour < 24) return `${diffHour}h ago`
      return formatTimestamp(timestamp)
    } catch {
      return timestamp
    }
  }

  if (loading && !systemsStatus) {
    return (
      <div className="bg-white rounded-lg shadow p-12 text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
        <p className="text-gray-600">Loading systems status...</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow p-6 flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-800">System Components Status</h2>
          <p className="text-sm text-gray-600 mt-1">Monitor all trading system components health</p>
        </div>
        <div className="flex items-center gap-4">
          {lastUpdated && (
            <span className="text-sm text-gray-500">
              Updated: {lastUpdated.toLocaleTimeString()}
            </span>
          )}
          <button
            onClick={loadSystemsStatus}
            className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 flex items-center gap-2"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800">{error}</p>
        </div>
      )}

      {systemsStatus && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Adaptive Learning Component */}
          <div className={`bg-white rounded-lg shadow border-2 ${getStatusColor(systemsStatus.adaptive_learning?.status, 'adaptive_learning')} p-6`}>
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <Brain className="w-6 h-6 text-purple-600" />
                <h3 className="text-lg font-semibold text-gray-800">Adaptive Learning</h3>
              </div>
              {getStatusIcon(systemsStatus.adaptive_learning?.status, 'adaptive_learning')}
            </div>
            <p className="text-sm text-gray-600 mb-4">{systemsStatus.adaptive_learning?.message || 'Unknown status'}</p>
            
            {systemsStatus.adaptive_learning?.current_parameters && Object.keys(systemsStatus.adaptive_learning.current_parameters).length > 0 && (
              <div className="space-y-2 mb-4">
                <div className="text-xs font-semibold text-gray-500 uppercase">Current Parameters</div>
                {systemsStatus.adaptive_learning.current_parameters.min_risk_reward_ratio && (
                  <div className="text-sm">
                    <span className="text-gray-600">R:R Ratio:</span>{' '}
                    <span className="font-semibold text-gray-800">{systemsStatus.adaptive_learning.current_parameters.min_risk_reward_ratio}</span>
                  </div>
                )}
                {systemsStatus.adaptive_learning.current_parameters.quality_filters && (
                  <div className="text-sm space-y-1">
                    <div>
                      <span className="text-gray-600">Min Confidence:</span>{' '}
                      <span className="font-semibold text-gray-800">
                        {systemsStatus.adaptive_learning.current_parameters.quality_filters.min_action_confidence}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-600">Min Quality:</span>{' '}
                      <span className="font-semibold text-gray-800">
                        {systemsStatus.adaptive_learning.current_parameters.quality_filters.min_quality_score}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            )}
            
            {/* Last Adjustment Information */}
            <div className="mt-4 pt-4 border-t border-gray-200 space-y-3">
              {systemsStatus.adaptive_learning?.last_adjustment ? (
                <>
                  <div>
                    <div className="text-xs font-semibold text-gray-500 uppercase mb-1">Last Adjustment</div>
                    <div className="text-sm font-semibold text-gray-800">
                      {formatTimeAgo(systemsStatus.adaptive_learning.last_adjustment.timestamp)}
                    </div>
                    {systemsStatus.adaptive_learning.last_adjustment.timestep && (
                      <div className="text-xs text-gray-500 mt-1">
                        At timestep: {systemsStatus.adaptive_learning.last_adjustment.timestep.toLocaleString()}
                      </div>
                    )}
                    {systemsStatus.adaptive_learning.last_adjustment.adjustments && 
                     Object.keys(systemsStatus.adaptive_learning.last_adjustment.adjustments).length > 0 && (
                      <div className="text-xs text-gray-600 mt-2">
                        <div className="font-semibold mb-1">Adjustments made:</div>
                        <div className="pl-2 space-y-1">
                          {Object.entries(systemsStatus.adaptive_learning.last_adjustment.adjustments).map(([key, value]) => (
                            <div key={key}>
                              {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}: {String(value)}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </>
              ) : (
                <div>
                  <div className="text-xs font-semibold text-gray-500 uppercase mb-1">Last Adjustment</div>
                  <div className="text-sm text-gray-500 italic">No adjustments made yet</div>
                </div>
              )}
              
              {systemsStatus.adaptive_learning?.total_adjustments !== undefined && (
                <div className="text-sm">
                  <span className="text-gray-600">Total adjustments:</span>{' '}
                  <span className="font-semibold text-gray-800">{systemsStatus.adaptive_learning.total_adjustments}</span>
                </div>
              )}
            </div>
          </div>

          {/* Training System Component */}
          <div className={`bg-white rounded-lg shadow border-2 ${getStatusColor(systemsStatus.training_system?.status, 'training_system')} p-6`}>
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <TrendingUp className="w-6 h-6 text-blue-600" />
                <h3 className="text-lg font-semibold text-gray-800">Training System</h3>
              </div>
              {getStatusIcon(systemsStatus.training_system?.status, 'training_system')}
            </div>
            <p className="text-sm text-gray-600 mb-4">{systemsStatus.training_system?.message || 'Unknown status'}</p>
            
            {systemsStatus.training_system?.gpu_status && (
              <div className="space-y-2">
                <div className="text-xs font-semibold text-gray-500 uppercase">GPU Status</div>
                <div className="text-sm">
                  <span className="text-gray-600">GPU:</span>{' '}
                  <span className={`font-semibold ${
                    systemsStatus.training_system.gpu_status === 'active' ? 'text-green-600' :
                    systemsStatus.training_system.gpu_status === 'available' ? 'text-yellow-600' :
                    'text-gray-600'
                  }`}>
                    {systemsStatus.training_system.gpu_status === 'active' ? 'Active' :
                     systemsStatus.training_system.gpu_status === 'available' ? 'Available (not used)' :
                     systemsStatus.training_system.gpu_status === 'available_not_used' ? 'Available (CPU used)' :
                     systemsStatus.training_system.gpu_status === 'not_available' ? 'Not Available' :
                     'Unknown'}
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Trading System Component */}
          <div className={`bg-white rounded-lg shadow border-2 ${getStatusColor(systemsStatus.trading_system?.status, 'trading_system')} p-6`}>
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <Play className="w-6 h-6 text-green-600" />
                <h3 className="text-lg font-semibold text-gray-800">Trading System</h3>
              </div>
              {getStatusIcon(systemsStatus.trading_system?.status, 'trading_system')}
            </div>
            <p className="text-sm text-gray-600">{systemsStatus.trading_system?.message || 'Unknown status'}</p>
          </div>

          {/* Data Pipeline Component */}
          <div className={`bg-white rounded-lg shadow border-2 ${getStatusColor(systemsStatus.data_pipeline?.status, 'data_pipeline')} p-6`}>
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <Database className="w-6 h-6 text-indigo-600" />
                <h3 className="text-lg font-semibold text-gray-800">Data Pipeline</h3>
              </div>
              {getStatusIcon(systemsStatus.data_pipeline?.status, 'data_pipeline')}
            </div>
            <p className="text-sm text-gray-600 mb-4">{systemsStatus.data_pipeline?.message || 'Unknown status'}</p>
            
            {systemsStatus.data_pipeline?.file_count !== undefined && (
              <div className="text-sm">
                <span className="text-gray-600">Data files:</span>{' '}
                <span className="font-semibold text-gray-800">{systemsStatus.data_pipeline.file_count}</span>
              </div>
            )}
          </div>

          {/* Environment Component */}
          <div className={`bg-white rounded-lg shadow border-2 ${getStatusColor(systemsStatus.environment?.status, 'environment')} p-6`}>
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <Settings className="w-6 h-6 text-orange-600" />
                <h3 className="text-lg font-semibold text-gray-800">Environment</h3>
              </div>
              {getStatusIcon(systemsStatus.environment?.status, 'environment')}
            </div>
            <p className="text-sm text-gray-600 mb-4">{systemsStatus.environment?.message || 'Unknown status'}</p>
            
            {systemsStatus.environment?.config && Object.keys(systemsStatus.environment.config).length > 0 && (
              <div className="space-y-1">
                {systemsStatus.environment.config.min_risk_reward_ratio && (
                  <div className="text-sm">
                    <span className="text-gray-600">R:R Ratio:</span>{' '}
                    <span className="font-semibold text-gray-800">{systemsStatus.environment.config.min_risk_reward_ratio}</span>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Decision Gate Component */}
          <div className={`bg-white rounded-lg shadow border-2 ${getStatusColor(systemsStatus.decision_gate?.status, 'decision_gate')} p-6`}>
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <Filter className="w-6 h-6 text-pink-600" />
                <h3 className="text-lg font-semibold text-gray-800">Decision Gate</h3>
              </div>
              {getStatusIcon(systemsStatus.decision_gate?.status, 'decision_gate')}
            </div>
            <p className="text-sm text-gray-600 mb-4">{systemsStatus.decision_gate?.message || 'Unknown status'}</p>
            
            {systemsStatus.decision_gate?.quality_filters && Object.keys(systemsStatus.decision_gate.quality_filters).length > 0 && (
              <div className="space-y-1">
                <div className="text-xs font-semibold text-gray-500 uppercase">Quality Filters</div>
                {systemsStatus.decision_gate.quality_filters.min_action_confidence && (
                  <div className="text-sm">
                    <span className="text-gray-600">Min Confidence:</span>{' '}
                    <span className="font-semibold text-gray-800">
                      {systemsStatus.decision_gate.quality_filters.min_action_confidence}
                    </span>
                  </div>
                )}
                {systemsStatus.decision_gate.quality_filters.min_quality_score && (
                  <div className="text-sm">
                    <span className="text-gray-600">Min Quality:</span>{' '}
                    <span className="font-semibold text-gray-800">
                      {systemsStatus.decision_gate.quality_filters.min_quality_score}
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Status Legend</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="flex items-center gap-3">
            <CheckCircle2 className="w-5 h-5 text-green-600" />
            <div>
              <div className="font-semibold text-gray-800">Healthy</div>
              <div className="text-sm text-gray-600">Component is active and working correctly</div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-yellow-600" />
            <div>
              <div className="font-semibold text-gray-800">Warning</div>
              <div className="text-sm text-gray-600">Component is not active or using defaults</div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <XCircle className="w-5 h-5 text-red-600" />
            <div>
              <div className="font-semibold text-gray-800">Error/Stopped</div>
              <div className="text-sm text-gray-600">Component has errors or is stopped</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default SystemsPanel

