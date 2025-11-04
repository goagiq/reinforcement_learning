import React, { useState, useEffect } from 'react'
import axios from 'axios'
import TrainingPanel from './TrainingPanel'
import BacktestPanel from './BacktestPanel'
import TradingPanel from './TradingPanel'
import MonitoringPanel from './MonitoringPanel'
import SettingsPanel from './SettingsPanel'
import { Activity, Brain, TrendingUp, Play, Settings, BarChart3 } from 'lucide-react'
import { getDefaultModel, isDefaultModel } from '../utils/modelUtils'

const Dashboard = ({ onSetupChange }) => {
  const [activeTab, setActiveTab] = useState('overview')
  const [models, setModels] = useState([])
  const [performance, setPerformance] = useState(null)
  const [ws, setWs] = useState(null)
  const [notifications, setNotifications] = useState([])
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [modelComparison, setModelComparison] = useState(null)

  // Listen for openSettings event from TrainingPanel
  useEffect(() => {
    const handleOpenSettings = () => {
      setSettingsOpen(true)
    }
    
    window.addEventListener('openSettings', handleOpenSettings)
    
    return () => {
      window.removeEventListener('openSettings', handleOpenSettings)
    }
  }, [])

  useEffect(() => {
    loadModels()
    loadPerformance()
    loadModelComparison()
    
    // Set up periodic refresh of model comparison while training is active
    // This catches best_model updates that happen during training (not just at the end)
    const comparisonRefreshInterval = setInterval(() => {
      // Only refresh if we're on the overview tab and models might be updating
      if (activeTab === 'overview') {
        loadModelComparison()
      }
    }, 30000) // Refresh every 30 seconds
    
    // Connect to WebSocket with proper error handling
    let websocket = null
    let reconnectAttempts = 0
    let reconnectTimeout = null
    let isMounted = true
    const maxReconnectAttempts = 5
    const reconnectDelay = 3000 // 3 seconds
    
    const connectWebSocket = () => {
      if (!isMounted) return
      
      try {
        // Use the Vite proxy by connecting through the same host
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
        const wsUrl = `${wsProtocol}//${window.location.host}/ws`
        
        // Suppress browser console warnings for WebSocket during StrictMode
        const originalError = console.error
        const errorFilter = (...args) => {
          const message = args.join(' ')
          // Suppress StrictMode-related WebSocket warnings
          if (message.includes('WebSocket is closed before the connection is established') ||
              (message.includes('WebSocket connection to') && message.includes('failed'))) {
            return // Suppress these specific warnings
          }
          originalError.apply(console, args)
        }
        console.error = errorFilter
        
        websocket = new WebSocket(wsUrl)
        
        // Restore console.error after connection attempt
        setTimeout(() => {
          console.error = originalError
        }, 200)
        
        websocket.onopen = () => {
          if (!isMounted) {
            websocket.close()
            return
          }
          console.log('WebSocket connected')
          reconnectAttempts = 0 // Reset on successful connection
        }

        websocket.onmessage = (event) => {
          if (!isMounted) return
          try {
            const data = JSON.parse(event.data)
            handleWebSocketMessage(data)
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error)
          }
        }

        websocket.onerror = (error) => {
          // Suppress - React StrictMode causes benign errors
        }

        websocket.onclose = (event) => {
          if (!isMounted) return
          // Only log unexpected closures (not during StrictMode cleanup)
          // Code 1006 = abnormal closure (common during StrictMode unmount)
          if (event.code !== 1000 && event.code !== 1001 && event.code !== 1006) {
            console.log('WebSocket closed', event.code, event.reason)
          }
          websocket = null
          
          // Attempt to reconnect if not a normal closure and haven't exceeded max attempts
          if (event.code !== 1000 && reconnectAttempts < maxReconnectAttempts && isMounted) {
            reconnectAttempts++
            console.log(`Attempting to reconnect WebSocket (${reconnectAttempts}/${maxReconnectAttempts})...`)
            reconnectTimeout = setTimeout(connectWebSocket, reconnectDelay)
          } else if (reconnectAttempts >= maxReconnectAttempts) {
            console.warn('WebSocket reconnection limit reached. Real-time updates will not be available.')
          }
        }

        setWs(websocket)
      } catch (error) {
        console.error('Failed to create WebSocket connection:', error)
        // App continues to work without WebSocket
      }
    }

    connectWebSocket()

    return () => {
      isMounted = false
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout)
      }
      if (websocket) {
        // Only close if WebSocket is in a state that can be closed
        // 0 = CONNECTING, 1 = OPEN, 2 = CLOSING, 3 = CLOSED
        if (websocket.readyState === WebSocket.CONNECTING || websocket.readyState === WebSocket.OPEN) {
          try {
            websocket.close(1000, 'Component unmounting')
          } catch (error) {
            // Ignore errors when closing - WebSocket might already be closed
            console.debug('WebSocket close error (ignored):', error)
          }
        }
      }
      clearInterval(comparisonRefreshInterval)
    }
  }, [])

  const handleWebSocketMessage = (data) => {
    setNotifications(prev => [...prev, {
      id: Date.now(),
      type: data.type,
      message: data.message,
      timestamp: new Date()
    }])

    // Auto-refresh on certain events
    if (data.type === 'training' && data.status === 'completed') {
      loadModels()
      loadModelComparison() // Refresh model comparison when training completes
    }
    // Also refresh when training starts (best_model might be updated during training)
    if (data.type === 'training' && (data.status === 'running' || data.status === 'started')) {
      // Refresh comparison after a short delay to allow model files to be written
      setTimeout(() => {
        loadModelComparison()
      }, 2000)
    }
  }

  const loadModels = async () => {
    try {
      const response = await axios.get('/api/models/list')
      setModels(response.data.models || [])
      // Log model counts for debugging
      if (response.data.trained_count !== undefined) {
        console.log(`Loaded ${response.data.trained_count} trained models and ${response.data.ollama_count || 0} Ollama models`)
      }
    } catch (error) {
      console.error('Failed to load models:', error)
    }
  }

  const loadPerformance = async () => {
    try {
      const response = await axios.get('/api/monitoring/performance')
      if (response.data.status === 'success') {
        setPerformance(response.data.metrics)
      }
    } catch (error) {
      console.error('Failed to load performance:', error)
    }
  }

  const loadModelComparison = async () => {
    try {
      const response = await axios.get('/api/models/compare')
      setModelComparison(response.data)
    } catch (error) {
      console.error('Failed to load model comparison:', error)
    }
  }

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Activity },
    { id: 'training', label: 'Training', icon: Brain },
    { id: 'backtest', label: 'Backtest', icon: BarChart3 },
    { id: 'trading', label: 'Trading', icon: Play },
    { id: 'monitoring', label: 'Monitoring', icon: TrendingUp },
  ]

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold text-gray-800">NT8 RL Trading System</h1>
            <div className="flex items-center gap-4">
              <button
                onClick={() => setSettingsOpen(true)}
                className="px-4 py-2 text-gray-600 hover:text-gray-800 flex items-center gap-2"
              >
                <Settings className="w-5 h-5" />
                Settings
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Tabs */}
      <div className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex gap-1">
            {tabs.map(tab => {
              const Icon = tab.icon
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`
                    px-6 py-4 flex items-center gap-2 border-b-2 transition-colors
                    ${activeTab === tab.id
                      ? 'border-primary-600 text-primary-600 font-semibold'
                      : 'border-transparent text-gray-600 hover:text-gray-800'
                    }
                  `}
                >
                  <Icon className="w-5 h-5" />
                  {tab.label}
                </button>
              )
            })}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {activeTab === 'overview' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-white rounded-lg shadow p-6">
                <div className="text-sm text-gray-600 mb-2">Trained Models</div>
                <div className="text-3xl font-bold text-gray-800">
                  {models.filter(m => m.type === 'trained' || !m.type).length}
                </div>
                <div className="text-sm text-gray-500 mt-2">
                  {(() => {
                    const trainedModels = models.filter(m => m.type === 'trained' || !m.type)
                    if (trainedModels.length === 0) return 'No models yet'
                    const defaultModelPath = getDefaultModel(trainedModels, 'trained')
                    const defaultModel = defaultModelPath ? trainedModels.find(m => m.path === defaultModelPath) : null
                    const displayModel = defaultModel || trainedModels[0]
                    return (
                      <span>
                        {displayModel.name}
                        {defaultModel && isDefaultModel(defaultModel) && (
                          <span className="text-green-600 ml-1">⭐</span>
                        )}
                      </span>
                    )
                  })()}
                </div>
              </div>
              
              <div className="bg-white rounded-lg shadow p-6">
                <div className="text-sm text-gray-600 mb-2">System Status</div>
                <div className="text-3xl font-bold text-green-600">Ready</div>
                <div className="text-sm text-gray-500 mt-2">All systems operational</div>
              </div>

              {performance && (
                <div className="bg-white rounded-lg shadow p-6">
                  <div className="text-sm text-gray-600 mb-2">Sharpe Ratio</div>
                  <div className="text-3xl font-bold text-gray-800">
                    {performance.sharpe_ratio?.toFixed(2) || 'N/A'}
                  </div>
                  <div className="text-sm text-gray-500 mt-2">Risk-adjusted returns</div>
                </div>
              )}
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-bold text-gray-800 mb-4">Quick Start</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 border rounded-lg hover:border-primary-500 transition-colors">
                  <div className="font-semibold text-gray-800 mb-2">1. Train Model</div>
                  <div className="text-sm text-gray-600">
                    Train your RL agent on historical data to learn trading patterns.
                  </div>
                </div>
                <div className="p-4 border rounded-lg hover:border-primary-500 transition-colors">
                  <div className="font-semibold text-gray-800 mb-2">2. Backtest</div>
                  <div className="text-sm text-gray-600">
                    Validate model performance on historical data before live trading.
                  </div>
                </div>
                <div className="p-4 border rounded-lg hover:border-primary-500 transition-colors">
                  <div className="font-semibold text-gray-800 mb-2">3. Paper Trading</div>
                  <div className="text-sm text-gray-600">
                    Test with live market data in paper trading mode (no real money).
                  </div>
                </div>
                <div className="p-4 border rounded-lg hover:border-primary-500 transition-colors">
                  <div className="font-semibold text-gray-800 mb-2">4. Go Live</div>
                  <div className="text-sm text-gray-600">
                    Start automated trading with real capital (use with caution).
                  </div>
                </div>
              </div>
            </div>

            {/* Model Comparison Section */}
            {modelComparison && (modelComparison.best_model.exists || modelComparison.final_model.exists) && (
              <div className="bg-white rounded-lg shadow p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-bold text-gray-800">Model Comparison</h2>
                  {modelComparison.recommendation && (
                    <div className="flex items-center gap-2 px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-semibold">
                      <span>⭐ Recommended: {modelComparison.recommendation === 'best_model' ? 'best_model.pt' : 'final_model.pt'}</span>
                    </div>
                  )}
                </div>
                {modelComparison.recommendation_reason && (
                  <div className="mb-4 p-3 bg-blue-50 text-blue-800 rounded-lg text-sm">
                    {modelComparison.recommendation_reason}
                  </div>
                )}
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Best Model */}
                  <div className={`border-2 rounded-lg p-4 ${modelComparison.recommendation === 'best_model' ? 'border-green-500 bg-green-50' : 'border-gray-200'}`}>
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-lg font-semibold text-gray-800">best_model.pt</h3>
                      {modelComparison.recommendation === 'best_model' && (
                        <span className="px-2 py-1 bg-green-500 text-white text-xs font-semibold rounded">RECOMMENDED</span>
                      )}
                    </div>
                    
                    {modelComparison.best_model.exists ? (
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-600">File Size:</span>
                          <span className="font-medium">{modelComparison.best_model.file_size_mb} MB</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Created:</span>
                          <span className="font-medium">
                            {new Date(modelComparison.best_model.created_at).toLocaleDateString()} {new Date(modelComparison.best_model.created_at).toLocaleTimeString()}
                          </span>
                        </div>
                        {modelComparison.best_model.timestep !== null && (
                          <div className="flex justify-between">
                            <span className="text-gray-600">Training Timesteps:</span>
                            <span className="font-medium">{modelComparison.best_model.timestep?.toLocaleString() || 'N/A'}</span>
                          </div>
                        )}
                        {modelComparison.best_model.episode !== null && (
                          <div className="flex justify-between">
                            <span className="text-gray-600">Episodes:</span>
                            <span className="font-medium">{modelComparison.best_model.episode || 'N/A'}</span>
                          </div>
                        )}
                        {modelComparison.best_model.mean_reward !== null && (
                          <div className="flex justify-between">
                            <span className="text-gray-600">Mean Reward:</span>
                            <span className={`font-medium ${modelComparison.best_model.mean_reward >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                              {modelComparison.best_model.mean_reward.toFixed(4)}
                            </span>
                          </div>
                        )}
                        <div className="flex justify-between">
                          <span className="text-gray-600">Backtest Status:</span>
                          <span className={`font-medium ${modelComparison.best_model.evaluation_needed ? 'text-orange-600' : 'text-green-600'}`}>
                            {modelComparison.best_model.evaluation_needed ? '⚠️ Evaluation Needed' : '✅ Evaluated'}
                          </span>
                        </div>
                        {modelComparison.best_model.backtest_results && (
                          <div className="mt-3 p-2 bg-gray-50 rounded text-xs">
                            <div className="font-semibold mb-1">Backtest Results:</div>
                            {modelComparison.best_model.backtest_results.sharpe_ratio !== undefined && (
                              <div>Sharpe Ratio: {modelComparison.best_model.backtest_results.sharpe_ratio.toFixed(2)}</div>
                            )}
                            {modelComparison.best_model.backtest_results.total_return !== undefined && (
                              <div>Total Return: {(modelComparison.best_model.backtest_results.total_return * 100).toFixed(2)}%</div>
                            )}
                          </div>
                        )}
                        <div className="mt-3 pt-3 border-t border-gray-200">
                          <div className="text-xs text-gray-600 mb-1">Use Cases:</div>
                          <ul className="text-xs text-gray-700 space-y-1">
                            {modelComparison.use_case_guidance.best_model.recommended_for.map((use, idx) => (
                              <li key={idx} className="flex items-center gap-1">
                                <span className="text-green-500">•</span> {use}
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    ) : (
                      <div className="text-sm text-gray-500">Model file not found</div>
                    )}
                  </div>

                  {/* Final Model */}
                  <div className={`border-2 rounded-lg p-4 ${modelComparison.recommendation === 'final_model' ? 'border-green-500 bg-green-50' : 'border-gray-200'}`}>
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-lg font-semibold text-gray-800">final_model.pt</h3>
                      {modelComparison.recommendation === 'final_model' && (
                        <span className="px-2 py-1 bg-green-500 text-white text-xs font-semibold rounded">RECOMMENDED</span>
                      )}
                    </div>
                    
                    {modelComparison.final_model.exists ? (
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-600">File Size:</span>
                          <span className="font-medium">{modelComparison.final_model.file_size_mb} MB</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Created:</span>
                          <span className="font-medium">
                            {new Date(modelComparison.final_model.created_at).toLocaleDateString()} {new Date(modelComparison.final_model.created_at).toLocaleTimeString()}
                          </span>
                        </div>
                        {modelComparison.final_model.timestep !== null && (
                          <div className="flex justify-between">
                            <span className="text-gray-600">Training Timesteps:</span>
                            <span className="font-medium">{modelComparison.final_model.timestep?.toLocaleString() || 'N/A'}</span>
                          </div>
                        )}
                        {modelComparison.final_model.episode !== null && (
                          <div className="flex justify-between">
                            <span className="text-gray-600">Episodes:</span>
                            <span className="font-medium">{modelComparison.final_model.episode || 'N/A'}</span>
                          </div>
                        )}
                        {modelComparison.final_model.mean_reward !== null && (
                          <div className="flex justify-between">
                            <span className="text-gray-600">Mean Reward:</span>
                            <span className={`font-medium ${modelComparison.final_model.mean_reward >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                              {modelComparison.final_model.mean_reward.toFixed(4)}
                            </span>
                          </div>
                        )}
                        <div className="flex justify-between">
                          <span className="text-gray-600">Backtest Status:</span>
                          <span className={`font-medium ${modelComparison.final_model.evaluation_needed ? 'text-orange-600' : 'text-green-600'}`}>
                            {modelComparison.final_model.evaluation_needed ? '⚠️ Evaluation Needed' : '✅ Evaluated'}
                          </span>
                        </div>
                        {modelComparison.final_model.backtest_results && (
                          <div className="mt-3 p-2 bg-gray-50 rounded text-xs">
                            <div className="font-semibold mb-1">Backtest Results:</div>
                            {modelComparison.final_model.backtest_results.sharpe_ratio !== undefined && (
                              <div>Sharpe Ratio: {modelComparison.final_model.backtest_results.sharpe_ratio.toFixed(2)}</div>
                            )}
                            {modelComparison.final_model.backtest_results.total_return !== undefined && (
                              <div>Total Return: {(modelComparison.final_model.backtest_results.total_return * 100).toFixed(2)}%</div>
                            )}
                          </div>
                        )}
                        <div className="mt-3 pt-3 border-t border-gray-200">
                          <div className="text-xs text-gray-600 mb-1">Use Cases:</div>
                          <ul className="text-xs text-gray-700 space-y-1">
                            {modelComparison.use_case_guidance.final_model.recommended_for.map((use, idx) => (
                              <li key={idx} className="flex items-center gap-1">
                                <span className="text-blue-500">•</span> {use}
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    ) : (
                      <div className="text-sm text-gray-500">Model file not found</div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {notifications.length > 0 && (
              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-xl font-bold text-gray-800 mb-4">Recent Activity</h2>
                <div className="space-y-2">
                  {notifications.slice(-5).reverse().map(notif => (
                    <div
                      key={notif.id}
                      className={`p-3 rounded-lg text-sm ${
                        notif.type === 'error' ? 'bg-red-50 text-red-700' :
                        notif.type === 'success' ? 'bg-green-50 text-green-700' :
                        'bg-blue-50 text-blue-700'
                      }`}
                    >
                      <div className="font-semibold">{notif.type}</div>
                      <div>{notif.message}</div>
                      <div className="text-xs opacity-75 mt-1">
                        {notif.timestamp.toLocaleTimeString()}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'training' && (
          <TrainingPanel models={models} onModelsChange={loadModels} />
        )}

        {activeTab === 'backtest' && (
          <BacktestPanel models={models} />
        )}

        {activeTab === 'trading' && (
          <TradingPanel models={models} />
        )}

        {activeTab === 'monitoring' && (
          <MonitoringPanel />
        )}
      </main>

      {/* Settings Panel */}
      <SettingsPanel 
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        models={models}
      />
    </div>
  )
}

export default Dashboard

