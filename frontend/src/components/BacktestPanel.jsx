import React, { useState } from 'react'
import axios from 'axios'
import { Play, Loader, CheckCircle, AlertCircle, BarChart3 } from 'lucide-react'
import { getDefaultModel, sortModelsWithDefaultFirst, isDefaultModel } from '../utils/modelUtils'
import CapabilityExplainer from './CapabilityExplainer'

const BacktestPanel = ({ models }) => {
  const [selectedModel, setSelectedModel] = useState('')
  const [episodes, setEpisodes] = useState(20)
  const [running, setRunning] = useState(false)
  const [results, setResults] = useState(null)
  const [messages, setMessages] = useState([])
  const [ws, setWs] = useState(null)

  React.useEffect(() => {
    // Auto-select model using universal default setting
    // Also listen for storage changes (when settings are updated)
    const handleStorageChange = () => {
      if (models.length > 0) {
        const defaultModelPath = getDefaultModel(models, 'trained')
        if (defaultModelPath && defaultModelPath !== selectedModel) {
          setSelectedModel(defaultModelPath)
        }
      }
    }
    
    if (models.length > 0 && !selectedModel) {
      const defaultModelPath = getDefaultModel(models, 'trained')
      if (defaultModelPath) {
        setSelectedModel(defaultModelPath)
      }
    }
    
    // Listen for storage changes (settings updates)
    window.addEventListener('storage', handleStorageChange)
    // Also listen for custom event from same window
    window.addEventListener('defaultModelChanged', handleStorageChange)
    
    return () => {
      window.removeEventListener('storage', handleStorageChange)
      window.removeEventListener('defaultModelChanged', handleStorageChange)
    }
  }, [models, selectedModel])

  React.useEffect(() => {
    let websocket = null
    let isMounted = true
    
    const connectWebSocket = () => {
      if (!isMounted) return
      
      try {
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
          console.log('BacktestPanel WebSocket connected')
        }

        websocket.onmessage = (event) => {
          if (!isMounted) return
          try {
            const data = JSON.parse(event.data)
            if (data.type === 'backtest') {
              setMessages(prev => [...prev, data])
              if (data.status === 'completed') {
                setRunning(false)
                if (data.results) {
                  setResults(data.results)
                }
              } else if (data.status === 'error') {
                setRunning(false)
              }
            }
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
            console.log('BacktestPanel WebSocket closed', event.code)
          }
          websocket = null
        }

        setWs(websocket)
      } catch (error) {
        console.error('Failed to create WebSocket connection:', error)
      }
    }

    connectWebSocket()

    return () => {
      isMounted = false
      if (websocket) {
        // Only close if WebSocket is in a state that can be closed
        if (websocket.readyState === WebSocket.CONNECTING || websocket.readyState === WebSocket.OPEN) {
          try {
            websocket.close(1000, 'Component unmounting')
          } catch (error) {
            // Ignore errors when closing - WebSocket might already be closed
            console.debug('WebSocket close error (ignored):', error)
          }
        }
      }
    }
  }, [])

  const handleRunBacktest = async () => {
    if (!selectedModel) {
      alert('Please select a model')
      return
    }

    setRunning(true)
    setResults(null)
    setMessages([])

    try {
      await axios.post('/api/backtest/run', {
        model_path: selectedModel,
        episodes
      })
    } catch (error) {
      setRunning(false)
      setMessages(prev => [...prev, {
        type: 'backtest',
        status: 'error',
        message: `Failed to start backtest: ${error.message}`
      }])
    }
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-6">Run Backtest</h2>

        <div className="space-y-6">
          {/* Model Selection */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Select Model
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={running}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            >
              <option value="">-- Select a model --</option>
              {(() => {
                const trainedModels = sortModelsWithDefaultFirst(
                  models.filter(m => m.type === 'trained' || !m.type),
                  'trained'
                )
                
                return trainedModels.length > 0 ? (
                  <optgroup label="Trained RL Models">
                    {trainedModels.map((model, idx) => (
                      <option key={idx} value={model.path}>
                        {isDefaultModel(model) ? `⭐ ${model.name} (Default)` : model.name}
                      </option>
                    ))}
                  </optgroup>
                ) : (
                  <optgroup label="Trained RL Models">
                    <option value="" disabled>No trained models available</option>
                  </optgroup>
                )
              })()}
              {(() => {
                const ollamaModels = sortModelsWithDefaultFirst(
                  models.filter(m => m.type === 'ollama'),
                  'ollama'
                )
                
                return ollamaModels.length > 0 && (
                  <optgroup label="Ollama Models (for reasoning)">
                    {ollamaModels.map((model, idx) => (
                      <option key={`ollama-${idx}`} value={model.path} disabled>
                        {isDefaultModel(model) ? `⭐ ${model.name} (Default - not for backtest)` : `${model.name} (Ollama - not for backtest)`}
                      </option>
                    ))}
                  </optgroup>
                )
              })()}
            </select>
            {models.filter(m => m.type === 'ollama').length > 0 && (
              <p className="text-sm text-gray-500 mt-1">
                Note: Ollama models are for reasoning/validation, not for direct backtesting. Select a trained RL model above.
              </p>
            )}
          </div>

          {/* Episodes */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Episodes
            </label>
            <input
              type="number"
              value={episodes}
              onChange={(e) => setEpisodes(parseInt(e.target.value))}
              disabled={running}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              min="1"
              max="100"
            />
            <p className="text-sm text-gray-500 mt-1">
              Number of episodes to run in the backtest
            </p>
          </div>

          {/* Run Button */}
          <button
            onClick={handleRunBacktest}
            disabled={running || !selectedModel}
            className="px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {running ? (
              <>
                <Loader className="w-5 h-5 animate-spin" />
                Running Backtest...
              </>
            ) : (
              <>
                <Play className="w-5 h-5" />
                Run Backtest
              </>
            )}
          </button>
        </div>
      </div>

      {/* Results */}
      {results && (
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5" />
              <h3 className="text-lg font-semibold text-gray-800">Backtest Results</h3>
            </div>
            <CapabilityExplainer
              capabilityId="backtest.results"
              context={{ results, selectedModel }}
            />
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(results).map(([key, value]) => (
              <div key={key} className="p-4 bg-gray-50 rounded-lg">
                <div className="text-sm text-gray-600 mb-1">{key.replace(/_/g, ' ').toUpperCase()}</div>
                <div className="text-2xl font-bold text-gray-800">
                  {typeof value === 'number' ? value.toFixed(2) : String(value)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Messages */}
      {messages.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Backtest Log</h3>
          <div className="bg-gray-50 rounded-lg p-4 max-h-64 overflow-y-auto space-y-2">
            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`
                  p-2 rounded flex items-center gap-2
                  ${msg.status === 'error' ? 'bg-red-50 text-red-700' : ''}
                  ${msg.status === 'completed' ? 'bg-green-50 text-green-700' : ''}
                  ${msg.status === 'running' ? 'bg-blue-50 text-blue-700' : ''}
                `}
              >
                {msg.status === 'running' && <Loader className="w-4 h-4 animate-spin" />}
                {msg.status === 'completed' && <CheckCircle className="w-4 h-4" />}
                {msg.status === 'error' && <AlertCircle className="w-4 h-4" />}
                <span>{msg.message}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default BacktestPanel

