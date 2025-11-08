import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { Play, Square, Loader, CheckCircle, AlertCircle, Server } from 'lucide-react'
import { getDefaultModel, sortModelsWithDefaultFirst, isDefaultModel } from '../utils/modelUtils'
import MonteCarloRiskPanel from './MonteCarloRiskPanel'
import VolatilityPanel from './VolatilityPanel'

const TradingPanel = ({ models }) => {
  const [selectedModel, setSelectedModel] = useState('')
  const [paperTrading, setPaperTrading] = useState(true)
  const [bridgeRunning, setBridgeRunning] = useState(false)
  const [bridgeStarting, setBridgeStarting] = useState(false)
  const [trading, setTrading] = useState(false)
  const [tradingStatus, setTradingStatus] = useState(null)
  const [messages, setMessages] = useState([])
  const [ws, setWs] = useState(null)

  useEffect(() => {
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

  useEffect(() => {
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
          console.log('TradingPanel WebSocket connected')
        }

        websocket.onmessage = (event) => {
          if (!isMounted) return
          try {
            const data = JSON.parse(event.data)
            if (data.type === 'bridge' || data.type === 'trading') {
              setMessages(prev => [...prev, data])
              if (data.type === 'bridge' && data.status === 'running') {
                setBridgeRunning(true)
              }
              if (data.type === 'trading') {
                if (data.status === 'running') {
                  setTrading(true)
                } else if (data.status === 'stopped' || data.status === 'error') {
                  setTrading(false)
                }
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
            console.log('TradingPanel WebSocket closed', event.code)
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

  useEffect(() => {
    const checkStatus = async () => {
      try {
        // Check bridge status
        const bridgeResponse = await axios.get('/api/trading/bridge-status')
        setBridgeRunning(bridgeResponse.data.running || false)
        
        // Check trading status
        const tradingResponse = await axios.get('/api/trading/status')
        setTradingStatus(tradingResponse.data)
        setTrading(tradingResponse.data.status === 'running')
      } catch (error) {
        // Ignore errors for now
        console.debug('Status check error:', error)
      }
    }

    checkStatus() // Check immediately
    const interval = setInterval(checkStatus, 2000)
    return () => clearInterval(interval)
  }, [])

  const handleStartBridge = async () => {
    if (bridgeStarting || bridgeRunning) {
      return // Prevent duplicate requests
    }
    
    setBridgeStarting(true)
    setMessages(prev => [...prev, {
      type: 'bridge',
      status: 'starting',
      message: 'Starting bridge server...'
    }])
    
    try {
      const response = await axios.post('/api/trading/start-bridge')
      // Status will be updated by WebSocket or next status check
    } catch (error) {
      setBridgeStarting(false)
      const errorMessage = error.response?.data?.detail || error.message || 'Unknown error'
      setMessages(prev => [...prev, {
        type: 'bridge',
        status: 'error',
        message: `Failed to start bridge: ${errorMessage}`
      }])
    } finally {
      // Reset starting state after a delay
      setTimeout(() => setBridgeStarting(false), 2000)
    }
  }
  
  const handleStopBridge = async () => {
    try {
      await axios.post('/api/trading/stop-bridge')
      setBridgeRunning(false)
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message || 'Unknown error'
      setMessages(prev => [...prev, {
        type: 'bridge',
        status: 'error',
        message: `Failed to stop bridge: ${errorMessage}`
      }])
    }
  }

  const handleStartTrading = async () => {
    if (!selectedModel) {
      alert('Please select a model')
      return
    }

    if (!bridgeRunning) {
      alert('Please start the NT8 bridge server first')
      return
    }

    setTrading(true)
    setMessages([])

    try {
      await axios.post('/api/trading/start', {
        model_path: selectedModel,
        paper_trading: paperTrading
      })
    } catch (error) {
      setTrading(false)
      setMessages(prev => [...prev, {
        type: 'trading',
        status: 'error',
        message: `Failed to start trading: ${error.message}`
      }])
    }
  }

  const handleStopTrading = async () => {
    try {
      await axios.post('/api/trading/stop')
      setTrading(false)
    } catch (error) {
      console.error('Failed to stop trading:', error)
    }
  }

  return (
    <div className="space-y-6">
      {/* Bridge Server */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-6">NT8 Bridge Server</h2>
        
        <div className="space-y-4">
          <div className={`
            p-4 rounded-lg flex items-center justify-between
            ${bridgeRunning ? 'bg-green-50 text-green-700' : 'bg-gray-50 text-gray-600'}
          `}>
            <div className="flex items-center gap-3">
              <Server className="w-5 h-5" />
              <span className="font-semibold">
                Bridge Server {bridgeRunning ? 'Running' : 'Stopped'}
              </span>
            </div>
            {!bridgeRunning && (
              <button
                onClick={handleStartBridge}
                disabled={bridgeStarting}
                className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {bridgeStarting ? (
                  <>
                    <Loader className="w-4 h-4 animate-spin" />
                    Starting...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" />
                    Start Bridge
                  </>
                )}
              </button>
            )}
            {bridgeRunning && (
              <button
                onClick={handleStopBridge}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 flex items-center gap-2"
              >
                <Square className="w-4 h-4" />
                Stop Bridge
              </button>
            )}
          </div>
          
          <p className="text-sm text-gray-600">
            The bridge server connects your Python trading system to NinjaTrader 8. 
            Make sure NT8 is running and the strategy is loaded before starting trading.
          </p>
        </div>
      </div>

      {/* Trading Configuration */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-6">Start Trading</h2>

        <div className="space-y-6">
          {/* Model Selection */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Select Model
            </label>
            <select
              value={selectedModel}
              onChange={(e) => {
                // Only allow selecting non-disabled options
                if (e.target.value) {
                  const selectedModelObj = models.find(m => m.path === e.target.value)
                  // Only set if it's a trained model (not Ollama)
                  if (selectedModelObj && (selectedModelObj.type === 'trained' || !selectedModelObj.type)) {
                    setSelectedModel(e.target.value)
                  }
                } else {
                  setSelectedModel('')
                }
              }}
              disabled={trading}
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
                  <optgroup label="Ollama Models (for reasoning only)">
                    {ollamaModels.map((model, idx) => (
                      <option key={`ollama-${idx}`} value={model.path} disabled>
                        {isDefaultModel(model) ? `⭐ ${model.name} (Default)` : model.name}
                      </option>
                    ))}
                  </optgroup>
                )
              })()}
            </select>
            {models.filter(m => m.type === 'trained' || !m.type).length === 0 ? (
              <div className="mt-2 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                <p className="text-sm text-yellow-800">
                  <strong>No trained RL models available.</strong> Train a model in the <strong>Training</strong> tab before starting trading.
                </p>
              </div>
            ) : models.filter(m => m.type === 'ollama').length > 0 && (
              <p className="text-sm text-gray-500 mt-1">
                Ollama models are used for reasoning validation. Select a trained RL model for trading.
              </p>
            )}
          </div>

          {/* Trading Mode */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Trading Mode
            </label>
            <div className="flex gap-4">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  checked={paperTrading}
                  onChange={() => setPaperTrading(true)}
                  disabled={trading}
                  className="w-4 h-4"
                />
                <span>Paper Trading (Recommended for testing)</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  checked={!paperTrading}
                  onChange={() => setPaperTrading(false)}
                  disabled={trading}
                  className="w-4 h-4"
                />
                <span className="text-red-600 font-semibold">Live Trading (Real Money)</span>
              </label>
            </div>
          </div>

          {/* Status */}
          {tradingStatus && (
            <div className={`
              p-4 rounded-lg flex items-center gap-3
              ${tradingStatus.status === 'running' ? 'bg-green-50 text-green-700' : 'bg-gray-50 text-gray-600'}
            `}>
              {tradingStatus.status === 'running' && <CheckCircle className="w-5 h-5" />}
              <span>{tradingStatus.message}</span>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-4">
            {!trading ? (
              <button
                onClick={handleStartTrading}
                disabled={!selectedModel || !bridgeRunning}
                className="px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                <Play className="w-5 h-5" />
                Start Trading
              </button>
            ) : (
              <button
                onClick={handleStopTrading}
                className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 flex items-center gap-2"
              >
                <Square className="w-5 h-5" />
                Stop Trading
              </button>
            )}
          </div>

          {!paperTrading && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
              <div className="flex items-start gap-2">
                <AlertCircle className="w-5 h-5 text-red-600 mt-0.5" />
                <div>
                  <div className="font-semibold text-red-800 mb-1">Warning: Live Trading</div>
                  <div className="text-sm text-red-700">
                    You are about to trade with real money. Make sure you have:
                  </div>
                  <ul className="list-disc list-inside text-sm text-red-700 mt-2 space-y-1">
                    <li>Thoroughly tested your model in paper trading</li>
                    <li>Set appropriate risk limits</li>
                    <li>Reviewed all trading parameters</li>
                    <li>Have sufficient capital for potential losses</li>
                  </ul>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Volatility Prediction & Adaptive Risk Management */}
      <VolatilityPanel />

      {/* Monte Carlo Risk Assessment */}
      <MonteCarloRiskPanel
        currentPrice={5000} // Example price - in production, this would come from market data
        proposedPosition={0.5} // Example position - in production, this would come from RL agent
        currentPosition={0.0}
      />

      {/* Trading Log */}
      {messages.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Trading Log</h3>
          <div className="bg-gray-50 rounded-lg p-4 max-h-96 overflow-y-auto space-y-2 font-mono text-sm">
            {messages.slice(-20).map((msg, idx) => (
              <div
                key={idx}
                className={`
                  p-2 rounded
                  ${msg.status === 'error' ? 'bg-red-50 text-red-700' : ''}
                  ${msg.status === 'running' ? 'bg-green-50 text-green-700' : ''}
                  ${msg.status === 'stopped' ? 'bg-gray-50 text-gray-700' : ''}
                `}
              >
                [{msg.type}] {msg.message}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default TradingPanel

