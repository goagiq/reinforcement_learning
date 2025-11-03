import React, { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import { Play, Square, Loader, CheckCircle, AlertCircle, Cpu, RefreshCw } from 'lucide-react'
import { getDefaultModel, sortModelsWithDefaultFirst, isDefaultModel } from '../utils/modelUtils'

// Animated number component that highlights when value changes
const AnimatedNumber = ({ value, format = (v) => v, className = '', colorClass = '' }) => {
  const [displayValue, setDisplayValue] = useState(value)
  const [isAnimating, setIsAnimating] = useState(false)
  const prevValueRef = useRef(value)

  // ALWAYS update displayValue when value prop changes - no conditions
  useEffect(() => {
    console.log(`[AnimatedNumber] Value prop changed: ${prevValueRef.current} -> ${value}`)
    if (value !== undefined && value !== null) {
      setDisplayValue(value)
      prevValueRef.current = value
    }
  }, [value])

  // Animate if value changed (for numeric values, use tolerance for floating point)
  useEffect(() => {
    if (prevValueRef.current === undefined || value === undefined) {
      return
    }

    const hasChanged = typeof value === 'number' && typeof prevValueRef.current === 'number'
      ? Math.abs(prevValueRef.current - value) > 0.0001  // Tolerance for floating point
      : prevValueRef.current !== value
    
    if (hasChanged) {
      setIsAnimating(true)
      const timer = setTimeout(() => setIsAnimating(false), 600)
      return () => clearTimeout(timer)
    }
  }, [value])

  // Debug: Log current display value
  useEffect(() => {
    console.log(`[AnimatedNumber] displayValue updated to: ${displayValue}`)
  }, [displayValue])

  return (
    <div
      className={`text-2xl font-bold transition-all duration-300 ${
        isAnimating ? 'scale-110 bg-blue-100 rounded px-2 py-1' : ''
      } ${colorClass} ${className}`}
    >
      {format(displayValue)}
    </div>
  )
}

// Animated metric component for smaller metrics
const AnimatedMetric = ({ value, format = (v) => v, className = '' }) => {
  const [displayValue, setDisplayValue] = useState(value)
  const [isAnimating, setIsAnimating] = useState(false)
  const prevValueRef = useRef(value)

  // ALWAYS update displayValue when value prop changes - no conditions
  useEffect(() => {
    if (value !== undefined && value !== null) {
      setDisplayValue(value)
      prevValueRef.current = value
    }
  }, [value])

  useEffect(() => {
    // Animate if value changed (for numeric values, use tolerance for floating point)
    if (prevValueRef.current === undefined || value === undefined) {
      prevValueRef.current = value
      return
    }

    const hasChanged = typeof value === 'number' && typeof prevValueRef.current === 'number'
      ? Math.abs(prevValueRef.current - value) > 0.0001  // Tolerance for floating point
      : prevValueRef.current !== value
    
    if (hasChanged) {
      setIsAnimating(true)
      const timer = setTimeout(() => setIsAnimating(false), 400)
      prevValueRef.current = value
      return () => clearTimeout(timer)
    } else {
      prevValueRef.current = value
    }
  }, [value])

  return (
    <div
      className={`text-lg font-semibold transition-all duration-300 ${
        isAnimating ? 'scale-105 text-blue-600' : 'text-gray-800'
      } ${className}`}
    >
      {format(displayValue)}
    </div>
  )
}

const TrainingPanel = ({ models = [], onModelsChange }) => {
  const [training, setTraining] = useState(false)
  const [trainingStatus, setTrainingStatus] = useState(null)
  const [trainingMetrics, setTrainingMetrics] = useState(null)
  const [renderKey, setRenderKey] = useState(0) // Force re-render key
  const [, forceUpdate] = useState(0) // Additional force update mechanism
  const onModelsChangeRef = useRef(onModelsChange)
  
  // Use refs to store interval IDs so they persist across re-renders
  const pollingIntervalRef = useRef(null)
  const watchdogIntervalRef = useRef(null)
  const prevMetricsRef = useRef(null) // Store previous metrics for comparison
  const lastPollTimeRef = useRef(Date.now()) // Store last poll time so refresh can update it
  
  // Keep ref updated without triggering re-renders
  useEffect(() => {
    onModelsChangeRef.current = onModelsChange
  }, [onModelsChange])
  
  // Keep metrics ref in sync with state
  useEffect(() => {
    prevMetricsRef.current = trainingMetrics
  }, [trainingMetrics])
  
  // Debug: Log whenever trainingMetrics state changes (to verify React detects it)
  useEffect(() => {
    if (trainingMetrics) {
      console.log(`[TrainingPanel] trainingMetrics state changed - Timestep: ${trainingMetrics.timestep}, Reward: ${trainingMetrics.latest_reward?.toFixed(2)}, Episode: ${trainingMetrics.episode}`)
    }
  }, [trainingMetrics])
  const [device, setDevice] = useState('cpu')
  const [cudaAvailable, setCudaAvailable] = useState(false)
  const [gpuInfo, setGpuInfo] = useState(null)
  const [totalTimesteps, setTotalTimesteps] = useState(1000000)
  const [reasoningModel, setReasoningModel] = useState('')
  const [configPath, setConfigPath] = useState('configs/train_config.yaml')
  const [availableConfigs, setAvailableConfigs] = useState([])
  const [checkpoints, setCheckpoints] = useState([])
  const [selectedCheckpoint, setSelectedCheckpoint] = useState('latest') // 'latest', 'none', or checkpoint path
  const [latestCheckpoint, setLatestCheckpoint] = useState(null)
  const [messages, setMessages] = useState([])
  const [ws, setWs] = useState(null)
  
  // Default reasoning model using universal default setting
  useEffect(() => {
    // Also listen for storage changes (when settings are updated)
    const handleStorageChange = () => {
      if (models.length > 0) {
        const defaultModelPath = getDefaultModel(models, 'ollama')
        if (defaultModelPath && defaultModelPath !== reasoningModel) {
          setReasoningModel(defaultModelPath)
        }
      }
    }
    
    if (models.length > 0 && !reasoningModel) {
      const defaultModelPath = getDefaultModel(models, 'ollama')
      if (defaultModelPath) {
        setReasoningModel(defaultModelPath)
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
  }, [models, reasoningModel])

  useEffect(() => {
    let websocket = null
    let isMounted = true
    let connectionTimeout = null
    
    const connectWebSocket = () => {
      if (!isMounted) return
      
      try {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
        const wsUrl = `${wsProtocol}//${window.location.host}/ws`
        
        // Small delay to avoid React StrictMode immediate cleanup issues
        connectionTimeout = setTimeout(() => {
          if (!isMounted) return
          
          try {
            // Suppress browser console warnings for WebSocket during StrictMode
            const originalError = console.error
            const errorFilter = (...args) => {
              const message = args.join(' ')
              // Suppress StrictMode-related WebSocket warnings in development
              if (message.includes('WebSocket is closed before the connection is established') ||
                  (message.includes('WebSocket connection to') && message.includes('failed') && 
                   message.includes('localhost:3200/ws'))) {
                // These are benign StrictMode warnings in development
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
              console.log('TrainingPanel WebSocket connected')
            }

            websocket.onmessage = (event) => {
              if (!isMounted) return
              try {
                const data = JSON.parse(event.data)
                if (data.type === 'training') {
                  setMessages(prev => [...prev, data])
                  if (data.status === 'completed' || data.status === 'error' || data.status === 'stopped') {
                    setTraining(false)
                    setTrainingStatus({ status: data.status, message: data.message })
                    // Refresh model list when training completes
                    if (data.status === 'completed') {
                      onModelsChange()
                    }
                  } else if (data.status === 'running' || data.status === 'starting') {
                    setTraining(true)
                    setTrainingStatus({ status: data.status, message: data.message })
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
                console.log('TrainingPanel WebSocket closed', event.code)
              }
              websocket = null
            }

            setWs(websocket)
          } catch (error) {
            if (isMounted) {
              console.error('Failed to create WebSocket connection:', error)
            }
          }
        }, 50)
      } catch (error) {
        if (isMounted) {
          console.error('Failed to setup WebSocket connection:', error)
        }
      }
    }

    connectWebSocket()

    return () => {
      isMounted = false
      if (connectionTimeout) {
        clearTimeout(connectionTimeout)
      }
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
  }, [onModelsChange])

  // Load available config files on mount
  useEffect(() => {
    const loadConfigs = async () => {
      try {
        const response = await axios.get('/api/config/list')
        const configs = response.data.configs || []
        setAvailableConfigs(configs)
        
        // Auto-select GPU-optimized config if available
        const optimizedConfig = configs.find(c => c.name.includes('gpu_optimized') || c.name.includes('optimized'))
        if (optimizedConfig) {
          setConfigPath(optimizedConfig.relative)
        } else if (configs.length > 0) {
          // Fall back to first available config
          setConfigPath(configs[0].relative)
        }
      } catch (error) {
        console.error('Failed to load configs:', error)
      }
    }
    
    loadConfigs()
  }, [])

  // Load available checkpoints on mount and refresh periodically
  useEffect(() => {
    const loadCheckpoints = async () => {
      try {
        const response = await axios.get('/api/models/list')
        const checkpointsList = response.data.checkpoints || []
        const latest = response.data.latest_checkpoint
        
        setCheckpoints(checkpointsList)
        setLatestCheckpoint(latest)
        
        // Auto-select latest checkpoint only on first load if none is selected yet
        // Don't change user's explicit selection
      } catch (error) {
        console.error('Failed to load checkpoints:', error)
      }
    }
    
    loadCheckpoints()
    // Refresh checkpoints every 10 seconds (in case new ones are created)
    const interval = setInterval(loadCheckpoints, 10000)
    return () => clearInterval(interval)
  }, []) // Only run on mount, not when selectedCheckpoint changes

  // Check CUDA availability on mount and auto-select if available
  useEffect(() => {
    const checkCudaAvailability = async () => {
      try {
        const response = await axios.get('/api/system/cuda-status')
        const { cuda_available, device: recommendedDevice, gpu_name, cuda_version } = response.data
        
        console.log('CUDA Status Check:', {
          cuda_available,
          recommendedDevice,
          gpu_name,
          cuda_version,
          currentDevice: device
        })
        
        setCudaAvailable(cuda_available)
        if (cuda_available && gpu_name) {
          setGpuInfo({
            name: gpu_name,
            version: cuda_version
          })
        }
        
        // Auto-select CUDA if available
        if (cuda_available === true) {
          console.log('✅ CUDA is available - auto-selecting GPU')
          setDevice('cuda')
          // Force a state update by logging
          console.log(`Device will be set to: cuda (was: ${device})`)
        } else {
          console.log('❌ CUDA not available - keeping device as cpu')
          console.log('Response data:', response.data)
        }
      } catch (error) {
        console.error('Failed to check CUDA status:', error)
        // Default to CPU if check fails
        setDevice('cpu')
      }
    }
    
    checkCudaAvailability()
  }, []) // Only run once on mount
  
        useEffect(() => {
      let isMounted = true
      let pollCount = 0
      let lastPollTime = Date.now()
      lastPollTimeRef.current = lastPollTime
      
      // Clear any existing intervals first
      if (pollingIntervalRef.current) {
        console.log('[TrainingPanel] Clearing existing polling interval')
        clearInterval(pollingIntervalRef.current)
        pollingIntervalRef.current = null
      }
      if (watchdogIntervalRef.current) {
        clearInterval(watchdogIntervalRef.current)
        watchdogIntervalRef.current = null
      }
      
      const checkStatus = async () => {
        if (!isMounted) {
          console.log('[TrainingPanel] Skipping checkStatus - component unmounted')
          return
        }
        
        pollCount++
        const pollId = pollCount
        
        try {
          // Add a timestamp to help debug if polling stops
          // Log every 15 polls (every 30 seconds) and also log first few polls for debugging
          if (pollCount % 15 === 0 || pollCount <= 5) {
            console.log(`[TrainingPanel] Polling check #${pollId} at ${new Date().toLocaleTimeString()}, interval ID: ${pollingIntervalRef.current}`)
          }
          
          // Retry logic for failed requests
          let response = null
          let retries = 0
          const maxRetries = 2
          
          while (retries <= maxRetries && isMounted) {
            try {
              response = await axios.get('/api/training/status', { 
                timeout: 5000,
                validateStatus: (status) => status < 500 // Don't throw on 4xx, only 5xx
              })
              break // Success, exit retry loop
            } catch (retryError) {
              retries++
              if (retries <= maxRetries && isMounted) {
                // Wait a bit before retry (exponential backoff)
                await new Promise(resolve => setTimeout(resolve, 500 * retries))
                continue
              } else {
                throw retryError // Give up after max retries
              }
            }
          }
          
          if (!response || !isMounted) return
          
          if (!isMounted) {
            console.log(`[TrainingPanel] Component unmounted during request #${pollId}`)
            return
          }
          
          const status = response.data.status
          setTrainingStatus(response.data)
          
          // Update training metrics if available
          // Only update if metrics actually contain data (not empty object during initialization)
          if (response.data.metrics && Object.keys(response.data.metrics).length > 0) {
            // Deep clone to ensure new reference - use JSON parse/stringify for complete deep copy
            const newMetrics = JSON.parse(JSON.stringify(response.data.metrics))
            
            // Check if values actually changed to avoid unnecessary re-renders
            const prevMetrics = prevMetricsRef.current
            const valuesChanged = !prevMetrics || 
                prevMetrics.timestep !== newMetrics.timestep ||
                Math.abs((prevMetrics.latest_reward || 0) - (newMetrics.latest_reward || 0)) > 0.001 ||
                prevMetrics.episode !== newMetrics.episode ||
                prevMetrics.current_episode_length !== newMetrics.current_episode_length
            
            // Only update state if values changed OR force update every 10 polls to show activity
            if (valuesChanged || pollCount % 10 === 0) {
              // Always update state when values change
              setTrainingMetrics(newMetrics)
              prevMetricsRef.current = newMetrics
              
              // ALWAYS increment render key to force re-render (even if values same, key changes)
              setRenderKey(prev => {
                const next = prev + 1
                // Debug: Log when metrics update
                if (pollCount % 15 === 0 || pollCount <= 5 || valuesChanged) {
                  console.log(`[TrainingPanel] State updated: Timestep=${newMetrics.timestep}, Reward=${newMetrics.latest_reward?.toFixed(2)}, RenderKey=${next}, ValuesChanged=${valuesChanged}`)
                }
                return next
              })
              
              // Also force update counter to trigger re-render
              forceUpdate(prev => prev + 1)
            } else {
              // Values haven't changed, but still update renderKey occasionally to show activity
              // This ensures UI stays responsive even when backend returns same values
              if (pollCount % 30 === 0) {
                setRenderKey(prev => prev + 1)
              }
            }
            
            lastPollTime = Date.now() // Update on successful response
            lastPollTimeRef.current = lastPollTime
          }
          
          // Update training state based on status
          if (status === 'running' || status === 'starting') {
            setTraining(true)
          } else if (status === 'completed' || status === 'error' || status === 'idle') {
            setTraining(false)
            
            // Add status message to log if not already present
            if (status === 'completed') {
              setMessages(prev => {
                // Check if completion message already exists
                const hasCompletion = prev.some(m => m.type === 'training' && m.status === 'completed')
                if (!hasCompletion) {
                  return [...prev, {
                    type: 'training',
                    status: 'completed',
                    message: response.data.message || 'Training completed successfully'
                  }]
                }
                return prev
              })
              // Only call onModelsChange if component is still mounted
              if (isMounted && onModelsChangeRef.current) {
                onModelsChangeRef.current()
              }
            } else if (status === 'error') {
              setMessages(prev => {
                const hasError = prev.some(m => m.type === 'training' && m.status === 'error' && m.message === response.data.message)
                if (!hasError) {
                  return [...prev, {
                    type: 'training',
                    status: 'error',
                    message: response.data.message || 'Training failed'
                  }]
                }
                return prev
              })
            }
          }
        } catch (error) {
          // Update last poll time even on error so watchdog knows we're still trying
          lastPollTime = Date.now()
          
          // Don't stop polling on error - just log it
          if (isMounted) {
            const errorMsg = error.response 
              ? `HTTP ${error.response.status}: ${error.response.statusText}` 
              : error.message || 'Unknown error'
            console.error(`[TrainingPanel] Failed to check training status (poll #${pollId}):`, errorMsg)
            if (error.code) {
              console.error(`[TrainingPanel] Error code:`, error.code)
            }
            // Still try to update with last known status if available
            // Don't clear metrics - keep showing last known values
          }
        }
      }

      console.log('[TrainingPanel] Setting up polling interval')
      
      // Watchdog to detect if polling stops
      const checkPollingHealth = () => {
        if (!isMounted) return
        
        const timeSinceLastPoll = Date.now() - lastPollTime
        if (timeSinceLastPoll > 10000) { // More than 10 seconds since last poll
          console.warn(`[TrainingPanel] Polling appears to have stopped! Last poll was ${Math.round(timeSinceLastPoll/1000)}s ago. Restarting...`)
          
          // Try to restart polling
          if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current)
            pollingIntervalRef.current = null
          }
          
          // Restart interval
          pollingIntervalRef.current = setInterval(() => {
            if (isMounted) {
              lastPollTime = Date.now()
              checkStatus()
            }
          }, 2000)
          
          console.log('[TrainingPanel] Polling restarted, new interval ID:', pollingIntervalRef.current)
        }
      }
      
      // Check immediately and then every 2 seconds (more frequent for real-time metrics)
      checkStatus()
      lastPollTime = Date.now()
      lastPollTimeRef.current = lastPollTime
      
          pollingIntervalRef.current = setInterval(() => {
            if (isMounted) {
              lastPollTime = Date.now()
              lastPollTimeRef.current = lastPollTime
              checkStatus()
            } else {
          console.log('[TrainingPanel] Interval callback called but component unmounted')
          if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current)
            pollingIntervalRef.current = null
          }
        }
      }, 2000)
      
      // Watchdog checks every 5 seconds
      watchdogIntervalRef.current = setInterval(checkPollingHealth, 5000)
      
      // Verify interval was set
      if (!pollingIntervalRef.current) {
        console.error('[TrainingPanel] Failed to create interval!')
      } else {
        console.log('[TrainingPanel] Polling interval created, ID:', pollingIntervalRef.current)
      }
      
      return () => {
        console.log('[TrainingPanel] Cleaning up polling interval')
        isMounted = false
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current)
          pollingIntervalRef.current = null
          console.log('[TrainingPanel] Polling interval cleared')
        }
        if (watchdogIntervalRef.current) {
          clearInterval(watchdogIntervalRef.current)
          watchdogIntervalRef.current = null
          console.log('[TrainingPanel] Watchdog cleared')
        }
      }
    }, []) // Empty deps - only run once on mount

  const handleStartTraining = async () => {
    console.log('[TrainingPanel] handleStartTraining called')
    setTraining(true)
    setMessages([])
    
    try {
      const requestData = {
        device,
        total_timesteps: totalTimesteps,
        config_path: configPath
      }
      
      // Include reasoning model if selected
      if (reasoningModel) {
        requestData.reasoning_model = reasoningModel
      }
      
      // Include checkpoint_path if a checkpoint is selected
      if (selectedCheckpoint === 'latest' && latestCheckpoint) {
        requestData.checkpoint_path = latestCheckpoint.path
        console.log('📂 Sending checkpoint_path (latest):', latestCheckpoint.path)
      } else if (selectedCheckpoint !== 'none' && selectedCheckpoint !== 'latest') {
        // A specific checkpoint was selected
        requestData.checkpoint_path = selectedCheckpoint
        console.log('📂 Sending checkpoint_path (specific):', selectedCheckpoint)
      } else {
        console.log('📂 No checkpoint selected - starting fresh training')
      }
      
      console.log('📤 Training request data:', JSON.stringify(requestData, null, 2))
      console.log('📡 Sending POST request to /api/training/start...')
      
      const response = await axios.post('/api/training/start', requestData)
      console.log('✅ Training start request successful:', response.data)
      
    } catch (error) {
      console.error('❌ Failed to start training:', error)
      console.error('   Error details:', {
        message: error.message,
        response: error.response?.data,
        status: error.response?.status,
        statusText: error.response?.statusText,
        code: error.code
      })
      
      setTraining(false)
      const errorMessage = error.response?.data?.detail || error.response?.data?.message || error.message || 'Unknown error'
      setMessages(prev => [...prev, {
        type: 'training',
        status: 'error',
        message: `Failed to start training: ${errorMessage}`
      }])
      
      // Also update training status
      setTrainingStatus({
        status: 'error',
        message: `Failed to start training: ${errorMessage}`
      })
    }
  }

  const handleStopTraining = async () => {
    try {
      await axios.post('/api/training/stop')
      setTraining(false)
    } catch (error) {
      console.error('Failed to stop training:', error)
    }
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-6">Train RL Model</h2>

        <div className="space-y-6">
          {/* Config File Selection */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Configuration File
            </label>
            <select
              value={configPath}
              onChange={(e) => setConfigPath(e.target.value)}
              disabled={training}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            >
              {availableConfigs.length > 0 ? (
                availableConfigs.map((config, idx) => (
                  <option key={idx} value={config.relative}>
                    {config.name}
                    {config.name.includes('gpu_optimized') || config.name.includes('optimized') ? ' ⚡ (Optimized for GPU)' : ''}
                  </option>
                ))
              ) : (
                <option value="configs/train_config.yaml">configs/train_config.yaml (Default)</option>
              )}
            </select>
            <p className="text-sm text-gray-500 mt-1">
              Select a training configuration. The GPU-optimized config provides 2-3x faster training.
            </p>
          </div>

          {/* Device Selection */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Device
            </label>
            <div className="flex gap-4">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="device"
                  value="cpu"
                  checked={device === 'cpu'}
                  onChange={(e) => setDevice(e.target.value)}
                  disabled={training}
                  className="w-4 h-4"
                />
                <Cpu className="w-5 h-5" />
                <span>CPU</span>
              </label>
              <label className={`flex items-center gap-2 cursor-pointer ${!cudaAvailable ? 'opacity-50' : ''}`}>
                <input
                  type="radio"
                  name="device"
                  value="cuda"
                  checked={device === 'cuda'}
                  onChange={(e) => setDevice(e.target.value)}
                  disabled={training || !cudaAvailable}
                  className="w-4 h-4"
                />
                <Cpu className="w-5 h-5" />
                <span>CUDA (GPU)</span>
                {cudaAvailable && gpuInfo && (
                  <span className="text-xs text-green-600 ml-1 font-semibold">
                    ({gpuInfo.name})
                  </span>
                )}
              </label>
            </div>
            {!cudaAvailable && (
              <p className="text-sm text-gray-500 mt-2">
                GPU not available. Install PyTorch with CUDA support to enable GPU training.
              </p>
            )}
            {cudaAvailable && device === 'cuda' && gpuInfo && (
              <p className="text-sm text-green-600 mt-2 flex items-center gap-1">
                <CheckCircle className="w-4 h-4" />
                Using GPU: {gpuInfo.name} (CUDA {gpuInfo.version})
              </p>
            )}
          </div>

          {/* Total Timesteps */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Total Timesteps
            </label>
            <input
              type="number"
              value={totalTimesteps}
              onChange={(e) => setTotalTimesteps(parseInt(e.target.value))}
              disabled={training}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              min="10000"
              step="10000"
            />
            <p className="text-sm text-gray-500 mt-1">
              More timesteps = longer training time but potentially better results
            </p>
          </div>

          {/* Reasoning Model Selection */}
          {models.filter(m => m.type === 'ollama').length > 0 && (
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Reasoning Model (for validation)
              </label>
              <select
                value={reasoningModel}
                onChange={(e) => setReasoningModel(e.target.value)}
                disabled={training}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              >
                <option value="">-- Select reasoning model --</option>
                {sortModelsWithDefaultFirst(
                  models.filter(m => m.type === 'ollama'),
                  'ollama'
                ).map((model, idx) => (
                  <option key={`ollama-${idx}`} value={model.path}>
                    {isDefaultModel(model) ? `⭐ ${model.name} (Default)` : model.name}
                  </option>
                ))}
              </select>
              <p className="text-sm text-gray-500 mt-1">
                DeepSeek-R1 models are recommended for reasoning validation during training
              </p>
            </div>
          )}

          {/* Checkpoint Selection */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Resume from Checkpoint
            </label>
            <select
              value={selectedCheckpoint}
              onChange={(e) => setSelectedCheckpoint(e.target.value)}
              disabled={training}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            >
              <option value="latest">
                {latestCheckpoint 
                  ? `Latest: checkpoint_${latestCheckpoint.timestep?.toLocaleString() || 'N/A'}.pt (${latestCheckpoint.timestep ? `${(latestCheckpoint.timestep / 1000).toFixed(0)}k timesteps` : 'Unknown'})`
                  : 'Latest (No checkpoints found)'}
              </option>
              <option value="none">Start Fresh Training</option>
              {checkpoints.length > 0 && (
                <optgroup label="Select Specific Checkpoint:">
                  {checkpoints.map((checkpoint, idx) => (
                    <option key={idx} value={checkpoint.path}>
                      checkpoint_{checkpoint.timestep?.toLocaleString() || 'N/A'}.pt ({checkpoint.timestep ? `${(checkpoint.timestep / 1000).toFixed(0)}k timesteps` : 'Unknown'})
                    </option>
                  ))}
                </optgroup>
              )}
            </select>
            <p className="text-sm text-gray-500 mt-1">
              {selectedCheckpoint === 'latest' && latestCheckpoint ? (
                <span className="text-blue-600 font-medium">
                  ✓ Will resume from latest checkpoint ({latestCheckpoint.timestep ? `${(latestCheckpoint.timestep / 1000).toFixed(0)}k timesteps` : 'Unknown timesteps'})
                </span>
              ) : selectedCheckpoint === 'none' ? (
                <span className="text-orange-600 font-medium">
                  ⚠ Will start fresh training from timestep 0
                </span>
              ) : (
                'Select a checkpoint to resume training from that point, or choose "Start Fresh Training" to begin from scratch'
              )}
            </p>
          </div>

          {/* Status */}
          {trainingStatus && (
            <div className={`
              p-4 rounded-lg flex items-center gap-3
              ${trainingStatus.status === 'running' ? 'bg-blue-50 text-blue-700' : ''}
              ${trainingStatus.status === 'completed' ? 'bg-green-50 text-green-700' : ''}
              ${trainingStatus.status === 'error' ? 'bg-red-50 text-red-700' : ''}
            `}>
              {trainingStatus.status === 'running' && <Loader className="w-5 h-5 animate-spin" />}
              {trainingStatus.status === 'completed' && <CheckCircle className="w-5 h-5" />}
              {trainingStatus.status === 'error' && <AlertCircle className="w-5 h-5" />}
              <span>{trainingStatus.message}</span>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-4">
            {!training ? (
              <button
                onClick={handleStartTraining}
                className="px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 flex items-center gap-2"
              >
                <Play className="w-5 h-5" />
                {selectedCheckpoint === 'latest' && latestCheckpoint 
                  ? `Resume Training (from ${latestCheckpoint.timestep ? `${(latestCheckpoint.timestep / 1000).toFixed(0)}k` : 'latest'} timesteps)`
                  : selectedCheckpoint !== 'none'
                  ? 'Resume Training'
                  : 'Start Training'}
              </button>
            ) : (
              <button
                onClick={handleStopTraining}
                className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 flex items-center gap-2"
              >
                <Square className="w-5 h-5" />
                Stop Training
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Training Log */}
      {messages.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Training Log</h3>
          <div className="bg-gray-50 rounded-lg p-4 max-h-96 overflow-y-auto space-y-2 font-mono text-sm">
            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`
                  p-2 rounded
                  ${msg.status === 'error' ? 'bg-red-50 text-red-700' : ''}
                  ${msg.status === 'completed' ? 'bg-green-50 text-green-700' : ''}
                  ${msg.status === 'running' ? 'bg-blue-50 text-blue-700' : ''}
                `}
              >
                {msg.message}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Detailed Training Status */}
      {training && trainingMetrics && (
        <div key={`training-metrics-${renderKey}`} className="bg-white rounded-lg shadow p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-gray-800">Training Progress</h3>
            <button
              onClick={async (e) => {
                e.preventDefault()
                e.stopPropagation()
                console.log('[TrainingPanel] Refresh button clicked')
                try {
                  // Force a fresh metrics fetch - independent of polling
                  const response = await axios.get('/api/training/status', {
                    timeout: 10000,
                    validateStatus: (status) => status < 500
                  })
                  
                  console.log('[TrainingPanel] Refresh - API response received:', response.status)
                  
                  if (response.data) {
                    // Update training status
                    setTrainingStatus(response.data)
                    
                    // Update training state
                    const status = response.data.status
                    if (status === 'running' || status === 'starting') {
                      setTraining(true)
                    } else if (status === 'completed' || status === 'error' || status === 'idle') {
                      setTraining(false)
                    }
                    
                    if (response.data.metrics && Object.keys(response.data.metrics).length > 0) {
                      // Deep clone to ensure new reference
                      const newMetrics = JSON.parse(JSON.stringify(response.data.metrics))
                      console.log('[TrainingPanel] Refresh - updating metrics:', {
                        timestep: newMetrics.timestep,
                        reward: newMetrics.latest_reward,
                        episode: newMetrics.episode
                      })
                      
                      // Force state updates - use function form to ensure we get latest state
                      setTrainingMetrics((prev) => {
                        console.log('[TrainingPanel] Refresh - setTrainingMetrics called, prev:', prev?.timestep, 'new:', newMetrics.timestep)
                        return newMetrics
                      })
                      
                      prevMetricsRef.current = newMetrics
                      
                      // Force re-render by incrementing render key
                      setRenderKey(prev => {
                        const next = prev + 1
                        console.log('[TrainingPanel] Refresh - RenderKey:', prev, '->', next)
                        return next
                      })
                      
                      // Force update counter
                      forceUpdate(prev => {
                        const next = prev + 1
                        console.log('[TrainingPanel] Refresh - forceUpdate:', prev, '->', next)
                        return next
                      })
                      
                      // Update lastPollTime so watchdog doesn't think polling stopped
                      lastPollTimeRef.current = Date.now()
                      
                      // Force a DOM update by directly manipulating the render
                      // Use requestAnimationFrame to ensure it happens after React's render cycle
                      requestAnimationFrame(() => {
                        console.log('[TrainingPanel] Refresh - requestAnimationFrame fired')
                        // Force one more tiny update to trigger render
                        setRenderKey(k => k === Math.floor(k) ? k + 0.001 : k + 1)
                      })
                    } else {
                      console.warn('[TrainingPanel] Refresh - No metrics in response:', response.data)
                    }
                  } else {
                    console.warn('[TrainingPanel] Refresh - No data in response')
                  }
                } catch (err) {
                  console.error('[TrainingPanel] Manual refresh failed:', err)
                  alert(`Refresh failed: ${err.message || 'Unknown error'}`)
                }
              }}
              className="px-3 py-1.5 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg flex items-center gap-2 text-gray-700 transition-colors"
              title="Refresh metrics"
            >
              <RefreshCw className="w-4 h-4" />
              Refresh
            </button>
          </div>
          
          {/* Progress Bar */}
          {trainingMetrics && trainingMetrics.progress_percent !== undefined && (
            <div className="mb-6">
              <div className="flex justify-between text-sm text-gray-600 mb-2">
                <span>Progress</span>
                <span>{trainingMetrics.progress_percent.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div
                  className="bg-primary-600 h-3 rounded-full transition-all duration-300"
                  style={{ width: `${trainingMetrics.progress_percent}%` }}
                />
              </div>
            </div>
          )}
          
          {trainingMetrics ? (
            <>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            {/* Episode */}
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="text-sm text-gray-600 mb-1">Current Episode</div>
              {trainingMetrics.episode !== undefined ? (
                <div
                  key={`episode-${renderKey}`}
                  className="text-2xl font-bold text-gray-800"
                >
                  {trainingMetrics.episode.toLocaleString()}
                </div>
              ) : (
                <div className="text-2xl font-bold text-gray-800">N/A</div>
              )}
            </div>
            
            {/* Timestep Progress */}
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="text-sm text-gray-600 mb-1">Timesteps</div>
              {trainingMetrics.timestep !== undefined && trainingMetrics.total_timesteps !== undefined ? (
                <div
                  key={`timestep-${renderKey}`}
                  className="text-2xl font-bold text-gray-800"
                >
                  {(trainingMetrics.timestep / 1000).toFixed(0)}k / {((trainingMetrics.total_timesteps || 0) / 1000).toFixed(0)}k
                </div>
              ) : (
                <div className="text-2xl font-bold text-gray-800">N/A</div>
              )}
            </div>
            
            {/* Latest Reward */}
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="text-sm text-gray-600 mb-1">Latest Reward</div>
              {trainingMetrics.latest_reward !== undefined ? (
                <div 
                  key={`reward-${renderKey}`}
                  className={`text-2xl font-bold ${trainingMetrics.latest_reward >= 0 ? 'text-green-600' : 'text-red-600'}`}
                  title={`Exact value: ${trainingMetrics.latest_reward}`}
                >
                  {Math.abs(trainingMetrics.latest_reward) < 0.01 && trainingMetrics.latest_reward !== 0
                    ? trainingMetrics.latest_reward.toFixed(4)  // Show 4 decimals for very small non-zero values
                    : trainingMetrics.latest_reward.toFixed(2)}
                </div>
              ) : (
                <div className="text-2xl font-bold text-gray-500">N/A</div>
              )}
            </div>
            
            {/* Mean Reward (Last 10) */}
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="text-sm text-gray-600 mb-1">Mean Reward (Last 10)</div>
              {trainingMetrics.mean_reward_10 !== undefined ? (
                <div
                  key={`mean-reward-${renderKey}`}
                  className={`text-2xl font-bold ${trainingMetrics.mean_reward_10 >= 0 ? 'text-green-600' : 'text-red-600'}`}
                >
                  {trainingMetrics.mean_reward_10.toFixed(2)}
                </div>
              ) : (
                <div className="text-2xl font-bold text-gray-500">N/A</div>
              )}
            </div>
          </div>
          
          {/* Training Metrics */}
          {trainingMetrics.training_metrics && (
            <div className="border-t pt-4">
              <h4 className="text-md font-semibold text-gray-700 mb-3">Training Metrics</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <div className="text-xs text-gray-600 mb-1">Loss</div>
                  {trainingMetrics.training_metrics.loss !== undefined ? (
                    <div
                      key={`loss-${renderKey}`}
                      className="text-lg font-semibold text-gray-800"
                    >
                      {trainingMetrics.training_metrics.loss.toFixed(4)}
                    </div>
                  ) : (
                    <div className="text-lg font-semibold text-gray-800">N/A</div>
                  )}
                </div>
                <div>
                  <div className="text-xs text-gray-600 mb-1">Policy Loss</div>
                  {trainingMetrics.training_metrics.policy_loss !== undefined ? (
                    <div
                      key={`policy-loss-${renderKey}`}
                      className="text-lg font-semibold text-gray-800"
                    >
                      {trainingMetrics.training_metrics.policy_loss.toFixed(4)}
                    </div>
                  ) : (
                    <div className="text-lg font-semibold text-gray-800">N/A</div>
                  )}
                </div>
                <div>
                  <div className="text-xs text-gray-600 mb-1">Value Loss</div>
                  {trainingMetrics.training_metrics.value_loss !== undefined ? (
                    <div
                      key={`value-loss-${renderKey}`}
                      className="text-lg font-semibold text-gray-800"
                    >
                      {trainingMetrics.training_metrics.value_loss.toFixed(4)}
                    </div>
                  ) : (
                    <div className="text-lg font-semibold text-gray-800">N/A</div>
                  )}
                </div>
                <div>
                  <div className="text-xs text-gray-600 mb-1">Entropy</div>
                  {trainingMetrics.training_metrics.entropy !== undefined ? (
                    <div
                      key={`entropy-${renderKey}`}
                      className="text-lg font-semibold text-gray-800"
                    >
                      {trainingMetrics.training_metrics.entropy.toFixed(4)}
                    </div>
                  ) : (
                    <div className="text-lg font-semibold text-gray-800">N/A</div>
                  )}
                </div>
              </div>
            </div>
              )}
            </>
          ) : (
            <div className="text-center py-12 text-gray-500">
              <p>Training is initializing...</p>
              <p className="text-sm mt-2">Metrics will appear once training starts</p>
            </div>
          )}
          
          {/* Episode Length */}
          {trainingMetrics && trainingMetrics.latest_episode_length !== undefined && (
            <div className="border-t pt-4 mt-4">
              <div className="flex justify-between items-center">
                <div>
                  <div className="text-sm text-gray-600">Latest Episode Length</div>
                  {trainingMetrics.latest_episode_length !== undefined ? (
                    <div
                      key={`latest-length-${renderKey}`}
                      className="text-gray-800 mt-1 text-xl font-semibold"
                    >
                      {trainingMetrics.latest_episode_length} steps
                    </div>
                  ) : (
                    <div className="text-xl font-semibold text-gray-800 mt-1">N/A</div>
                  )}
                </div>
                {trainingMetrics.mean_episode_length !== undefined && (
                  <div className="text-right">
                    <div className="text-sm text-gray-600">Mean Length</div>
                    <div
                      key={`mean-length-${renderKey}`}
                      className="text-gray-800 mt-1 text-xl font-semibold"
                    >
                      {trainingMetrics.mean_episode_length.toFixed(1)} steps
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default TrainingPanel

