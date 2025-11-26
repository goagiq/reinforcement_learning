import React, { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import { Play, Square, Loader, CheckCircle, AlertCircle, Cpu, RefreshCw, Star } from 'lucide-react'
import { getDefaultModel, sortModelsWithDefaultFirst, isDefaultModel } from '../utils/modelUtils'
import CapabilityExplainer from './CapabilityExplainer'

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
  const DEFAULT_CONFIG_PATH = 'configs/train_config_adaptive.yaml'
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
  const startupPhaseRef = useRef(true) // Track if we're in startup phase (first 30 seconds)
  const startupStartTimeRef = useRef(Date.now()) // Track when polling started
  
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
      // DEBUG: Log timestep diagnostic information to browser console (F12)
      if (trainingMetrics.debug) {
        console.log('[TIMESTEP DEBUG]', {
          timestep: trainingMetrics.timestep,
          episode: trainingMetrics.episode,
          debug: trainingMetrics.debug,
          warning: trainingMetrics.debug.warning
        })
        if (trainingMetrics.debug.warning) {
          console.warn('[TIMESTEP ISSUE]', trainingMetrics.debug.warning)
        }
      }
      console.log(`[TrainingPanel] trainingMetrics state changed - Timestep: ${trainingMetrics.timestep}, Reward: ${trainingMetrics.latest_reward?.toFixed(2)}, Episode: ${trainingMetrics.episode}`)
    }
  }, [trainingMetrics])
  const [device, setDevice] = useState('cpu')
  const [cudaAvailable, setCudaAvailable] = useState(false)
  const [gpuInfo, setGpuInfo] = useState(null)
  const [totalTimesteps, setTotalTimesteps] = useState(20000000)
  const [reasoningModel, setReasoningModel] = useState('')
  const [configPath, setConfigPathState] = useState('')
  const [availableConfigs, setAvailableConfigs] = useState([])
  const [checkpoints, setCheckpoints] = useState([])
  const [bestModel, setBestModel] = useState(null)
  const [selectedCheckpoint, setSelectedCheckpoint] = useState('latest') // 'latest', 'none', or checkpoint path
  const [latestCheckpoint, setLatestCheckpoint] = useState(null)
  const [flushOldData, setFlushOldData] = useState(false) // Clear old training data for fresh start
  const [messages, setMessages] = useState([])
  const [ws, setWs] = useState(null)
  const [trainingMode, setTrainingMode] = useState('quiet') // 'quiet', 'performance', or 'turbo'
  const [turboModeEnabled, setTurboModeEnabled] = useState(false)
  const [manualTriggerLoading, setManualTriggerLoading] = useState(false)
  const [promoteLoading, setPromoteLoading] = useState(false)
  const [transferStrategy, setTransferStrategy] = useState('copy_and_extend') // Transfer learning strategy
  const [checkpointInfo, setCheckpointInfo] = useState(null) // Architecture info for selected checkpoint
  const [configArchitecture, setConfigArchitecture] = useState(null) // Architecture from config
  const [architectureMismatch, setArchitectureMismatch] = useState(false) // Whether architectures differ
  const normalizePath = (path) => (path ? path.replace(/\\/g, '/') : '')
  const setConfigPath = (path) => {
    const normalized = normalizePath(path)
    setConfigPathState(normalized)
    try {
      if (normalized) {
        localStorage.setItem('trainingConfigPath', normalized)
      } else {
        localStorage.removeItem('trainingConfigPath')
      }
    } catch (error) {
      // Ignore storage errors (e.g., privacy mode)
    }
  }

  useEffect(() => {
    try {
      const stored = localStorage.getItem('trainingConfigPath')
      if (stored) {
        setConfigPathState(normalizePath(stored))
        return
      }
    } catch (error) {
      // Ignore storage errors (e.g., privacy mode)
    }
    setConfigPath(DEFAULT_CONFIG_PATH)
  }, [])
  
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

  // Load training mode from settings
  const loadTrainingMode = async () => {
    try {
      const response = await axios.get('/api/settings/get')
      if (response.data.performance_mode) {
        setTrainingMode(response.data.performance_mode)
      }
      // Check for turbo_training_mode - it might be missing if setting wasn't saved yet
      const turboEnabled = response.data.turbo_training_mode === true
      setTurboModeEnabled(turboEnabled)
      
      console.log('[TrainingPanel] Loaded training mode from settings:', {
        performance_mode: response.data.performance_mode,
        turbo_training_mode: response.data.turbo_training_mode,
        turboEnabled: turboEnabled
      })
      
      // If turbo mode is enabled but not in settings, log a warning
      if (response.data.turbo_training_mode === undefined && turboEnabled) {
        console.warn('[TrainingPanel] Turbo mode enabled in UI but not saved in settings.json. Please save settings again.')
      }
    } catch (error) {
      console.error('Failed to load training mode:', error)
    }
  }
  
  useEffect(() => {
    let mounted = true
    
    const loadOnce = async () => {
      if (mounted) {
        await loadTrainingMode()
      }
    }
    
    loadOnce()
    
    // Reload when settings might have changed (listen to storage events)
    const handleSettingsChange = () => {
      if (mounted) {
        loadTrainingMode()
      }
    }
    
    window.addEventListener('storage', handleSettingsChange)
    window.addEventListener('defaultModelChanged', handleSettingsChange)
    
    // Also poll settings periodically during training (every 10 seconds, not 5)
    // This ensures the indicator updates when turbo mode is toggled
    const settingsPollInterval = setInterval(() => {
      if (mounted && training) {
        loadTrainingMode()
      }
    }, 10000) // Increased from 5000 to reduce spam
    
    return () => {
      mounted = false
      window.removeEventListener('storage', handleSettingsChange)
      window.removeEventListener('defaultModelChanged', handleSettingsChange)
      clearInterval(settingsPollInterval)
    }
  }, [training]) // Re-run when training state changes
  
  // Load available config files on mount
  useEffect(() => {
    const loadConfigs = async () => {
      try {
        const response = await axios.get('/api/config/list')
        const rawConfigs = response.data.configs || []
        const normalizedConfigs = rawConfigs.map(cfg => ({
          ...cfg,
          path: normalizePath(cfg.path),
          relative: normalizePath(cfg.relative)
        }))
        setAvailableConfigs(normalizedConfigs)
        
        if (normalizedConfigs.length > 0) {
          const currentNormalized = normalizePath(configPath)
          const hasCurrent = currentNormalized && normalizedConfigs.some(c => c.relative === currentNormalized)
          if (!hasCurrent) {
            const lowerName = (name) => name.toLowerCase()
            const adaptiveConfig = normalizedConfigs.find(c => lowerName(c.name).includes('adaptive'))
            const fullConfig = normalizedConfigs.find(c => lowerName(c.name).includes('full'))
            const gpuConfig = normalizedConfigs.find(c => lowerName(c.name).includes('gpu'))
            const optimizedConfig = normalizedConfigs.find(c => {
              const name = lowerName(c.name)
              return name.includes('gpu_optimized') || name.includes('optimized')
            })
            const fallback = adaptiveConfig || fullConfig || gpuConfig || optimizedConfig || normalizedConfigs[0]
            setConfigPath(fallback.relative)
          }
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
        const allModels = response.data.models || []
        const foundBestModel = allModels.find((model) => (model.name || '').toLowerCase() === 'best_model.pt')
        
        setCheckpoints(checkpointsList)
        setLatestCheckpoint(latest)
        if (foundBestModel) {
          const normalizedPath = normalizePath(foundBestModel.path || 'models/best_model.pt')
          setBestModel({
            ...foundBestModel,
            path: normalizedPath,
          })
        } else {
          setBestModel(null)
        }
        
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

  useEffect(() => {
    if (
      (!latestCheckpoint || !latestCheckpoint.path) &&
      bestModel &&
      selectedCheckpoint === 'latest'
    ) {
      setSelectedCheckpoint(bestModel.path)
    }
  }, [bestModel, latestCheckpoint, selectedCheckpoint])

  // Load config architecture when config changes
  useEffect(() => {
    const loadConfigArchitecture = async () => {
      try {
        const targetPath = normalizePath(configPath)
        if (!targetPath) {
          return
        }
        const response = await axios.get(`/api/config/read?path=${encodeURIComponent(targetPath)}`)
        const config = response.data.config || {}
        const hiddenDims = config.model?.hidden_dims || [256, 256, 128]
        const configuredStateDim = config.environment?.state_features
        const timeframes = config.environment?.timeframes
        const lookbackBars = config.environment?.lookback_bars
        const derivedStateDim = Array.isArray(timeframes) && timeframes.length > 0 && lookbackBars
          ? timeframes.length * 15 * lookbackBars
          : undefined
        const stateDim = configuredStateDim ?? derivedStateDim ?? 200
        
        setConfigArchitecture({
          hidden_dims: hiddenDims,
          state_dim: stateDim
        })
      } catch (error) {
        console.error('Failed to load config architecture:', error)
      }
    }
    
    if (configPath) {
      loadConfigArchitecture()
    }
  }, [configPath])

  // Load checkpoint info when checkpoint selection changes
  useEffect(() => {
    const loadCheckpointInfo = async () => {
      if (selectedCheckpoint === 'none' || !selectedCheckpoint) {
        setCheckpointInfo(null)
        setArchitectureMismatch(false)
        return
      }
      
      try {
        let checkpointPath = selectedCheckpoint
        if (selectedCheckpoint === 'latest' && latestCheckpoint) {
          checkpointPath = latestCheckpoint.path
        }
        
        if (checkpointPath && checkpointPath !== 'latest' && checkpointPath !== 'none') {
          const response = await axios.get(`/api/models/checkpoint/info?checkpoint_path=${encodeURIComponent(checkpointPath)}`)
          if (response.data.error) {
            setCheckpointInfo(null)
            setArchitectureMismatch(false)
            return
          }
          
          setCheckpointInfo(response.data)
          
          // Check for architecture mismatch
          if (response.data.architecture && configArchitecture) {
            const checkpointArch = response.data.architecture
            const configArch = configArchitecture
            
            const hiddenDimsMatch = JSON.stringify(checkpointArch.hidden_dims) === JSON.stringify(configArch.hidden_dims)
            const stateDimMatch = checkpointArch.state_dim === configArch.state_dim
            
            setArchitectureMismatch(!hiddenDimsMatch || !stateDimMatch)
          } else {
            setArchitectureMismatch(false)
          }
        }
      } catch (error) {
        console.error('Failed to load checkpoint info:', error)
        setCheckpointInfo(null)
        setArchitectureMismatch(false)
      }
    }
    
    loadCheckpointInfo()
  }, [selectedCheckpoint, latestCheckpoint, configArchitecture])

  // Check CUDA availability on mount and auto-select if available
  useEffect(() => {
    const checkCudaAvailability = async () => {
      try {
        console.log('🔍 Checking CUDA availability...')
        const response = await axios.get('/api/system/cuda-status')
        
        console.log('📦 Full API Response:', response)
        console.log('📦 Response Data:', response.data)
        
        const data = response.data || {}
        const cuda_available = Boolean(data.cuda_available) // Ensure boolean
        const recommendedDevice = data.device || 'cpu'
        const gpu_name = data.gpu_name
        const cuda_version = data.cuda_version
        const error = data.error
        
        console.log('🔍 Parsed CUDA Status:', {
          cuda_available,
          type: typeof cuda_available,
          recommendedDevice,
          gpu_name,
          cuda_version,
          error,
          currentDevice: device
        })
        
        setCudaAvailable(cuda_available)
        
        if (cuda_available && gpu_name) {
          setGpuInfo({
            name: gpu_name,
            version: cuda_version || 'Unknown'
          })
          console.log(`✅ GPU Info Set: ${gpu_name} (CUDA ${cuda_version})`)
        } else if (error) {
          console.warn('⚠️ CUDA Check Warning:', error)
        }
        
        // Auto-select CUDA if available (check both boolean true and string "true")
        if (cuda_available === true || cuda_available === 'true') {
          console.log('✅ CUDA is available - auto-selecting GPU')
          setDevice('cuda')
          console.log(`✅ Device set to: cuda (was: ${device})`)
        } else {
          console.log('❌ CUDA not available - keeping device as cpu')
          if (error) {
            console.log('   Error details:', error)
          }
        }
      } catch (error) {
        console.error('❌ Failed to check CUDA status:', error)
        console.error('   Error details:', error.response?.data || error.message)
        // Default to CPU if check fails
        setCudaAvailable(false)
        setDevice('cpu')
      }
    }
    
    // Add a small delay to ensure backend is ready
    const timeoutId = setTimeout(() => {
      checkCudaAvailability()
    }, 1000)
    
    return () => clearTimeout(timeoutId)
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
                timeout: 10000, // Increased to 10 seconds to handle slow backend responses during startup
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
          
          // Update training mode from backend if provided (more accurate than settings)
          if (response.data.training_mode && response.data.training_mode.performance_mode) {
            const backendMode = response.data.training_mode.performance_mode
            if (backendMode === 'turbo') {
              setTurboModeEnabled(true)
              setTrainingMode('performance') // Keep performance as base, turbo overrides
            } else {
              setTrainingMode(backendMode)
              if (backendMode !== 'turbo') {
                setTurboModeEnabled(false)
              }
            }
          } else if (status === 'running' || status === 'starting') {
            // Fallback: refresh from settings if backend doesn't provide mode
            loadTrainingMode()
          }
          
          // Update training metrics if available
          // Only update if metrics actually contain data (not empty object during initialization)
          // Debug: Log metrics received
          if (pollCount % 10 === 0 || pollCount <= 3) {
            const metrics = response.data.metrics || {}
            console.log(`[TrainingPanel] Status check #${pollCount}: status=${status}, metrics keys=${Object.keys(metrics).length}`)
            console.log(`[TrainingPanel] Trade counts - total=${metrics.total_trades || 0}, winning=${metrics.total_winning_trades || 0}, losing=${metrics.total_losing_trades || 0}, current_episode=${metrics.current_episode_trades || 0}`)
            console.log(`[TrainingPanel] Full metrics:`, response.data.metrics)
          }
          
          if (response.data.metrics && Object.keys(response.data.metrics).length > 0) {
            // Deep clone to ensure new reference - use JSON parse/stringify for complete deep copy
            const newMetrics = JSON.parse(JSON.stringify(response.data.metrics))
            
            // Check if values actually changed to avoid unnecessary re-renders
            // IMPORTANT: Check ALL current episode metrics to ensure real-time updates
            const prevMetrics = prevMetricsRef.current
            const valuesChanged = !prevMetrics || 
                prevMetrics.timestep !== newMetrics.timestep ||
                Math.abs((prevMetrics.latest_reward || 0) - (newMetrics.latest_reward || 0)) > 0.001 ||
                prevMetrics.episode !== newMetrics.episode ||
                prevMetrics.current_episode_length !== newMetrics.current_episode_length ||
                // Check current episode trading metrics for changes
                prevMetrics.current_episode_trades !== newMetrics.current_episode_trades ||
                Math.abs((prevMetrics.current_episode_pnl || 0) - (newMetrics.current_episode_pnl || 0)) > 0.01 ||
                Math.abs((prevMetrics.current_episode_equity || 0) - (newMetrics.current_episode_equity || 0)) > 0.01 ||
                Math.abs((prevMetrics.current_episode_win_rate || 0) - (newMetrics.current_episode_win_rate || 0)) > 0.1 ||
                Math.abs((prevMetrics.current_episode_max_drawdown || 0) - (newMetrics.current_episode_max_drawdown || 0)) > 0.1 ||
                // Check aggregate trading metrics for changes (total trades, winning, losing)
                (prevMetrics.total_trades || 0) !== (newMetrics.total_trades || 0) ||
                (prevMetrics.total_winning_trades || 0) !== (newMetrics.total_winning_trades || 0) ||
                (prevMetrics.total_losing_trades || 0) !== (newMetrics.total_losing_trades || 0) ||
                Math.abs((prevMetrics.overall_win_rate || 0) - (newMetrics.overall_win_rate || 0)) > 0.1
            
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
          
          // Track if we're still in startup phase (first 30 seconds or first 10 polls)
          const isStartupPhase = Date.now() - startupStartTimeRef.current < 30000 || pollCount < 10
          startupPhaseRef.current = isStartupPhase
          
          // Don't stop polling on error - just log it
          if (isMounted) {
            const errorMsg = error.response 
              ? `HTTP ${error.response.status}: ${error.response.statusText}` 
              : error.message || 'Unknown error'
            
            // Only log errors after startup phase to reduce console noise
            // Timeouts during startup are normal and expected
            if (!isStartupPhase || error.code !== 'ECONNABORTED') {
              console.error(`[TrainingPanel] Failed to check training status (poll #${pollId}):`, errorMsg)
              if (error.code) {
                console.error(`[TrainingPanel] Error code:`, error.code)
              }
            } else {
              // Debug level for startup timeouts
              console.debug(`[TrainingPanel] Startup timeout (poll #${pollId}) - will retry`)
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
      startupStartTimeRef.current = Date.now() // Record when polling starts
      
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
      
      // Include flush_old_data if starting fresh (no checkpoint)
      if (selectedCheckpoint === 'none') {
        requestData.flush_old_data = flushOldData
        console.log('🗑️ Flush old data:', flushOldData)
      }
      
      // Include transfer_strategy if architecture mismatch detected or explicitly set
      if (architectureMismatch || selectedCheckpoint !== 'none') {
        requestData.transfer_strategy = transferStrategy
        console.log('🔄 Using transfer learning strategy:', transferStrategy)
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

  const handleManualRetrainTrigger = async () => {
    setManualTriggerLoading(true)
    try {
      const response = await axios.post('/api/settings/auto-retrain/trigger-manual')
      console.log('Manual retrain triggered:', response.data)
      
      // Show success message
      setMessages(prev => [...prev, {
        status: 'running',
        message: `✅ Manual retraining triggered: ${response.data.message || 'Success'}. ${response.data.files_found || 0} file(s) found.`
      }])
      
      // If training isn't already running, start checking status
      if (!training) {
        setTraining(true)
      }
    } catch (error) {
      console.error('Failed to trigger manual retraining:', error)
      const errorMessage = error.response?.data?.detail || error.message || 'Unknown error'
      setMessages(prev => [...prev, {
        status: 'error',
        message: `❌ Failed to trigger manual retraining: ${errorMessage}`
      }])
    } finally {
      setManualTriggerLoading(false)
    }
  }

  const handlePromoteCheckpoint = async () => {
    if (!selectedCheckpoint || selectedCheckpoint === 'none') {
      return
    }

    let checkpointPath = selectedCheckpoint
    if (selectedCheckpoint === 'latest') {
      if (!latestCheckpoint?.path) {
        setMessages(prev => [...prev, {
          status: 'error',
          message: '❌ No latest checkpoint available to promote.'
        }])
        return
      }
      checkpointPath = latestCheckpoint.path
    }

    checkpointPath = normalizePath(checkpointPath)
    if (!checkpointPath) {
      setMessages(prev => [...prev, {
        status: 'error',
        message: '❌ Invalid checkpoint path. Unable to promote.'
      }])
      return
    }

    setPromoteLoading(true)
    try {
      const payload = {
        source_path: checkpointPath,
        target_name: 'best_model.pt',
        overwrite: true
      }
      const response = await axios.post('/api/models/promote', payload)
      const successMessage = response.data?.message || `⭐ Promoted ${checkpointPath} to best_model.pt`
      setMessages(prev => [...prev, {
        status: 'success',
        message: successMessage
      }])

      if (typeof onModelsChangeRef.current === 'function') {
        onModelsChangeRef.current()
      }
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message || 'Failed to promote checkpoint'
      setMessages(prev => [...prev, {
        status: 'error',
        message: `❌ Failed to promote checkpoint: ${errorMessage}`
      }])
    } finally {
      setPromoteLoading(false)
    }
  }

  const resolvedCheckpointPath = (() => {
    if (!selectedCheckpoint || selectedCheckpoint === 'none') {
      return null
    }
    if (selectedCheckpoint === 'latest') {
      return latestCheckpoint?.path ? normalizePath(latestCheckpoint.path) : null
    }
    return normalizePath(selectedCheckpoint)
  })()

  const isBestModelSelected =
    resolvedCheckpointPath && bestModel && normalizePath(bestModel.path) === resolvedCheckpointPath

  const canPromoteCheckpoint = Boolean(resolvedCheckpointPath) && !isBestModelSelected

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
              value={configPath || ''}
              onChange={(e) => setConfigPath(normalizePath(e.target.value))}
              disabled={training}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            >
              {availableConfigs.length > 0 ? (
                availableConfigs.map((config, idx) => (
                  <option key={idx} value={config.relative}>
                    {config.name}
                    {config.name.toLowerCase().includes('adaptive')
                      ? ' 🔄 (Adaptive Training - Recommended)'
                      : config.name.toLowerCase().includes('gpu') || config.name.toLowerCase().includes('optimized')
                        ? ' ⚡ (Optimized for GPU)'
                        : config.name.toLowerCase().includes('full')
                          ? ' (Full Capacity)'
                          : ''}
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
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm font-semibold text-gray-700">
                Device
              </label>
              <CapabilityExplainer
                capabilityId="training.device"
                context={{
                  cudaAvailable,
                  selectedDevice: device,
                  gpuName: gpuInfo?.name || null,
                  cudaVersion: gpuInfo?.version || null,
                }}
              />
            </div>
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
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm font-semibold text-gray-700">
                Resume from Checkpoint
              </label>
              <CapabilityExplainer
                capabilityId="training.checkpoints"
                context={{
                  checkpoints: checkpoints.map(c => ({
                    timestep: c.timestep,
                    path: c.path,
                  })),
                  latestCheckpoint: latestCheckpoint,
                  architectureMismatch,
                  transferStrategy,
                }}
              />
            </div>
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
              {bestModel && (
                <option value={bestModel.path}>
                  best_model.pt (Current Best Model)
                </option>
              )}
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
            
            {/* Checkpoint Architecture Info */}
            {checkpointInfo && checkpointInfo.has_architecture && (
              <div className="mt-2 p-3 bg-blue-50 rounded-lg border border-blue-200">
                <div className="text-sm font-semibold text-blue-800 mb-1">Checkpoint Architecture:</div>
                <div className="text-sm text-blue-700">
                  Hidden Layers: [{checkpointInfo.architecture.hidden_dims?.join(', ') || 'N/A'}]
                  {checkpointInfo.architecture.state_dim && (
                    <> | State Dim: {checkpointInfo.architecture.state_dim}</>
                  )}
                  {checkpointInfo.training.timestep > 0 && (
                    <> | Timestep: {checkpointInfo.training.timestep.toLocaleString()}</>
                  )}
                </div>
              </div>
            )}
            
            {/* Config Architecture Info */}
            {configArchitecture && (
              <div className="mt-2 p-3 bg-gray-50 rounded-lg border border-gray-200">
                <div className="text-sm font-semibold text-gray-700 mb-1">Config Architecture:</div>
                <div className="text-sm text-gray-600">
                  Hidden Layers: [{configArchitecture.hidden_dims?.join(', ') || 'N/A'}]
                  {configArchitecture.state_dim && (
                    <> | State Dim: {configArchitecture.state_dim}</>
                  )}
                </div>
              </div>
            )}
            
            {canPromoteCheckpoint && (
              <div className="mt-3">
                <button
                  type="button"
                  onClick={handlePromoteCheckpoint}
                  disabled={promoteLoading || !canPromoteCheckpoint}
                  className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {promoteLoading ? (
                    <>
                      <Loader className="w-4 h-4 animate-spin" />
                      Promoting...
                    </>
                  ) : (
                    <>
                      <Star className="w-4 h-4" />
                      Promote to Best Model
                    </>
                  )}
                </button>
                <p className="text-xs text-gray-500 mt-2">
                  Copies the selected checkpoint to <code>models/best_model.pt</code> for backtests and deployment.
                </p>
              </div>
            )}
            
            {/* Architecture Mismatch Warning */}
            {architectureMismatch && (
              <div className="mt-2 p-3 bg-yellow-50 rounded-lg border border-yellow-300">
                <div className="flex items-start gap-2">
                  <span className="text-yellow-600 font-bold text-lg">⚠️</span>
                  <div className="flex-1">
                    <div className="text-sm font-semibold text-yellow-800 mb-1">Architecture Mismatch Detected</div>
                    <div className="text-sm text-yellow-700 mb-2">
                      The checkpoint architecture differs from the config. Transfer learning will be used to preserve learned knowledge.
                    </div>
                    <div className="text-xs text-yellow-600">
                      <div>Checkpoint: [{checkpointInfo.architecture.hidden_dims?.join(', ') || 'N/A'}]</div>
                      <div>Config: [{configArchitecture.hidden_dims?.join(', ') || 'N/A'}]</div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            <p className="text-sm text-gray-500 mt-1">
              {selectedCheckpoint === 'latest' && latestCheckpoint ? (
                <span className="text-blue-600 font-medium">
                  ✓ Will resume from latest checkpoint ({latestCheckpoint.timestep ? `${(latestCheckpoint.timestep / 1000).toFixed(0)}k timesteps` : 'Unknown timesteps'})
                  {architectureMismatch && ' with transfer learning'}
                </span>
              ) : selectedCheckpoint === 'none' ? (
                <span className="text-orange-600 font-medium">
                  ⚠ Will start fresh training from timestep 0
                </span>
              ) : (
                'Select a checkpoint to resume training from that point, or choose "Start Fresh Training" to begin from scratch'
              )}
            </p>
            
            {/* Flush Old Data Option (only shown for fresh training) */}
            {selectedCheckpoint === 'none' && (
              <div className="mt-4 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={flushOldData}
                    onChange={(e) => setFlushOldData(e.target.checked)}
                    className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                  />
                  <span className="font-semibold text-yellow-800">
                    🗑️ Clear old training data (database, caches)
                  </span>
                </label>
                <p className="text-sm text-yellow-700 mt-2 ml-6">
                  When enabled, this will:
                  <ul className="list-disc list-inside mt-1 space-y-1">
                    <li>Archive and clear the trading journal database</li>
                    <li>Clear cache files (known_files_cache.json, etc.)</li>
                    <li>Ensure metrics show only new training data</li>
                  </ul>
                  <span className="font-semibold">Note: Old data will be backed up to logs/trading_journal_archive/ before clearing.</span>
                </p>
              </div>
            )}
          </div>

          {/* Transfer Learning Strategy (shown when checkpoint selected) */}
          {selectedCheckpoint !== 'none' && (
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Transfer Learning Strategy
                {architectureMismatch && <span className="text-yellow-600 ml-2">(Required for architecture mismatch)</span>}
              </label>
              <select
                value={transferStrategy}
                onChange={(e) => setTransferStrategy(e.target.value)}
                disabled={training}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              >
                <option value="copy_and_extend">
                  Copy and Extend (Recommended) - Preserves learned weights, initializes new dimensions with small random values
                </option>
                <option value="interpolate">
                  Interpolate - Averages existing neurons to fill new dimensions
                </option>
                <option value="zero_pad">
                  Zero Pad - Copies weights, initializes new dimensions with zeros
                </option>
              </select>
              <p className="text-sm text-gray-500 mt-1">
                {transferStrategy === 'copy_and_extend' && (
                  <>✅ <strong>Recommended:</strong> Best balance between preserving knowledge and allowing new capacity to learn.</>
                )}
                {transferStrategy === 'interpolate' && (
                  <>🔄 Uses average of existing neurons - good for similar architectures.</>
                )}
                {transferStrategy === 'zero_pad' && (
                  <>⚡ Fastest but may require more training for new dimensions to learn effectively.</>
                )}
                {!architectureMismatch && (
                  <> This strategy will be used if architecture changes are detected during checkpoint loading.</>
                )}
              </p>
            </div>
          )}

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
          <div className="flex gap-4 flex-wrap">
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
            
            {/* Manual Auto-Retrain Trigger Button */}
            <button
              onClick={handleManualRetrainTrigger}
              disabled={manualTriggerLoading}
              className="px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center gap-2"
              title="Manually trigger auto-retraining with all files in the configured directory (bypasses file detection)"
            >
              {manualTriggerLoading ? (
                <>
                  <Loader className="w-5 h-5 animate-spin" />
                  <span>Triggering...</span>
                </>
              ) : (
                <>
                  <RefreshCw className="w-5 h-5" />
                  <span>Trigger Auto-Retrain</span>
                </>
              )}
            </button>
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

      {/* Training Mode Indicator */}
      {training && (
        <div className="bg-white rounded-lg shadow p-4 mb-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="text-sm font-semibold text-gray-700">Training Mode:</span>
              {turboModeEnabled ? (
                <span className="px-3 py-1.5 bg-gradient-to-r from-yellow-400 to-orange-500 text-white text-sm font-bold rounded-lg shadow-md flex items-center gap-2">
                  <span>🔥</span>
                  <span>TURBO MODE</span>
                  <span className="text-xs opacity-90">(Max GPU)</span>
                </span>
              ) : trainingMode === 'performance' ? (
                <span className="px-3 py-1.5 bg-blue-500 text-white text-sm font-semibold rounded-lg shadow flex items-center gap-2">
                  <span>🚀</span>
                  <span>PERFORMANCE</span>
                </span>
              ) : (
                <span className="px-3 py-1.5 bg-gray-500 text-white text-sm font-semibold rounded-lg shadow flex items-center gap-2">
                  <span>✅</span>
                  <span>QUIET MODE</span>
                </span>
              )}
            </div>
            <button
              onClick={() => {
                // Open settings by triggering a custom event or navigating
                window.dispatchEvent(new CustomEvent('openSettings'))
              }}
              className="text-xs text-gray-500 hover:text-gray-700 underline"
            >
              Change in Settings
            </button>
          </div>
        </div>
      )}

      {/* Detailed Training Status */}
      {training && trainingMetrics && (
        <div key={`training-metrics-${renderKey}`} className="bg-white rounded-lg shadow p-6">
          <div className="flex justify-between items-center mb-4">
            <div className="flex items-center gap-3">
              <h3 className="text-lg font-semibold text-gray-800">Training Progress</h3>
              <CapabilityExplainer
                capabilityId="training.progress"
                context={{
                  metrics: trainingMetrics,
                  performanceMode: trainingMode,
                  turboModeEnabled,
                }}
              />
            </div>
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
                        // DEBUG: Log timestep diagnostic if available
                        if (newMetrics.debug) {
                          console.log('[TIMESTEP DEBUG]', {
                            timestep: newMetrics.timestep,
                            episode: newMetrics.episode,
                            episode_lengths_count: newMetrics.debug.episode_lengths_count,
                            last_episode_length: newMetrics.debug.last_episode_length,
                            current_episode_length: newMetrics.debug.current_episode_length,
                            warning: newMetrics.debug.warning
                          })
                          if (newMetrics.debug.warning) {
                            console.warn('[TIMESTEP ISSUE]', newMetrics.debug.warning)
                          }
                        }
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
                  {trainingMetrics.timestep >= 1000 
                    ? `${(trainingMetrics.timestep / 1000).toFixed(0)}k` 
                    : trainingMetrics.timestep.toLocaleString()} / {trainingMetrics.total_timesteps >= 1000 
                    ? `${(trainingMetrics.total_timesteps / 1000).toFixed(0)}k` 
                    : trainingMetrics.total_timesteps.toLocaleString()}
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
          
          {/* Trading Metrics */}
          {(trainingMetrics.total_trades !== undefined || trainingMetrics.current_episode_trades !== undefined) && (
            <div className="border-t pt-4 mb-6">
              <h4 className="text-md font-semibold text-gray-700 mb-3">Trading Metrics</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                {/* Total Trades */}
                <div className="bg-blue-50 rounded-lg p-4">
                  <div className="text-xs text-gray-600 mb-1">Total Trades</div>
                  {trainingMetrics.total_trades !== undefined ? (
                    <div
                      key={`total-trades-${renderKey}`}
                      className="text-2xl font-bold text-blue-700"
                    >
                      {trainingMetrics.total_trades.toLocaleString()}
                    </div>
                  ) : (
                    <div className="text-2xl font-bold text-gray-500">0</div>
                  )}
                </div>
                
                {/* Winning Trades */}
                <div className="bg-green-50 rounded-lg p-4">
                  <div className="text-xs text-gray-600 mb-1">Winning Trades</div>
                  {trainingMetrics.total_winning_trades !== undefined ? (
                    <div
                      key={`winning-trades-${renderKey}`}
                      className="text-2xl font-bold text-green-700"
                    >
                      {trainingMetrics.total_winning_trades.toLocaleString()}
                    </div>
                  ) : (
                    <div className="text-2xl font-bold text-gray-500">0</div>
                  )}
                </div>
                
                {/* Losing Trades */}
                <div className="bg-red-50 rounded-lg p-4">
                  <div className="text-xs text-gray-600 mb-1">Losing Trades</div>
                  {trainingMetrics.total_losing_trades !== undefined ? (
                    <div
                      key={`losing-trades-${renderKey}`}
                      className="text-2xl font-bold text-red-700"
                    >
                      {trainingMetrics.total_losing_trades.toLocaleString()}
                    </div>
                  ) : (
                    <div className="text-2xl font-bold text-gray-500">0</div>
                  )}
                </div>
                
                {/* Overall Win Rate */}
                <div className="bg-purple-50 rounded-lg p-4">
                  <div className="text-xs text-gray-600 mb-1">Overall Win Rate</div>
                  {trainingMetrics.overall_win_rate !== undefined ? (
                    <div
                      key={`win-rate-${renderKey}`}
                      className="text-2xl font-bold text-purple-700"
                    >
                      {trainingMetrics.overall_win_rate.toFixed(1)}%
                    </div>
                  ) : (
                    <div className="text-2xl font-bold text-gray-500">0%</div>
                  )}
                </div>
              </div>
              
              {/* Current Episode Trading Metrics */}
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                <div className="bg-gray-50 rounded-lg p-3">
                  <div className="text-xs text-gray-600 mb-1">Current Episode Trades</div>
                  {trainingMetrics.current_episode_trades !== undefined ? (
                    <div className="text-lg font-semibold text-gray-800">
                      {trainingMetrics.current_episode_trades}
                    </div>
                  ) : (
                    <div className="text-lg font-semibold text-gray-500">0</div>
                  )}
                </div>
                
                <div className="bg-gray-50 rounded-lg p-3">
                  <div className="text-xs text-gray-600 mb-1">Current PnL</div>
                  {trainingMetrics.current_episode_pnl !== undefined ? (
                    <div className={`text-lg font-semibold ${trainingMetrics.current_episode_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      ${trainingMetrics.current_episode_pnl.toFixed(2)}
                    </div>
                  ) : (
                    <div className="text-lg font-semibold text-gray-500">$0.00</div>
                  )}
                </div>
                
                <div className="bg-gray-50 rounded-lg p-3">
                  <div className="text-xs text-gray-600 mb-1">Current Equity</div>
                  {trainingMetrics.current_episode_equity !== undefined ? (
                    <div className="text-lg font-semibold text-gray-800">
                      ${trainingMetrics.current_episode_equity.toFixed(2)}
                    </div>
                  ) : (
                    <div className="text-lg font-semibold text-gray-500">$0.00</div>
                  )}
                </div>
                
                <div className="bg-gray-50 rounded-lg p-3">
                  <div className="text-xs text-gray-600 mb-1">Current Win Rate</div>
                  {trainingMetrics.current_episode_win_rate !== undefined ? (
                    <div className="text-lg font-semibold text-gray-800">
                      {trainingMetrics.current_episode_win_rate.toFixed(1)}%
                    </div>
                  ) : (
                    <div className="text-lg font-semibold text-gray-500">0%</div>
                  )}
                </div>
                
                <div className="bg-gray-50 rounded-lg p-3">
                  <div className="text-xs text-gray-600 mb-1">Max Drawdown</div>
                  {trainingMetrics.current_episode_max_drawdown !== undefined ? (
                    <div className="text-lg font-semibold text-red-600">
                      {trainingMetrics.current_episode_max_drawdown.toFixed(1)}%
                    </div>
                  ) : (
                    <div className="text-lg font-semibold text-gray-500">0%</div>
                  )}
                </div>
              </div>
              
              {/* Mean Metrics (Last 10 Episodes) */}
              {(trainingMetrics.mean_pnl_10 !== undefined || trainingMetrics.mean_win_rate_10 !== undefined) && (
                <div className="mt-4 pt-4 border-t">
                  <div className="text-xs text-gray-500 mb-2">Mean (Last 10 Episodes)</div>
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <div className="text-xs text-gray-600 mb-1">Mean PnL</div>
                      {trainingMetrics.mean_pnl_10 !== undefined ? (
                        <div className={`text-sm font-semibold ${trainingMetrics.mean_pnl_10 >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          ${trainingMetrics.mean_pnl_10.toFixed(2)}
                        </div>
                      ) : (
                        <div className="text-sm font-semibold text-gray-500">$0.00</div>
                      )}
                    </div>
                    <div>
                      <div className="text-xs text-gray-600 mb-1">Mean Equity</div>
                      {trainingMetrics.mean_equity_10 !== undefined ? (
                        <div className="text-sm font-semibold text-gray-800">
                          ${trainingMetrics.mean_equity_10.toFixed(2)}
                        </div>
                      ) : (
                        <div className="text-sm font-semibold text-gray-500">$0.00</div>
                      )}
                    </div>
                    <div>
                      <div className="text-xs text-gray-600 mb-1">Mean Win Rate</div>
                      {trainingMetrics.mean_win_rate_10 !== undefined ? (
                        <div className="text-sm font-semibold text-gray-800">
                          {trainingMetrics.mean_win_rate_10.toFixed(1)}%
                        </div>
                      ) : (
                        <div className="text-sm font-semibold text-gray-500">0%</div>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
          
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

