import React, { useState, useEffect } from 'react'
import { X, Save, Brain, CheckCircle } from 'lucide-react'

const SettingsPanel = ({ isOpen, onClose, models = [] }) => {
  const [defaultModel, setDefaultModel] = useState('')
  const [nt8DataPath, setNt8DataPath] = useState('')
  const [performanceMode, setPerformanceMode] = useState('quiet')
  const [turboTrainingMode, setTurboTrainingMode] = useState(false)
  const [autoRetrainEnabled, setAutoRetrainEnabled] = useState(true)
  const [loading, setLoading] = useState(false)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    if (isOpen) {
      // Load saved settings from localStorage
      const savedDefault = localStorage.getItem('defaultModel') || ''
      const savedNt8Path = localStorage.getItem('nt8DataPath') || ''
      setDefaultModel(savedDefault)
      setNt8DataPath(savedNt8Path)
      setSaved(false)
      
      // Also try to load from backend settings.json
      fetch('/api/settings/get')
        .then(res => res.json())
        .then(data => {
          if (data.nt8_data_path && !savedNt8Path) {
            setNt8DataPath(data.nt8_data_path)
          }
          if (data.performance_mode) {
            setPerformanceMode(data.performance_mode)
          }
          if (data.turbo_training_mode !== undefined) {
            setTurboTrainingMode(data.turbo_training_mode)
          }
          if (data.auto_retrain_enabled !== undefined) {
            setAutoRetrainEnabled(data.auto_retrain_enabled)
          }
        })
        .catch(() => {
          // Ignore if endpoint doesn't exist yet
        })
    }
  }, [isOpen])

  const handleSave = async () => {
    // Save to localStorage
    localStorage.setItem('defaultModel', defaultModel)
    localStorage.setItem('nt8DataPath', nt8DataPath)
    
    // Save settings to backend
    try {
      const settingsPayload = { 
        nt8_data_path: nt8DataPath || null,
        performance_mode: performanceMode,
        turbo_training_mode: turboTrainingMode,
        auto_retrain_enabled: autoRetrainEnabled
      }
      
      console.log('[SettingsPanel] Saving settings:', settingsPayload)
      
      const response = await fetch('/api/settings/set', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settingsPayload)
      })
      
      const result = await response.json()
      console.log('[SettingsPanel] Settings save response:', result)
      
      if (!response.ok) {
        console.error('[SettingsPanel] Failed to save settings:', result)
      }
    } catch (error) {
      console.error('Could not save settings to backend:', error)
    }
    
    // Dispatch custom event to notify other components in the same window
    window.dispatchEvent(new Event('defaultModelChanged'))
    setSaved(true)
    setTimeout(() => {
      setSaved(false)
      onClose()
    }, 1000)
  }

  if (!isOpen) return null

  // Get available models for selection
  const trainedModels = models.filter(m => m.type === 'trained' || !m.type)
  const ollamaModels = models.filter(m => m.type === 'ollama')

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        <div className="sticky top-0 bg-white border-b px-6 py-4 flex items-center justify-between">
          <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
            <Brain className="w-6 h-6" />
            Settings
          </h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        <div className="p-6 space-y-6">
          {/* NT8 Data Folder Path */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              NT8 Data Folder Path (Optional)
            </label>
            <input
              type="text"
              value={nt8DataPath}
              onChange={(e) => setNt8DataPath(e.target.value)}
              placeholder="e.g., \\\\server\\share\\NT8Export or C:\\Users\\...\\NinjaTrader 8\\Export"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent font-mono text-sm"
            />
            <p className="text-sm text-gray-500 mt-2">
              Map directly to NT8's data export folder. Supports UNC paths (\\\\server\\share) and local paths.
              Leave empty to use local data/raw folder only.
            </p>
            {nt8DataPath && (
              <p className="text-sm text-blue-600 mt-1">
                Current path: {nt8DataPath}
              </p>
            )}
          </div>

          {/* Performance Mode Toggle */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Training Performance Mode
            </label>
            <div className="flex items-center gap-4">
              <label className="flex items-center cursor-pointer">
                <input
                  type="radio"
                  name="performanceMode"
                  value="quiet"
                  checked={performanceMode === 'quiet'}
                  onChange={(e) => setPerformanceMode(e.target.value)}
                  className="w-4 h-4 text-primary-600 focus:ring-primary-500"
                />
                <span className="ml-2 text-sm text-gray-700">Quiet (Resource-Friendly)</span>
              </label>
              <label className="flex items-center cursor-pointer">
                <input
                  type="radio"
                  name="performanceMode"
                  value="performance"
                  checked={performanceMode === 'performance'}
                  onChange={(e) => setPerformanceMode(e.target.value)}
                  className="w-4 h-4 text-primary-600 focus:ring-primary-500"
                />
                <span className="ml-2 text-sm text-gray-700">Performance (Faster Training)</span>
              </label>
            </div>
            <p className="text-sm text-gray-500 mt-2">
              {turboTrainingMode 
                ? "⚠️ Turbo mode is enabled - Performance mode setting is overridden"
                : performanceMode === 'quiet' 
                  ? "✅ Quiet mode: Uses configured batch size and epochs. Recommended for daytime use when you're using your computer."
                  : "🚀 Performance mode: Doubles batch size and increases epochs for faster training. Use when you're away (e.g., at night)."
              }
            </p>
          </div>

          {/* Turbo Training Mode Toggle */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Turbo Training Mode
            </label>
            <div className="flex items-center gap-3">
              <input
                type="checkbox"
                id="turboTraining"
                checked={turboTrainingMode}
                onChange={(e) => setTurboTrainingMode(e.target.checked)}
                className="w-5 h-5 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
              />
              <label htmlFor="turboTraining" className="text-sm text-gray-700 cursor-pointer">
                Enable Turbo Mode (Maximum GPU Utilization)
              </label>
            </div>
            <p className="text-sm text-gray-500 mt-2">
              {turboTrainingMode 
                ? "🔥 Turbo mode: 4x batch size, 2x epochs for maximum GPU utilization. Perfect for overnight training when GPU resources are available. Overrides Performance mode."
                : "💤 Turbo mode: Disabled. Use Performance mode for faster training."
              }
            </p>
            {turboTrainingMode && (
              <div className="mt-2 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                <p className="text-sm text-yellow-800">
                  ⚠️ <strong>Note:</strong> Turbo mode maximizes GPU usage. Ensure your GPU has sufficient VRAM and cooling. Monitor GPU temperature during extended training sessions.
                </p>
              </div>
            )}
          </div>

          {/* Auto-Retrain Toggle */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Automatic Retraining
            </label>
            <div className="flex items-center gap-3">
              <input
                type="checkbox"
                id="autoRetrain"
                checked={autoRetrainEnabled}
                onChange={(e) => setAutoRetrainEnabled(e.target.checked)}
                className="w-5 h-5 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
              />
              <label htmlFor="autoRetrain" className="text-sm text-gray-700 cursor-pointer">
                Automatically retrain when new data is detected
              </label>
            </div>
            <p className="text-sm text-gray-500 mt-2">
              {autoRetrainEnabled 
                ? "✅ Auto-retrain: Will detect new CSV/TXT files in NT8 export folder and trigger retraining. Won't interrupt ongoing training."
                : "❌ Auto-retrain: Disabled. No automatic retraining will occur."
              }
            </p>
          </div>

          {/* Default Model Selection */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Default Model (used across Trading, Backtest, and Training)
            </label>
            <select
              value={defaultModel}
              onChange={(e) => setDefaultModel(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            >
              <option value="">-- Auto-select first available --</option>
              {trainedModels.length > 0 && (
                <optgroup label="Trained RL Models">
                  {trainedModels.map((model, idx) => (
                    <option key={idx} value={model.path}>
                      {model.name}
                    </option>
                  ))}
                </optgroup>
              )}
              {ollamaModels.length > 0 && (
                <optgroup label="Ollama Models (for reasoning)">
                  {ollamaModels.map((model, idx) => (
                    <option key={`ollama-${idx}`} value={model.path}>
                      {model.name}
                    </option>
                  ))}
                </optgroup>
              )}
            </select>
            <p className="text-sm text-gray-500 mt-2">
              This model will be automatically selected in Trading, Backtest, and Training panels.
              Leave empty to use the first available model.
            </p>
            {defaultModel && (
              <p className="text-sm text-blue-600 mt-2">
                Current default: {models.find(m => m.path === defaultModel)?.name || defaultModel}
              </p>
            )}
          </div>

          {/* Save Button */}
          <div className="flex justify-end gap-4 pt-4 border-t">
            <button
              onClick={onClose}
              className="px-4 py-2 text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-50"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={loading}
              className="px-6 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {saved ? (
                <>
                  <CheckCircle className="w-5 h-5" />
                  Saved!
                </>
              ) : (
                <>
                  <Save className="w-5 h-5" />
                  Save Settings
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default SettingsPanel
