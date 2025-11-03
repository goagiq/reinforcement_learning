/**
 * Utility functions for model selection
 */

/**
 * Get the default model path from settings, or find the best match
 * @param {Array} models - Array of model objects
 * @param {string} preferredType - 'trained' or 'ollama' or null for any
 * @returns {string|null} - Model path or null
 */
export function getDefaultModel(models, preferredType = null) {
  if (!models || models.length === 0) return null

  // First, check if user has set a default model in settings
  const savedDefault = localStorage.getItem('defaultModel')
  if (savedDefault) {
    const savedModel = models.find(m => m.path === savedDefault)
    if (savedModel) {
      // Check if it matches the preferred type (if specified)
      if (!preferredType || savedModel.type === preferredType || (!savedModel.type && preferredType === 'trained')) {
        return savedModel.path
      }
    }
  }

  // Filter by preferred type if specified
  let availableModels = models
  if (preferredType === 'trained') {
    availableModels = models.filter(m => m.type === 'trained' || !m.type)
  } else if (preferredType === 'ollama') {
    availableModels = models.filter(m => m.type === 'ollama')
  }

  if (availableModels.length === 0) return null

  // Look for deepseek-r1:8b specifically
  const deepseekR1Model = availableModels.find(m => 
    m.name.toLowerCase().includes('deepseek-r1:8b') || 
    m.path.toLowerCase().includes('deepseek-r1:8b') ||
    m.name.toLowerCase() === 'deepseek-r1:8b' ||
    m.path.toLowerCase() === 'deepseek-r1:8b'
  )

  if (deepseekR1Model) return deepseekR1Model.path

  // Look for any deepseek model
  const deepseekModel = availableModels.find(m => 
    m.name.toLowerCase().includes('deepseek') || 
    m.path.toLowerCase().includes('deepseek')
  )

  if (deepseekModel) return deepseekModel.path

  // Fall back to first available model
  return availableModels[0].path
}

/**
 * Sort models with default model first
 * @param {Array} models - Array of model objects
 * @param {string} preferredType - 'trained' or 'ollama' or null for any
 * @returns {Array} - Sorted array of models with default first
 */
export function sortModelsWithDefaultFirst(models, preferredType = null) {
  if (!models || models.length === 0) return []
  
  const defaultModelPath = getDefaultModel(models, preferredType)
  if (!defaultModelPath) return models
  
  // Separate default model from others
  const defaultModel = models.find(m => m.path === defaultModelPath)
  const otherModels = models.filter(m => m.path !== defaultModelPath)
  
  // Return default first, then others
  return defaultModel ? [defaultModel, ...otherModels] : models
}

/**
 * Check if a model is the default model
 * @param {Object} model - Model object
 * @returns {boolean} - True if this is the default model
 */
export function isDefaultModel(model) {
  if (!model) return false
  const savedDefault = localStorage.getItem('defaultModel')
  return savedDefault && model.path === savedDefault
}

