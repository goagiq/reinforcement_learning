import React, { useEffect, useMemo, useState } from 'react'
import { Info, Loader2, Sparkles, ThumbsUp, ThumbsDown, RefreshCcw } from 'lucide-react'
import { fetchCapabilityAnalysis, submitCapabilityFeedback } from '../utils/capabilityApi'

const ICON_SIZE = 16

function Tooltip({ message, align = 'center' }) {
  if (!message) return null
  let alignmentClass = 'left-1/2 -translate-x-1/2'
  if (align === 'left') {
    alignmentClass = 'left-0 translate-x-0'
  } else if (align === 'right') {
    alignmentClass = 'right-0 translate-x-0'
  }
  return (
    <div className={`absolute z-20 w-64 top-8 ${alignmentClass}`}>
      <div className="relative bg-gray-900 text-white text-xs rounded-lg shadow-lg p-3">
        {message}
        <div className="absolute -top-2 left-1/2 -translate-x-1/2 w-0 h-0 border-l-8 border-r-8 border-b-8 border-l-transparent border-r-transparent border-b-gray-900" />
      </div>
    </div>
  )
}

export default function CapabilityExplainer({
  capabilityId,
  context = {},
  className = '',
  align = 'right',
  variant = 'icon',
}) {
  const [analysisData, setAnalysisData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [expanded, setExpanded] = useState(false)
  const [showTooltip, setShowTooltip] = useState(false)
  const [feedbackState, setFeedbackState] = useState({ status: 'idle', lastRating: null })
  const [detailsEnabled, setDetailsEnabled] = useState(() => {
    if (typeof window === 'undefined') return false
    return localStorage.getItem('aiInsightsEnabled') === 'true'
  })
  const [tooltipsEnabled, setTooltipsEnabled] = useState(() => {
    if (typeof window === 'undefined') return false
    return localStorage.getItem('aiTooltipsEnabled') === 'true'
  })

  const anchorClass = useMemo(() => {
    if (variant === 'badge') {
      return 'inline-flex items-center gap-1 px-2 py-1 bg-indigo-50 text-indigo-700 text-xs font-semibold rounded-full hover:bg-indigo-100 transition-colors'
    }
    return 'inline-flex items-center justify-center text-indigo-600 hover:text-indigo-800 transition-colors'
  }, [variant])

  const contextKey = useMemo(() => {
    try {
      return JSON.stringify(context || {})
    } catch {
      return '{}'
    }
  }, [context])

  useEffect(() => {
    let mounted = true
    async function load() {
      if (!capabilityId || !detailsEnabled) return
      setLoading(true)
      setError(null)
      try {
        const response = await fetchCapabilityAnalysis(capabilityId, context)
        if (!mounted) return
        const resolved = response.analysis?.data || response.analysis || null
        setAnalysisData(resolved)
      } catch (err) {
        if (mounted) {
          setError(err?.response?.data?.detail || err.message || 'Failed to load AI analysis')
        }
      } finally {
        if (mounted) {
          setLoading(false)
        }
      }
    }
    load()
    return () => {
      mounted = false
    }
  }, [capabilityId, contextKey, detailsEnabled])

  useEffect(() => {
    const handleSettingsChange = () => {
      if (typeof window === 'undefined') return
      setDetailsEnabled(localStorage.getItem('aiInsightsEnabled') === 'true')
      setTooltipsEnabled(localStorage.getItem('aiTooltipsEnabled') === 'true')
    }
    window.addEventListener('storage', handleSettingsChange)
    window.addEventListener('aiSettingsUpdated', handleSettingsChange)
    return () => {
      window.removeEventListener('storage', handleSettingsChange)
      window.removeEventListener('aiSettingsUpdated', handleSettingsChange)
    }
  }, [])

  useEffect(() => {
    if (!detailsEnabled) {
      setAnalysisData(null)
      setLoading(false)
      setError(null)
      setExpanded(false)
    }
  }, [detailsEnabled])

  if (!detailsEnabled) {
    return null
  }

  const tooltipMessage = analysisData?.tooltip
  const fullAnalysis = analysisData?.analysis || ''

  async function handleFeedback(rating) {
    try {
      setFeedbackState({ status: 'submitting', lastRating: rating })
      await submitCapabilityFeedback(capabilityId, { rating })
      setFeedbackState({ status: 'submitted', lastRating: rating })
    } catch (err) {
      setFeedbackState({ status: 'error', lastRating: rating })
    }
  }

  async function handleRefresh(e) {
    e.stopPropagation()
    setLoading(true)
    setError(null)
    try {
      const response = await fetchCapabilityAnalysis(capabilityId, context, { force_refresh: true })
      const resolved = response.analysis?.data || response.analysis || null
      setAnalysisData(resolved)
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || 'Failed to refresh AI analysis')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div
      className={`relative ${className}`}
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
    >
      <button
        type="button"
        className={anchorClass}
        onClick={() => setExpanded(true)}
        aria-label="Show AI explanation"
      >
        {variant === 'badge' && <Sparkles className="w-3 h-3" />}
        {variant === 'badge' ? 'AI Insight' : <Info className="w-4 h-4" />}
      </button>
      {tooltipsEnabled && showTooltip && !loading && tooltipMessage && <Tooltip message={tooltipMessage} align={align} />}
      {tooltipsEnabled && showTooltip && loading && (
        <div className="absolute z-20 left-1/2 -translate-x-1/2 top-8 bg-gray-900 text-white text-xs rounded-lg shadow-lg p-3 flex items-center gap-2">
          <Loader2 className="w-3 h-3 animate-spin" />
          <span>Generating insight…</span>
        </div>
      )}
      {tooltipsEnabled && showTooltip && error && (
        <div className="absolute z-20 left-1/2 -translate-x-1/2 top-8 bg-red-600 text-white text-xs rounded-lg shadow-lg p-3">
          {error}
        </div>
      )}

      {expanded && (
        <div className="fixed inset-0 z-40 flex items-start justify-center bg-black/40 backdrop-blur-sm p-4">
          <div className="relative max-w-2xl w-full bg-white rounded-xl shadow-2xl border border-gray-200">
            <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100">
              <div className="flex items-center gap-2 text-indigo-700">
                <Sparkles className="w-4 h-4" />
                <span className="text-sm font-semibold">AI Analysis</span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={handleRefresh}
                  className="inline-flex items-center gap-1 text-xs text-gray-500 hover:text-indigo-600"
                >
                  <RefreshCcw className="w-3 h-3" />
                  Refresh
                </button>
                <button
                  type="button"
                  className="text-gray-400 hover:text-gray-600"
                  onClick={() => setExpanded(false)}
                >
                  ✕
                </button>
              </div>
            </div>
            <div className="px-5 py-4 max-h-[60vh] overflow-y-auto">
              {loading && (
                <div className="flex items-center gap-2 text-gray-500 text-sm">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Generating updated insight…
                </div>
              )}
              {!loading && error && (
                <div className="text-sm text-red-600 bg-red-50 border border-red-100 rounded-lg px-3 py-2">
                  {error}
                </div>
              )}
              {!loading && !error && (
                <pre className="whitespace-pre-wrap text-sm text-gray-800 leading-relaxed font-sans">{fullAnalysis}</pre>
              )}
            </div>
            <div className="px-5 py-3 border-t border-gray-100 flex items-center justify-between bg-gray-50 rounded-b-xl">
              <span className="text-xs text-gray-500">
                How helpful was this insight?
              </span>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  className={`inline-flex items-center gap-1 px-2 py-1 rounded-md border text-xs ${
                    feedbackState.lastRating === 1
                      ? 'border-green-500 text-green-600 bg-green-50'
                      : 'border-gray-200 text-gray-600 hover:border-green-500 hover:text-green-600'
                  }`}
                  onClick={() => handleFeedback(1)}
                  disabled={feedbackState.status === 'submitting'}
                >
                  <ThumbsUp className="w-3 h-3" />
                  Helpful
                </button>
                <button
                  type="button"
                  className={`inline-flex items-center gap-1 px-2 py-1 rounded-md border text-xs ${
                    feedbackState.lastRating === -1
                      ? 'border-red-500 text-red-600 bg-red-50'
                      : 'border-gray-200 text-gray-600 hover:border-red-500 hover:text-red-600'
                  }`}
                  onClick={() => handleFeedback(-1)}
                  disabled={feedbackState.status === 'submitting'}
                >
                  <ThumbsDown className="w-3 h-3" />
                  Needs work
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}


