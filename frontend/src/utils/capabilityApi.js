import axios from 'axios'

const locale = typeof navigator !== 'undefined' ? navigator.language || 'en-US' : 'en-US'

export async function fetchCapabilityAnalysis(capabilityId, context = {}, options = {}) {
  const payload = {
    capability_id: capabilityId,
    locale,
    context,
    ...options,
  }

  const response = await axios.post('/api/ai/capabilities/generate', payload)
  return response.data
}

export async function submitCapabilityFeedback(capabilityId, { rating, comment, source = 'frontend', userId }) {
  const payload = {
    capability_id: capabilityId,
    rating,
    comment,
    source,
  }
  if (userId) {
    payload.user_id = userId
  }
  payload.locale = locale

  await axios.post('/api/ai/capabilities/feedback', payload)
}


