import React, { useState, useEffect } from 'react'
import SetupWizard from './components/SetupWizard'
import Dashboard from './components/Dashboard'
import { CheckCircle, XCircle, Loader } from 'lucide-react'

function App() {
  const [setupComplete, setSetupComplete] = useState(false)
  const [checkingSetup, setCheckingSetup] = useState(true)
  const [setupStatus, setSetupStatus] = useState(null)

  useEffect(() => {
    checkSetup()
  }, [])

  const checkSetup = async () => {
    try {
      const response = await fetch('/api/setup/check')
      
      // Check if response is OK
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      
      // Check if response has content
      const text = await response.text()
      if (!text || text.trim() === '') {
        throw new Error('Empty response from server')
      }
      
      // Parse JSON
      let data
      try {
        data = JSON.parse(text)
      } catch (parseError) {
        console.error('Failed to parse JSON:', parseError, 'Response text:', text)
        throw new Error('Invalid JSON response from server')
      }
      
      setSetupStatus(data)
      setSetupComplete(data.ready)
    } catch (error) {
      console.error('Failed to check setup:', error)
      setSetupStatus({ 
        ready: false, 
        issues: [`Failed to connect to API server: ${error.message}. Make sure the backend is running on port 8200.`] 
      })
    } finally {
      setCheckingSetup(false)
    }
  }

  if (checkingSetup) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <Loader className="w-12 h-12 animate-spin mx-auto text-primary-600 mb-4" />
          <p className="text-gray-600">Checking setup...</p>
        </div>
      </div>
    )
  }

  if (!setupComplete) {
    return (
      <div className="min-h-screen p-8">
        <div className="max-w-6xl mx-auto">
          <div className="bg-white rounded-lg shadow-xl p-8 mb-6">
            <div className="flex items-center gap-4 mb-6">
              <h1 className="text-4xl font-bold text-gray-800">NT8 RL Trading System</h1>
              {setupStatus?.ready ? (
                <CheckCircle className="w-8 h-8 text-green-500" />
              ) : (
                <XCircle className="w-8 h-8 text-red-500" />
              )}
            </div>
            <p className="text-gray-600 mb-8">
              Welcome! Let's get your trading system set up and running.
            </p>
          </div>
          <SetupWizard 
            setupStatus={setupStatus} 
            onSetupComplete={checkSetup}
          />
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen">
      <Dashboard onSetupChange={checkSetup} />
    </div>
  )
}

export default App

