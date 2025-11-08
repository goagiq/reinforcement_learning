import React, { useState, useEffect } from 'react'
import { CheckCircle, XCircle, Upload, Download, Loader, AlertCircle, ArrowRight } from 'lucide-react'
import axios from 'axios'

const SetupWizard = ({ setupStatus, onSetupComplete }) => {
  const [currentStep, setCurrentStep] = useState(1)
  const [messages, setMessages] = useState([])
  const [ws, setWs] = useState(null)
  const [installingDeps, setInstallingDeps] = useState(false)
  const [uploadingFiles, setUploadingFiles] = useState(false)

  useEffect(() => {
    // Connect to WebSocket for real-time updates
    const websocket = new WebSocket(`ws://${window.location.hostname}:8200/ws`)
    
    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data)
      setMessages(prev => [...prev, data])
    }

    websocket.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    setWs(websocket)

    return () => {
      websocket.close()
    }
  }, [])

  const steps = [
    { id: 1, title: 'Environment Setup', description: 'Install dependencies' },
    { id: 2, title: 'Upload Data', description: 'Upload historical market data' },
    { id: 3, title: 'Train Model', description: 'Train your RL agent' },
    { id: 4, title: 'Backtest', description: 'Validate performance' },
    { id: 5, title: 'Paper Trading', description: 'Test with live data' },
    { id: 6, title: 'Go Live', description: 'Start automated trading' }
  ]

  const handleInstallDependencies = async () => {
    setInstallingDeps(true)
    try {
      await axios.post('/api/setup/install-dependencies')
      setMessages(prev => [...prev, { type: 'info', message: 'Installation started...' }])
    } catch (error) {
      setMessages(prev => [...prev, { type: 'error', message: `Installation failed: ${error.message}` }])
    }
  }

  const handleFileUpload = async (event) => {
    const files = Array.from(event.target.files)
    if (files.length === 0) return

    setUploadingFiles(true)
    const formData = new FormData()
    files.forEach(file => formData.append('files', file))

    try {
      const response = await axios.post('/api/data/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setMessages(prev => [...prev, { 
        type: 'success', 
        message: `Uploaded ${files.length} file(s) successfully` 
      }])
      // Move to next step after upload
      setTimeout(() => setCurrentStep(3), 1000)
    } catch (error) {
      setMessages(prev => [...prev, { 
        type: 'error', 
        message: `Upload failed: ${error.message}` 
      }])
    } finally {
      setUploadingFiles(false)
    }
  }

  const isStepComplete = (stepId) => {
    if (stepId === 1) {
      return setupStatus?.dependencies_installed && setupStatus?.venv_exists
    }
    if (stepId === 2) {
      return setupStatus?.data_directory_exists
    }
    return false
  }

  const canProceedToStep = (stepId) => {
    if (stepId === 1) return true
    if (stepId === 2) return isStepComplete(1)
    if (stepId === 3) return isStepComplete(2)
    return false
  }

  return (
    <div className="bg-white rounded-lg shadow-xl p-8">
      {/* Progress Steps */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          {steps.map((step, index) => (
            <React.Fragment key={step.id}>
              <div className="flex flex-col items-center flex-1">
                <div className={`
                  w-12 h-12 rounded-full flex items-center justify-center mb-2
                  ${currentStep === step.id ? 'bg-primary-600 text-white' : ''}
                  ${currentStep > step.id ? 'bg-green-500 text-white' : 'bg-gray-200 text-gray-600'}
                  ${!canProceedToStep(step.id) ? 'opacity-50' : ''}
                `}>
                  {currentStep > step.id ? (
                    <CheckCircle className="w-6 h-6" />
                  ) : (
                    <span>{step.id}</span>
                  )}
                </div>
                <div className="text-center">
                  <div className={`font-semibold text-sm ${currentStep === step.id ? 'text-primary-600' : 'text-gray-600'}`}>
                    {step.title}
                  </div>
                  <div className="text-xs text-gray-500">{step.description}</div>
                </div>
              </div>
              {index < steps.length - 1 && (
                <div className={`
                  h-1 flex-1 mx-2 mb-6
                  ${currentStep > step.id ? 'bg-green-500' : 'bg-gray-200'}
                `} />
              )}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Step Content */}
      <div className="border-t pt-8">
        {currentStep === 1 && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-gray-800">Environment Setup</h2>
            
            <div className="space-y-4">
              <div className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg">
                {setupStatus?.venv_exists ? (
                  <CheckCircle className="w-5 h-5 text-green-500" />
                ) : (
                  <XCircle className="w-5 h-5 text-red-500" />
                )}
                <span className="flex-1">
                  Virtual Environment
                  {setupStatus?.venv_type && (
                    <span className="text-xs text-gray-500 ml-2">({setupStatus.venv_type})</span>
                  )}
                </span>
                {!setupStatus?.venv_exists && (
                  <span className="text-sm text-gray-500">
                    Run: {setupStatus?.venv_message || "python -m venv .venv"}
                  </span>
                )}
              </div>

              <div className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg">
                {setupStatus?.dependencies_installed ? (
                  <CheckCircle className="w-5 h-5 text-green-500" />
                ) : (
                  <XCircle className="w-5 h-5 text-red-500" />
                )}
                <span className="flex-1">Python Dependencies</span>
                {!setupStatus?.dependencies_installed && (
                  <button
                    onClick={handleInstallDependencies}
                    disabled={installingDeps}
                    className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 flex items-center gap-2"
                  >
                    {installingDeps ? (
                      <>
                        <Loader className="w-4 h-4 animate-spin" />
                        Installing...
                      </>
                    ) : (
                      <>
                        <Download className="w-4 h-4" />
                        Install Dependencies
                      </>
                    )}
                  </button>
                )}
              </div>

              <div className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg">
                {setupStatus?.config_exists ? (
                  <CheckCircle className="w-5 h-5 text-green-500" />
                ) : (
                  <XCircle className="w-5 h-5 text-red-500" />
                )}
                <span className="flex-1">Configuration File</span>
              </div>
            </div>

            {setupStatus?.issues && setupStatus.issues.length > 0 && (
              <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                <div className="flex items-start gap-2">
                  <AlertCircle className="w-5 h-5 text-yellow-600 mt-0.5" />
                  <div>
                    <div className="font-semibold text-yellow-800 mb-2">Issues Found:</div>
                    <ul className="list-disc list-inside space-y-1 text-sm text-yellow-700">
                      {setupStatus.issues.map((issue, idx) => (
                        <li key={idx}>{issue}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            )}

            <div className="flex justify-end">
              <button
                onClick={() => setCurrentStep(2)}
                disabled={!isStepComplete(1)}
                className="px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                Next Step
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}

        {currentStep === 2 && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-gray-800">Upload Historical Data</h2>
            <p className="text-gray-600">
              Upload your historical market data files (CSV format). 
              Files should be named: ES_1min.csv, ES_5min.csv, ES_15min.csv
            </p>

            <div className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center hover:border-primary-500 transition-colors">
              <input
                type="file"
                id="file-upload"
                multiple
                accept=".csv"
                onChange={handleFileUpload}
                className="hidden"
                disabled={uploadingFiles}
              />
              <label
                htmlFor="file-upload"
                className="cursor-pointer flex flex-col items-center gap-4"
              >
                {uploadingFiles ? (
                  <>
                    <Loader className="w-12 h-12 text-primary-600 animate-spin" />
                    <span className="text-gray-600">Uploading files...</span>
                  </>
                ) : (
                  <>
                    <Upload className="w-12 h-12 text-primary-600" />
                    <div>
                      <span className="text-primary-600 font-semibold">Click to upload</span>
                      <span className="text-gray-600"> or drag and drop</span>
                    </div>
                    <span className="text-sm text-gray-500">CSV files only</span>
                  </>
                )}
              </label>
            </div>

            <div className="flex justify-between">
              <button
                onClick={() => setCurrentStep(1)}
                className="px-6 py-3 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300"
              >
                Previous
              </button>
              <button
                onClick={() => setCurrentStep(3)}
                disabled={!isStepComplete(2)}
                className="px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                Next Step
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}

        {currentStep >= 3 && (
          <div className="text-center py-12">
            <h2 className="text-2xl font-bold text-gray-800 mb-4">Setup Complete!</h2>
            <p className="text-gray-600 mb-6">
              Basic setup is complete. You can now proceed to the dashboard to:
            </p>
            <ul className="list-disc list-inside space-y-2 text-left max-w-md mx-auto text-gray-600 mb-6">
              <li>Train your RL model</li>
              <li>Run backtests</li>
              <li>Start paper trading</li>
              <li>Monitor performance</li>
            </ul>
            <button
              onClick={onSetupComplete}
              className="px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
            >
              Go to Dashboard
            </button>
          </div>
        )}
      </div>

      {/* Message Log */}
      {messages.length > 0 && (
        <div className="mt-8 border-t pt-8">
          <h3 className="font-semibold mb-4">Activity Log</h3>
          <div className="bg-gray-50 rounded-lg p-4 max-h-48 overflow-y-auto space-y-2">
            {messages.slice(-10).map((msg, idx) => (
              <div
                key={idx}
                className={`text-sm p-2 rounded ${
                  msg.type === 'error' ? 'bg-red-50 text-red-700' :
                  msg.type === 'success' ? 'bg-green-50 text-green-700' :
                  'bg-blue-50 text-blue-700'
                }`}
              >
                {msg.message || JSON.stringify(msg)}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default SetupWizard

