import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Determine if Kong Gateway should be used
// Set VITE_USE_KONG=true to use Kong (port 8300), otherwise use direct backend (port 8200)
const USE_KONG = process.env.VITE_USE_KONG === 'true' || process.env.VITE_USE_KONG === '1'
const API_TARGET = USE_KONG ? 'http://localhost:8300' : 'http://localhost:8200'
const WS_TARGET = USE_KONG ? 'ws://localhost:8300' : 'ws://localhost:8200'
const KONG_API_KEY = process.env.VITE_KONG_API_KEY || ''

console.log(`🔌 Using ${USE_KONG ? 'Kong Gateway' : 'Direct Backend'}: ${API_TARGET}`)
if (USE_KONG && KONG_API_KEY) {
  console.log('🔑 Kong API key configured')
}

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',  // Listen on all interfaces (IPv4 and IPv6)
    port: 3200,
    proxy: {
      '/api': {
        target: API_TARGET,
        changeOrigin: true,
        // Add Kong API key header if using Kong
        ...(USE_KONG && KONG_API_KEY ? {
          configure: (proxy, _options) => {
            proxy.on('proxyReq', (proxyReq, req, res) => {
              proxyReq.setHeader('apikey', KONG_API_KEY)
            })
          }
        } : {}),
        onError: (err, req, res) => {
          if (USE_KONG) {
            console.warn('Kong Gateway connection failed. Ensure Kong is running on port 8300')
          } else {
            console.warn('Direct backend connection failed. Ensure backend is running on port 8200')
          }
        }
      },
      '/ws': {
        target: WS_TARGET,
        ws: true,
        changeOrigin: true,
        // Add Kong API key header for WebSocket if using Kong
        // Note: WebSocket authentication may need to be handled differently
        ...(USE_KONG && KONG_API_KEY ? {
          configure: (proxy, _options) => {
            proxy.on('proxyReqWs', (proxyReq, req, socket, options, head) => {
              proxyReq.setHeader('apikey', KONG_API_KEY)
            })
          }
        } : {})
      }
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: true
  }
})

