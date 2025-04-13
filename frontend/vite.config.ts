import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0', // Allow access from any host
    port: 53940, // Use one of the assigned ports
    strictPort: true, // Ensure this port is used
    hmr: {
        clientPort: 53940 // Specify HMR client port if needed behind proxy/docker
    },
    watch: {
      usePolling: true, // Necessary in some containerized environments
    },
    cors: true, // Enable CORS
    // Allow connections from any host (useful in containerized/proxied setups)
    // Note: Vite 5 uses `server.host = true` or `server.host = '0.0.0.0'`
    // `allowedHosts` might be needed for older versions or specific proxy setups
    // allowedHosts: ['*'], // Uncomment if host: '0.0.0.0' isn't sufficient
  }
})
