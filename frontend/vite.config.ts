import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Dev proxy: the api client fetches relative "/api/..." paths; without
    // this every dev fetch silently returned the SPA shell (Lab 11 review
    // Critical). SSE passes through untouched.
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
