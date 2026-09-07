import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// RiskLattice web - local dev only. VITE_API_URL defaults to :8000 in the client.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
})