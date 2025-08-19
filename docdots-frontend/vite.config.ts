import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    headers: {
      // Allow this page and the Adobe viewer iframe to use the Clipboard API
      'Permissions-Policy':
        'clipboard-read=(self), clipboard-write=(self "https://acrobatservices.adobe.com")'
    }
  }
})