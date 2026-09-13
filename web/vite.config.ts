import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// three pages: the landing at /, the admin panel at /admin/, the latency demo at /demo/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: { rollupOptions: { input: { main: 'index.html', admin: 'admin/index.html', demo: 'demo/index.html' } } },
})
