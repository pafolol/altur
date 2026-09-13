import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// three pages: the landing at /, the admin panel at /admin/, the latency demo at /demo/
export default defineConfig({
  // The project keeps ONE .env at the repository root; without this Vite would look for web/.env and
  // find nothing. Only VITE_-prefixed variables ever reach the browser bundle - the API keys and the
  // Twilio token in that file have no prefix, so Vite loads them for config and never ships them.
  envDir: '..',
  plugins: [react(), tailwindcss()],
  build: { rollupOptions: { input: { main: 'index.html', admin: 'admin/index.html', demo: 'demo/index.html' } } },
})
