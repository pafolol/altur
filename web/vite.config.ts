import { defineConfig, type Plugin, type ViteDevServer, type PreviewServer } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Three pages: the landing at /, the admin panel at /admin/, the latency demo at /demo/.
// One source of truth - the build inputs and the redirect below are both derived from this.
const PAGES = { main: 'index.html', admin: 'admin/index.html', demo: 'demo/index.html' }

// The sub-pages, by directory name: ['admin', 'demo'].
const SUBPAGES = Object.values(PAGES).filter((p) => p.includes('/')).map((p) => p.split('/')[0])

/**
 * `/demo` -> `/demo/`, `/admin` -> `/admin/`.
 *
 * This is a multi-page app: every page but the landing one is a directory with its own index.html, so
 * only the trailing-slash URL resolves to a file. Without this, Vite answers `/demo` with its SPA
 * history fallback, which is the ROOT index.html - so typing `/demo` silently served the LANDING PAGE,
 * with HTTP 200 and no error anywhere to say the demo had not loaded.
 *
 * A redirect rather than a rewrite, because the page must actually sit at `/demo/` for its relative
 * asset URLs to resolve. The dev server and `vite preview` get the same rule; a static host in front of
 * `dist/` normally does this itself, but do not rely on that.
 */
function mpaTrailingSlash(): Plugin {
  const install = (server: ViteDevServer | PreviewServer) => {
    // Registered from inside the hook, so it runs BEFORE Vite's internal middlewares - the html
    // fallback among them. Added after them, it would never see these paths.
    server.middlewares.use((req, res, next) => {
      const path = (req.url || '').split('?')[0]
      const name = path.replace(/^\/+/, '')
      if (!SUBPAGES.includes(name)) return next()
      res.writeHead(301, { Location: `/${name}/${(req.url || '').slice(path.length)}` })
      res.end()
    })
  }
  return {
    name: 'mpa-trailing-slash',
    configureServer: install,
    configurePreviewServer: install,
  }
}

export default defineConfig({
  // The project keeps ONE .env at the repository root; without this Vite would look for web/.env and
  // find nothing. Only VITE_-prefixed variables ever reach the browser bundle - the API keys and the
  // Twilio token in that file have no prefix, so Vite loads them for config and never ships them.
  envDir: '..',
  // 127.0.0.1 explicitly. Vite's default "localhost" bound to ::1 only here, so every URL the project
  // documents - http://127.0.0.1:5173/ - was refused while http://localhost:5173/ worked. Binding to
  // IPv4 serves both: a browser asked for "localhost" falls back to 127.0.0.1 on its own.
  server: { host: '127.0.0.1' },
  preview: { host: '127.0.0.1' },
  plugins: [react(), tailwindcss(), mpaTrailingSlash()],
  build: { rollupOptions: { input: PAGES } },
})
