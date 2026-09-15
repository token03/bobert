import { defineConfig, type Plugin, type PreviewServer, type ViteDevServer } from 'vite'
import babel from '@rolldown/plugin-babel'
import react, { reactCompilerPreset } from '@vitejs/plugin-react'
import { tanstackRouter } from '@tanstack/router-plugin/vite'

function searchArtifact(): Plugin {
  const serve = (server: ViteDevServer | PreviewServer) => {
    server.middlewares.use('/search.bin', (_request, response, next) => {
      response.setHeader('Content-Type', 'application/octet-stream')
      response.setHeader('Content-Encoding', 'br')
      response.setHeader('Cache-Control', 'no-cache')
      next()
    })
  }
  return {
    name: 'search-artifact',
    configureServer: serve,
    configurePreviewServer: serve,
  }
}

export default defineConfig({
  worker: { format: 'es' },
  plugins: [
    searchArtifact(),
    tanstackRouter({
      target: 'react',
      autoCodeSplitting: true,
    }),
    react(),
    babel({ presets: [reactCompilerPreset()] }),
  ],
  server: {
    host: '0.0.0.0',
    proxy: {
      '/api': 'http://127.0.0.1:8008',
    },
  },
})
