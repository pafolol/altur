import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './demo.css'
import { Demo } from './Demo'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Demo />
  </StrictMode>,
)
