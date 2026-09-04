import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '../styles/global.css'
import './simulator.css'
import { Simulator } from './Simulator'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Simulator />
  </StrictMode>,
)
