import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '../styles/global.css'
import '../simulator/simulator.css'
import './dsv41.css'
import { Dsv41Simulator } from './Dsv41Simulator'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Dsv41Simulator />
  </StrictMode>,
)
