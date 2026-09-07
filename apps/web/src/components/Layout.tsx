import React, { useEffect, useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { getHealth } from '../api/client'

const NAV = [
  { to: '/', label: 'Risk Overview', end: true },
  { to: '/campaigns', label: 'Campaigns' },
  { to: '/audit', label: 'Audit Trail' },
]

export function Layout() {
  const [dataset, setDataset] = useState('synthetic')
  const [status, setStatus] = useState('Loading…')

  useEffect(() => {
    getHealth()
      .then((h) => {
        setDataset(h.dataset)
        setStatus(h.status)
      })
      .catch(() => setStatus('API unreachable'))
  }, [])

  return (
    <div className="app-shell">
      <nav className="sidebar" aria-label="RiskLattice navigation">
        <div className="brand">
          Risk<span>Lattice</span>
        </div>
        {NAV.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end ?? false}
            className="nav-link"
          >
            {item.label}
          </NavLink>
        ))}
      </nav>
      <main className="main">
        <div className="topbar">
          <div className="title">Fraud Containment Console</div>
          <div className="tags">
            <span className="tag test">Test Mode</span>
            <span className="tag dataset">Dataset: {dataset}</span>
            <span className="tag ok">{status}</span>
          </div>
        </div>
        <Outlet />
      </main>
    </div>
  )
}