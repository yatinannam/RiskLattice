import React from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import './styles.css'
import { Layout } from './components/Layout'
import { OverviewPage } from './pages/OverviewPage'
import { CampaignsPage } from './pages/CampaignsPage'
import { InvestigationPage } from './pages/InvestigationPage'
import { AuditPage } from './pages/AuditPage'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<OverviewPage />} />
          <Route path="/campaigns" element={<CampaignsPage />} />
          <Route path="/campaigns/:campaignId" element={<InvestigationPage />} />
          <Route path="/audit" element={<AuditPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}