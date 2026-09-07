import React, { useState } from 'react'
import { getCampaigns } from '../api/client'
import { useAsync } from '../hooks/useAsync'
import type { CampaignSummary } from '../types/api'
import { CampaignTable } from '../components/CampaignTable'

export function CampaignsPage() {
  const { data, error, loading } = useAsync(getCampaigns, [])

  const [riskLevel, setRiskLevel] = useState('')
  const [minScore, setMinScore] = useState('')
  const [action, setAction] = useState('')

  let filtered: CampaignSummary[] = data ?? []
  if (riskLevel) filtered = filtered.filter((c) => c.risk_level === riskLevel)
  if (minScore) filtered = filtered.filter((c) => c.risk_score >= parseFloat(minScore))
  if (action) filtered = filtered.filter((c) => c.recommended_action.includes(action))

  if (loading) return <div className="loading">Loading campaigns…</div>
  if (error) return <div className="error">{error}</div>

  return (
    <>
      <div className="card">
        <h3>Filters</h3>
        <table style={{ width: '100%' }}>
          <tbody>
            <tr>
              <td>Risk level</td>
              <td>
                <select
                  aria-label="Filter by risk level"
                  value={riskLevel}
                  onChange={(e) => setRiskLevel(e.target.value)}
                >
                  <option value="">All</option>
                  <option value="LOW">LOW</option>
                  <option value="MEDIUM">MEDIUM</option>
                  <option value="HIGH">HIGH</option>
                  <option value="CRITICAL">CRITICAL</option>
                </select>
              </td>
              <td>Min score</td>
              <td>
                <input
                  aria-label="Minimum risk score"
                  type="number"
                  value={minScore}
                  onChange={(e) => setMinScore(e.target.value)}
                />
              </td>
              <td>Action contains</td>
              <td>
                <input
                  aria-label="Recommended action contains"
                  value={action}
                  onChange={(e) => setAction(e.target.value.toUpperCase())}
                />
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <div className="card">
        <h3>{filtered.length} assessed campaigns</h3>
        <CampaignTable campaigns={filtered} />
      </div>
    </>
  )
}