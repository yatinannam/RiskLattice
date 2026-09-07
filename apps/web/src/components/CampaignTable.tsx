import React from 'react'
import type { CampaignSummary } from '../types/api'
import { Link } from 'react-router-dom'
import { fmtInr } from '../hooks/useAsync'

export function RiskBadge({ level }: { level: string }) {
  return <span className={`risk-badge ${level}`}>{level}</span>
}

export function CampaignTable({ campaigns }: { campaigns: CampaignSummary[] }) {
  if (!campaigns.length) {
    return <div className="empty">No campaigns match the current selection.</div>
  }
  return (
    <table className="data">
      <thead>
        <tr>
          <th>Campaign</th>
          <th>Risk</th>
          <th>Score</th>
          <th>Txns</th>
          <th>Users</th>
          <th>Exposure</th>
          <th>Recommended Action</th>
        </tr>
      </thead>
      <tbody>
        {campaigns.map((c) => (
          <tr
            key={c.campaign_id}
            className="clickable"
            onClick={() => {
              window.location.href = `/campaigns/${c.campaign_id}`
            }}
          >
            <td>
              <Link to={`/campaigns/${c.campaign_id}`}>{c.campaign_id}</Link>
            </td>
            <td>
              <RiskBadge level={c.risk_level} />
            </td>
            <td className="mono">{c.risk_score.toFixed(1)}</td>
            <td className="mono">{c.transaction_count}</td>
            <td className="mono">{c.user_count}</td>
            <td className="mono">{fmtInr(c.exposure)}</td>
            <td className="mono">{c.recommended_action}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}