import React from 'react'
import { getOverview } from '../api/client'
import { useAsync, fmtInr, pct } from '../hooks/useAsync'
import { CampaignTable } from '../components/CampaignTable'

export function OverviewPage() {
  const { data, error, loading } = useAsync(getOverview, [])
  if (loading) return <div className="loading">Loading risk overview…</div>
  if (error || !data) return <div className="error">{error ?? 'No data'}</div>

  const dist = data.risk_distribution
  const total = Math.max(
    (dist.LOW ?? 0) + (dist.MEDIUM ?? 0) + (dist.HIGH ?? 0) + (dist.CRITICAL ?? 0),
    1,
  )

  return (
    <>
      <div className="kpi-row">
        <div className="kpi">
          <div className="label">Active Campaigns</div>
          <div className="value">{data.active_campaigns}</div>
        </div>
        <div className="kpi">
          <div className="label">High / Critical</div>
          <div className="value" style={{ color: 'var(--warn)' }}>
            {data.high_critical_campaigns}
          </div>
        </div>
        <div className="kpi">
          <div className="label">Fraud Exposure</div>
          <div className="value">{fmtInr(data.fraud_exposure)}</div>
        </div>
        <div className="kpi">
          <div className="label">Containment Rate</div>
          <div className="value">{pct(data.average_containment)}</div>
        </div>
      </div>

      <div className="card">
        <h3>Risk Distribution</h3>
        <div className="risk-dist">
          <div
            className="seg LOW"
            style={{ flex: dist.LOW ?? 0 }}
            title={`LOW ${dist.LOW ?? 0}`}
          />
          <div
            className="seg MEDIUM"
            style={{ flex: dist.MEDIUM ?? 0 }}
            title={`MEDIUM ${dist.MEDIUM ?? 0}`}
          />
          <div
            className="seg HIGH"
            style={{ flex: dist.HIGH ?? 0 }}
            title={`HIGH ${dist.HIGH ?? 0}`}
          />
          <div
            className="seg CRITICAL"
            style={{ flex: dist.CRITICAL ?? 0 }}
            title={`CRITICAL ${dist.CRITICAL ?? 0}`}
          />
        </div>
        <div className="legend">
          <span><span className="dot LOW" />LOW {dist.LOW ?? 0} ({`${Math.round(((dist.LOW ?? 0) / total) * 100)}%`})</span>
          <span><span className="dot MEDIUM" />MEDIUM {dist.MEDIUM ?? 0}</span>
          <span><span className="dot HIGH" />HIGH {dist.HIGH ?? 0}</span>
          <span><span className="dot CRITICAL" />CRITICAL {dist.CRITICAL ?? 0}</span>
        </div>
      </div>

      <div className="card">
        <h3>Active Campaigns</h3>
        <CampaignTable campaigns={data.recent_campaigns} />
      </div>
    </>
  )
}