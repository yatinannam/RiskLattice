import React from 'react'
import { getAudit } from '../api/client'
import { useAsync } from '../hooks/useAsync'
import { Link } from 'react-router-dom'

export function AuditPage() {
  const { data, error, loading } = useAsync(getAudit, [])
  if (loading) return <div className="loading">Loading audit trail…</div>
  if (error || !data) return <div className="error">{error ?? 'No audit data'}</div>

  return (
    <div className="card">
      <h3>Audit Trail — SIMULATED / TEST MODE</h3>
      <div className="audit-list">
        <div className="audit-head">
          <span>Timestamp</span><span>Campaign</span><span>Event</span>
          <span>Details</span><span>Validation</span>
        </div>
        {data.map((e, i) => (
          <div key={i} className="ev">
            <span>{e.timestamp === 'startup' ? 'startup' : e.timestamp.slice(0, 19)}</span>
            <span>
              {e.campaign === '-' ? '-' : <Link to={`/campaigns/${e.campaign}`}>{e.campaign}</Link>}
            </span>
            <span className="main">{e.event}</span>
            <span>{e.provider} · {e.action}</span>
            <span>{e.validation}</span>
          </div>
        ))}
      </div>
    </div>
  )
}