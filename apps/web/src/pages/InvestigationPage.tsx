import React, { useState } from 'react'
import { useParams } from 'react-router-dom'
import {
  getCampaign,
  getContainment,
  getGraph,
  getInvestigation,
  simulateContainment,
} from '../api/client'
import { useAsync, fmtInr, pct } from '../hooks/useAsync'
import { GraphView } from '../components/GraphView'
import { RiskBadge } from '../components/CampaignTable'
import type { SimulationResult } from '../types/api'

type Opt = {
  action_types: string[]
  fraud_containment_rate: number
  legitimate_users_affected: number
  collateral_level: string
  action_count: number
  fraud_exposure_contained?: number
}

export function InvestigationPage() {
  const { campaignId = '' } = useParams()
  const campaign = useAsync(() => getCampaign(campaignId), [campaignId])
  const containment = useAsync(() => getContainment(campaignId), [campaignId])
  const investigation = useAsync(() => getInvestigation(campaignId), [campaignId])
  const graph = useAsync(() => getGraph(campaignId), [campaignId])

  const [sim, setSim] = useState<SimulationResult | null>(null)
  const [simError, setSimError] = useState<string | null>(null)

  if (campaign.loading) return <div className="loading">Loading campaign…</div>
  if (campaign.error || !campaign.data)
    return <div className="error">{campaign.error ?? 'Campaign not found'}</div>

  const detail = campaign.data
  const containmentData = containment.data

  async function runSimulate(type: string) {
    const targetMap: Record<string, string | undefined> = {
      BLOCK_USER: detail.user_ids[0],
      REVIEW_USER: detail.user_ids[0],
      RESTRICT_DEVICE: detail.device_ids[0],
      REVIEW_DEVICE: detail.device_ids[0],
      RESTRICT_PAYMENT_INSTRUMENT: detail.payment_instrument_ids[0],
      REVIEW_PAYMENT_INSTRUMENT: detail.payment_instrument_ids[0],
      BLOCK_TRANSACTION: detail.transaction_ids[0],
      REVIEW_TRANSACTION: detail.transaction_ids[0],
    }
    const target = targetMap[type]
    if (!target) {
      setSimError(`No candidate target for ${type}`)
      return
    }
    setSimError(null)
    setSim(null)
    try {
      setSim(await simulateContainment(campaignId, type, target))
    } catch (e) {
      setSimError(e instanceof Error ? e.message : 'Simulation failed')
    }
  }

  return (
    <>
      <div className="card">
        <div style={{ display: 'flex', gap: 24, alignItems: 'center' }}>
          <span className="mono" style={{ fontSize: 16 }}>{detail.campaign_id}</span>
          <RiskBadge level={detail.risk_level} />
          <div className="coverage-grid">
            <div className="cell"><div className="k">Risk score</div><div className="v">{detail.risk_score.toFixed(1)}</div></div>
            <div className="cell"><div className="k">Confidence</div><div className="v">{detail.confidence.toFixed(2)}</div></div>
            <div className="cell"><div className="k">Exposure</div><div className="v">{fmtInr(detail.exposure)}</div></div>
            <div className="cell"><div className="k">Txns / Users</div><div className="v">{detail.transaction_count} / {detail.user_count}</div></div>
          </div>
        </div>
      </div>

      {containmentData?.recommendation === 'NO_SAFE_ACTION' ? (
        <div className="warn-box">
          <h4>NO SAFE AUTOMATED ACTION</h4>
          <div>{containmentData.reason}</div>
          <p>
            RiskLattice did not identify an intervention satisfying the
            configured collateral constraints. No execute-anyway path is offered.
          </p>
        </div>
      ) : null}

      <div className="grid-2">
        <div className="card">
          <h3>Relationship Graph</h3>
          {graph.data ? <GraphView graph={graph.data} /> : (
            <div className={graph.error ? 'error' : 'empty'}>
              {graph.error ?? 'Graph unavailable'}
            </div>
          )}
          <p style={{ color: 'var(--text-dim)' }}>
            Nodes are "associated with the campaign"; association is not proof of fraud.
          </p>
        </div>

        <div className="card">
          <h3>AI Investigation</h3>
          {investigation.error ? <div className="error">{investigation.error}</div> :
           !investigation.data ? <div className="loading">Loading…</div> : (
            <InvestigationBody inv={investigation.data} />
          )}
        </div>
      </div>

      <div className="card">
        <h3>Risk Dimensions</h3>
        {detail.risk_dimensions.map((d) => (
          <div key={d.name} className="dim-row">
            <div className="name">{d.name}</div>
            <div className="bar-bg">
              <div className="bar-fill" style={{ width: `${Math.round(d.value * 100)}%` }} />
            </div>
            <div className="value">{d.value.toFixed(2)}</div>
          </div>
        ))}
      </div>

      <ContainmentPanel containment={containmentData} onSimulate={runSimulate} />

      <div className="card" style={{ display: sim || simError ? 'block' : 'none' }}>
        <h3>Simulation Result</h3>
        {simError ? <div className="error">{simError}</div> : sim ? (
          <>
            <div className="coverage-grid">
              <div className="cell"><div className="k">Action</div><div className="v">{sim.action_type} {sim.target_id}</div></div>
              <div className="cell"><div className="k">Fraud tx affected</div><div className="v">{sim.fraud_transactions_affected}</div></div>
              <div className="cell"><div className="k">Legit tx affected</div><div className="v">{sim.legitimate_transactions_affected}</div></div>
              <div className="cell"><div className="k">Fraud amount</div><div className="v">{fmtInr(sim.fraud_amount_affected)}</div></div>
              <div className="cell"><div className="k">Containment</div><div className="v">{pct(sim.fraud_containment_rate)}</div></div>
              <div className="cell"><div className="k">Collateral</div><div className="v">{sim.collateral_level}</div></div>
            </div>
            <p className="mono" style={{ color: 'var(--warn)' }}>SIMULATION / TEST MODE — no real action executed.</p>
          </>
        ) : null}
      </div>

      <div className="card">
        <h3>Audit</h3>
        <div className="coverage-grid">
          <div className="cell"><div className="k">Evidence items</div><div className="v">{investigation.data?.evidence_count ?? '-'}</div></div>
          <div className="cell"><div className="k">Validation</div><div className="v">{investigation.data?.validation_status ?? '-'}</div></div>
          <div className="cell"><div className="k">Provider</div><div className="v">{investigation.data?.provider ?? '-'}</div></div>
          <div className="cell"><div className="k">Recommendation</div><div className="v">{containmentData?.recommendation ?? '-'}</div></div>
        </div>
      </div>
    </>
  )
}

function InvestigationBody({ inv }: { inv: {
  executive_summary: string
  why_flagged: { type: string; text: string; evidence_ids: string[] }[]
  risk_assessment: string
  uncertainty: string[]
  questions_for_reviewer: string[]
} }) {
  return (
    <>
      <p>{inv.executive_summary}</p>
      <h4>WHY FLAGGED</h4>
      {inv.why_flagged.map((f, i) => (
        <div key={i} className={`finding ${f.type}`}>
          <span className={`tag ${f.type.toLowerCase()}`}>{f.type}</span>
          <div className="text">{f.text}</div>
          <div className="refs">
            Evidence:{' '}
            {f.evidence_ids.map((eid) => (
              <span key={eid} className="evidence-id">{eid}</span>
            ))}
          </div>
        </div>
      ))}
      <h4>Risk Assessment</h4>
      <p className="mono">{inv.risk_assessment}</p>
      <h4>Uncertainty</h4>
      <ul>{inv.uncertainty.map((u, i) => <li key={i}>{u}</li>)}</ul>
      <h4>Questions for Reviewer</h4>
      <ul>{inv.questions_for_reviewer.map((q, i) => <li key={i}>{q}</li>)}</ul>
    </>
  )
}

function ContainmentOption({
  title, option, onSimulate,
}: {
  title: string
  option: Opt
  onSimulate: (type: string) => void
}) {
  const type = option.action_types[0] ?? ''
  return (
    <div
      style={{
        background: 'var(--bg-2)', border: '1px solid var(--border)',
        padding: 10, borderRadius: 6, marginBottom: 8,
        display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center',
      }}
    >
      <strong className="mono">{title}: {type}</strong>
      <span className="mono">containment {pct(option.fraud_containment_rate)}</span>
      <span className="mono">exposure {fmtInr(option.fraud_exposure_contained ?? 0)}</span>
      <span className="mono">legit users {option.legitimate_users_affected}</span>
      <span className="mono">collateral {option.collateral_level}</span>
      <button
        className="btn primary"
        disabled={!type}
        onClick={() => onSimulate(type)}
      >
        Simulate
      </button>
    </div>
  )
}

function ContainmentPanel({
  containment, onSimulate,
}: {
  containment: {
    recommendation?: string
    recommended_action?: Opt | null
    alternative_actions?: Opt[]
    reason?: string
  } | null
  onSimulate: (type: string) => void
}) {
  if (!containment)
    return <div className="card"><div className="loading">Loading containment…</div></div>
  return (
    <div className="card">
      <h3>Containment</h3>
      {containment.recommended_action ? (
        <ContainmentOption
          title="RECOMMENDED"
          option={containment.recommended_action}
          onSimulate={onSimulate}
        />
      ) : (
        <div className="warn-box">
          <h4>NO SAFE AUTOMATED ACTION</h4>
          <div>{containment.reason}</div>
        </div>
      )}
      <h4>Alternative Actions</h4>
      {(containment.alternative_actions ?? []).length
        ? containment.alternative_actions!.map((opt) => (
          <ContainmentOption
            key={opt.action_types.join('-')}
            title="Alternative"
            option={opt}
            onSimulate={onSimulate}
          />
        ))
        : <div className="empty">No alternative strategies remain after dominance pruning.</div>}
    </div>
  )
}