// Typed API client. All values come from the backend — nothing is computed here.
import type {
  AuditEvent,
  CampaignDetail,
  CampaignSummary,
  Containment,
  Graph,
  HealthResponse,
  Investigation,
  Overview,
  SimulationResult,
} from '../types/api'

const API_URL = import.meta.env?.VITE_API_URL ?? 'http://localhost:8000'

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, init)
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const body = await res.json()
      if (body?.detail) detail = String(body.detail)
    } catch {
      /* ignore */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export function getHealth(): Promise<HealthResponse> {
  return req<HealthResponse>('/health')
}

export function getOverview(): Promise<Overview> {
  return req<Overview>('/api/overview')
}

export function getCampaigns(): Promise<CampaignSummary[]> {
  return req<CampaignSummary[]>('/api/campaigns')
}

export function getCampaign(id: string): Promise<CampaignDetail> {
  return req<CampaignDetail>(`/api/campaigns/${encodeURIComponent(id)}`)
}

export function getGraph(id: string): Promise<Graph> {
  return req<Graph>(`/api/campaigns/${encodeURIComponent(id)}/graph`)
}

export function getInvestigation(id: string): Promise<Investigation> {
  return req<Investigation>(
    `/api/campaigns/${encodeURIComponent(id)}/investigation`,
  )
}

export function getContainment(id: string): Promise<Containment> {
  return req<Containment>(`/api/campaigns/${encodeURIComponent(id)}/containment`)
}

export function simulateContainment(
  id: string,
  actionType: string,
  targetId: string,
): Promise<SimulationResult> {
  return req<SimulationResult>(
    `/api/campaigns/${encodeURIComponent(id)}/containment/simulate`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action_type: actionType, target_id: targetId }),
    },
  )
}

export function getAudit(): Promise<AuditEvent[]> {
  return req<AuditEvent[]>('/api/audit')
}