// API response types — mirror app/schemas.py. No ground-truth fields.

export interface HealthResponse {
  status: string
  service: string
  version: string
  dataset: string
  mode: string
}

export interface CampaignSummary {
  campaign_id: string
  risk_score: number
  risk_level: string
  confidence: number
  transaction_count: number
  user_count: number
  device_count: number
  ip_count: number
  payment_instrument_count: number
  exposure: number
  recommended_action: string
  collateral_level: string
}

export interface RiskDimension {
  name: string
  value: number
}

export interface EvidenceItem {
  type: string
  severity: string
  description: string
  entities: string[]
  supporting_transactions: string[]
}

export interface CampaignDetail extends CampaignSummary {
  start_time: string
  end_time: string
  duration_seconds: number
  high_risk_transaction_count: number
  risk_dimensions: RiskDimension[]
  evidence: EvidenceItem[]
  transaction_ids: string[]
  user_ids: string[]
  device_ids: string[]
  ip_ids: string[]
  payment_instrument_ids: string[]
}

export interface GraphNode {
  id: string
  type: string
  degree: number
  transaction_count: number
}

export interface GraphEdge {
  source: string
  target: string
  type: string
  weight: number
}

export interface Graph {
  campaign_id: string
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export interface Finding {
  type: string // FACT | INFERENCE | UNCERTAINTY
  text: string
  evidence_ids: string[]
}

export interface Investigation {
  campaign_id: string
  executive_summary: string
  why_flagged: Finding[]
  risk_assessment: string
  recommended_action: Record<string, unknown>
  alternative_actions: Record<string, unknown>[]
  collateral_warning: string
  uncertainty: string[]
  questions_for_reviewer: string[]
  evidence_count: number
  provider: string
  validation_status: string
  evidence_hash: string
}

export interface ContainmentOption {
  action_ids: string[]
  action_types: string[]
  fraud_containment_rate: number
  fraud_exposure_contained: number
  legitimate_users_affected: number
  legitimate_transactions_affected: number
  collateral_level: string
  collateral_risk: number
  action_count: number
  total_cost: number
}

export interface Containment {
  campaign_id: string
  recommendation: string
  recommended_action: ContainmentOption | null
  alternative_actions: ContainmentOption[]
  collateral_metrics: Record<string, unknown>
  reason: string
  constraints: Record<string, unknown>
  test_mode: boolean
}

export interface SimulationResult {
  campaign_id: string
  action_type: string
  target_id: string
  fraud_transactions_affected: number
  fraud_amount_affected: number
  legitimate_transactions_affected: number
  legitimate_amount_affected: number
  users_affected: number
  fraud_containment_rate: number
  collateral_level: string
  test_mode: boolean
  message: string
}

export interface AuditEvent {
  timestamp: string
  campaign: string
  event: string
  provider: string
  validation: string
  action: string
}

export interface Overview {
  dataset: string
  active_campaigns: number
  high_critical_campaigns: number
  fraud_exposure: number
  average_containment: number
  no_safe_action_count: number
  risk_distribution: Record<string, number>
  recent_campaigns: CampaignSummary[]
}