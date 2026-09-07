import React from 'react'
import type { Graph, GraphNode } from '../types/api'

const TYPE_COLORS: Record<string, string> = {
  USER: '#5aa2f8',
  DEVICE: '#d78be8',
  IP: '#3ecf8e',
  PAYMENT_INSTRUMENT: '#f0b35c',
  TRANSACTION: '#e0e6f0',
  MERCHANT: '#9aa5b8',
}

// Simple deterministic radial layout to avoid a heavy graph library.
function layout(nodes: GraphNode[]): Record<string, [number, number]> {
  const pos: Record<string, [number, number]> = {}
  const byType: Record<string, GraphNode[]> = {}
  for (const n of nodes) byType[n.type] = (byType[n.type] ?? []).concat(n)
  const W = 440
  const H = 400
  let offset = 0
  Object.entries(byType).forEach(([type, list]) => {
    const r = type === 'TRANSACTION' ? 40 : 130
    const cy = type === 'TRANSACTION' ? H / 2 : H / 2
    list.forEach((n, i) => {
      const angle = offset + (i / Math.max(list.length, 1)) * 2 * Math.PI
      const cx = W / 2 + r * Math.cos(angle)
      const cy2 = cy + r * 0.6 * Math.sin(angle)
      pos[n.id] = [Math.round(cx), Math.round(cy2)]
    })
    offset += 0.4
  })
  return pos
}

export function GraphView({ graph }: { graph: Graph }) {
  if (!graph.nodes.length) {
    return <div className="empty">Graph unavailable for this campaign.</div>
  }
  const pos = layout(graph.nodes)
  const selected = new Set<string>()

  return (
    <svg
      className="node-svg"
      viewBox="0 0 440 400"
      role="img"
      aria-label="Campaign relationship graph. Nodes represent users, devices, IPs and payment instruments; edges show relationships."
    >
      {graph.edges.map((e, i) => (
        <line
          key={`e${i}`}
          x1={pos[e.source]?.[0] ?? 0}
          y1={pos[e.source]?.[1] ?? 0}
          x2={pos[e.target]?.[0] ?? 0}
          y2={pos[e.target]?.[1] ?? 0}
          stroke="#2f3a4f"
          strokeWidth={Math.min(1 + Math.log(e.weight + 1), 4)}
        />
      ))}
      {graph.nodes.map((n) => (
        <g key={n.id}>
          <circle
            cx={pos[n.id][0]}
            cy={pos[n.id][1]}
            r={n.type === 'TRANSACTION' ? 6 : 9}
            fill={TYPE_COLORS[n.type] ?? '#9aa5b8'}
            opacity={selected.size && !selected.has(n.id) ? 0.25 : 1}
            aria-label={`${n.type} ${n.id}`}
          />
          <title>{`${n.id} · ${n.type} · degree ${n.degree}`}</title>
          <text
            x={pos[n.id][0] + 10}
            y={pos[n.id][1] + 3}
            fontSize="8"
            fill={TYPE_COLORS[n.type] ?? '#9aa5b8'}
          >
            {n.id}
          </text>
        </g>
      ))}
    </svg>
  )
}

export function nodeSummary(nodes: GraphNode[], nodeId: string) {
  const node = nodes.find((n) => n.id === nodeId)
  if (!node) return null
  return {
    id: node.id,
    type: node.type,
    degree: node.degree,
    transaction_count: node.transaction_count,
  }
}