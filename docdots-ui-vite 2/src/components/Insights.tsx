import React from 'react'
import type { InsightsResponse } from '../lib/types'

export default function Insights({ data }: { data?: InsightsResponse }) {
  if (!data) return <div className="text-sm text-muted p-6 border border-slate-200 rounded-xl bg-white">No insight generated.</div>
  return (
    <div className="space-y-3">
      <div className="bg-primary-subtle border border-primary rounded-xl p-3">
        <div className="text-primary text-xs font-semibold mb-1">KEY INSIGHT</div>
        <div className="text-sm">{data.insight}</div>
      </div>
      <div className="text-xs text-muted">Sources</div>
      <div className="space-y-2">
        {data.used_items.map((u, i) => (
          <div key={i} className="bg-white border border-slate-200 rounded-lg p-2">
            <div className="font-medium text-sm truncate">{u.document}</div>
            <div className="text-xs text-muted">p{u.page_number} • {u.section_title}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
