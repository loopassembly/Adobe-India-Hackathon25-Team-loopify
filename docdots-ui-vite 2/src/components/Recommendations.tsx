import React from 'react'
import type { RecommendationItem } from '../lib/types'

export default function Recommendations({
  items, onJump
}: { items: RecommendationItem[], onJump: (doc: string, page: number) => void }) {
  return (
    <div className="space-y-3">
      {items.map((it, idx) => (
        <div key={idx} className="bg-white border border-slate-200 rounded-xl p-3">
          <div className="flex items-center justify-between gap-2">
            <div className="font-medium truncate">{it.document}</div>
            <button className="btn btn-secondary h-8 px-3" onClick={() => onJump(it.document, it.page_number)}>Jump</button>
          </div>
          <div className="text-xs text-muted mt-0.5">p{it.page_number} • {(it.score*100).toFixed(0)}%</div>
          <div className="text-sm mt-1 line-clamp-2">{it.snippet}</div>
        </div>
      ))}
      {items.length === 0 && (
        <div className="text-sm text-muted p-6 border border-slate-200 rounded-xl bg-white">No related content yet.</div>
      )}
    </div>
  )
}
