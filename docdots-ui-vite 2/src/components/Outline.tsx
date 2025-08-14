import React from 'react'
import type { OutlineItem } from '../lib/types'

export default function Outline({
  items, onJump
}: { items: OutlineItem[], onJump: (page: number) => void }) {
  return (
    <div className="card p-3 mt-3">
      <div className="flex items-center justify-between">
        <div className="font-semibold">Outline</div>
        <div className="text-xs text-muted">{items.length} sections</div>
      </div>
      <div className="mt-2 space-y-2 max-h-[48vh] overflow-auto pr-1">
        {items.length === 0 && (
          <div className="text-sm text-muted py-6">No outline detected.</div>
        )}
        {items.map((it, idx) => (
          <div key={idx} className="flex items-center justify-between bg-white border border-slate-200 rounded-lg px-2 py-2">
            <div className="flex items-center gap-2 min-w-0">
              <span className="badge">{it.level}</span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-100 border border-slate-200">p{it.page}</span>
              <div className="truncate">{it.text}</div>
            </div>
            <button className="btn btn-secondary h-8 px-3" onClick={() => onJump(it.page)}>Jump</button>
          </div>
        ))}
      </div>
    </div>
  )
}
