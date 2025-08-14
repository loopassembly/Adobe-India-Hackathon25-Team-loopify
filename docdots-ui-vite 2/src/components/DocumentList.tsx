import React from 'react'
import clsx from 'clsx'

export default function DocumentList({
  docs, current, onSelect
}: { docs: string[], current?: string, onSelect: (name: string) => void }) {
  return (
    <div className="card p-3">
      <div className="font-semibold mb-2">Documents</div>
      <div className="space-y-2 max-h-[68vh] overflow-auto pr-1">
        {docs.length === 0 && (
          <div className="text-sm text-muted py-8 text-center">
            No documents found<br/>Upload PDFs to get started
          </div>
        )}
        {docs.map(d => (
          <button
            key={d}
            className={clsx(
              'w-full text-left px-3 py-2 border rounded-lg hover:bg-white',
              current === d ? 'border-primary bg-white' : 'border-slate-200'
            )}
            onClick={() => onSelect(d)}
            title={d}
          >
            <div className="truncate">{d}</div>
          </button>
        ))}
      </div>
    </div>
  )
}
