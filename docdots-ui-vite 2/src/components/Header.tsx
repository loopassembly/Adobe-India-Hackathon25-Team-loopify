import React from 'react'

export default function Header({ onUpload }: { onUpload: (files: File[]) => void }) {
  const inputRef = React.useRef<HTMLInputElement>(null)
  return (
    <div className="sticky top-0 z-20 bg-white border-b border-slate-200">
      <div className="mx-auto max-w-7xl px-4 py-3 flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-primary"></div>
        <div className="font-semibold">DocDots</div>
        <span className="text-[11px] px-2 py-0.5 rounded-full bg-primary-subtle text-primary ml-1">food</span>

        <div className="ml-auto flex items-center gap-3">
          <input
            ref={inputRef}
            type="file"
            accept="application/pdf"
            multiple
            className="hidden"
            onChange={(e) => {
              if (e.target.files && e.target.files.length) onUpload(Array.from(e.target.files))
              e.currentTarget.value = ''
            }}
          />
          <button className="btn btn-primary" onClick={() => inputRef.current?.click()}>
            ⬆ Upload PDFs
          </button>
        </div>
      </div>
    </div>
  )
}
