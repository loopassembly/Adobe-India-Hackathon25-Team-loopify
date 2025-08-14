import React from 'react'
import Header from './components/Header'
import PdfViewer from './components/PdfViewer'
import DocumentList from './components/DocumentList'
import Outline from './components/Outline'
import Recommendations from './components/Recommendations'
import Insights from './components/Insights'
import { listDocs, uploadDocs, getOutline, getRecommendations, getInsights, pdfUrl } from './lib/api'
import type { OutlineItem, RecommendationsResponse, InsightsResponse } from './lib/types'

export default function App() {
  const [docs, setDocs] = React.useState<string[]>([])
  const [doc, setDoc] = React.useState<string>('')
  const [page, setPage] = React.useState(0)
  const [title, setTitle] = React.useState<string>('')
  const [outline, setOutline] = React.useState<OutlineItem[]>([])
  const [tab, setTab] = React.useState<'related'|'insights'>('related')
  const [recs, setRecs] = React.useState<RecommendationsResponse | null>(null)
  const [ins, setIns] = React.useState<InsightsResponse | null>(null)
  const [error, setError] = React.useState<string| null>(null)

  React.useEffect(() => {
    listDocs().then(setDocs).catch(e => setError(String(e)))
  }, [])

  React.useEffect(() => {
    if (!doc) return
    getOutline(doc).then(o => {
      setTitle(o.title || '')
      setOutline(o.outline || [])
      setPage(o.outline?.[0]?.page ?? 0)
    }).catch(e => setError(String(e)))
  }, [doc])

  React.useEffect(() => {
    if (!doc) return
    getRecommendations(doc, page, title, 5).then(setRecs).catch(() => {})
    if (tab === 'insights') getInsights(doc, page, 3).then(setIns).catch(() => {})
  }, [doc, page, title, tab])

  function onUpload(files: File[]) {
    uploadDocs(files).then(() => listDocs().then(setDocs)).catch(e => setError(String(e)))
  }

  function onJumpRelated(d: string, p: number) {
    if (d === doc) {
      setPage(p)
    } else {
      setDoc(d)
      setPage(p)
    }
  }

  return (
    <div className="min-h-screen">
      <Header onUpload={onUpload}/>

      <div className="mx-auto max-w-7xl px-4 py-4 grid grid-cols-12 gap-4">
        {/* Left column */}
        <div className="col-span-3 space-y-3">
          <DocumentList docs={docs} current={doc} onSelect={(d) => setDoc(d)} />
          <div className="card p-3">
            <div className="text-xs text-muted">EXTRACTED TITLE</div>
            <div className="mt-1 text-sm">{title || '—'}</div>
          </div>
          <Outline items={outline} onJump={(p) => setPage(p)} />
        </div>

        {/* Middle column */}
        <div className="col-span-6 space-y-3">
          <PdfViewer src={doc ? pdfUrl(doc) : ''} page={page} onPageChange={(p) => setPage(p)} />
        </div>

        {/* Right column */}
        <div className="col-span-3 space-y-3">
          <div className="bg-white border border-slate-200 rounded-xl p-2">
            <div className="grid grid-cols-2 gap-2 p-1 bg-slate-100 rounded-lg">
              <button className={tab==='related' ? 'tab tab-active' : 'tab'} onClick={() => setTab('related')}>Related</button>
              <button className={tab==='insights' ? 'tab tab-active' : 'tab'} onClick={() => setTab('insights')}>Insights</button>
            </div>
          </div>

          {tab === 'related' ? (
            <Recommendations items={recs?.items || []} onJump={onJumpRelated} />
          ) : (
            <Insights data={ins || undefined} />
          )}

          {error && (
            <div className="bg-red-50 border border-red-300 text-red-600 rounded-xl p-3 text-sm">
              {error} <button className="underline" onClick={() => setError(null)}>dismiss</button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
