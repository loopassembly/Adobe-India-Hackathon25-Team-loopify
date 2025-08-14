import type { OutlineResponse, RecommendationsResponse, InsightsResponse } from './types'

const BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8080'
const PDF_BASE = import.meta.env.VITE_PDF_BASE || '/data/pdfs'

export function pdfUrl(name: string) {
  return `${PDF_BASE}/${encodeURIComponent(name)}`
}

async function jsonOrThrow(res: Response) {
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  const ct = res.headers.get('content-type') || ''
  if (!ct.includes('application/json')) {
    const text = await res.text()
    throw new Error(`Expected JSON but got '${ct}'. First bytes: ${text.slice(0, 80)}`)
  }
  return res.json()
}

export async function listDocs(): Promise<string[]> {
  const r = await fetch(`${BASE}/docs`)
  const j = await jsonOrThrow(r)
  return j.docs || j || []
}

export async function uploadDocs(files: File[]) {
  const fd = new FormData()
  for (const f of files) fd.append('files', f)
  const r = await fetch(`${BASE}/index`, { method: 'POST', body: fd })
  return jsonOrThrow(r)
}

export async function getOutline(document: string) {
  const r = await fetch(`${BASE}/outline?document=${encodeURIComponent(document)}`)
  return jsonOrThrow(r) as Promise<OutlineResponse>
}

export async function getRecommendations(document: string, page: number, title?: string, top_k = 5) {
  const body: any = { document, page, top_k }
  if (title) body.title = title
  const r = await fetch(`${BASE}/recommendations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  return jsonOrThrow(r) as Promise<RecommendationsResponse>
}

export async function getInsights(document: string, page: number, top_k = 3) {
  const r = await fetch(`${BASE}/insights`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ document, page, top_k }),
  })
  return jsonOrThrow(r) as Promise<InsightsResponse>
}

export async function makePodcast(text: string, voice: string, format: string) {
  const r = await fetch(`${BASE}/podcast`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, voice, format }),
  })
  return jsonOrThrow(r)
}
