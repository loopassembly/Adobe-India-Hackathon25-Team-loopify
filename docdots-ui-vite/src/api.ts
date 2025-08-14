export function computeBackend() {
  const env = import.meta.env.VITE_BACKEND as string | undefined;
  if (env) return env.replace(/\/$/, '');

  try {
    const loc = window.location;
    const host = loc.hostname || 'localhost';
    return `http://${host}:8080`;
  } catch {
    return 'http://localhost:8080';
  }
}
export const BACKEND = computeBackend();

export type DocInfo = { name: string; pages: number };
export type DocsResp = { documents: DocInfo[]; domain: string; persona: string; job: string };

async function withTimeout<T>(p: Promise<T>, ms=15000): Promise<T> {
  const ac = new AbortController();
  const t = setTimeout(() => ac.abort('timeout'), ms);
  try {
    const res = await p;
    return res as any;
  } finally {
    clearTimeout(t);
  }
}

export async function listDocs(): Promise<DocsResp> {
  const r = await withTimeout(fetch(`${BACKEND}/docs`));
  if (!r.ok) { throw new Error(await r.text()); }
  return r.json();
}

export async function uploadIndex(files: File[], persona: string, job: string) {
  const fd = new FormData();
  fd.append('persona', persona || 'Food lover');
  fd.append('job_to_be_done', job || 'Explore recipes');
  for (const f of files) fd.append('files', f);
  const r = await withTimeout(fetch(`${BACKEND}/index`, { method: 'POST', body: fd }));
  if (!r.ok) { throw new Error(await r.text()); }
  return r.json();
}

export async function fetchOutline(docName: string) {
  const r = await withTimeout(fetch(`${BACKEND}/outline?document=${encodeURIComponent(docName)}`));
  if (!r.ok) { throw new Error(await r.text()); }
  return r.json();
}

export async function fetchRecommendations(document: string, page: number | null, title?: string, top_k=5) {
  const r = await withTimeout(fetch(`${BACKEND}/recommendations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ document, page, title, top_k })
  }));
  if (!r.ok) { throw new Error(await r.text()); }
  return r.json();
}

export async function fetchInsights(document: string, page: number, top_k=3) {
  const r = await withTimeout(fetch(`${BACKEND}/insights`, {
    method:'POST',
    headers:{ 'Content-Type': 'application/json' },
    body: JSON.stringify({ document, page, top_k })
  }));
  if (!r.ok) { throw new Error(await r.text()); }
  return r.json();
}

export async function makePodcast(text: string, voice='en-US-JennyNeural', format='audio-48khz-192kbitrate-mono-mp3') {
  const r = await withTimeout(fetch(`${BACKEND}/podcast`, {
    method:'POST',
    headers:{ 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, voice, format })
  }));
  if (!r.ok) { throw new Error(await r.text()); }
  return r.json();
}

export function pdfUrl(name: string) {
  // server.py serves PDFs under /data/uploads/<name>
  return `${BACKEND}/data/uploads/${encodeURIComponent(name)}`;
}

export function audioUrl(pathFromApi: string) {
  // API returns "/data/audio/tts_...mp3" -> we need absolute
  if (pathFromApi.startsWith('http')) return pathFromApi;
  return `${BACKEND}${pathFromApi}`;
}