// src/api.ts
export type Status = {
  phase: "idle" | "checking-cache" | "downloading" | "loading" | "ready" | "error";
  progress: number;
  message: string;
  embedding_model?: string | null;
};

const BASE =
  (import.meta as any).env?.VITE_API_BASE?.replace(/\/+$/, "") ||
  "http://localhost:8080";

async function j<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export async function getStatus(): Promise<Status> {
  const res = await fetch(`${BASE}/status`, { cache: "no-store" });
  return j<Status>(res);
}

export async function fetchDocs(): Promise<{ docs: string[] }> {
  const res = await fetch(`${BASE}/docs`, { cache: "no-store" });
  return j(res);
}

export async function indexPdfFiles(files?: FileList | File[] | null): Promise<any> {
  if (!files || (Array.isArray(files) ? files.length === 0 : files?.length === 0)) {
    const res = await fetch(`${BASE}/index`, { method: "POST" });
    return j(res);
  }
  const fd = new FormData();
  if (files instanceof FileList) {
    Array.from(files).forEach((f) => fd.append("files", f, f.name));
  } else {
    files.forEach((f) => fd.append("files", f, f.name));
  }
  const res = await fetch(`${BASE}/index`, { method: "POST", body: fd });
  return j(res);
}

export async function fetchOutline(document: string): Promise<any> {
  const url = new URL(`${BASE}/outline`);
  url.searchParams.set("document", document);
  const res = await fetch(url.toString());
  return j(res);
}

export async function fetchRecommendations(
  document: string,
  page: number,
  top_k = 5,
  selection = ""
): Promise<any> {
  const res = await fetch(`${BASE}/recommendations`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ document, page, top_k, selection }),
  });
  return j(res);
}

export async function fetchInsights(
  document: string,
  page: number,
  top_k = 3,
  selection = ""
): Promise<any> {
  const res = await fetch(`${BASE}/insights`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ document, page, top_k, selection }),
  });
  return j(res);
}

export async function selectRelated(selection: string, top_k = 5): Promise<any> {
  const res = await fetch(`${BASE}/select`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ selection, top_k }),
  });
  return j(res);
}

export function pdfUrl(name: string): string {
  return `${BASE}/data/pdfs/${encodeURIComponent(name)}`;
}

/* ---------- NEW: podcast ---------- */
export type PodcastRequest = {
  document?: string | null;
  page?: number;
  selection?: string;
  style?: "podcast" | "overview";
  speakers?: 1 | 2;
  duration_min?: number;
  voices?: string[];
  format?: string;
  // fast?: boolean; // uncomment if your backend supports it
};

export type PodcastResponse = {
  audio_url: string;
  title: string;
  script: { speaker: "S1" | "S2"; text: string; refs?: number[] }[];
  used_items?: any[];
  provider?: string;
  mode?: "podcast" | "overview";
  estimated_duration_sec?: number;
  timings?: { [k: string]: number }; // if backend returns it
};

export async function createPodcast(body: PodcastRequest): Promise<PodcastResponse> {
  const res = await fetch(`${BASE}/podcast`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  return j<PodcastResponse>(res);
}

export function mediaUrl(rel: string): string {
  // helper to turn /data/audio/xyz.mp3 into absolute
  if (!rel) return "";
  if (rel.startsWith("http")) return rel;
  return `${BASE}${rel}`;
}