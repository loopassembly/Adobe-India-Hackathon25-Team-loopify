// src/components/Podcast.tsx
import _, { useCallback, useEffect, useMemo, useState } from "react";

type Props = {
  document: string | null;
  page: number;
  selection: string;

  // optional app-level hooks (kept as-is)
  onStart?: () => void;
  onComplete?: (res?: { audio_url?: string; title?: string }) => void;
  onError?: (message?: string) => void;

  // Provided by App so results persist across tab switches
  externalAudioUrl?: string | null;
  externalTitle?: string | null;
  generatingExternal?: boolean;
};

// same base resolution as api.ts
const BASE: string =
  ((import.meta as any).env?.VITE_API_BASE?.replace(/\/+$/, "")) ||
  "http://localhost:8080";

export default function Podcast({
  document,
  page,
  selection,
  onStart,
  onComplete,
  onError,
  externalAudioUrl,
  externalTitle,
  generatingExternal,
}: Props) {
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // local display state (synced with external values)
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [title, setTitle] = useState<string>("");

  const [etaSec, setEtaSec] = useState<number | null>(null);

  // prefer external values if present (they survive remounts)
  const displayAudioUrl = externalAudioUrl ?? audioUrl;
  const displayTitle = externalTitle ?? title;

  const busy = generating || !!generatingExternal;
  const canGenerate = useMemo(() => !!document && !busy, [document, busy]);

  // keep local view in sync with app-level values
  useEffect(() => {
    if (externalAudioUrl != null) setAudioUrl(externalAudioUrl);
    if (externalTitle != null) setTitle(externalTitle || "");
  }, [externalAudioUrl, externalTitle]);

  const fireStart = useCallback(() => {
    try {
      window.dispatchEvent(new CustomEvent("docdots:podcast-start"));
    } catch {}
    onStart?.();
  }, [onStart]);

  const fireDone = useCallback((detail?: { audio_url?: string; title?: string }) => {
    try {
      window.dispatchEvent(new CustomEvent("docdots:podcast-done", { detail }));
    } catch {}
    onComplete?.(detail);
  }, [onComplete]);

  const fireError = useCallback((message?: string) => {
    try {
      window.dispatchEvent(new CustomEvent("docdots:podcast-error", { detail: { message } }));
    } catch {}
    onError?.(message);
  }, [onError]);

  const generate = useCallback(async () => {
    if (!document) {
      const msg = "Open a PDF first before generating a podcast.";
      setError(msg);
      fireError(msg);
      return;
    }

    setGenerating(true);
    setError(null);
    // keep previous audio visible while a new one is being generated
    setTitle("");
    setEtaSec(null);
    fireStart();

    try {
      const body = {
        document,
        page,
        selection: selection || "",
        style: "podcast",
        speakers: 2,
        duration_min: 3.0,
        voices: ["alloy", "verse"],
        format: "audio-48khz-192kbitrate-mono-mp3",
      };

      const res = await fetch(`${BASE}/podcast`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const msg = `${res.status} ${res.statusText}`;
        setError(msg);
        setGenerating(false);
        fireError(msg);
        return;
      }

      const data = await res.json();
      const url: string | null = data?.audio_url || null;
      const t: string = data?.title || "Audio overview";

      setAudioUrl(url);
      setTitle(t);
      if (typeof data?.estimated_duration_sec === "number") {
        setEtaSec(data.estimated_duration_sec);
      }
      setGenerating(false);
      fireDone({ audio_url: url ?? undefined, title: t });
    } catch (e: any) {
      const msg = e?.message || "Network error";
      setError(msg);
      setGenerating(false);
      fireError(msg);
    }
  }, [document, page, selection, fireStart, fireDone, fireError]);

  const hasSelection = !!selection.trim();

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <div className="flex items-center justify-between gap-2">
        <div className="min-w-0">
          <div className="text-sm font-semibold text-slate-900">Podcast</div>
          <div className="text-xs text-slate-500 truncate">
            {document ? `From: ${document} • p${page + 1}` : "No document open"}
          </div>
        </div>
        <button
          className="h-9 px-3 rounded-lg bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-700 disabled:opacity-60"
          onClick={generate}
          disabled={!canGenerate}
          title={document ? "Generate a short audio from this page" : "Open a PDF first"}
        >
          {busy ? "Generating…" : displayAudioUrl ? "Regenerate" : "Generate Podcast"}
        </button>
      </div>

      <div className="mt-3 text-xs text-slate-600">
        {hasSelection ? (
          <div className="flex items-center gap-2">
            <span className="badge bg-indigo-50 text-indigo-700 border border-indigo-200">
              Using selection ({selection.trim().length} chars)
            </span>
            <span className="text-slate-400">•</span>
            <span>We’ll craft an audio overview from the highlighted text.</span>
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <span className="badge bg-slate-100 text-slate-700 border border-slate-200">Using page</span>
            <span className="text-slate-400">•</span>
            <span>No text selected — we’ll use this page’s content.</span>
          </div>
        )}
      </div>

      {(busy || etaSec != null) && (
        <div className="mt-2 text-[11px] text-slate-500">
          {busy ? "Generating podcast… this can take a few minutes. " : ""}
          {etaSec != null && <>Estimated length: ~{Math.round(etaSec / 60)} min.</>}
        </div>
      )}

      {displayAudioUrl && (
        <div className="mt-3">
          <div className="text-xs font-semibold text-slate-700">Last generated</div>
          <div className="mt-2 flex items-center gap-3">
            <audio className="w-full" src={displayAudioUrl} controls />
          </div>
          {displayTitle && (
            <div className="mt-2 text-xs text-slate-700">
              Title: <span className="font-medium">{displayTitle}</span>
            </div>
          )}
          <div className="mt-2">
            <a
              className="text-xs inline-flex items-center gap-1 px-2 py-1 rounded-lg border border-slate-300 hover:bg-slate-50"
              href={displayAudioUrl}
              target="_blank"
              rel="noopener noreferrer"
            >
              Open audio
              <svg width="14" height="14" viewBox="0 0 24 24">
                <path fill="currentColor" d="M14 3h7v7h-2V6.41l-9.29 9.3-1.42-1.42 9.3-9.29H14V3zM5 5h5V3H3v7h2V5z" />
              </svg>
            </a>
          </div>
        </div>
      )}

      {error && <div className="mt-3 text-xs text-rose-600">{error}</div>}
    </div>
  );
}