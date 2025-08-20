// src/components/Podcast.tsx
import _, { useCallback, useEffect, useMemo, useState } from "react";

/* --------------------------- Types & constants --------------------------- */

type ScriptTurn = {
  speaker: "S1" | "S2" | string;
  text: string;
  refs?: number[];
};

type HistoryItemLite = {
  id: string;
  title?: string | null;
  audio_url?: string | null;
  created_at?: number;
  voices?: string[];
  duration_min?: number;
};

type Props = {
  document: string | null;
  page: number;
  selection: string;

  // app-level optional hooks
  onStart?: () => void;
  onComplete?: (res?: {
    audio_url?: string;
    title?: string;
    script?: ScriptTurn[];
    voices?: string[];
    duration_min?: number;
  }) => void;
  onError?: (message?: string) => void;

  // provided by App (persist across tab switches)
  externalAudioUrl?: string | null;
  externalTitle?: string | null;
  externalScript?: ScriptTurn[] | null;
  generatingExternal?: boolean;

  // history (for this doc)
  history?: HistoryItemLite[];
  onLoadFromHistory?: (id: string) => void;
};

// same base resolution as api.ts
const BASE: string =
  ((import.meta as any).env?.VITE_API_BASE?.replace(/\/+$/, "")) ||
  "http://localhost:8080";

// ensure these match your TTS backend
const VOICE_CHOICES = ["alloy", "verse", "aria", "coral", "sage", "vivid"] as const;

/* -------------------------------- Component ------------------------------ */

export default function Podcast({
  document,
  page,
  selection,
  onStart,
  onComplete,
  onError,
  externalAudioUrl,
  externalTitle,
  externalScript,
  generatingExternal,
  history = [],
  onLoadFromHistory,
}: Props) {
  // request/response state
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [etaSec, setEtaSec] = useState<number | null>(null);

  // local display (synced with external)
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [title, setTitle] = useState<string>("");
  const [script, setScript] = useState<ScriptTurn[] | null>(null);

  // UI controls
  const [voice1, setVoice1] = useState<string>(VOICE_CHOICES[0]);
  const [voice2, setVoice2] = useState<string>(VOICE_CHOICES[1] || VOICE_CHOICES[0]);
  const [durationMin, setDurationMin] = useState<3 | 4 | 5>(3);
  const [transcriptCollapsed, setTranscriptCollapsed] = useState(false);
  const [viewMode, setViewMode] = useState<"bubbles" | "list">("bubbles");

  // prefer app state when present (so results survive remounts/tab switches)
  const displayAudioUrl = externalAudioUrl ?? audioUrl;
  const displayTitle = externalTitle ?? title;
  const displayScript = externalScript ?? script;

  const busy = generating || !!generatingExternal;
  const canGenerate = useMemo(() => !!document && !busy, [document, busy]);
  const hasSelection = !!selection.trim();

  // sync down from App
  useEffect(() => {
    if (externalAudioUrl !== undefined) setAudioUrl(externalAudioUrl ?? null);
    if (externalTitle !== undefined) setTitle(externalTitle || "");
    if (externalScript !== undefined) setScript(externalScript ?? null);
  }, [externalAudioUrl, externalTitle, externalScript]);

  /* ------------------------------ Event helpers ------------------------------ */

  const fireStart = useCallback(() => {
    try {
      window.dispatchEvent(new CustomEvent("docdots:podcast-start"));
    } catch {}
    onStart?.();
  }, [onStart]);

  const fireDone = useCallback(
    (detail?: {
      audio_url?: string;
      title?: string;
      script?: ScriptTurn[];
      voices?: string[];
      duration_min?: number;
    }) => {
      try {
        window.dispatchEvent(new CustomEvent("docdots:podcast-done", { detail }));
      } catch {}
      onComplete?.(detail);
    },
    [onComplete]
  );

  const fireError = useCallback(
    (message?: string) => {
      try {
        window.dispatchEvent(new CustomEvent("docdots:podcast-error", { detail: { message } }));
      } catch {}
      onError?.(message);
    },
    [onError]
  );

  /* --------------------------------- Actions -------------------------------- */

  const generate = useCallback(async () => {
    if (!document) {
      const msg = "Open a PDF first before generating a podcast.";
      setError(msg);
      fireError(msg);
      return;
    }

    setGenerating(true);
    setError(null);
    setEtaSec(null);
    fireStart();

    try {
      const body = {
        document,
        page,
        selection: selection || "",
        style: "podcast",
        speakers: 2,
        duration_min: durationMin,
        voices: [voice1, voice2],
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
      const s: ScriptTurn[] | undefined = data?.script;

      setAudioUrl(url);
      setTitle(t);
      setScript(s || null);

      if (typeof data?.estimated_duration_sec === "number") {
        setEtaSec(data.estimated_duration_sec);
      }

      setGenerating(false);
      fireDone({
        audio_url: url ?? undefined,
        title: t,
        script: s,
        voices: [voice1, voice2],
        duration_min: durationMin,
      });
    } catch (e: any) {
      const msg = e?.message || "Network error";
      setError(msg);
      setGenerating(false);
      fireError(msg);
    }
  }, [document, page, selection, durationMin, voice1, voice2, fireStart, fireDone, fireError]);

  const swapVoices = () => {
    setVoice1(voice2);
    setVoice2(voice1);
  };

  const clearCurrent = () => {
    setAudioUrl(null);
    setTitle("");
    setScript(null);
    setEtaSec(null);
    setError(null);
  };

  const copyLink = async () => {
    const url = displayAudioUrl;
    if (!url) return;
    try {
      await navigator.clipboard?.writeText(url);
    } catch {}
  };

  /* ---------------------------------- UI ---------------------------------- */

  return (
    <div className="w-full max-w-full overflow-x-auto rounded-2xl border border-slate-200 bg-white p-3">
      {/* Header */}
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div className="min-w-0">
          <div className="text-sm font-semibold text-slate-900">Podcast</div>
          <div className="text-xs text-slate-500 truncate">
            {document ? `From: ${document} • p${page + 1}` : "No document open"}
          </div>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          {displayAudioUrl && (
            <button
              onClick={clearCurrent}
              className="h-9 px-3 rounded-lg border border-slate-300 bg-white text-slate-700 text-sm hover:bg-slate-50"
              title="Clear current audio"
            >
              Clear
            </button>
          )}
          <button
            className="h-9 px-3 rounded-lg bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-700 disabled:opacity-60"
            onClick={generate}
            disabled={!canGenerate}
            title={document ? "Generate an audio overview" : "Open a PDF first"}
          >
            {busy ? "Generating…" : displayAudioUrl ? "Regenerate" : "Generate Podcast"}
          </button>
        </div>
      </div>

      {/* Controls */}
      <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-5 w-full">
        {/* Speakers */}
        <div className="sm:col-span-2">
          <div className="mb-1 flex items-center justify-between text-xs">
            <span className="font-medium text-slate-700">Speaker 1</span>
          </div>
          <select
            className="input h-9 w-full"
            value={voice1}
            onChange={(e) => setVoice1(e.target.value)}
          >
            {VOICE_CHOICES.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </div>

        <div className="flex items-end justify-center sm:justify-start">
          <button
            type="button"
            onClick={swapVoices}
            className="h-9 w-9 rounded-lg border border-slate-300 bg-white hover:bg-slate-50 grid place-items-center mt-5"
            title="Swap voices"
          >
            <svg width="16" height="16" viewBox="0 0 24 24">
              <path
                fill="currentColor"
                d="M7 7h11l-3.5-3.5L16 2l6 6l-6 6l-1.5-1.5L18 9H7V7Zm10 10H6l3.5 3.5L8 22l-6-6l6-6l1.5 1.5L6 15h11v2Z"
              />
            </svg>
          </button>
        </div>

        <div className="sm:col-span-2">
          <div className="mb-1 text-xs font-medium text-slate-700">Speaker 2</div>
          <select
            className="input h-9 w-full"
            value={voice2}
            onChange={(e) => setVoice2(e.target.value)}
          >
            {VOICE_CHOICES.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </div>

        {/* Duration */}
        <div className="sm:col-span-5">
          <div className="mb-1 text-xs font-medium text-slate-700">Duration</div>
          <div className="flex flex-wrap items-center gap-1">
            {[3, 4, 5].map((d) => (
              <button
                key={d}
                className={`h-9 px-3 rounded-lg border text-sm ${
                  durationMin === d
                    ? "bg-indigo-600 text-white border-indigo-600"
                    : "bg-white text-slate-700 border-slate-300 hover:bg-slate-50"
                }`}
                onClick={() => setDurationMin(d as 3 | 4 | 5)}
              >
                {d} min
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Context note */}
      <div className="mt-3 text-xs text-slate-600">
        {hasSelection ? (
          <div className="flex flex-wrap items-center gap-2">
            <span className="px-2 py-1 rounded-md bg-indigo-50 text-indigo-700 border border-indigo-200">
              Using selection ({selection.trim().length} chars)
            </span>
            <span className="text-slate-400">•</span>
            <span>We’ll craft an audio overview from the highlighted text.</span>
          </div>
        ) : (
          <div className="flex flex-wrap items-center gap-2">
            <span className="px-2 py-1 rounded-md bg-slate-100 text-slate-700 border border-slate-200">
              Using page
            </span>
            <span className="text-slate-400">•</span>
            <span>No text selected — we’ll use the current page content.</span>
          </div>
        )}
      </div>

      {/* Status */}
      {(busy || etaSec != null || error) && (
        <div className="mt-2">
          {busy && (
            <div className="text-[11px] text-slate-500">
              Generating podcast… this can take a few minutes.
            </div>
          )}
          {etaSec != null && (
            <div className="text-[11px] text-slate-500">
              Estimated length: ~{Math.round(etaSec / 60)} min
            </div>
          )}
          {error && (
            <div className="mt-2 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-700 flex items-start justify-between gap-2">
              <span className="leading-5">{error}</span>
              <button
                className="shrink-0 rounded-md border border-rose-300 bg-white px-2 py-1 text-rose-700 hover:bg-rose-100"
                onClick={() => {
                  setError(null);
                  generate();
                }}
                title="Retry"
              >
                Retry
              </button>
            </div>
          )}
        </div>
      )}

      {/* Player + meta */}
      {displayAudioUrl && (
        <div className="mt-4">
          <div className="text-xs font-semibold text-slate-700">Latest podcast</div>

          <div className="mt-2 rounded-xl border border-slate-200 bg-white p-3">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
              <div className="min-w-0">
                <div className="text-sm font-medium text-slate-900 truncate">
                  {displayTitle || "Audio overview"}
                </div>
                <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-slate-600">
                  <span className="px-2 py-[2px] rounded-md bg-slate-100 border border-slate-200">
                    {voice1 === voice2
                      ? `Both speakers: ${voice1}`
                      : `S1: ${voice1} • S2: ${voice2}`}
                  </span>
                  <span className="px-2 py-[2px] rounded-md bg-slate-100 border border-slate-200">
                    {durationMin} min target
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <a
                  className="text-xs px-2 py-1 rounded-lg border border-slate-300 hover:bg-slate-50"
                  href={displayAudioUrl}
                  download
                  title="Download MP3"
                >
                  Download
                </a>
                <button
                  className="text-xs px-2 py-1 rounded-lg border border-slate-300 hover:bg-slate-50"
                  onClick={copyLink}
                  title="Copy link"
                >
                  Copy link
                </button>
              </div>
            </div>

            <audio className="mt-2 w-full" src={displayAudioUrl} controls />

            {/* Transcript header */}
            {displayScript && displayScript.length > 0 && (
              <div className="mt-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="text-xs font-semibold text-slate-700">Transcript</div>
                  <div className="flex items-center gap-2">
                    <div className="hidden sm:flex rounded-lg border border-slate-200 bg-slate-50 p-1">
                      <button
                        className={`px-2 py-1 text-[11px] rounded ${
                          viewMode === "bubbles"
                            ? "bg-white border border-slate-200"
                            : "text-slate-600"
                        }`}
                        onClick={() => setViewMode("bubbles")}
                        title="Chat bubbles"
                      >
                        Bubbles
                      </button>
                      <button
                        className={`ml-1 px-2 py-1 text-[11px] rounded ${
                          viewMode === "list"
                            ? "bg-white border border-slate-200"
                            : "text-slate-600"
                        }`}
                        onClick={() => setViewMode("list")}
                        title="Compact list"
                      >
                        List
                      </button>
                    </div>

                    <button
                      className="text-[11px] text-indigo-700 hover:underline"
                      onClick={() => setTranscriptCollapsed((v) => !v)}
                      title={transcriptCollapsed ? "Expand transcript" : "Collapse transcript"}
                    >
                      {transcriptCollapsed ? "Expand" : "Collapse"}
                    </button>
                  </div>
                </div>

                {!transcriptCollapsed && (
                  <>
                    {viewMode === "bubbles" ? (
            <div className="mt-2 space-y-2">
                        {displayScript.map((turn, idx) => {
                          const isS1 = turn.speaker === "S1";
                          return (
                            <div
                              key={idx}
                              className={`flex ${isS1 ? "justify-start" : "justify-end"}`}
                            >
                              <div
                                className={`max-w-[85%] sm:max-w-[80%] rounded-2xl px-3 py-2 border text-xs leading-relaxed break-words ${
                                  isS1
                                    ? "bg-indigo-50 border-indigo-200 text-slate-900"
                                    : "bg-fuchsia-50 border-fuchsia-200 text-slate-900"
                                }`}
                              >
                                <div className="mb-1 flex items-center gap-2">
                                  <span
                                    className={`inline-flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-semibold text-white ${
                                      isS1 ? "bg-indigo-600" : "bg-fuchsia-600"
                                    }`}
                                  >
                                    {turn.speaker}
                                  </span>
                                  <span className="text-[11px] text-slate-600">
                                    {isS1 ? "Speaker 1" : "Speaker 2"}
                                  </span>
                                </div>
                                <div className="break-words">{turn.text}</div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <div className="mt-2 divide-y divide-slate-200 rounded-lg border border-slate-200">
                        {displayScript.map((turn, idx) => {
                          const isS1 = turn.speaker === "S1";
                          return (
                            <div key={idx} className="grid grid-cols-[56px,1fr] gap-2 p-2 text-xs">
                              <div className="flex items-start">
                                <span
                                  className={`inline-flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-semibold text-white mt-0.5 ${
                                    isS1 ? "bg-indigo-600" : "bg-fuchsia-600"
                                  }`}
                                >
                                  {turn.speaker}
                                </span>
                                <span className="ml-2 mt-0.5 text-[11px] text-slate-600">
                                  {isS1 ? "S1" : "S2"}
                                </span>
                              </div>
                              <div className="leading-relaxed break-words">{turn.text}</div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* History */}
      {history.length > 0 && (
        <div className="mt-5">
          <div className="text-xs font-semibold text-slate-700 mb-2">Previous podcasts</div>
          <div className="flex flex-col gap-2 w-full">
            {history.map((h) => (
              <button
                key={h.id}
                onClick={() => onLoadFromHistory?.(h.id)}
                className="flex items-center justify-between gap-3 flex-wrap w-full rounded-lg border border-slate-200 bg-white hover:bg-slate-50 px-3 py-2 text-left"
                title="Load this podcast"
              >
                <div className="min-w-0">
                  <div className="text-xs font-medium text-slate-900 truncate break-all">
                    {h.title || "Audio overview"}
                  </div>
                  <div className="text-[11px] text-slate-500 break-all">
                    {(h.voices && h.voices.join(" • ")) || "2 speakers"} • {h.duration_min ?? 3} min
                    {h.created_at ? ` • ${new Date(h.created_at).toLocaleString()}` : ""}
                  </div>
                </div>
                <div className="shrink-0 text-[11px] font-medium text-indigo-700">Load</div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}