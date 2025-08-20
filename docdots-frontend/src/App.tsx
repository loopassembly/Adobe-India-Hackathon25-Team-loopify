import React, { useEffect, useRef, useState } from "react";
import {
  fetchDocs,
  fetchOutline,
  fetchRecommendations,
  fetchInsights,
  indexPdfFiles,
  getStatus,
  pdfUrl,
  selectRelated,
} from "./api";
import PdfViewer, { PdfViewerHandle } from "./components/PdfViewer";
import Outline from "./components/Outline";
import Recommendations from "./components/Recommendations";
import Insights from "./components/Insights";
import Welcome from "./components/Welcome";
import Podcast from "./components/Podcast";

type OutlineItem = { level: string; text: string; page: number };

// Toast state for bottom-right notifier (independent from actual results)
type PodcastStatus =
  | { state: "idle" }
  | { state: "generating"; message: string }
  | { state: "ready"; message: string; title?: string }
  | { state: "error"; message: string };

// Saved results
type ScriptTurn = { speaker: "S1" | "S2" | string; text: string; refs?: number[] };
type PodcastResult = {
  audio_url: string | null;
  title: string | null;
  script: ScriptTurn[] | null;
  voices?: string[];
  duration_min?: number;
};

// History (per document)
type PodcastHistoryItem = PodcastResult & {
  id: string;
  doc: string | null;
  page: number;
  selection_len: number;
  created_at: number;
};

const HISTORY_KEY = "docdots.podcastHistory.v1";

export default function App() {
  const [docs, setDocs] = useState<string[]>([]);
  const [selectedDoc, setSelectedDoc] = useState<string | null>(null);
  const [outline, setOutline] = useState<OutlineItem[]>([]);
  const [title, setTitle] = useState<string>("");
  const [page, setPage] = useState<number>(0);

  const [freshDoc, setFreshDoc] = useState<string | null>(null);
  const [freshPage, setFreshPage] = useState<number>(0);

  const [tab, setTab] = useState<"related" | "insights" | "podcast">("related");
  const [recs, setRecs] = useState<any[]>([]);
  const [insightText, setInsightText] = useState<string>("");

  const [selection, setSelection] = useState("");
  const [, setSelectionSource] = useState<"pdf" | "input">("input");
  const [selectLoading, setSelectLoading] = useState(false);

  const [isIndexing, setIsIndexing] = useState(false);
  const [statusText, setStatusText] = useState("Starting…");
  const [statusPct, setStatusPct] = useState(5);
  const [statusTitle, setStatusTitle] = useState<"open" | "bulk">("bulk");
  const pollRef = useRef<number | null>(null);
  const timeoutRef = useRef<number | null>(null);
  // Progress trickle helpers
  const trickleRef = useRef<number | null>(null);
  const lastServerPctRef = useRef<number>(5);
  const capTo100Ref = useRef<boolean>(false);

  const viewerRef = useRef<PdfViewerHandle>(null);
  const bulkInputRef = useRef<HTMLInputElement>(null);
  const openInputRef = useRef<HTMLInputElement>(null);

  const [rightTopPct, setRightTopPct] = useState(56);
  const rightRailRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ startY: number; startPct: number } | null>(null);

  // one-shot flag to prevent related/insights refetch when returning via Back to Reading
  const preserveViewRef = useRef(false);

  // ---- Podcast: global toast + persisted result + history
  const [podcastStatus, setPodcastStatus] = useState<PodcastStatus>({ state: "idle" });
  const [toastMinimized, setToastMinimized] = useState(false);
  const [podcastResult, setPodcastResult] = useState<PodcastResult>({
    audio_url: null,
    title: null,
    script: null,
  });
  const [podcastHistory, setPodcastHistory] = useState<PodcastHistoryItem[]>([]);

  // load history from localStorage
  useEffect(() => {
    try {
      const raw = localStorage.getItem(HISTORY_KEY);
      if (raw) {
        const parsed = JSON.parse(raw) as PodcastHistoryItem[];
        setPodcastHistory(Array.isArray(parsed) ? parsed : []);
      }
    } catch {}
  }, []);

  // persist history to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(podcastHistory));
    } catch {}
  }, [podcastHistory]);

  function onDragStart(e: React.MouseEvent) {
    e.preventDefault();
    dragRef.current = { startY: e.clientY, startPct: rightTopPct };
    window.addEventListener("mousemove", onDragMove);
    window.addEventListener("mouseup", onDragEnd);
  }
  function onDragMove(e: MouseEvent) {
    if (!dragRef.current || !rightRailRef.current) return;
    const rect = rightRailRef.current.getBoundingClientRect();
    const next = Math.min(
      85,
      Math.max(15, dragRef.current.startPct + ((e.clientY - dragRef.current.startY) / rect.height) * 100)
    );
    setRightTopPct(next);
  }
  function onDragEnd() {
    dragRef.current = null;
    window.removeEventListener("mousemove", onDragMove);
    window.removeEventListener("mouseup", onDragEnd);
  }

  useEffect(() => {
    (async () => {
      try {
        const { docs } = await fetchDocs();
        setDocs(docs || []);
      } catch (e) {
        console.error(e);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // when selectedDoc changes
  useEffect(() => {
    if (!selectedDoc) return;
    let cancelled = false;
    const preserve = preserveViewRef.current;

    (async () => {
      try {
        const o = await fetchOutline(selectedDoc);
        if (cancelled) return;
        setTitle(o.title || "");
        setOutline(o.outline || []);
        if (!preserve) setPage(0);
      } catch {
        if (!cancelled) {
          setOutline([]);
          setTitle("");
        }
      }

      try {
        if (!preserve) {
          const r = await fetchRecommendations(selectedDoc, 2);
          if (!cancelled) setRecs(r?.results || []);
        }
      } catch {
        if (!cancelled && !preserve) setRecs([]);
      }

      try {
        if (!preserve) {
          const i = await fetchInsights(selectedDoc, 2);
          if (!cancelled) setInsightText(i?.text || "");
        }
      } catch {
        if (!cancelled && !preserve) setInsightText("");
      }

      if (preserve) preserveViewRef.current = false;
    })();

    return () => {
      cancelled = true;
    };
  }, [selectedDoc]);

  // when page changes (for current selectedDoc)
  useEffect(() => {
    if (!selectedDoc) return;
    let cancelled = false;

    if (preserveViewRef.current) {
      // returning to previous view: keep right-rail content intact
      preserveViewRef.current = false;
      return;
    }

    (async () => {
      try {
        const r = await fetchRecommendations(selectedDoc, page + 2);
        if (!cancelled) setRecs(r?.results || []);
      } catch {
        if (!cancelled) setRecs([]);
      }
      try {
        const i = await fetchInsights(selectedDoc, page + 2);
        if (!cancelled) setInsightText(i?.text || "");
      } catch {
        if (!cancelled) setInsightText("");
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [page, selectedDoc]);

  useEffect(() => {
    if (freshDoc && selectedDoc === freshDoc) setFreshPage(page);
  }, [page, selectedDoc, freshDoc]);

  function startStatusPolling(kind: "open" | "bulk") {
    stopStatusPolling();
    setStatusTitle(kind);

    // reset trickle helpers
    lastServerPctRef.current = Math.max(5, statusPct || 5);
    capTo100Ref.current = false;

    // 1) Poll backend status periodically to update text + server-reported % (if any)
    pollRef.current = window.setInterval(async () => {
      try {
        const s = await getStatus();

        // Update status text from server
        setStatusText(s.message || s.phase || (kind === "open" ? "Opening & indexing…" : "Uploading & indexing…"));

        // Remember the latest server % but never go backwards
        const serverPct = Number.isFinite((s as any).progress) ? (s as any).progress as number : 0;
        if (serverPct > 0) {
          lastServerPctRef.current = Math.max(lastServerPctRef.current, serverPct);
        }

        // If backend says ready, allow cap to reach 100 (but still keep overlay until POST /index says OK)
        if (s.phase === "ready") {
          capTo100Ref.current = true;
        }
      } catch {
        // Keep previous text; trickle keeps UI alive even if status endpoint hiccups
      }
    }, 1200) as unknown as number;

    // 2) Independent smooth trickle — always move a bit until final OK
    trickleRef.current = window.setInterval(() => {
      setStatusPct((prev) => {
        const cap = capTo100Ref.current ? 100 : 92;
        const base = Math.max(prev || 5, lastServerPctRef.current || 5);

        if (base >= cap) return base;

        // Easing: faster early on, slower near cap
        const remaining = cap - base;
        const accel =
          base < 30 ? 1.4 :
          base < 60 ? 1.0 :
          0.6;

        const bump = Math.max(0.6, Math.min(2.2, remaining * 0.05 * accel));
        const next = Math.min(cap, base + bump);
        return Number(next.toFixed(2));
      });
    }, 700) as unknown as number;

    // NOTE: do not auto-hide overlay on timeout; /index success will do that.
    timeoutRef.current = window.setTimeout(() => {
      // Stop polling but keep showing overlay with informative text
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
      setStatusText("Still working… finishing setup (first run can take a few minutes)");
    }, 180_000) as unknown as number;
  }
  function stopStatusPolling() {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    if (trickleRef.current) {
      clearInterval(trickleRef.current);
      trickleRef.current = null;
    }
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
  }

  async function doIndex(files: FileList | null | undefined, kind: "open" | "bulk") {
    if (!files || files.length === 0) return;
    try {
      setIsIndexing(true);
      setStatusPct((p) => Math.max(5, p || 5));
      setStatusText(kind === "open" ? "Opening & indexing…" : "Uploading & indexing…");
      startStatusPolling(kind);

      const res = await indexPdfFiles(files);

      // Only hide overlay when backend confirms success
      if (res && res.status === "ok") {
        // Snap to 100, brief "finalizing", then hide
        capTo100Ref.current = true;
        setStatusText("Finalizing…");
        setStatusPct(100);

        const fetched = await fetchDocs();
        setDocs(fetched?.docs || []);

        if (kind === "open") {
          const just = files.item(0)?.name;
          if (just) {
            setFreshDoc(just);
            setFreshPage(0);
            setSelectedDoc(just);
          }
        }

        stopStatusPolling();
        setIsIndexing(false);
      } else {
        // Keep overlay visible; show some context so user knows we are waiting on backend
        setStatusText("Almost there… awaiting confirmation from the server");
        // do not change isIndexing here
      }
    } catch (e) {
      console.error(e);
      setStatusText("Error during indexing. You can dismiss and retry.");
      // keep overlay up so user can read the error until dismissed
    } finally {
      // Always clear file inputs
      if (openInputRef.current) openInputRef.current.value = "";
      if (bulkInputRef.current) bulkInputRef.current.value = "";
    }
  }

  function onBulkUpload() {
    bulkInputRef.current?.click();
  }
  async function onBulkInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    await doIndex(e.target.files, "bulk");
  }
  function onOpenOne() {
    openInputRef.current?.click();
  }
  async function onOpenInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    await doIndex(e.target.files, "open");
  }

  // --------- HIGHLIGHT (yellow) via PdfViewer ref ----------
  async function highlightSelection(t: string) {
    try {
      const q = (t ?? "").toString().trim();
      if (!q) return;
      await viewerRef.current?.highlight?.(q);
    } catch (err) {
      console.warn("[DocDots] highlight failed:", err);
    }
  }

  /*
  // NOTE: "Use from PDF" temporarily disabled.
  // Keeping the implementation here for future re-enable.
  async function useFromPdf() {
    if (!selectedDoc) return;

    setSelectLoading(true);
    const token = Date.now();
    (window as any).__selToken = token;

    try {
      const t = (await viewerRef.current?.getSelection?.()) || "";
      const q = t.trim();

      setSelection(t);
      setSelectionSource("pdf");
      setTab("related");

      if (q) {
        await highlightSelection(q);
      }

      // Fire both calls in parallel; pass selection (can be empty string)
      const [r, i] = await Promise.all([
        fetchRecommendations(selectedDoc, page + 2, 5, q),
        fetchInsights(selectedDoc, page + 2, 5, q),
      ]);

      if ((window as any).__selToken !== token) return;

      setRecs(r?.results || []);
      setInsightText(i?.text || "");
    } catch (e) {
      console.error(e);
    } finally {
      if ((window as any).__selToken === token) {
        setSelectLoading(false);
      }
    }
  }
  */

  async function findRelatedFromSelection() {
    const q = selection.trim();
    if (!q) return;
    setSelectLoading(true);
    try {
      await highlightSelection(q);

      const res = await selectRelated(q, 5);
      setRecs(res?.results || []);
      setInsightText(res?.insight || "");
      setTab("related");
    } catch (e) {
      console.error(e);
    } finally {
      setSelectLoading(false);
    }
  }

  function backToReading() {
    if (!freshDoc) return;
    // one-time skip: do not reload right rail, restore page
    preserveViewRef.current = true;
    setSelectedDoc(freshDoc);
    setPage(freshPage || 0);
  }

  // ---- Global podcast events
  useEffect(() => {
    const onStart = () => {
      setPodcastStatus({
        state: "generating",
        message: "Generating podcast… this can take a few minutes. You can keep browsing.",
      });
      setToastMinimized(false);
    };

    const onDone = (e: Event) => {
      const detail = (e as CustomEvent).detail || {};
      // persist "latest"
      setPodcastResult({
        audio_url: detail.audio_url ?? null,
        title: detail.title ?? "Audio overview",
        script: detail.script ?? null,
        voices: detail.voices,
        duration_min: detail.duration_min,
      });

      // add to history
      const item: PodcastHistoryItem = {
        id: `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
        doc: selectedDoc ?? null,
        page,
        selection_len: (selection || "").trim().length,
        created_at: Date.now(),
        audio_url: detail.audio_url ?? null,
        title: detail.title ?? "Audio overview",
        script: detail.script ?? null,
        voices: detail.voices,
        duration_min: detail.duration_min,
      };
      setPodcastHistory((prev) => [item, ...prev].slice(0, 50));

      setPodcastStatus({
        state: "ready",
        message: "Podcast is ready!",
        title: detail.title ?? "Audio overview",
      });
      setToastMinimized(false);
    };

    const onError = (e: Event) => {
      const detail = (e as CustomEvent).detail || {};
      setPodcastStatus({
        state: "error",
        message: detail.message || "Podcast generation failed. Please try again.",
      });
      setToastMinimized(false);
    };

    window.addEventListener("docdots:podcast-start", onStart as EventListener);
    window.addEventListener("docdots:podcast-done", onDone as EventListener);
    window.addEventListener("docdots:podcast-error", onError as EventListener);
    return () => {
      window.removeEventListener("docdots:podcast-start", onStart as EventListener);
      window.removeEventListener("docdots:podcast-done", onDone as EventListener);
      window.removeEventListener("docdots:podcast-error", onError as EventListener);
    };
  }, [page, selectedDoc, selection]);

  // ---- Optional: callbacks passed to <Podcast />
  const handlePodcastStart = () => {
    setPodcastStatus({
      state: "generating",
      message: "Generating podcast… this can take a few minutes. You can keep browsing.",
    });
    setToastMinimized(false);
  };
  const handlePodcastComplete = (res?: {
    audio_url?: string;
    title?: string;
    script?: ScriptTurn[];
    voices?: string[];
    duration_min?: number;
  }) => {
    setPodcastResult({
      audio_url: res?.audio_url ?? null,
      title: res?.title ?? "Audio overview",
      script: res?.script ?? null,
      voices: res?.voices,
      duration_min: res?.duration_min,
    });
    const item: PodcastHistoryItem = {
      id: `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      doc: selectedDoc ?? null,
      page,
      selection_len: (selection || "").trim().length,
      created_at: Date.now(),
      audio_url: res?.audio_url ?? null,
      title: res?.title ?? "Audio overview",
      script: res?.script ?? null,
      voices: res?.voices,
      duration_min: res?.duration_min,
    };
    setPodcastHistory((prev) => [item, ...prev].slice(0, 50));

    setPodcastStatus({
      state: "ready",
      message: "Podcast is ready!",
      title: res?.title ?? "Audio overview",
    });
    setToastMinimized(false);
  };
  const handlePodcastError = (message?: string) => {
    setPodcastStatus({
      state: "error",
      message: message || "Podcast generation failed. Please try again.",
    });
    setToastMinimized(false);
  };

  const isPodcastGenerating = podcastStatus.state === "generating";

  // helper: history filtered for current doc
  const historyForDoc = podcastHistory.filter((h) => h.doc === selectedDoc);

  // handler: load a previous item into the player
  function loadHistoryById(id: string) {
    const item = podcastHistory.find((h) => h.id === id);
    if (!item) return;
    setPodcastResult({
      audio_url: item.audio_url,
      title: item.title,
      script: item.script,
      voices: item.voices,
      duration_min: item.duration_min,
    });
    setTab("podcast");
  }

  return (
    <div className="h-screen overflow-hidden bg-slate-50">
      {/* hidden inputs */}
      <input
        ref={openInputRef}
        type="file"
        accept="application/pdf"
        className="hidden"
        onChange={onOpenInputChange}
      />
      <input
        ref={bulkInputRef}
        type="file"
        accept="application/pdf"
        multiple
        className="hidden"
        onChange={onBulkInputChange}
      />

      {/* TOP BAR */}
      <div className="h-16 border-b border-slate-200/80 bg-white/90 backdrop-blur supports-[backdrop-filter]:bg-white/70">
        <div className="mx-auto max-w-[1400px] h-full px-4 flex items-center gap-4">
          {/* Brand */}
          <div className="flex items-center gap-3 min-w-[160px]">
            <div className="h-8 w-8 rounded-xl bg-indigo-600 shadow-inner" />
            <div className="text-[17px] font-semibold text-slate-900 tracking-tight">DocDots</div>
          </div>

          {/* Command bar */}
          <div className="flex-1">
            <div className="flex items-center gap-3 rounded-2xl border border-slate-200 bg-white shadow-sm px-2.5 py-2">
              <div className="flex items-center gap-2.5 flex-1">
                <svg viewBox="0 0 24 24" width="18" height="18" className="text-slate-400" aria-hidden>
                  <path
                    fill="currentColor"
                    d="M15.5 14h-.79l-.28-.27A6.471 6.471 0 0 0 16 9.5 6.5 6.5 0 1 0 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79L20 21.5 21.5 20l-6-6zM4 9.5C4 6.46 6.46 4 9.5 4S15 6.46 15 9.5 12.54 15 9.5 15 4 12.54 4 9.5Z"
                  />
                </svg>
                <input
                  className="input flex-1 h-9"
                  placeholder={selectedDoc ? "Select text in the PDF, or paste here…" : "Upload PDFs, then choose a PDF to read…"}
                  value={selection}
                  onChange={(e) => {
                    setSelection(e.target.value);
                    setSelectionSource("input");
                  }}
                />
              </div>

              <div className="hidden sm:block h-6 w-px bg-slate-200" />

              <div className="flex flex-wrap items-center gap-2">
                <button
                  className="px-3 h-9 rounded-lg bg-indigo-600 text-white font-medium hover:bg-indigo-700 disabled:opacity-60"
                  onClick={findRelatedFromSelection}
                  disabled={!selectedDoc || selectLoading || !selection.trim()}
                  title={selectedDoc ? "Search related sections & get insights" : "Upload & choose a PDF first"}
                >
                  {selectLoading ? "Finding…" : "Find Related"}
                </button>
              </div>
            </div>
          </div>

          {/* Library actions */}
          <div className="hidden sm:flex items-center gap-2">
            <button
              className="h-10 px-3 rounded-xl bg-indigo-600 text-white font-medium hover:bg-indigo-700"
              onClick={onOpenOne}
              title="Open one PDF to read now"
            >
              Open PDF
            </button>
            <button
              className="h-10 px-3 rounded-xl border border-slate-300 bg-white text-slate-700 font-medium hover:bg-slate-50"
              onClick={onBulkUpload}
              title="Upload multiple PDFs (your library)"
            >
              Upload PDFs
            </button>
          </div>
        </div>
      </div>

      {/* INITIAL WELCOME (full) */}
      {docs.length === 0 ? (
        <div className="mx-auto max-w-[1400px] px-4 py-6">
          <Welcome onUpload={(files) => doIndex(files || null, "bulk")} />
        </div>
      ) : (
        /* MAIN LAYOUT */
        <div
          className="mx-auto max-w-[1400px] px-4 py-4 grid gap-4"
          style={{ gridTemplateColumns: "280px 1fr 400px", height: "calc(100vh - 64px - 32px)" }}
        >
          {/* LEFT */}
          <aside className="rounded-2xl border border-slate-200 bg-white p-3 overflow-hidden grid grid-rows-[auto,auto,1fr,auto]">
            <div className="text-sm font-semibold text-slate-800">Documents</div>

            {freshDoc && selectedDoc !== freshDoc && (
              <div className="mt-2">
                <button
                  onClick={backToReading}
                  className="inline-flex items-center gap-1 text-xs font-medium text-indigo-700 hover:text-indigo-900 px-2 py-1 rounded-lg hover:bg-indigo-50"
                  title={`Back to “${freshDoc}” p${freshPage + 1}`}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24">
                    <path fill="currentColor" d="M20 11H7.83l5.59-5.59L12 4l-8 8l8 8l1.41-1.41L7.83 13H20v-2z" />
                  </svg>
                  Back to Reading
                </button>
              </div>
            )}

            <div className="min-h-0 overflow-auto pr-1 mt-2 flex flex-col gap-2">
              {docs.map((d) => {
                const isSelected = d === selectedDoc;
                const isFresh = d === freshDoc;
                return (
                  <button
                    key={d}
                    onClick={() => {
                      // user explicitly chooses what to read
                      setSelectedDoc(d);
                      setFreshDoc(d);
                      setFreshPage(d === selectedDoc ? page : 0);
                    }}
                    className={`w-full rounded-xl border px-3 py-2 text-left text-sm flex items-center gap-2 transition ${
                      isSelected
                        ? "border-indigo-300 bg-indigo-50 text-indigo-700"
                        : "border-slate-200 hover:bg-slate-50"
                    }`}
                  >
                    <span className="truncate">{d}</span>
                    {isFresh && (
                      <span className="ml-auto text-[10px] font-medium text-indigo-700 bg-indigo-50 border border-indigo-200 rounded-full px-2 py-[2px]">
                        Reading
                      </span>
                    )}
                  </button>
                );
              })}
            </div>

            <div className="mt-3 rounded-xl border border-slate-200 bg-slate-50 p-3">
              <div className="text-xs font-semibold text-slate-400">EXTRACTED TITLE</div>
              <div className="mt-1 text-sm text-slate-700 min-h-[22px]">{title || "—"}</div>
            </div>
          </aside>

          {/* CENTER */}
          <main className="rounded-2xl border border-slate-200 bg-white p-2 overflow-hidden">
            {selectedDoc ? (
              <div className="h-full">
                <PdfViewer
                  ref={viewerRef}
                  src={pdfUrl(selectedDoc)}
                  page={page}
                  onPageChange={(p) => setPage(p)}
                  onSelection={async (text) => {
                    const t = text.trim();
                    if (!t || t.length < 3) return;

                    await highlightSelection(t);
                    setSelection(t);
                    setTab("related");
                    setSelectLoading(true);

                    const token = Date.now();
                    (window as any).__selToken = token;

                    try {
                      const res = await selectRelated(t, 5);
                      if ((window as any).__selToken !== token) return;
                      setRecs(res?.results || []);
                      setInsightText(res?.insight || "");
                    } catch (e) {
                      console.error(e);
                    } finally {
                      if ((window as any).__selToken === token) setSelectLoading(false);
                    }
                  }}
                />
              </div>
            ) : (
              <div className="h-full grid place-items-center text-slate-400">
                Choose a PDF on the left to start reading
              </div>
            )}
          </main>

          {/* RIGHT */}
          <aside
            ref={rightRailRef}
            className="rounded-2xl border border-slate-200 bg-white overflow-hidden"
            style={{
              display: "grid",
              gridTemplateRows: `${rightTopPct}% 8px ${100 - rightTopPct}%`,
              height: "100%",
            }}
          >
            <div className="min-h-0 overflow-auto">
              <div className="sticky top-0 z-10 bg-white border-b border-slate-200 px-3 py-2">
                <div className="text-sm font-medium text-slate-700">
                  Outline <span className="text-slate-400">({outline?.length || 0})</span>
                </div>
              </div>
              <div className="p-2">
                <Outline outline={outline} onJump={(p) => setPage(p)} currentPage={page} />
              </div>
            </div>

            <div
              className="cursor-row-resize bg-slate-200 hover:bg-slate-300"
              onMouseDown={onDragStart}
              title="Drag to resize"
            />

            <div className="min-h-0 overflow-auto">
              <div className="sticky top-0 z-10 bg-white border-b border-slate-200 px-2 py-2 flex items-center gap-2">
                <button
                  disabled={!selectedDoc}
                  className={`flex-1 rounded-xl px-3 py-2 text-sm font-medium disabled:opacity-60 disabled:pointer-events-none ${
                    tab === "related"
                      ? "bg-indigo-600 text-white"
                      : "bg-white text-slate-600 border border-slate-200"
                  }`}
                  onClick={() => setTab("related")}
                >
                  Related
                </button>
                <button
                  disabled={!selectedDoc}
                  className={`flex-1 rounded-xl px-3 py-2 text-sm font-medium disabled:opacity-60 disabled:pointer-events-none ${
                    tab === "insights"
                      ? "bg-indigo-600 text-white"
                      : "bg-white text-slate-600 border border-slate-200"
                  }`}
                  onClick={() => setTab("insights")}
                >
                  Insights
                </button>
                <button
                  disabled={!selectedDoc}
                  className={`flex-1 rounded-xl px-3 py-2 text-sm font-medium relative disabled:opacity-60 disabled:pointer-events-none ${
                    tab === "podcast"
                      ? "bg-indigo-600 text-white"
                      : "bg-white text-slate-600 border border-slate-200"
                  }`}
                  onClick={() => setTab("podcast")}
                  title="Generate a podcast from the current page/selection"
                >
                  Podcast
                  {isPodcastGenerating && (
                    <span
                      className="absolute -right-2 -top-1 w-2 h-2 rounded-full bg-amber-500 animate-pulse"
                      aria-hidden
                    />
                  )}
                </button>
              </div>

              <div className="p-2 pb-5">
                {tab === "related" ? (
                  <Recommendations
                    items={recs}
                    loading={selectLoading}
                    onJump={(doc: string, p: number) => {
                      // Jump to another doc/page; Back to Reading still knows where to go
                      setSelectedDoc(doc);
                      setPage(p);
                    }}
                  />
                ) : tab === "insights" ? (
                  <Insights text={insightText} loading={selectLoading} />
                ) : (
                  <Podcast
                    document={selectedDoc ?? null}
                    page={page}
                    selection={selection}
                    externalAudioUrl={podcastResult.audio_url}
                    externalTitle={podcastResult.title}
                    externalScript={podcastResult.script}
                    generatingExternal={isPodcastGenerating}
                    history={historyForDoc}
                    onLoadFromHistory={loadHistoryById}
                    onStart={handlePodcastStart}
                    onComplete={handlePodcastComplete}
                    onError={handlePodcastError}
                  />
                )}
              </div>
            </div>
          </aside>
        </div>
      )}

      {/* 🔔 Bottom-right toast (collapsible) */}
      {podcastStatus.state !== "idle" && (
        <>
          {!toastMinimized ? (
            <div className="fixed bottom-4 right-4 z-[70]">
              <div className="max-w-sm rounded-xl border border-slate-200 bg-white shadow-lg p-3">
                <div className="flex items-start gap-3">
                  {podcastStatus.state === "generating" && (
                    <svg className="w-5 h-5 mt-0.5 animate-spin text-indigo-600" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 0 1 8-8v4a4 4 0 0 0-4 4H4z"/>
                    </svg>
                  )}
                  {podcastStatus.state === "ready" && (
                    <svg className="w-5 h-5 mt-0.5 text-emerald-600" viewBox="0 0 24 24" fill="currentColor">
                      <path d="M9 16.2 4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4z" />
                    </svg>
                  )}
                  {podcastStatus.state === "error" && (
                    <svg className="w-5 h-5 mt-0.5 text-rose-600" viewBox="0 0 24 24" fill="currentColor">
                      <path d="M1 21h22L12 2 1 21zm12-3h-2v2h2v-2zm0-8h-2v6h2V10z"/>
                    </svg>
                  )}

                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <div className="text-sm font-medium text-slate-900 truncate">
                          {podcastStatus.state === "generating" && "Generating podcast…"}
                          {podcastStatus.state === "ready" && (podcastStatus.title || "Podcast ready")}
                          {podcastStatus.state === "error" && "Something went wrong"}
                        </div>
                        <div className="text-xs text-slate-600 mt-0.5">{podcastStatus.message}</div>
                      </div>

                      <div className="shrink-0 flex items-center gap-1">
                        <button
                          className="text-slate-500 hover:text-slate-700 rounded p-1"
                          title="Minimize"
                          onClick={() => setToastMinimized(true)}
                          aria-label="Minimize notification"
                        >
                          <svg width="16" height="16" viewBox="0 0 24 24">
                            <path fill="currentColor" d="M19 13H5v-2h14v2z" />
                          </svg>
                        </button>
                        <button
                          className="text-slate-500 hover:text-slate-700 rounded p-1"
                          title="Dismiss"
                          onClick={() => setPodcastStatus({ state: "idle" })}
                          aria-label="Dismiss notification"
                        >
                          <svg width="16" height="16" viewBox="0 0 24 24">
                            <path fill="currentColor" d="M18.3 5.71 12 12.01 5.7 5.7 4.29 7.11 10.59 13.4 4.29 19.7l1.41 1.41L12 14.82l6.29 6.3 1.41-1.42-6.29-6.29 6.29-6.29z"/>
                          </svg>
                        </button>
                      </div>
                    </div>

                    <div className="mt-2 flex items-center gap-2">
                      {podcastStatus.state === "ready" && podcastResult.audio_url && (
                        <>
                          <a
                            href={podcastResult.audio_url}
                            className="text-xs px-2 py-1 rounded-lg bg-indigo-600 text-white hover:bg-indigo-700"
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            Open audio
                          </a>
                          <button
                            className="text-xs px-2 py-1 rounded-lg border border-slate-300 hover:bg-slate-50"
                            onClick={() => setTab("podcast")}
                          >
                            Go to Podcast
                          </button>
                        </>
                      )}
                      {podcastStatus.state === "generating" && (
                        <button
                          className="text-xs px-2 py-1 rounded-lg border border-slate-300 hover:bg-slate-50"
                          onClick={() => setTab("podcast")}
                        >
                          View section
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            // Minimized pill
            <button
              className="fixed bottom-4 right-4 z-[70] rounded-full shadow-lg border border-slate-200 bg-white px-3 h-10 flex items-center gap-2"
              onClick={() => setToastMinimized(false)}
              title="Expand podcast notification"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" className="text-indigo-600">
                <path
                  fill="currentColor"
                  d="M12 3a6 6 0 0 0-6 6v4a4 4 0 0 0 4 4v4h2v-4h2v4h2v-4a4 4 0 0 0 4-4V9a6 6 0 0 0-6-6Zm0 2a4 4 0 0 1 4 4v4a2 2 0 0 1-2 2H10a2 2 0 0 1-2-2V9a4 4 0 0 1 4-4Z"
                />
              </svg>
              <span className="text-xs text-slate-800">
                {podcastStatus.state === "generating"
                  ? "Podcast… (working)"
                  : podcastStatus.state === "ready"
                  ? "Podcast ready"
                  : "Podcast"}
              </span>
              {podcastStatus.state === "generating" && (
                <span className="ml-1 w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
              )}
            </button>
          )}
        </>
      )}

      {/* Indexing overlay */}
      {isIndexing && (
        <div className="fixed inset-0 z-50 bg-black/20 backdrop-blur-sm grid place-items-center">
          <div className="w-[520px] max-w-[90vw] rounded-2xl bg-white p-6 shadow-xl border border-slate-200">
            <div className="text-lg font-semibold text-slate-900 mb-3">
              {statusTitle === "open" ? "Opening your PDF…" : "Indexing your PDFs…"}
            </div>
            <div className="h-2 rounded bg-slate-200 overflow-hidden mb-2">
              <div
                className="h-2 bg-indigo-600 transition-all"
                style={{ width: `${Math.min(100, Math.max(5, statusPct))}%` }}
              />
            </div>
            <div className="text-sm text-slate-700">{statusText}</div>
            <div className="text-xs mt-2 text-slate-500">
              Tip: first run may download an embedding model; progress shows here.
            </div>
            <div className="mt-4 flex justify-end">
              <button
                onClick={() => {
                  stopStatusPolling();
                  setIsIndexing(false);
                }}
                className="text-sm rounded-lg border border-slate-300 px-3 py-1.5 hover:bg-slate-50"
              >
                Dismiss
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}