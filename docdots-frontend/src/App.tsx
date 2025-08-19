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

// Persisted last result so Podcast tab can always show it
type PodcastResult = { audio_url: string | null; title: string | null };

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
  const [_, setSelectionSource] = useState<"pdf" | "input">("input");
  const [selectLoading, setSelectLoading] = useState(false);

  const [isIndexing, setIsIndexing] = useState(false);
  const [statusText, setStatusText] = useState("Starting…");
  const [statusPct, setStatusPct] = useState(5);
  const [statusTitle, setStatusTitle] = useState<"open" | "bulk">("bulk");
  const pollRef = useRef<number | null>(null);
  const timeoutRef = useRef<number | null>(null);

  const viewerRef = useRef<PdfViewerHandle>(null);
  const openInputRef = useRef<HTMLInputElement>(null);
  const bulkInputRef = useRef<HTMLInputElement>(null);

  const [rightTopPct, setRightTopPct] = useState(56);
  const rightRailRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ startY: number; startPct: number } | null>(null);

  // ---- Podcast: global toast + persisted result
  const [podcastStatus, setPodcastStatus] = useState<PodcastStatus>({ state: "idle" });
  const [toastMinimized, setToastMinimized] = useState(false);
  const [podcastResult, setPodcastResult] = useState<PodcastResult>({
    audio_url: null,
    title: null,
  });

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
        if (!selectedDoc && docs?.length) setSelectedDoc(docs[0]);
      } catch (e) {
        console.error(e);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!selectedDoc) return;
    let cancelled = false;
    (async () => {
      try {
        const o = await fetchOutline(selectedDoc);
        if (cancelled) return;
        setTitle(o.title || "");
        setOutline(o.outline || []);
        setPage(0);
      } catch {
        if (!cancelled) {
          setOutline([]);
          setTitle("");
        }
      }
      try {
        const r = await fetchRecommendations(selectedDoc, 0);
        if (!cancelled) setRecs(r?.results || []);
      } catch {
        if (!cancelled) setRecs([]);
      }
      try {
        const i = await fetchInsights(selectedDoc, 0);
        if (!cancelled) setInsightText(i?.text || "");
      } catch {
        if (!cancelled) setInsightText("");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedDoc]);

  useEffect(() => {
    if (!selectedDoc) return;
    let cancelled = false;
    (async () => {
      try {
        const r = await fetchRecommendations(selectedDoc, page);
        if (!cancelled) setRecs(r?.results || []);
      } catch {
        if (!cancelled) setRecs([]);
      }
      try {
        const i = await fetchInsights(selectedDoc, page);
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
    pollRef.current = window.setInterval(async () => {
      try {
        const s = await getStatus();
        setStatusText(s.message || s.phase);
        setStatusPct(s.progress ?? 10);
        if (s.phase === "ready" || s.phase === "error") {
          stopStatusPolling();
          setIsIndexing(false);
        }
      } catch {}
    }, 800) as unknown as number;

    timeoutRef.current = window.setTimeout(() => {
      stopStatusPolling();
      setIsIndexing(false);
    }, 120_000) as unknown as number;
  }
  function stopStatusPolling() {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
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
      setStatusText(kind === "open" ? "Opening & indexing…" : "Uploading & indexing…");
      startStatusPolling(kind);

      await indexPdfFiles(files);

      const { docs } = await fetchDocs();
      setDocs(docs || []);

      if (kind === "open") {
        const just = files.item(0)?.name;
        if (just) {
          setFreshDoc(just);
          setFreshPage(0);
          setSelectedDoc(just);
        }
      } else if (!selectedDoc && docs?.length) {
        setSelectedDoc(docs[0]);
      }
    } catch (e) {
      console.error(e);
    } finally {
      stopStatusPolling();
      setIsIndexing(false);
      if (openInputRef.current) openInputRef.current.value = "";
      if (bulkInputRef.current) bulkInputRef.current.value = "";
    }
  }

  function onOpenOne() {
    openInputRef.current?.click();
  }
  function onBulkUpload() {
    bulkInputRef.current?.click();
  }
  async function onOpenInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    await doIndex(e.target.files, "open");
  }
  async function onBulkInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    await doIndex(e.target.files, "bulk");
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

  // selection from the toolbar button
  async function useFromPdf() {
    const t = (await viewerRef.current?.getSelection?.()) || "";
    const q = t.trim();
    setSelection(t);
    setSelectionSource("pdf");
    if (q) {
      await highlightSelection(q);
    }
  }

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
    setSelectedDoc(freshDoc);
    setPage(freshPage || 0);
  }

  const outlineCount = outline?.length || 0;

  // ---- Global podcast events (component also calls these via props)
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
      setPodcastResult({
        audio_url: detail.audio_url ?? null,
        title: detail.title ?? "Audio overview",
      });
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
  }, []);

  // ---- Optional: callbacks passed to <Podcast /> (use either props OR events)
  const handlePodcastStart = () => {
    setPodcastStatus({
      state: "generating",
      message: "Generating podcast… this can take a few minutes. You can keep browsing.",
    });
    setToastMinimized(false);
  };
  const handlePodcastComplete = (res?: { audio_url?: string; title?: string }) => {
    setPodcastResult({
      audio_url: res?.audio_url ?? null,
      title: res?.title ?? "Audio overview",
    });
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
                  placeholder="Select text in the PDF, or paste here…"
                  value={selection}
                  onChange={(e) => {
                    setSelection(e.target.value);
                    setSelectionSource("input");
                  }}
                />
              </div>

              <div className="h-6 w-px bg-slate-200" />

              <button
                className="px-3 h-9 rounded-lg text-indigo-600 hover:text-indigo-700 hover:bg-indigo-50"
                onClick={useFromPdf}
                title="Grab selection from the PDF"
              >
                Use from PDF
              </button>
              <button
                className="px-3 h-9 rounded-lg bg-indigo-600 text-white font-medium hover:bg-indigo-700 disabled:opacity-60"
                onClick={findRelatedFromSelection}
                disabled={selectLoading || !selection.trim()}
                title="Search related sections & get insights"
              >
                {selectLoading ? "Finding…" : "Find Related"}
              </button>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              className="h-10 px-3 rounded-xl bg-indigo-600 text-white font-medium hover:bg-indigo-700"
              onClick={onOpenOne}
              title="Open one PDF (current)"
            >
              Open PDF
            </button>
            <button
              className="h-10 px-3 rounded-xl border border-slate-300 bg-white text-slate-700 font-medium hover:bg-slate-50"
              onClick={onBulkUpload}
              title="Upload multiple PDFs (past docs)"
            >
              Upload PDFs
            </button>
          </div>
        </div>
      </div>

      {/* MAIN LAYOUT */}
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
            {docs.length === 0 && (
              <Welcome onUpload={(files?: FileList | null) => doIndex(files || null, "bulk")} />
            )}
            {docs.map((d) => {
              const isSelected = d === selectedDoc;
              const isFresh = d === freshDoc;
              return (
                <button
                  key={d}
                  onClick={() => {
                    setSelectedDoc(d);
                    if (isFresh) setPage(freshPage);
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
                onPageChange={(p) => {
                  setPage(p);
                }}
                onSelection={async (text) => {
                  const t = text.trim();
                  if (!t || t.length < 3) return;

                  // yellow highlight in the viewer
                  await highlightSelection(t);

                  // UI + backend fetch
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
              Open a PDF or upload your library
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
                Outline <span className="text-slate-400">({outlineCount})</span>
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
                className={`flex-1 rounded-xl px-3 py-2 text-sm font-medium ${
                  tab === "related"
                    ? "bg-indigo-600 text-white"
                    : "bg-white text-slate-600 border border-slate-200"
                }`}
                onClick={() => setTab("related")}
              >
                Related
              </button>
              <button
                className={`flex-1 rounded-xl px-3 py-2 text-sm font-medium ${
                  tab === "insights"
                    ? "bg-indigo-600 text-white"
                    : "bg-white text-slate-600 border border-slate-200"
                }`}
                onClick={() => setTab("insights")}
              >
                Insights
              </button>
              <button
                className={`flex-1 rounded-xl px-3 py-2 text-sm font-medium relative ${
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
                  // pass persisted result + generating flag
                  externalAudioUrl={podcastResult.audio_url}
                  externalTitle={podcastResult.title}
                  generatingExternal={isPodcastGenerating}
                  // optional prop callbacks (use these OR the window events in Podcast.tsx)
                  onStart={handlePodcastStart}
                  onComplete={handlePodcastComplete}
                  onError={handlePodcastError}
                />
              )}
            </div>
          </div>
        </aside>
      </div>

      {/* 🔔 Bottom-right toast for long-running podcast generation (collapsible) */}
      {podcastStatus.state !== "idle" && (
        <>
          {!toastMinimized ? (
            <div className="fixed bottom-4 right-4 z-[70]">
              <div className="max-w-sm rounded-xl border border-slate-200 bg-white shadow-lg p-3">
                <div className="flex items-start gap-3">
                  {/* icon */}
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
                        {/* Minimize */}
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
                        {/* Close */}
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
                {podcastStatus.state === "generating" ? "Podcast… (working)" : podcastStatus.state === "ready" ? "Podcast ready" : "Podcast"}
              </span>
              {podcastStatus.state === "generating" && <span className="ml-1 w-2 h-2 rounded-full bg-amber-500 animate-pulse" />}
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