import React, { useEffect, useMemo, useState } from "react";
import {
  BACKEND,
  listDocs,
  uploadIndex,
  fetchOutline,
  fetchRecommendations,
  fetchInsights,
  makePodcast,
  pdfUrl,
  type DocInfo,
} from "./api";
import PdfViewer from "./components/PdfViewer";
import Outline from "./components/Outline";
import Recommendations from "./components/Recommendations";
import Insights from "./components/Insights";
import Welcome from "./components/Welcome";

type OutlineItem = { level: string; text: string; page: number };
type OutlineResp = { title: string; outline: OutlineItem[] };

export default function App() {
  // global
  const [domain, setDomain] = useState<string>("");
  const [persona, setPersona] = useState<string>("Food lover");
  const [job, setJob] = useState<string>("Explore recipes");

  // docs
  const [docs, setDocs] = useState<DocInfo[]>([]);
  const [loadingDocs, setLoadingDocs] = useState(false);
  const [error, setError] = useState<string>("");

  // selection
  const [selectedDoc, setSelectedDoc] = useState<string | null>(null);
  const [selectedDocPages, setSelectedDocPages] = useState<number>(0);
  const [outline, setOutline] = useState<OutlineResp | null>(null);
  const [page, setPage] = useState<number>(0);

  // right pane tabs
  const [tab, setTab] = useState<"outline" | "related" | "insights">("related");

  // recommendations / insights
  const [recs, setRecs] = useState<any[]>([]);
  const [recsLoading, setRecsLoading] = useState(false);

  const [insight, setInsight] = useState<string>("");
  const [insightLoading, setInsightLoading] = useState(false);

  const [audioURL, setAudioURL] = useState<string>(""); // podcast

  // -------- initial docs load --------
  useEffect(() => {
    (async () => {
      try {
        setLoadingDocs(true);
        const res = await listDocs();
        setDocs(res.documents || []);
        setDomain(res.domain || "");
        if (!selectedDoc && res.documents?.length) {
          const first = res.documents[0];
          await handleSelectDoc(first.name, first.pages);
        }
      } catch (e: any) {
        setError(e?.message || "Failed to load docs");
      } finally {
        setLoadingDocs(false);
      }
    })();
  }, []); // eslint-disable-line

  // keyboard: ← / → to flip pages when a doc is open
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!selectedDoc) return;
      if (e.key === "ArrowLeft") setPage((p) => Math.max(0, p - 1));
      if (e.key === "ArrowRight") setPage((p) => Math.min(selectedDocPages - 1, p + 1));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectedDoc, selectedDocPages]);

  // -------- select doc helper --------
  const handleSelectDoc = async (name: string, pages: number) => {
    try {
      setSelectedDoc(name);
      setSelectedDocPages(pages);
      setPage(0);
      setOutline(null);
      setAudioURL("");
      setInsight("");
      const o = await fetchOutline(name);
      setOutline(o);
    } catch (e: any) {
      setError(e?.message || "Failed to load outline");
    }
  };

  // -------- upload + index --------
  const onUpload = async (files: FileList | null) => {
    if (!files || !files.length) return;
    try {
      setError("");
      const fs = Array.from(files);
      await uploadIndex(fs, persona, job);
      const res = await listDocs();
      setDocs(res.documents || []);
      setDomain(res.domain || "");
      if (res.documents?.length) {
        const last = res.documents[res.documents.length - 1];
        await handleSelectDoc(last.name, last.pages);
      }
    } catch (e: any) {
      setError(e?.message || "Upload/index failed");
    }
  };

  // -------- computed helpers --------
  const pdfSrc = useMemo(() => (selectedDoc ? pdfUrl(selectedDoc) : ""), [selectedDoc]);

  const sectionTitle = useMemo(() => {
    if (!outline) return "";
    const match = outline.outline.find((o) => o.page === page);
    return match?.text || "";
  }, [outline, page]);

  // -------- fetch related / insights on change --------
  useEffect(() => {
    const loadRecs = async () => {
      if (!selectedDoc) return;
      try {
        setRecsLoading(true);
        const data = await fetchRecommendations(
          selectedDoc,
          page,
          sectionTitle || undefined,
          5
        );
        setRecs(data.items || []);
      } catch (e: any) {
        // keep quiet on first screen
        setError(selectedDoc ? (e?.message || "Failed to load recommendations") : "");
      } finally {
        setRecsLoading(false);
      }
    };
    loadRecs();
  }, [selectedDoc, page, sectionTitle]);

  useEffect(() => {
    const loadInsight = async () => {
      if (!selectedDoc) return;
      try {
        setInsightLoading(true);
        const data = await fetchInsights(selectedDoc, page, 3);
        setInsight(data.insight || "");
      } catch (e: any) {
        setError(selectedDoc ? (e?.message || "Failed to load insights") : "");
      } finally {
        setInsightLoading(false);
      }
    };
    if (tab === "insights") loadInsight();
  }, [tab, selectedDoc, page]);

  // -------- podcast --------
  const onMakePodcast = async () => {
    try {
      setAudioURL("");
      if (!insight) return;
      const r = await makePodcast(insight);
      const url = r.audio_url?.startsWith("http") ? r.audio_url : `${BACKEND}${r.audio_url || ""}`;
      setAudioURL(url);
    } catch (e: any) {
      setError(e?.message || "Podcast failed");
    }
  };

  const showWelcome = !loadingDocs && docs.length === 0;

  return (
    <div className="min-h-screen bg-white text-slate-900">
      {/* Header */}
      <header className="h-16 border-b border-slate-200 bg-slate-50 flex items-center px-6 justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-indigo-600" />
          <h1 className="font-bold text-lg">DocDots</h1>
          {domain && (
            <span className="ml-3 text-xs font-medium text-indigo-700 bg-indigo-50 border border-indigo-200 rounded-full px-2 py-0.5">
              {domain}
            </span>
          )}
        </div>

        <div className="hidden md:flex items-center gap-3">
          <input
            type="text"
            value={persona}
            onChange={(e) => setPersona(e.target.value)}
            placeholder="Persona"
            className="h-9 w-40 rounded-xl border border-slate-200 px-3 text-sm focus:ring-2 focus:ring-indigo-300"
          />
          <input
            type="text"
            value={job}
            onChange={(e) => setJob(e.target.value)}
            placeholder="Job to be done"
            className="h-9 w-56 rounded-xl border border-slate-200 px-3 text-sm focus:ring-2 focus:ring-indigo-300"
          />
          <label className="cursor-pointer inline-flex items-center gap-2 px-4 h-9 rounded-xl bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-700">
            <svg width="16" height="16" viewBox="0 0 24 24" className="opacity-90">
              <path fill="currentColor" d="M19 13v6H5v-6H3v8h18v-8zM11 3v10.17l-3.59-3.58L6 11l6 6l6-6l-1.41-1.41L13 13.17V3z"/>
            </svg>
            Upload PDFs
            <input type="file" multiple className="hidden" onChange={(e) => onUpload(e.target.files)} />
          </label>
        </div>
      </header>

      {/* Welcome screen when no docs yet */}
      {showWelcome ? (
        <div className="p-6">
          <Welcome
            onUpload={onUpload}
            persona={persona}
            setPersona={setPersona}
            job={job}
            setJob={setJob}
          />
        </div>
      ) : (
        // App layout
        <div className="grid grid-cols-1 lg:grid-cols-[320px_minmax(0,1fr)_360px] gap-6 p-6">
          {/* Left: docs & extracted title */}
          <aside className="space-y-4">
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              <div className="flex items-center justify-between mb-3">
                <h2 className="font-semibold">Documents</h2>
                {loadingDocs && <span className="text-xs text-slate-500">Loading…</span>}
              </div>

              {docs.length === 0 ? (
                <div className="text-center py-10">
                  <div className="mx-auto w-12 h-12 rounded-full border border-indigo-300 bg-indigo-50 flex items-center justify-center mb-2">
                    <div className="w-5 h-5 border-2 border-indigo-600 rounded-sm"></div>
                  </div>
                  <div className="text-sm font-medium">No documents found</div>
                  <div className="text-xs text-slate-500">Upload PDFs to get started</div>
                </div>
              ) : (
                <ul className="space-y-2">
                  {docs.map((d) => {
                    const active = d.name === selectedDoc;
                    return (
                      <li key={d.name}>
                        <button
                          className={`w-full text-left px-3 py-2 rounded-lg border text-sm transition ${
                            active ? "border-indigo-300 bg-indigo-50" : "border-slate-200 hover:bg-slate-50"
                          }`}
                          onClick={() => handleSelectDoc(d.name, d.pages)}
                        >
                          <div className="flex items-center justify-between">
                            <span className="font-medium line-clamp-1">{d.name}</span>
                            <span className="text-[10px] text-slate-600 bg-slate-100 border border-slate-200 rounded px-1 py-0.5">
                              {d.pages}p
                            </span>
                          </div>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>

            <div className="rounded-xl border border-slate-200 p-4">
              <div className="text-xs font-semibold text-slate-500 mb-1">EXTRACTED TITLE</div>
              <div className="text-sm">
                {outline?.title ? (
                  <span className="font-medium">{outline.title}</span>
                ) : (
                  <span className="text-slate-500">—</span>
                )}
              </div>
            </div>
          </aside>

          {/* Middle */}
          <main className="space-y-4">
            <div className="rounded-2xl border border-slate-200 overflow-hidden">
              {selectedDoc ? (
                <PdfViewer src={pdfSrc} page={page} onPageChange={(p: number) => setPage(p)} />
              ) : (
                <div className="p-12 text-center text-slate-500 text-sm">Select a document</div>
              )}
            </div>
          </main>

          {/* Right: tabs */}
          <aside className="flex flex-col gap-4 h-[calc(100vh-6rem)]">
            <div className="rounded-xl border border-slate-200 p-3">
              <div className="flex p-1 bg-slate-50 rounded-lg border border-slate-200 w-full">
                <button
                  className={`flex-1 h-9 rounded-md text-sm font-medium ${
                    tab === "outline" ? "bg-indigo-600 text-white" : "text-slate-700"
                  }`}
                  onClick={() => setTab("outline")}
                >
                  Outline
                </button>
                <button
                  className={`flex-1 h-9 rounded-md text-sm font-medium ${
                    tab === "related" ? "bg-indigo-600 text-white" : "text-slate-700"
                  }`}
                  onClick={() => setTab("related")}
                >
                  Related
                </button>
                <button
                  className={`flex-1 h-9 rounded-md text-sm font-medium ${
                    tab === "insights" ? "bg-indigo-600 text-white" : "text-slate-700"
                  }`}
                  onClick={() => setTab("insights")}
                >
                  Insights
                </button>
              </div>
            </div>

            <div className="flex-1 min-h-0 overflow-y-auto p-2">
              {tab === "outline" ? (
                <div className="rounded-xl border border-slate-200 p-4 h-full flex flex-col">
                  <div className="flex items-center justify-between mb-3">
                    <h2 className="font-semibold">Outline</h2>
                    <div className="text-xs text-slate-500">
                      {outline?.outline?.length || 0} sections
                    </div>
                  </div>
                  <div className="flex-1 min-h-0 overflow-y-auto pr-1">
                    <Outline
                      outline={outline?.outline || []}
                      onJump={(p: number) =>
                        setPage(Math.min(Math.max(p, 0), Math.max(0, selectedDocPages - 1)))
                      }
                    />
                  </div>
                </div>
              ) : tab === "related" ? (
                <div className="rounded-xl border border-slate-200 p-4">
                  <Recommendations
                    items={recs}
                    loading={recsLoading}
                    onJump={(docName: string, pageNo: number) => {
                      const doc = docs.find((d) => d.name === docName);
                      if (doc) {
                        handleSelectDoc(doc.name, doc.pages).then(() =>
                          setPage(Math.min(Math.max(pageNo, 0), Math.max(0, doc.pages - 1)))
                        );
                      }
                    }}
                  />
                </div>
              ) : (
                <div className="rounded-xl border border-slate-200 p-4 space-y-4">
                  <Insights text={insight} loading={insightLoading} />
                  <div className="flex items-center gap-3">
                    <button
                      onClick={onMakePodcast}
                      className="h-9 px-4 rounded-xl bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-700"
                    >
                      Make Podcast
                    </button>
                    {audioURL && (
                      <audio controls src={audioURL} className="w-full">
                        Your browser does not support the audio element.
                      </audio>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Error toast: show only when a doc is selected */}
            {error && selectedDoc && (
              <div className="rounded-xl border border-rose-200 bg-rose-50 p-3 text-rose-700 text-sm">
                {error}
                <button className="ml-2 text-rose-800 underline" onClick={() => setError("")}>
                  dismiss
                </button>
              </div>
            )}
          </aside>
        </div>
      )}
    </div>
  );
}