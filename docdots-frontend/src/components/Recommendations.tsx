// Removed unused React import

type RecItem = {
  document: string;
  section_title: string | null;
  page_number: number;   // zero-based
  score?: number;        // similarity score (any scale)
  snippet?: string;      // will be trimmed to 2–4 sentences for display
};

type Props = {
  items: RecItem[];
  loading: boolean;
  onJump: (docName: string, page: number) => void;
};

/** Keep 2–4 sentences; fall back gracefully if snippet is short. */
function trimToTwoToFourSentences(snippet?: string): string {
  const s = (snippet || "").trim();
  if (!s) return "";
  // Split on end-of-sentence periods/question/exclamation (handles “. ”, “? ”, “! ”)
  const parts = s.split(/(?<=[.!?])\s+/).filter(Boolean);
  if (parts.length <= 4) return s;
  return parts.slice(0, 4).join(" ");
}

/** Score to pretty pill (green/gray/red). We keep it generic since sources differ. */
// inside Recommendations.tsx, replace ScorePill with:
function ScorePill({ score }: { score: number }) {
  // Treat 0..1 as cosine-ish; 1..100 as percent
  let display: string;
  if (score >= 0 && score <= 1) display = `${Math.round(score * 100)}%`;
  else if (score > 1 && score <= 100) display = `${Math.round(score)}%`;
  else display = `${Math.round(score)}`;

  const tone = score > 0 ? "bg-emerald-600" : score < 0 ? "bg-rose-600" : "bg-slate-500";
  return (
    <span className={`text-[10px] font-medium ${tone} text-white rounded px-1.5 py-0.5`}>
      {display}
    </span>
  );
}

export default function Recommendations({ items, loading, onJump }: Props) {
  if (loading) {
    return (
      <div className="space-y-3">
        {Array.from({ length: 3 }).map((_, i) => (
          <div
            key={i}
            className="rounded-lg border border-slate-200 p-3 bg-white"
          >
            <div className="h-4 w-1/2 bg-slate-200 rounded mb-2 animate-pulse" />
            <div className="h-3 w-5/6 bg-slate-200 rounded mb-1 animate-pulse" />
            <div className="h-3 w-1/2 bg-slate-200 rounded animate-pulse" />
          </div>
        ))}
      </div>
    );
  }

  if (!items || items.length === 0) {
    return <div className="text-sm text-slate-500">No related content yet.</div>;
  }

  return (
    <div className="space-y-3">
      {items.map((it, idx) => {
        const snippet = trimToTwoToFourSentences(it.snippet);
        return (
          <article
            key={`${it.document}-${it.page_number}-${idx}`}
            className="rounded-lg border border-slate-200 p-3 bg-slate-50 hover:bg-slate-100/60 transition"
          >
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                {/* Document title (clickable jump) */}
                <button
                  className="text-left text-sm font-semibold line-clamp-1 hover:underline"
                  title={`Open ${it.document} • p${it.page_number + 1}`}
                  onClick={() => onJump(it.document, it.page_number)}
                >
                  {it.document}
                </button>

                {/* Section title */}
                <div className="mt-1 text-xs text-slate-600 line-clamp-2">
                  {it.section_title || "—"}
                </div>

                {/* Snippet */}
                {snippet && (
                  <div className="mt-1 text-xs text-slate-700 line-clamp-3">
                    {snippet}
                  </div>
                )}

                {/* Meta row */}
                <div className="mt-2 flex items-center gap-2">
                  <span className="text-[10px] text-slate-700 bg-slate-100 border border-slate-200 rounded px-1 py-0.5">
                    p{it.page_number + 1}
                  </span>
                  {typeof it.score === "number" && <ScorePill score={it.score} />}
                </div>
              </div>

              {/* Primary action */}
              <button
                className="shrink-0 h-8 px-3 rounded-lg text-sm border border-indigo-300 text-indigo-700 hover:bg-indigo-50"
                onClick={() => onJump(it.document, it.page_number)}
                aria-label={`Jump to p${it.page_number + 1} in ${it.document}`}
              >
                Jump
              </button>
            </div>
          </article>
        );
      })}
    </div>
  );
}
