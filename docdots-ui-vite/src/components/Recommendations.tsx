import React from "react";

type RecItem = {
  document: string;
  section_title: string | null;
  page_number: number;
  score?: number;
  snippet?: string;
};

type Props = {
  items: RecItem[];
  loading: boolean;
  onJump: (docName: string, page: number) => void;
};

export default function Recommendations({ items, loading, onJump }: Props) {
  if (loading) {
    return (
      <div className="space-y-3">
        {Array.from({ length: 3 }).map((_, i) => (
          <div key={i} className="rounded-lg border border-slate-200 p-3">
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
      {items.map((it, idx) => (
        <div key={idx} className="rounded-lg border border-slate-200 p-3 bg-slate-50">
          <div className="flex items-start justify-between">
            <div className="min-w-0">
              <div className="text-sm font-semibold line-clamp-1">{it.document}</div>
              <div className="mt-1 text-xs text-slate-600 line-clamp-2">
                {it.section_title || "—"}
              </div>
              {it.snippet && (
                <div className="mt-1 text-xs text-slate-700 line-clamp-2">{it.snippet}</div>
              )}
              <div className="mt-2 flex items-center gap-2">
                <span className="text-[10px] text-slate-700 bg-slate-100 border border-slate-200 rounded px-1 py-0.5">
                  p{it.page_number + 1}
                </span>
                {typeof it.score === "number" && (
                  <span className="text-[10px] font-medium bg-emerald-600 text-white rounded px-1.5 py-0.5">
                    {Math.round(it.score * 100)}%
                  </span>
                )}
              </div>
            </div>
            <button
              className="shrink-0 h-8 px-3 rounded-lg text-sm border border-indigo-300 text-indigo-700 hover:bg-indigo-50"
              onClick={() => onJump(it.document, it.page_number)}
            >
              Jump
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}