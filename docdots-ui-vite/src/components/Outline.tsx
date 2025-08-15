import React from "react";

export type OutlineItem = { level: string; text: string; page: number };

type Props = {
  outline: OutlineItem[];
  onJump: (page: number) => void;
};

const levelColor: Record<string, string> = {
  H1: "border-indigo-500 text-indigo-700 bg-indigo-50",
  H2: "border-indigo-500 text-indigo-700 bg-indigo-50",
  H3: "border-slate-300 text-slate-700 bg-slate-50",
  H4: "border-slate-300 text-slate-700 bg-slate-50",
};

export default function Outline({ outline, onJump }: Props) {
  if (!outline || outline.length === 0) {
    return <div className="text-sm text-slate-500">No outline detected.</div>;
  }

  return (
    <ul className="space-y-2">
      {outline.map((item, idx) => {
        const badge = item.level?.toUpperCase() || "H1";
        const badgeStyle = levelColor[badge] || levelColor.H3;

        return (
          <li key={`${item.page}-${idx}`}>
            <div className="flex items-start justify-between gap-3 rounded-lg border border-slate-200 p-3 hover:bg-slate-50">
              <div className="min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span
                    className={`inline-flex items-center justify-center h-5 px-2 text-[10px] font-semibold rounded ${badgeStyle} border`}
                  >
                    {badge}
                  </span>
                  <span className="text-[10px] text-slate-700 bg-slate-100 border border-slate-200 rounded px-1 py-0.5">
                    p{item.page + 1}
                  </span>
                </div>
                <div className="text-sm font-medium line-clamp-2">{item.text}</div>
              </div>

              <button
                onClick={() => onJump(item.page)}
                className="shrink-0 h-8 px-3 rounded-lg text-sm border border-indigo-300 text-indigo-700 hover:bg-indigo-50"
              >
                Jump
              </button>
            </div>
          </li>
        );
      })}
    </ul>
  );
}