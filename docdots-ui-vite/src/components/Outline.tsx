// src/components/Outline.tsx
import React from "react";

type Item = { level: string; text: string; page: number };
type Props = {
  outline: Item[];
  onJump: (p: number) => void;
  currentPage?: number;
};

export default function Outline({ outline, onJump, currentPage = -1 }: Props) {
  if (!outline || outline.length === 0) {
    return <div className="text-sm text-slate-500 p-2">No outline detected.</div>;
  }

  function levelPad(level: string) {
    switch (level) {
      case "H1":
        return "pl-1";
      case "H2":
        return "pl-4";
      case "H3":
        return "pl-7";
      case "H4":
        return "pl-10";
      default:
        return "pl-1";
    }
  }

  return (
    <div className="space-y-1">
      {outline.map((it, idx) => {
        const active = it.page === currentPage;
        return (
          <button
            key={idx}
            className={`w-full text-left rounded-lg px-2 py-1 text-sm border ${
              active
                ? "border-indigo-300 bg-indigo-50 text-indigo-800"
                : "border-transparent hover:bg-slate-50"
            } ${levelPad(it.level)}`}
            onClick={() => onJump(it.page)}
            title={`Go to p${it.page + 1}`}
          >
            <div className="flex items-center gap-2">
              <span className="badge-level">{it.level}</span>
              <span className="truncate">{it.text}</span>
              <span className="ml-auto text-[10px] text-slate-500">p{it.page + 1}</span>
            </div>
          </button>
        );
      })}
    </div>
  );
}