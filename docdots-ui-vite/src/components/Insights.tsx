// src/components/Insights.tsx
import React from "react";

type Props = {
  text: string;
  loading: boolean;
};

export default function Insights({ text, loading }: Props) {
  if (loading) {
    return (
      <div className="space-y-3">
        <div className="rounded-lg border border-indigo-300 bg-indigo-50 p-3">
          <div className="h-4 w-1/4 bg-indigo-200 rounded mb-2 animate-pulse" />
          <div className="h-3 w-5/6 bg-indigo-200 rounded mb-1 animate-pulse" />
          <div className="h-3 w-2/3 bg-indigo-200 rounded animate-pulse" />
        </div>
      </div>
    );
  }

  if (!text) {
    return <div className="text-sm text-slate-500">No insights yet.</div>;
  }

  // Try to split a "Did you know?" if present
  const dykIdx = text.toLowerCase().indexOf("did you know");
  const main = dykIdx >= 0 ? text.slice(0, dykIdx).trim() : text;
  const dyk = dykIdx >= 0 ? text.slice(dykIdx).trim() : "";

  return (
    <div className="space-y-3">
      <div className="rounded-lg border border-indigo-300 bg-indigo-50 p-3">
        <div className="flex items-center gap-2 mb-1">
          <span className="w-2 h-2 rounded-full bg-indigo-600" />
          <span className="text-xs font-semibold text-indigo-700">KEY INSIGHT</span>
        </div>
        <div className="text-sm text-slate-900">{main}</div>
      </div>

      {dyk && (
        <div className="rounded-lg border border-emerald-300 bg-emerald-50 p-3">
          <div className="text-xs font-semibold text-emerald-700 mb-1">DID YOU KNOW?</div>
          <div className="text-sm text-slate-900">{dyk}</div>
        </div>
      )}
    </div>
  );
}