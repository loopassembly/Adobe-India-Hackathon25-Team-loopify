// src/components/Insights.tsx
// Removed unused React import

type Props = {
  text: string;
  loading: boolean;
};

function parseSections(md: string) {
  const sections = {
    key: [] as string[],
    dyk: [] as string[],
    contra: [] as string[],
    insp: [] as string[],
  };

  // super-light parsing for our fixed headings
  const blocks = md.split(/\n(?=### )/g);
  for (const b of blocks) {
    const title = (b.match(/^###\s+(.+)$/m)?.[1] || "").toLowerCase();
    const bullets = (b.match(/^- .+$/gms) || []).map((x) => x.replace(/^- /, "").trim());
    if (title.startsWith("key")) sections.key = bullets;
    else if (title.includes("did you know")) sections.dyk = bullets;
    else if (title.includes("contradiction") || title.includes("counterpoint")) sections.contra = bullets;
    else if (title.includes("inspirations")) sections.insp = bullets;
  }
  return sections;
}

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

  if (!text?.trim()) {
    return <div className="text-sm text-slate-500">No insights yet.</div>;
  }

  const { key, dyk, contra, insp } = parseSections(text);

  return (
    <div className="space-y-3">
      {/* Key insights */}
      <div className="rounded-lg border border-indigo-300 bg-indigo-50 p-3">
        <div className="flex items-center gap-2 mb-1">
          <span className="w-2 h-2 rounded-full bg-indigo-600" />
          <span className="text-xs font-semibold text-indigo-700">KEY INSIGHTS</span>
        </div>
        <ul className="list-disc pl-5 text-sm text-slate-900 space-y-1">
          {(key.length ? key : ["—"]).map((b, i) => (
            <li key={i}>{b}</li>
          ))}
        </ul>
      </div>

      {/* Did you know */}
      <div className="rounded-lg border border-emerald-300 bg-emerald-50 p-3">
        <div className="text-xs font-semibold text-emerald-700 mb-1">DID YOU KNOW?</div>
        <ul className="list-disc pl-5 text-sm text-slate-900 space-y-1">
          {(dyk.length ? dyk : ["—"]).map((b, i) => (
            <li key={i}>{b}</li>
          ))}
        </ul>
      </div>

      {/* Contradictions / counterpoints */}
      <div className="rounded-lg border border-amber-300 bg-amber-50 p-3">
        <div className="text-xs font-semibold text-amber-700 mb-1">CONTRADICTIONS / COUNTERPOINTS</div>
        <ul className="list-disc pl-5 text-sm text-slate-900 space-y-1">
          {(contra.length ? contra : ["—"]).map((b, i) => (
            <li key={i}>{b}</li>
          ))}
        </ul>
      </div>

      {/* Inspirations */}
      <div className="rounded-lg border border-slate-300 bg-slate-50 p-3">
        <div className="text-xs font-semibold text-slate-700 mb-1">INSPIRATIONS & CONNECTIONS</div>
        <ul className="list-disc pl-5 text-sm text-slate-900 space-y-1">
          {(insp.length ? insp : ["—"]).map((b, i) => (
            <li key={i}>{b}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}