import React, { useCallback, useState } from "react";

type Props = {
  onUpload: (files?: FileList | null) => void | Promise<void>;
};

export default function Welcome({ onUpload }: Props) {
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      if (e.dataTransfer?.files?.length) onUpload(e.dataTransfer.files);
    },
    [onUpload]
  );

  return (
    <div className="h-[72vh] rounded-2xl border border-slate-200 bg-white overflow-hidden grid md:grid-cols-2">
      {/* Left: hero */}
      <div className="p-10 flex flex-col justify-center gap-6 bg-gradient-to-br from-indigo-50 to-white">
        <div className="inline-flex items-center gap-2">
          <div className="w-9 h-9 rounded-xl bg-indigo-600" />
          <div className="text-xs font-medium text-indigo-700 bg-indigo-50 border border-indigo-200 rounded-full px-2 py-0.5">
            DocDots
          </div>
        </div>
        <h1 className="text-3xl font-bold text-slate-900">
          Read PDFs beautifully. Jump to related sections. Get insights. Make a podcast.
        </h1>
        <p className="text-slate-600">
          Drop your PDFs to build a private library on this device. Then pick a PDF to start reading, explore related sections,
          and generate an audio overview.
        </p>
        <ul className="text-sm text-slate-600 list-disc pl-5">
          <li>Drop multiple PDFs or click to upload</li>
          <li>All processing stays on your machine</li>
          <li>Navigate with ← → keys once open</li>
        </ul>
      </div>

      {/* Right: dropzone */}
      <div
        className={`m-6 rounded-2xl border-2 border-dashed flex items-center justify-center
        ${dragOver ? "border-indigo-400 bg-indigo-50/60" : "border-slate-300 bg-slate-50"}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        <label className="cursor-pointer flex flex-col items-center gap-3 p-10 text-center">
          <div className="w-14 h-14 rounded-2xl border border-indigo-200 bg-indigo-50 grid place-items-center">
            <svg width="22" height="22" viewBox="0 0 24 24" className="text-indigo-700">
              <path fill="currentColor" d="M19 13v6H5v-6H3v8h18v-8zM11 3v10.17l-3.59-3.58L6 11l6 6l6-6l-1.41-1.41L13 13.17V3z"/>
            </svg>
          </div>
          <div className="space-y-1">
            <div className="font-semibold">Upload PDFs</div>
            <div className="text-sm text-slate-500">Drag & drop or click to choose files</div>
          </div>
          <input type="file" multiple accept="application/pdf" className="hidden" onChange={(e) => onUpload(e.target.files)} />
        </label>
      </div>
    </div>
  );
}