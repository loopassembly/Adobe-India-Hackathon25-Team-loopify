import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";

type Props = {
  src: string;
  page: number; // zero-based
  onPageChange: (p: number) => void;
  /** fired whenever the user completes a text selection in the embedded viewer */
  onSelection?: (text: string) => void;
};

export type PdfViewerHandle = {
  /** Best-effort plain text selection from the embedded viewer */
  getSelection: () => Promise<string>;
  /** Yellow-highlight occurrences using Adobe’s built-in search (preserves current page) */
  highlight: (query: string) => Promise<void>;
  /** Raw SDK APIs (if you still want to access them) */
  getAPIs: () => Promise<any | null>;
};

declare global {
  interface Window {
    AdobeDC?: any;
    _adobeViewSDKPromise?: Promise<void>;
  }
}

/** Load the Embed API only once */
function loadAdobeSDK(): Promise<void> {
  if (window.AdobeDC) return Promise.resolve();
  if (window._adobeViewSDKPromise) return window._adobeViewSDKPromise;

  window._adobeViewSDKPromise = new Promise<void>((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "https://acrobatservices.adobe.com/view-sdk/viewer.js";
    script.async = true;
    script.onload = () => {
      const wait = () => (window.AdobeDC ? resolve() : setTimeout(wait, 50));
      wait();
    };
    script.onerror = () => reject(new Error("Failed to load Adobe PDF SDK"));
    document.head.appendChild(script);
  });

  return window._adobeViewSDKPromise;
}

function fileNameFromUrl(u: string) {
  try {
    const url = new URL(u, window.location.origin);
    return decodeURIComponent(url.pathname.split("/").pop() || "document.pdf");
  } catch {
    const parts = u.split("/");
    return decodeURIComponent(parts[parts.length - 1] || "document.pdf");
  }
}

/** Sanitize long/complex selections before sending to the search APIs */
function sanitizeQuery(raw: string, maxLen = 160) {
  if (!raw) return "";
  let s = String(raw)
    // bullets & common list markers → space
    .replace(/[\u2022\u2023\u25E6\u2043\u2219•·◦▪●]/g, " ")
    // smart quotes → straight quotes
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[\u201C\u201D]/g, '"')
    // strip non-ascii (Embed search behaves better with ascii)
    .replace(/[^\x20-\x7E]/g, " ")
    // collapse whitespace
    .replace(/\s+/g, " ")
    .trim();

  // hard cap
  if (s.length > maxLen) s = s.slice(0, maxLen);

  // drop super short tokens to reduce noise
  s = s
    .split(" ")
    .filter((w) => w.length >= 2)
    .join(" ")
    .trim();

  return s;
}

const PdfViewer = forwardRef<PdfViewerHandle, Props>(function PdfViewer(
  { src, page, onPageChange, onSelection }: Props,
  ref
) {
  const containerId = useRef(`adobe-${Math.random().toString(36).slice(2)}`).current;

  const [error, setError] = useState<string | null>(null);
  const [lastResolvedUrl, setLastResolvedUrl] = useState<string>("");

  // Adobe instances/APIs
  const viewRef = useRef<any>(null);   // AdobeDC.View (registerCallback lives here)
  const viewerRef = useRef<any>(null); // previewFile() return (getAPIs lives here)
  const apisRef = useRef<any>(null);   // viewer.getAPIs() cache

  const isHighlightingRef = useRef(false);
  const suppressPageEventsUntilRef = useRef(0);
  const lastPageRef = useRef<number>(1);

  /** Ensure & cache SDK APIs */
  async function ensureAPIs(): Promise<any | null> {
    if (apisRef.current) return apisRef.current;
    const apis = await viewerRef.current?.getAPIs?.();
    if (apis) {
      apisRef.current = apis;
      // Log once so we know exactly what this build supports
      try {
        const keys = Object.keys(apis).filter((k) => typeof (apis as any)[k] === "function");
        console.log("[PDF] APIs ready:", keys);
      } catch {}
    }
    return apis || null;
  }

  // Expose methods to parent
  useImperativeHandle(ref, () => ({
    async getSelection() {
      try {
        const apis = await ensureAPIs();
        if (!apis) return "";
        const res = await apis.getSelectedContent?.();
        const text =
          (res?.data as string) ??
          res?.content?.map?.((c: any) => c?.str).join(" ") ??
          (res?.text as string) ??
          "";
        return (text || "").toString().trim();
      } catch (e) {
        console.warn("[PdfViewer] getSelection failed:", e);
        return "";
      }
    },

    async highlight(query: string) {
      // Normalize & bound the query to avoid SDK errors on huge/complex selections
      const q = sanitizeQuery(query ?? "");
      if (!q) return;

      const apis = await ensureAPIs();
      if (!apis) return;

      // The officially supported API is `apis.search(<string>)`. Avoid `performSearch` and
      // avoid manual navigation; search highlights from the CURRENT page in view.
      const hasSearch = typeof apis.search === "function";
      if (!hasSearch) {
        console.info("[PdfViewer] highlight(): search APIs not available on this build");
        return;
      }

      isHighlightingRef.current = true;
      try {
        if (typeof apis.clearPageSelection === "function") {
          await apis.clearPageSelection();
        }

        // Run the search; do NOT call gotoLocation afterwards. The viewer will
        // highlight occurrences starting at the current page without changing pages.
        await apis.search(q);
      } catch (e) {
        console.warn("[PdfViewer] highlight failed:", e);
      } finally {
        // brief suppression so PAGE_VIEW handlers don't echo back immediately
        suppressPageEventsUntilRef.current = Date.now() + 400;
        setTimeout(() => {
          isHighlightingRef.current = false;
        }, 400);
      }
    },

    async getAPIs() {
      try {
        return await ensureAPIs();
      } catch {
        return null;
      }
    },
  }));

  // Init (or re-init) when src changes
  useEffect(() => {
    if (!src) return;

    let cancelled = false;
    setError(null);

    (async () => {
      try {
        await loadAdobeSDK();
        if (cancelled) return;

        const clientId =
          ((import.meta as any).env?.VITE_ADOBE_CLIENT_ID ||
            (import.meta as any).env?.VITE_ADOBE_EMBED_API_KEY) as string | undefined;

        if (!clientId) {
          setError("Missing VITE_ADOBE_CLIENT_ID or VITE_ADOBE_EMBED_API_KEY in your environment.");
          return;
        }
        if (!window.AdobeDC) {
          setError("Adobe PDF SDK failed to load.");
          return;
        }

        // Reset host node if re-mounting
        const host = document.getElementById(containerId);
        if (host) host.innerHTML = "";

        // Absolutize src
        let absoluteSrc = src;
        if (!(src.startsWith("http://") || src.startsWith("https://"))) {
          const BACKEND = (import.meta as any).env?.VITE_BACKEND_BASE || "";
          const base = BACKEND ? BACKEND.replace(/\/$/, "") : window.location.origin;
          absoluteSrc = `${base}${src.startsWith("/") ? src : `/${src}`}`;
        }
        try {
          absoluteSrc = new URL(absoluteSrc).toString();
        } catch {}
        setLastResolvedUrl(absoluteSrc);

        // Create view + preview
        const view = new window.AdobeDC.View({ clientId, divId: containerId });
        viewRef.current = view;

        // (Optional) log SDK-level errors
        try {
          view.registerCallback(
            window.AdobeDC.View.Enum.CallbackType.ERROR,
            (e: any) => console.error("[Adobe Embed ERROR]", e)
          );
        } catch {}

        const fileName = fileNameFromUrl(absoluteSrc);
        const viewer = await view.previewFile(
          {
            content: { location: { url: absoluteSrc } },
            metaData: { fileName },
          },
          {
            embedMode: "SIZED_CONTAINER",
            defaultViewMode: "FIT_WIDTH",
            showDownloadPDF: true,
            showPrintPDF: true,
            // needed so apis.search / performSearch are available
            enableSearchAPIs: true,
          }
        );
        if (cancelled) return;

        viewerRef.current = viewer;

        // Grab APIs for navigation + selection
        try {
          const apis = await viewer.getAPIs();
          apisRef.current = apis;

          // Jump to initial page (SDK is 1-based; clamp to >= 1)
          if (typeof page === "number" && page >= 0 && apis?.gotoLocation) {
            const target = Math.max(1, Number(page) + 1);
            await apis.gotoLocation(target);
          }
          // Track our last known page for safe snap-back after search
          lastPageRef.current = Math.max(1, Number(page) + 1);

          // Ensure text selection is enabled if the API exists
          if (typeof apis.enableTextSelection === "function") {
            apis.enableTextSelection(true);
          }
        } catch (e) {
          console.warn("[PdfViewer] getAPIs failed:", e);
        }

        // --- Register EVENTS on the *view* ---
        const { AdobeDC } = window;
        const previewEvents = AdobeDC.View.Enum.FilePreviewEvents;

        // 1) Page change -> notify React
        try {
          view.registerCallback(
            AdobeDC.View.Enum.CallbackType.EVENT_LISTENER,
            (event: any) => {
              if (
                event?.type === previewEvents.PREVIEW_PAGE_VIEW &&
                typeof event.data?.pageNumber === "number"
              ) {
                // Track the last page we saw from the viewer (SDK is 1-based)
                const cur = Math.max(1, Number(event.data.pageNumber) || 1);
                lastPageRef.current = cur;

                // Ignore page events while we're running a highlight or briefly after
                if (isHighlightingRef.current || Date.now() < suppressPageEventsUntilRef.current) {
                  return;
                }

                const newPage = Math.max(0, cur - 1);
                onPageChange?.(newPage);
              }
            },
            { enablePDFAnalytics: true, listenOn: [previewEvents.PREVIEW_PAGE_VIEW] }
          );
        } catch (e) {
          console.warn("[PdfViewer] failed to register PAGE_VIEW listener:", e);
        }

        // 2) Selection end -> read selected content and bubble up
        try {
          view.registerCallback(
            AdobeDC.View.Enum.CallbackType.EVENT_LISTENER,
            async (event: any) => {
              if (event?.type !== previewEvents.PREVIEW_SELECTION_END) return;
              try {
                const apis = await ensureAPIs();
                if (!apis?.getSelectedContent) return;
                const res = await apis.getSelectedContent();
                const text =
                  (res?.data as string) ??
                  res?.content?.map?.((c: any) => c?.str).join(" ") ??
                  (res?.text as string) ??
                  "";
                const t = (text || "").toString().trim();
                console.log("[PdfViewer] selection:", { length: t.length, text: t });
                if (t && onSelection) onSelection(t);
              } catch (err) {
                console.warn("[PdfViewer] selection handler failed:", err);
              }
            },
            {
              enableFilePreviewEvents: true,
              listenOn: [previewEvents.PREVIEW_SELECTION_END],
            }
          );
        } catch (e) {
          console.warn("[PdfViewer] failed to register SELECTION listener:", e);
        }
      } catch (e: any) {
        if (!cancelled) setError(String(e?.message || e));
      }
    })();

    return () => {
      cancelled = true;
      viewRef.current = null;
      viewerRef.current = null;
      apisRef.current = null;
    };
  }, [src]);

  // React -> Viewer page sync (when `page` changes after init)
  useEffect(() => {
    (async () => {
      try {
        const apis = await ensureAPIs();
        const now = Date.now();
        if (isHighlightingRef.current || now < suppressPageEventsUntilRef.current) return;
        if (apis?.gotoLocation && typeof page === "number" && page >= 0) {
          const target = Math.max(1, Number(page) + 1);
          await apis.gotoLocation(target);
          lastPageRef.current = target;
        }
      } catch {}
    })();
  }, [page]);

  return (
    <div className="bg-white rounded-xl border border-slate-200 overflow-hidden h-full flex flex-col">
      <div className="flex items-center justify-between px-4 py-2 border-b border-slate-200">
        <div className="text-sm text-slate-700">Adobe PDF preview</div>
        <div className="text-xs text-slate-500">
          Client:{" "}
          {(import.meta as any).env?.VITE_ADOBE_CLIENT_ID ||
          (import.meta as any).env?.VITE_ADOBE_EMBED_API_KEY
            ? "configured"
            : "not set"}
        </div>
      </div>

      {error ? (
        <div className="p-8 text-sm text-red-600 space-y-2">
          <div>{error}</div>
          {lastResolvedUrl && (
            <div className="text-slate-600">
              Try opening the PDF directly:{" "}
              <a
                className="text-indigo-700 underline"
                href={lastResolvedUrl}
                target="_blank"
                rel="noreferrer"
              >
                {lastResolvedUrl}
              </a>
            </div>
          )}
        </div>
      ) : (
        <div id={containerId} className="flex-1 min-h-0 w-full bg-slate-50" />
      )}
    </div>
  );
});

export default PdfViewer;