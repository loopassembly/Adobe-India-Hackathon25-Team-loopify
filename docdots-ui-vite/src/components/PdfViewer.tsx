import React, { useEffect, useRef, useState } from "react";

/**
 * Adobe PDF Embed API based viewer
 * 
 * Props:
 *  - src: URL to the PDF (same origin as app or CORS-enabled)
 *  - page: zero-based page index to display
 *  - onPageChange: callback when user navigates in the embed viewer (zero-based)
 *
 * Requires env var: VITE_ADOBE_CLIENT_ID
 */
type Props = {
  src: string;
  page: number;
  onPageChange: (p: number) => void;
};

declare global {
  interface Window {
    AdobeDC?: any;
    _adobeViewSDKPromise?: Promise<void>;
  }
}

function loadAdobeSDK(): Promise<void> {
  if (window.AdobeDC) return Promise.resolve();
  if (window._adobeViewSDKPromise) return window._adobeViewSDKPromise;
  
  window._adobeViewSDKPromise = new Promise<void>((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "https://acrobatservices.adobe.com/view-sdk/viewer.js";
    script.async = true;
    
    script.onload = () => {
      // Wait a bit for AdobeDC to be available
      const checkAdobeDC = () => {
        if (window.AdobeDC) {
          resolve();
        } else {
          setTimeout(checkAdobeDC, 100);
        }
      };
      checkAdobeDC();
    };
    
    script.onerror = () => {
      reject(new Error("Failed to load Adobe PDF SDK"));
    };
    
    document.head.appendChild(script);
  });
  
  return window._adobeViewSDKPromise;
}

export default function PdfViewer({ src, page, onPageChange }: Props) {
  const [error, setError] = useState<string | null>(null);
  const viewerRef = useRef<any>(null);
  const apisRef = useRef<any>(null);
  const containerId = "adobe-viewer";

  // Initialize viewer when SDK + src ready
  useEffect(() => {
    let cancelled = false;
    setError(null);

    if (!src) return;

    loadAdobeSDK().then(async () => {
      if (cancelled) return;

      const clientId = ((import.meta as any).env?.VITE_ADOBE_CLIENT_ID) as string;
      if (!clientId) {
        setError("Missing VITE_ADOBE_CLIENT_ID in your environment.");
        return;
      }

      if (!window.AdobeDC) {
        setError("Adobe PDF SDK failed to load properly.");
        return;
      }

      // Reset container content if reusing
      const host = document.getElementById(containerId);
      if (host) host.innerHTML = "";

      // Create the viewer host
      const view = new window.AdobeDC.View({
        clientId,
        divId: containerId,
      });

      const fileName = decodeURIComponent(src.split("/").pop() || "document.pdf");

      // previewFile returns a Viewer instance (not the same as AdobeDC.View)
      const viewer = await view.previewFile(
        {
          content: { location: { url: src } },
          metaData: { fileName },
        },
        {
          embedMode: "SIZED_CONTAINER",
          showDownloadPDF: true,
          showPrintPDF: true,
          defaultViewMode: "FIT_PAGE",
        }
      );

      // Keep a ref to the viewer instance
      viewerRef.current = viewer;

      // Obtain Viewer APIs from the viewer (not from AdobeDC.View)
      try {
        const apis = await viewer.getAPIs();
        apisRef.current = apis;

        // Sync external state -> viewer (API is 1-based)
        if (typeof page === "number" && page >= 0) {
          await apis.gotoLocation(page + 1);
        }
      } catch (e) {
        // If APIs are unavailable, fail gracefully
        console.warn("Adobe PDF APIs unavailable:", e);
      }

      // Register event listener for page changes
      try {
        viewer.registerCallback(
          window.AdobeDC.View.Enum.CallbackType.EVENT_LISTENER,
          (event: any) => {
            // PAGE_VIEW events provide the current pageNumber (1-based)
            if (event?.type === "PAGE_VIEW" && typeof event.data?.pageNumber === "number") {
              onPageChange(Math.max(0, event.data.pageNumber - 1));
            }
          },
          { enablePDFAnalytics: true }
        );
      } catch (e) {
        console.warn("Failed to register Adobe viewer callbacks:", e);
      }
    }).catch((e) => {
      setError(String(e));
    });

    return () => {
      cancelled = true;
    };
  }, [src]); // re-init if file changes

  // Respond to external page changes after init
  useEffect(() => {
    if (apisRef.current && typeof page === "number" && page >= 0) {
      apisRef.current.gotoLocation(page + 1).catch(() => {});
    }
  }, [page]);

  return (
    <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 border-b border-slate-200">
        <div className="text-sm text-slate-700">Adobe PDF preview</div>
        <div className="text-xs text-slate-500">Client: {((import.meta as any).env?.VITE_ADOBE_CLIENT_ID) ? "configured" : "not set"}</div>
      </div>

      {error ? (
        <div className="p-8 text-sm text-red-600">{error}</div>
      ) : (
        <div id={containerId} className="h-[75vh] w-full bg-slate-50" />
      )}
    </div>
  );
}
