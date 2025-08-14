import React, { useEffect, useRef, useState } from 'react'

type Props = {
  src: string
  page: number
  onPageChange: (p: number) => void
}

declare global {
  interface Window {
    AdobeDC?: any
    _adobeViewSDKPromise?: Promise<void>
  }
}

function loadAdobeSDK(): Promise<void> {
  if (window.AdobeDC) return Promise.resolve()
  if (window._adobeViewSDKPromise) return window._adobeViewSDKPromise
  window._adobeViewSDKPromise = new Promise<void>((resolve, reject) => {
    const s = document.createElement('script')
    s.src = 'https://acrobatservices.adobe.com/view-sdk/viewer.js'
    s.async = true
    s.onload = () => {
      const check = () => (window.AdobeDC ? resolve() : setTimeout(check, 50))
      check()
    }
    s.onerror = () => reject(new Error('Failed to load Adobe SDK'))
    document.head.appendChild(s)
  })
  return window._adobeViewSDKPromise
}

export default function PdfViewer({ src, page, onPageChange }: Props) {
  const [error, setError] = useState<string | null>(null)
  const containerId = 'adobe-viewer'
  const apisRef = useRef<any>(null)

  useEffect(() => {
    let cancelled = false
    setError(null)
    if (!src) return

    ;(async () => {
      await loadAdobeSDK()
      if (cancelled) return
      const clientId = import.meta.env.VITE_ADOBE_CLIENT_ID
      if (!clientId) { setError('Missing VITE_ADOBE_CLIENT_ID'); return }
      const host = document.getElementById(containerId)
      if (host) host.innerHTML = ''

      const view = new window.AdobeDC.View({ clientId, divId: containerId })
      const fileName = decodeURIComponent(src.split('/').pop() || 'document.pdf')

      const previewPromise = view.previewFile(
        { content: { location: { url: src } }, metaData: { fileName } },
        { embedMode: 'SIZED_CONTAINER', defaultViewMode: 'FIT_PAGE', showDownloadPDF: true, showPrintPDF: true }
      )

      previewPromise.then((viewer: any) => {
        if (!viewer || !viewer.getAPIs) return
        viewer.getAPIs().then((apis: any) => {
          apisRef.current = apis
          if (typeof page === 'number' && page >= 0) apis.gotoLocation(page + 1).catch(() => {})
          apis.on('PAGE_VIEW_CHANGE', (e: any) => {
            if (e && typeof e.pageNumber === 'number') onPageChange(Math.max(0, e.pageNumber - 1))
          })
        })
      })
    })().catch((e) => setError(String(e)))

    return () => { cancelled = true }
  }, [src])

  useEffect(() => {
    if (apisRef.current && typeof page === 'number' && page >= 0) {
      apisRef.current.gotoLocation(page + 1).catch(() => {})
    }
  }, [page])

  return (
    <div className="card overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 border-b border-slate-200">
        <div className="text-sm text-slate-700">Adobe PDF preview</div>
        <div className="text-xs text-slate-500">Client: {import.meta.env.VITE_ADOBE_CLIENT_ID ? 'configured' : 'not set'}</div>
      </div>
      {error ? <div className="p-8 text-red-600 text-sm">{error}</div> : <div id={containerId} className="h-[70vh] w-full bg-white"/>}
    </div>
  )
}
