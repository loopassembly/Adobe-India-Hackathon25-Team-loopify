/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE: string
  readonly VITE_PDF_BASE: string
  readonly VITE_ADOBE_CLIENT_ID: string
}
interface ImportMeta {
  readonly env: ImportMetaEnv
}
