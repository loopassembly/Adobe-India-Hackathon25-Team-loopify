/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_BACKEND?: string;
  readonly VITE_ADOBE_CLIENT_ID?: string;
  readonly VITE_ADOBE_EMBED_API_KEY?: string;
  // add more VITE_ variables here as needed
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}