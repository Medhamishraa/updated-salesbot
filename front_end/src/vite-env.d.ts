/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_GOOGLE_CLIENT_ID: string;
  readonly VITE_FACEBOOK_APP_ID: string;
  readonly VITE_BACKEND_BASE_URL: string;
  // Add other environment variables here if you use them
  // For example, if you have VITE_API_KEY, add:
  // readonly VITE_API_KEY: string;

  // Vite's built-in env variables (these should be recognized by default with <reference types="vite/client" />
  // but explicitly defining them here can sometimes help if there are conflicts or older configurations)
  readonly PROD: boolean;
  readonly DEV: boolean;
  readonly SSR: boolean;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}