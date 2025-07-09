import { createRoot } from 'react-dom/client'
import App from './App.tsx'
import './index.css'

// ⭐ NEW IMPORTS
import React from 'react'; // React is implicitly used by JSX, good practice to import
import { GoogleOAuthProvider } from '@react-oauth/google';

// ⭐ Get your Google Client ID from environment variables
// For Vite, use import.meta.env.VITE_YOUR_ENV_VAR_NAME
// Ensure this variable is set in your frontend's .env file (e.g., .env.local)
const GOOGLE_CLIENT_ID_FRONTEND = import.meta.env.VITE_GOOGLE_CLIENT_ID_FRONTEND;

// Optional: Add a check to ensure the ID is set
// This will log an error in the console if the ID is missing during development
if (!GOOGLE_CLIENT_ID_FRONTEND) {
  console.error("VITE_GOOGLE_CLIENT_ID_FRONTEND is not set in your frontend's .env file! Google login may not work.");
  // In a production app, you might want more robust error handling or a build-time check
}
console.log("DEBUG: Google Client ID loaded in main.tsx:", GOOGLE_CLIENT_ID_FRONTEND);

createRoot(document.getElementById("root")!).render(
  // ⭐ Wrap your App component with GoogleOAuthProvider
  <React.StrictMode>
    <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID_FRONTEND}>
      <App />
    </GoogleOAuthProvider>
  </React.StrictMode>
);