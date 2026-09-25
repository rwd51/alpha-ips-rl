import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base "./" keeps the built site working from any sub-path (e.g. GitHub Pages)
export default defineConfig({
  plugins: [react()],
  base: "./",
  // three.js alone is ~700 kB minified; one bundle is fine for a local demo
  build: { chunkSizeWarningLimit: 1200 },
});
