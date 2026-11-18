import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.RASTA_API_URL || "http://127.0.0.1:8000";
  const proxy = { "/api": target, "/media": target };
  return {
    esbuild: { jsx: "automatic" },
    server: { proxy },
    preview: { proxy },
  };
});
