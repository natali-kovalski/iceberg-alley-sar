import { defineConfig, loadEnv } from "vite";
import vue from "@vitejs/plugin-vue";
import { viteStaticCopy } from "vite-plugin-static-copy";

// Cesium loads workers, widgets CSS and assets at runtime from CESIUM_BASE_URL,
// so they are copied next to the app instead of being bundled.
const cesiumSource = "node_modules/cesium/Build/Cesium";
const cesiumDir = "cesium";

// GitHub Pages serves the repo under /iceberg-alley-sar/; dev serves at /. `vite preview` runs
// as "serve" too, so it gets the production base explicitly.
// VIEWER_BASE (env or .env) overrides the production base, e.g. "/" for another host.
const PAGES_BASE = "/iceberg-alley-sar/";

export default defineConfig(({ command, mode, isPreview }) => {
  const production = command === "build" || isPreview;
  const base = production ? (loadEnv(mode, ".", "").VIEWER_BASE ?? PAGES_BASE) : "/";
  return {
    base,
    build: { chunkSizeWarningLimit: 6000 }, // Cesium alone is ~4 MB
    define: { CESIUM_BASE_URL: JSON.stringify(`${base}${cesiumDir}`) },
    plugins: [
      vue(),
      viteStaticCopy({
        targets: ["ThirdParty", "Workers", "Assets", "Widgets"].map((dir) => ({
          src: `${cesiumSource}/${dir}`,
          dest: cesiumDir,
          rename: { stripBase: 4 }, // drop node_modules/cesium/Build/Cesium
        })),
      }),
    ],
  };
});
