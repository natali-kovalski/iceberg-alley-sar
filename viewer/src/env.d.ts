/// <reference types="vite/client" />

declare const CESIUM_BASE_URL: string;

interface ImportMetaEnv {
  /** Optional Cesium ion token. Without it the viewer uses Esri imagery + Esri World Elevation terrain and no ion services. */
  readonly VITE_CESIUM_ION_TOKEN?: string;
}

declare module "*.vue" {
  import type { DefineComponent } from "vue";
  const component: DefineComponent<object, object, unknown>;
  export default component;
}
