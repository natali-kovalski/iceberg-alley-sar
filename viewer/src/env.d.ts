/// <reference types="vite/client" />

declare const CESIUM_BASE_URL: string;
/** Cesium ion token from the repo-root .env or the build env; "" = no ion (no SAR imagery layer). */
declare const CESIUM_ION_TOKEN: string;
/** SAR imagery tiled by ion: "<scene_id>=<asset id>,..."; "" = none. */
declare const SAR_ION_ASSETS: string;

declare module "*.vue" {
  import type { DefineComponent } from "vue";
  const component: DefineComponent<object, object, unknown>;
  export default component;
}
