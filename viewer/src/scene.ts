// Cesium side of the viewer: globe, CZML loading, layer toggles, scene lookup by time.
// Kept out of Vue reactivity on purpose: Cesium objects must not be wrapped in proxies.
import {
  ArcGISTiledElevationTerrainProvider,
  CzmlDataSource,
  Credit,
  ImageryLayer,
  Ion,
  JulianDate,
  Rectangle,
  Terrain,
  UrlTemplateImageryProvider,
  Viewer,
  type Entity,
} from "cesium";

// Everything the viewer shows is listed in data/manifest.json, written by `cli.py publish`.
// URLs are built from BASE_URL so the same build works at / and under /iceberg-alley-sar/ (Pages).
export const DATA_URL = `${import.meta.env.BASE_URL}data/`;

// One CZML per scene and detector (schema: iceberg_sar.publish.Manifest).
export interface ManifestScene {
  scene_id: string;
  date: string;
  polarization: string;
  detector: "cfar" | "thor";
  czml: string; // relative to the manifest
  quicklook: string | null;
  quicklook_bounds: [number, number, number, number] | null; // W, S, E, N
  n_detections: number;
}

export interface Manifest {
  generated: string;
  base_czml: string;
  compare_index: string | null;
  scenes: ManifestScene[];
}

// Iceberg Alley, Bonavista to St. Anthony and offshore (matches aoi.geojson roughly).
const HOME = Rectangle.fromDegrees(-57.5, 48.3, -50.5, 52.8);

// NL relief is modest (Long Range Mountains ~800 m), so a little exaggeration helps the 3D feel.
// Detections sit at sea level, where exaggeration changes nothing.
const TERRAIN_EXAGGERATION = 2.0;

const ESRI_ELEVATION_URL =
  "https://elevation3d.arcgis.com/arcgis/rest/services/WorldElevation3D/Terrain3D/ImageServer";

// One satellite pass: the frames acquired seconds apart on one day, shown as one timeline step.
export interface SceneInfo {
  id: string; // "pass/<date>"
  frames: string[]; // product names
  platform: string;
  start: JulianDate;
  end: JulianDate;
  date: string;
  nDetections: number;
  nOpenWater: number;
  nNearIce: number;
  iceAreaKm2: number;
  nThor?: number; // THOR detections in open water; undefined when THOR wasn't run on the scene
  coveragePct?: number; // share of the AOI the pass observed; undefined without footprints
  observedKm2?: number;
  compare?: Compare; // THOR vs CFAR head-to-head on this pass (Milestone 5d)
  // THOR training frame (5b): "train" = the head was fitted on it, "val" = pack-ice set, never
  // trained on. Undefined for every other pass.
  split?: string;
  nTruth?: number; // IIP analyst labels shown on a training frame
}

export interface Compare {
  figureUrl: string;
  labels: number; // IIP labels compared
}

export interface Legend {
  nearIceKm: number;
  sizeBreaksM: [number, number]; // structure_m breaks small | medium | large (CFAR and THOR alike)
  sizePx: [number, number, number];
  thorScore?: number;
}

export interface Layers {
  nearIce: boolean;
  seaIce: boolean;
  footprints: boolean;
  unobserved: boolean;
  thor: boolean;
  truth: boolean;
}

function esriImagery(): ImageryLayer {
  return new ImageryLayer(
    new UrlTemplateImageryProvider({
      url: "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      maximumLevel: 18,
      credit: new Credit("Imagery © Esri, Maxar, Earthstar Geographics"),
    }),
  );
}

// Cesium World Terrain with an ion token; otherwise Esri's World Elevation, which needs no key.
function worldTerrain(hasIonToken: boolean): Terrain {
  return hasIonToken
    ? Terrain.fromWorldTerrain()
    : new Terrain(ArcGISTiledElevationTerrainProvider.fromUrl(ESRI_ELEVATION_URL));
}

export function createViewer(container: HTMLElement): Viewer {
  const token = import.meta.env.VITE_CESIUM_ION_TOKEN;
  if (token) Ion.defaultAccessToken = token;

  const viewer = new Viewer(container, {
    baseLayer: esriImagery(),
    terrain: worldTerrain(Boolean(token)),
    baseLayerPicker: false,
    geocoder: false,
    navigationHelpButton: false,
    sceneModePicker: true,
    fullscreenButton: false,
    animation: true,
    timeline: true,
    infoBox: true,
    selectionIndicator: true,
  });
  viewer.scene.verticalExaggeration = TERRAIN_EXAGGERATION;
  // Detections and polygons are at height 0 (ellipsoid); draw them over the terrain instead
  // of letting coastal relief hide them. At sea the terrain is within ~20 m of the ellipsoid.
  viewer.scene.globe.depthTestAgainstTerrain = false;
  viewer.camera.setView({ destination: HOME });
  viewer.homeButton.viewModel.command.beforeExecute.addEventListener((e) => {
    e.cancel = true;
    viewer.camera.flyTo({ destination: HOME });
  });
  return viewer;
}

// A text button in Cesium's toolbar, left of Home, styled like the built-in ones.
export function addToolbarButton(viewer: Viewer, label: string, title: string, onClick: () => void): void {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "cesium-button cesium-toolbar-button text-toolbar-button";
  button.textContent = label;
  button.title = title;
  button.setAttribute("aria-label", title);
  button.addEventListener("click", onClick);
  viewer.container.querySelector(".cesium-viewer-toolbar")?.prepend(button);
}

function prop(entity: Entity, name: string): unknown {
  return entity.properties?.[name]?.getValue(JulianDate.now());
}

export async function loadDetections(viewer: Viewer): Promise<{
  dataSource: CzmlDataSource;
  scenes: SceneInfo[];
  legend: Legend;
}> {
  const manifest = await fetchJson<Manifest>(`${DATA_URL}manifest.json`);
  // Fetch in parallel, process in manifest order: a scene's CFAR file holds the `scene/` parent
  // of its THOR points. Every file starts with a document packet; only base.czml has the clock.
  const [base, ...rest] = await Promise.all(
    [manifest.base_czml, ...manifest.scenes.map((s) => s.czml)].map((f) => fetchJson<object[]>(DATA_URL + f)),
  );
  const dataSource = await CzmlDataSource.load(base);
  for (const czml of rest) await dataSource.process(czml);
  await viewer.dataSources.add(dataSource);
  viewer.clockTrackedDataSource = dataSource;
  viewer.timeline.zoomTo(viewer.clock.startTime, viewer.clock.stopTime);

  const frames = dataSource.entities.values
    .filter((e) => e.id.startsWith("scene/"))
    .map((e) => ({
      sceneId: prop(e, "scene_id") as string,
      platform: prop(e, "platform") as string,
      start: JulianDate.fromIso8601(prop(e, "start") as string),
      end: JulianDate.fromIso8601(prop(e, "end") as string),
      date: (prop(e, "start") as string).slice(0, 10),
      nDetections: prop(e, "n_detections") as number,
      nOpenWater: prop(e, "n_open_water") as number,
      // 0 when the scene is ice-free (summer) or no pack-ice mask was built.
      nNearIce: (prop(e, "n_near_ice") as number | undefined) ?? 0,
      iceAreaKm2: (prop(e, "ice_area_km2") as number | undefined) ?? 0,
      nThor: prop(e, "n_thor") as number | undefined,
      split: prop(e, "thor_split") as string | undefined,
      nTruth: prop(e, "n_truth") as number | undefined,
    }))
    .sort((a, b) => JulianDate.compare(a.start, b.start));

  const byDate = new Map<string, SceneInfo>();
  for (const f of frames) {
    const s = byDate.get(f.date);
    if (!s) {
      const { sceneId, ...first } = f;
      const pass = dataSource.entities.getById(`pass/${f.date}`);
      byDate.set(f.date, {
        ...first,
        id: `pass/${f.date}`,
        frames: [sceneId],
        coveragePct: pass ? (prop(pass, "aoi_coverage_pct") as number | undefined) : undefined,
        observedKm2: pass ? (prop(pass, "observed_km2") as number | undefined) : undefined,
      });
      continue;
    }
    s.frames.push(f.sceneId);
    if (!s.platform.includes(f.platform)) s.platform += `+${f.platform}`;
    s.nDetections += f.nDetections;
    s.nOpenWater += f.nOpenWater;
    s.nNearIce += f.nNearIce;
    s.iceAreaKm2 += f.iceAreaKm2;
    if (f.nThor !== undefined) s.nThor = (s.nThor ?? 0) + f.nThor;
    if (f.nTruth !== undefined) s.nTruth = (s.nTruth ?? 0) + f.nTruth;
    s.split ??= f.split;
  }
  const scenes = [...byDate.values()];
  const compare = await loadCompare(manifest.compare_index);
  for (const s of scenes) s.compare = compare.get(s.date);
  const l = dataSource.entities.getById("legend");
  const legend: Legend = {
    nearIceKm: l ? (prop(l, "near_ice_km") as number) : 5,
    sizeBreaksM: arrayProp(l, "size_breaks_m", 2) ?? [180, 270],
    sizePx: arrayProp(l, "size_px", 3) ?? [6, 13, 22],
    thorScore: frames
      .map((f) => dataSource.entities.getById(`scene/${f.sceneId}`))
      .map((e) => (e ? (prop(e, "thor_score") as number | undefined) : undefined))
      .find((v) => v !== undefined),
  };
  return { dataSource, scenes, legend };
}

// A list-valued property, or undefined if it isn't one of length n. CZML written as a bare
// list is read as time samples and comes back as a single number.
function arrayProp<T extends number[]>(entity: Entity | undefined, name: string, n: number): T | undefined {
  const v = entity ? prop(entity, name) : undefined;
  return Array.isArray(v) && v.length === n ? (v as T) : undefined;
}

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return (await res.json()) as T;
}

// Per-pass THOR vs CFAR recall figures; the viewer works without them.
async function loadCompare(indexPath: string | null): Promise<Map<string, Compare>> {
  if (!indexPath) return new Map();
  const dir = DATA_URL + indexPath.slice(0, indexPath.lastIndexOf("/") + 1);
  try {
    const index = await fetchJson<Record<string, { figure: string; labels: number }>>(DATA_URL + indexPath);
    return new Map(
      Object.entries(index).map(([date, c]) => [date, { figureUrl: dir + c.figure, labels: c.labels }]),
    );
  } catch {
    return new Map();
  }
}

export function applyLayers(dataSource: CzmlDataSource, layers: Layers): void {
  for (const e of dataSource.entities.values) {
    if (e.id.startsWith("det/")) e.show = layers.nearIce || prop(e, "near_ice") !== true;
    else if (e.id.startsWith("thor/"))
      e.show = layers.thor && (layers.nearIce || prop(e, "near_ice") !== true);
    else if (e.id.startsWith("truth/")) e.show = layers.truth;
    else if (e.id.startsWith("seaice/")) e.show = layers.seaIce;
    else if (e.id.startsWith("footprint/")) e.show = layers.footprints;
    else if (e.id.startsWith("unobserved/")) e.show = layers.unobserved;
  }
}

export function sceneAt(scenes: SceneInfo[], time: JulianDate): SceneInfo | undefined {
  return scenes.find(
    (s) => JulianDate.greaterThanOrEquals(time, s.start) && JulianDate.lessThan(time, s.end),
  );
}

export function jumpTo(viewer: Viewer, scene: SceneInfo): void {
  viewer.clock.shouldAnimate = false;
  viewer.clock.currentTime = JulianDate.clone(scene.start);
}
