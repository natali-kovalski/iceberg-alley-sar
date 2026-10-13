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

export const CZML_URL = "data/iceberg_alley.czml";
// Per-scene THOR vs CFAR recall figures, written by `thor-compare` / `czml` (optional).
export const COMPARE_DIR = "data/compare/";

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

function prop(entity: Entity, name: string): unknown {
  return entity.properties?.[name]?.getValue(JulianDate.now());
}

export async function loadDetections(viewer: Viewer): Promise<{
  dataSource: CzmlDataSource;
  scenes: SceneInfo[];
  legend: Legend;
}> {
  const dataSource = await CzmlDataSource.load(CZML_URL);
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
  }
  const scenes = [...byDate.values()];
  const compare = await loadCompare();
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

async function loadCompare(): Promise<Map<string, Compare>> {
  try {
    const res = await fetch(`${COMPARE_DIR}index.json`);
    if (!res.ok) return new Map();
    const index = (await res.json()) as Record<string, { figure: string; labels: number }>;
    return new Map(
      Object.entries(index).map(([date, c]) => [
        date,
        { figureUrl: COMPARE_DIR + c.figure, labels: c.labels },
      ]),
    );
  } catch {
    return new Map(); // no comparison run: the viewer works without it
  }
}

export function applyLayers(dataSource: CzmlDataSource, layers: Layers): void {
  for (const e of dataSource.entities.values) {
    if (e.id.startsWith("det/")) e.show = layers.nearIce || prop(e, "near_ice") !== true;
    else if (e.id.startsWith("thor/"))
      e.show = layers.thor && (layers.nearIce || prop(e, "near_ice") !== true);
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
