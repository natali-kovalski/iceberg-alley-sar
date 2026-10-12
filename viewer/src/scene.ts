// Cesium side of the viewer: globe, CZML loading, layer toggles, scene lookup by time.
// Kept out of Vue reactivity on purpose: Cesium objects must not be wrapped in proxies.
import {
  CzmlDataSource,
  Credit,
  ImageryLayer,
  Ion,
  JulianDate,
  Rectangle,
  UrlTemplateImageryProvider,
  Viewer,
  type Entity,
} from "cesium";

export const CZML_URL = "data/iceberg_alley.czml";

// Iceberg Alley, Bonavista to St. Anthony and offshore (matches aoi.geojson roughly).
const HOME = Rectangle.fromDegrees(-57.5, 48.3, -50.5, 52.8);

export interface SceneInfo {
  id: string;
  sceneId: string;
  platform: string;
  start: JulianDate;
  end: JulianDate;
  date: string;
  nDetections: number;
  nOpenWater: number;
  nNearIce: number;
  iceAreaKm2: number;
  nThor?: number; // THOR detections in open water; undefined when THOR wasn't run on the scene
}

export interface Legend {
  nearIceKm: number;
  contrastDbMin: number;
  contrastDbMax: number;
  thorScore?: number;
}

export interface Layers {
  nearIce: boolean;
  seaIce: boolean;
  footprints: boolean;
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

export function createViewer(container: HTMLElement): Viewer {
  const token = import.meta.env.VITE_CESIUM_ION_TOKEN;
  if (token) Ion.defaultAccessToken = token;

  const viewer = new Viewer(container, {
    baseLayer: esriImagery(),
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

  const scenes = dataSource.entities.values
    .filter((e) => e.id.startsWith("scene/"))
    .map((e): SceneInfo => {
      const start = JulianDate.fromIso8601(prop(e, "start") as string);
      return {
        id: e.id,
        sceneId: prop(e, "scene_id") as string,
        platform: prop(e, "platform") as string,
        start,
        end: JulianDate.fromIso8601(prop(e, "end") as string),
        date: (prop(e, "start") as string).slice(0, 10),
        nDetections: prop(e, "n_detections") as number,
        nOpenWater: prop(e, "n_open_water") as number,
        // 0 when the scene is ice-free (summer) or no pack-ice mask was built.
        nNearIce: (prop(e, "n_near_ice") as number | undefined) ?? 0,
        iceAreaKm2: (prop(e, "ice_area_km2") as number | undefined) ?? 0,
        nThor: prop(e, "n_thor") as number | undefined,
      };
    })
    .sort((a, b) => JulianDate.compare(a.start, b.start));
  const l = dataSource.entities.getById("legend");
  const legend: Legend = {
    nearIceKm: l ? (prop(l, "near_ice_km") as number) : 5,
    contrastDbMin: l ? (prop(l, "contrast_db_min") as number) : 10,
    contrastDbMax: l ? (prop(l, "contrast_db_max") as number) : 20,
    thorScore: scenes
      .map((s) => dataSource.entities.getById(s.id))
      .map((e) => (e ? (prop(e, "thor_score") as number | undefined) : undefined))
      .find((v) => v !== undefined),
  };
  return { dataSource, scenes, legend };
}

export function applyLayers(dataSource: CzmlDataSource, layers: Layers): void {
  for (const e of dataSource.entities.values) {
    if (e.id.startsWith("det/")) e.show = layers.nearIce || prop(e, "near_ice") !== true;
    else if (e.id.startsWith("thor/"))
      e.show = layers.thor && (layers.nearIce || prop(e, "near_ice") !== true);
    else if (e.id.startsWith("seaice/")) e.show = layers.seaIce;
    else if (e.id.startsWith("footprint/")) e.show = layers.footprints;
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
