<script setup lang="ts">
import { onBeforeUnmount, onMounted, reactive, ref, shallowRef, watch } from "vue";
import type { CzmlDataSource, Viewer } from "cesium";
import ScenePanel from "./components/ScenePanel.vue";
import {
  applyLayers,
  createViewer,
  jumpTo,
  loadDetections,
  sceneAt,
  type Layers,
  type Legend,
  type SceneInfo,
} from "./scene";

const container = ref<HTMLDivElement>();
const scenes = shallowRef<SceneInfo[]>([]);
const currentId = ref<string>();
const legend = ref<Legend>({ nearIceKm: 5, contrastDbMin: 10, contrastDbMax: 20 });
const error = ref<string>();
const layers = reactive<Layers>({ nearIce: true, seaIce: true, footprints: true, thor: true });

// Plain variables, not refs: Cesium objects must stay out of Vue's reactivity.
let viewer: Viewer | undefined;
let dataSource: CzmlDataSource | undefined;

onMounted(async () => {
  viewer = createViewer(container.value!);
  try {
    ({ dataSource, scenes: scenes.value, legend: legend.value } = await loadDetections(viewer));
  } catch (e) {
    error.value = `Could not load detections (${String(e)}). Run "python -m iceberg_sar.cli czml" first.`;
    return;
  }
  applyLayers(dataSource, layers);
  viewer.clock.onTick.addEventListener((clock) => {
    const id = sceneAt(scenes.value, clock.currentTime)?.id;
    if (id !== currentId.value) currentId.value = id;
  });
});

onBeforeUnmount(() => viewer?.destroy());

watch(layers, () => dataSource && applyLayers(dataSource, layers));

function jump(scene: SceneInfo): void {
  if (viewer) jumpTo(viewer, scene);
}
</script>

<template>
  <div ref="container" class="globe" />
  <ScenePanel v-model:layers="layers" :scenes="scenes" :legend="legend" :current-id="currentId" @jump="jump" />
  <div v-if="error" class="error">{{ error }}</div>
</template>

<style scoped>
.globe { position: absolute; inset: 0; }
.error {
  position: absolute;
  top: 10px;
  right: 10px;
  max-width: 420px;
  padding: 10px 12px;
  border-radius: 6px;
  background: #5a1d1d;
  color: #fde2e2;
  font-size: 13px;
}
</style>
