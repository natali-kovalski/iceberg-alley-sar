<script setup lang="ts">
import type { SceneInfo } from "../scene";

defineProps<{ scene: SceneInfo }>();
const emit = defineEmits<{ close: [] }>();
</script>

<template>
  <aside v-if="scene.compare" class="compare">
    <header>
      <h2>THOR vs CFAR · {{ scene.date }}</h2>
      <button type="button" aria-label="Close" @click="emit('close')">×</button>
    </header>
    <p class="sub">Recall against {{ scene.compare.labels }} same-pass IIP labels, at matched detection counts.</p>
    <a :href="scene.compare.figureUrl" target="_blank" rel="noopener">
      <img :src="scene.compare.figureUrl" :alt="`Recall vs. detections, THOR and CFAR, ${scene.date}`" />
    </a>
  </aside>
</template>

<style scoped>
.compare {
  position: absolute;
  top: 50px;
  right: 10px;
  width: 460px;
  max-width: calc(100% - 300px);
  max-height: calc(100% - 140px);
  overflow-y: auto;
  padding: 10px 12px 12px;
  border-radius: 6px;
  background: rgba(16, 24, 38, 0.92);
  color: #e6edf5;
  font-size: 13px;
  box-sizing: border-box;
}
header { display: flex; justify-content: space-between; align-items: center; }
h2 { margin: 0; font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: rgb(0, 229, 255); }
button { background: none; border: none; color: #8fa3bb; font-size: 20px; line-height: 1; cursor: pointer; }
button:hover { color: #fff; }
.sub { margin: 4px 0 8px; color: #8fa3bb; font-size: 12px; }
img { display: block; width: 100%; border-radius: 4px; }
</style>
