<script setup lang="ts">
import type { Layers, Legend, SceneInfo } from "../scene";

import { computed } from "vue";

const props = defineProps<{ scenes: SceneInfo[]; legend: Legend; currentId?: string }>();
const layers = defineModel<Layers>("layers", { required: true });
const emit = defineEmits<{ jump: [scene: SceneInfo] }>();

// Ice controls only mean something when the scene on screen has pack ice mapped.
const hasIce = (s: SceneInfo) => s.iceAreaKm2 > 0;
const hasCoverage = computed(() => props.scenes.some((s) => s.coveragePct !== undefined));
const sizeClasses = computed(() => {
  const [a, b] = props.legend.sizeBreaksM;
  const [s, m, l] = props.legend.sizePx;
  return [
    { label: "small", range: `< ${a} m`, px: s },
    { label: "medium", range: `${a}–${b} m`, px: m },
    { label: "large", range: `≥ ${b} m`, px: l },
  ];
});
const hasThor = computed(() => props.scenes.some((s) => s.nThor !== undefined));
const currentHasIce = computed(() => {
  const s = props.scenes.find((x) => x.id === props.currentId);
  return s ? hasIce(s) : props.scenes.some(hasIce);
});
</script>

<template>
  <aside class="panel">
    <h1>Iceberg Alley SAR</h1>
    <p class="sub">Sentinel-1 HH/HV, CA-CFAR candidates<template v-if="hasThor"> + THOR</template></p>

    <h2>Scenes</h2>
    <ul class="scenes">
      <li
        v-for="s in scenes"
        :key="s.id"
        :class="{ active: s.id === currentId, compare: s.compare }"
        :title="s.compare ? 'THOR vs CFAR compared on this pass: click for the recall chart' : undefined"
        @click="emit('jump', s)"
      >
        <span class="date">{{ s.date }}</span>
        <span v-if="s.compare" class="badge">THOR vs CFAR</span>
      </li>
    </ul>

    <h2>Layers</h2>
    <label :class="{ off: !currentHasIce }">
      <input v-model="layers.nearIce" type="checkbox" :disabled="!currentHasIce" /> Near-ice candidates (likely floes)
    </label>
    <label :class="{ off: !currentHasIce }">
      <input v-model="layers.seaIce" type="checkbox" :disabled="!currentHasIce" /> Pack-ice mask
      <span v-if="!currentHasIce" class="muted">(none in this scene)</span>
    </label>
    <label><input v-model="layers.footprints" type="checkbox" /> Scene footprint</label>
    <label v-if="hasCoverage"><input v-model="layers.unobserved" type="checkbox" /> Dim area not imaged</label>
    <label v-if="hasThor"><input v-model="layers.thor" type="checkbox" /> THOR detections (foundation model)</label>

    <h2>Legend</h2>
    <div class="key"><span class="dot cfar" /> CA-CFAR candidate</div>
    <div v-if="hasThor" class="key"><span class="ring" /> THOR candidate, score ≥ {{ legend.thorScore ?? 0.35 }}</div>
    <div v-if="hasThor" class="key"><span class="ring both"><span class="dot cfar" /></span> found by both</div>
    <div class="sizes-title">Size = target extent, same classes for both</div>
    <div class="sizes">
      <div v-for="c in sizeClasses" :key="c.label" class="size">
        <span class="slot"><span class="dot cfar" :style="{ width: `${c.px}px`, height: `${c.px}px` }" /></span>
        <span>{{ c.label }}</span><span class="muted">{{ c.range }}</span>
      </div>
    </div>
    <div class="key" :class="{ off: !currentHasIce }"><span class="dot grey" /> within {{ legend.nearIceKm }} km of pack ice (either detector)</div>
    <div class="key" :class="{ off: !currentHasIce }"><span class="swatch ice" /> pack-ice mask</div>
    <div class="key"><span class="swatch footprint" /> scene footprint</div>
    <div v-if="hasCoverage" class="key"><span class="swatch aoi" /> AOI, dimmed where the pass didn't image it</div>
    <p class="note">Click a point for its attributes. Candidates are not verified icebergs; ships also
      appear as bright targets.</p>
  </aside>
</template>

<style scoped>
.panel {
  position: absolute;
  top: 10px;
  left: 10px;
  width: 270px;
  max-height: calc(100% - 140px);
  overflow-y: auto;
  padding: 12px 14px;
  border-radius: 6px;
  background: rgba(16, 24, 38, 0.88);
  color: #e6edf5;
  font-size: 13px;
  box-sizing: border-box;
}
h1 { margin: 0; font-size: 16px; }
h2 { margin: 14px 0 6px; font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: #8fa3bb; }
.sub { margin: 2px 0 0; color: #8fa3bb; }
.scenes { list-style: none; margin: 0; padding: 0; }
.scenes li {
  padding: 6px 8px;
  margin-bottom: 4px;
  border-radius: 4px;
  border: 1px solid transparent;
  cursor: pointer;
}
.scenes li:hover { background: rgba(255, 255, 255, 0.06); }
.scenes li.compare { display: flex; justify-content: space-between; align-items: center; border-left: 3px solid rgb(0, 229, 255); }
.scenes li.active { border-color: #fba238; background: rgba(251, 162, 56, 0.12); }
.badge { font-size: 10px; padding: 1px 6px; border-radius: 8px; color: rgb(0, 229, 255); border: 1px solid rgba(0, 229, 255, 0.6); }
.date { font-weight: 600; }
.muted { color: #8fa3bb; }
label { display: block; margin: 4px 0; cursor: pointer; }
.off { opacity: 0.45; }
label.off { cursor: default; }
.key { display: flex; align-items: center; gap: 6px; margin: 4px 0; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; border: 1px solid #000; flex: none; }
.dot.cfar { background: rgb(251, 162, 56); }
.dot.grey { width: 8px; height: 8px; background: #969696; }
.sizes-title { margin: 8px 0 2px; }
.sizes { display: flex; gap: 10px; margin-bottom: 6px; }
.size { display: flex; flex-direction: column; align-items: center; font-size: 11px; }
.slot { display: flex; align-items: center; justify-content: center; height: 26px; }
.swatch { display: inline-block; width: 14px; height: 10px; }
.swatch.ice { background: rgba(120, 200, 255, 0.35); border: 1px solid rgb(120, 200, 255); }
.swatch.footprint { border: 1px solid rgb(255, 210, 60); }
.swatch.aoi { border: 1px solid #fff; background: rgba(8, 12, 20, 0.6); }
.ring { display: inline-block; width: 10px; height: 10px; border-radius: 50%; border: 2px solid rgb(0, 229, 255); flex: none; }
.ring.both { display: inline-flex; align-items: center; justify-content: center; width: 12px; height: 12px; }
.ring.both .dot { width: 8px; height: 8px; }
.note { margin: 10px 0 0; font-size: 11px; color: #8fa3bb; }
</style>
