<script setup lang="ts">
import type { Layers, Legend, SceneInfo } from "../scene";

import { computed } from "vue";

const props = defineProps<{ scenes: SceneInfo[]; legend: Legend; currentId?: string }>();
const layers = defineModel<Layers>("layers", { required: true });
const emit = defineEmits<{ jump: [scene: SceneInfo] }>();

// Ice controls only mean something when the scene on screen has pack ice mapped.
const hasIce = (s: SceneInfo) => s.iceAreaKm2 > 0;
const hasThor = computed(() => props.scenes.some((s) => s.nThor !== undefined));
const currentHasIce = computed(() => {
  const s = props.scenes.find((x) => x.id === props.currentId);
  return s ? hasIce(s) : props.scenes.some(hasIce);
});
</script>

<template>
  <aside class="panel">
    <h1>Iceberg Alley SAR</h1>
    <p class="sub">Sentinel-1 HV, CA-CFAR candidates<template v-if="hasThor"> + THOR</template></p>

    <h2>Scenes</h2>
    <ul class="scenes">
      <li
        v-for="s in scenes"
        :key="s.id"
        :class="{ active: s.id === currentId }"
        @click="emit('jump', s)"
      >
        <div class="row">
          <span class="date">{{ s.date }}</span>
          <span class="sat">{{ s.platform }}</span>
        </div>
        <div class="stats">
          <span><b>{{ s.nOpenWater }}</b> open water</span>
          <span v-if="s.nThor !== undefined"><b>{{ s.nThor }}</b> THOR (open water)</span>
          <template v-if="hasIce(s)">
            <span class="muted">{{ s.nNearIce }} near ice</span>
            <span class="muted">{{ s.iceAreaKm2.toLocaleString() }} km² pack ice</span>
          </template>
          <span v-else class="muted">no pack ice</span>
        </div>
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
    <label v-if="hasThor"><input v-model="layers.thor" type="checkbox" /> THOR detections (foundation model)</label>

    <h2>Legend</h2>
    <div class="ramp" />
    <div class="ramp-labels">
      <span>≤{{ legend.contrastDbMin }} dB</span><span>contrast over sea</span><span>≥{{ legend.contrastDbMax }} dB</span>
    </div>
    <div class="key" :class="{ off: !currentHasIce }"><span class="dot grey" /> within {{ legend.nearIceKm }} km of pack ice</div>
    <div class="key"><span class="dot small" /><span class="dot big" /> size = grown target extent</div>
    <div class="key" :class="{ off: !currentHasIce }"><span class="swatch ice" /> pack-ice mask</div>
    <div class="key"><span class="swatch footprint" /> scene footprint</div>
    <div v-if="hasThor" class="key"><span class="ring" /> THOR, score ≥ {{ legend.thorScore ?? 0.35 }}</div>
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
.scenes li.active { border-color: #fba238; background: rgba(251, 162, 56, 0.12); }
.row { display: flex; justify-content: space-between; }
.date { font-weight: 600; }
.sat, .muted { color: #8fa3bb; }
.stats { display: flex; flex-direction: column; margin-top: 2px; }
label { display: block; margin: 4px 0; cursor: pointer; }
.off { opacity: 0.45; }
label.off { cursor: default; }
.ramp {
  height: 10px;
  border-radius: 2px;
  background: linear-gradient(to right, #5601a4, #a51f99, #dc5d67, #fba238, #f0f921);
}
.ramp-labels { display: flex; justify-content: space-between; font-size: 11px; color: #8fa3bb; margin-bottom: 6px; }
.key { display: flex; align-items: center; gap: 6px; margin: 4px 0; }
.dot { display: inline-block; border-radius: 50%; background: #dc5d67; border: 1px solid #000; }
.dot.grey { width: 8px; height: 8px; background: #969696; }
.dot.small { width: 6px; height: 6px; }
.dot.big { width: 14px; height: 14px; }
.swatch { display: inline-block; width: 14px; height: 10px; }
.swatch.ice { background: rgba(120, 200, 255, 0.35); border: 1px solid rgb(120, 200, 255); }
.swatch.footprint { border: 1px solid rgb(255, 210, 60); }
.ring { display: inline-block; width: 10px; height: 10px; border-radius: 50%; border: 2px solid rgb(0, 229, 255); }
.note { margin: 10px 0 0; font-size: 11px; color: #8fa3bb; }
</style>
