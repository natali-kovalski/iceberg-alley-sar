<script setup lang="ts">
import type { Layers, Legend, SceneInfo } from "../scene";

import { computed, ref } from "vue";

const props = defineProps<{ scenes: SceneInfo[]; legend: Legend; currentId?: string }>();
const layers = defineModel<Layers>("layers", { required: true });
const emit = defineEmits<{ jump: [scene: SceneInfo] }>();

// On a phone the panel would cover the globe, so it starts collapsed to its title row.
const collapsed = ref(window.matchMedia("(max-width: 600px)").matches);

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
const hasTruth = computed(() => props.scenes.some((s) => s.nTruth !== undefined));
const SPLIT_BADGE: Record<string, string> = { train: "THOR train", val: "THOR val" };
function splitTitle(s: SceneInfo): string | undefined {
  if (!s.split) return undefined;
  const what = s.split === "train"
    ? "THOR training frame: the head was fitted on these labels, so THOR results here are in-sample"
    : "THOR pack-ice validation frame: labelled, never trained on";
  return `${what}. ${s.nTruth ?? 0} IIP analyst labels shown.`;
}
const current = computed(() => props.scenes.find((x) => x.id === props.currentId));
const currentHasIce = computed(() => (current.value ? hasIce(current.value) : props.scenes.some(hasIce)));
</script>

<template>
  <aside class="panel">
    <div class="title">
      <h1>Iceberg Alley SAR</h1>
      <span class="actions">
        <button type="button" class="toggle" :aria-expanded="!collapsed"
                :aria-label="collapsed ? 'Show legend and scenes' : 'Hide legend and scenes'"
                @click="collapsed = !collapsed">{{ collapsed ? "▸" : "▾" }}</button>
      </span>
    </div>
    <p class="sub">Sentinel-1 HH/HV · <template v-if="hasThor">THOR detections, <span class="nw">CA-CFAR baseline</span></template><span v-else class="nw">CA-CFAR candidates</span></p>
    <!-- Open-water counts of the pass on screen, THOR first: near-ice candidates are likely floes. -->
    <p v-if="current" class="sub pass">
      <span class="nw">Pass {{ current.date }}</span> · <span class="nw">open water:</span>&#32;<template
        v-if="current.nThor !== undefined"><span class="n-thor">THOR {{ current.nThor }}</span> · </template><span
        class="n-cfar">CFAR {{ current.nOpenWater }}</span>
    </p>
    <div v-show="!collapsed">

    <h2>Legend</h2>
    <div v-if="hasThor" class="key"><span class="ring" /> THOR detection, score ≥ {{ legend.thorScore ?? 0.35 }}</div>
    <div class="key"><span class="dot cfar" /> CA-CFAR <template v-if="hasThor">baseline </template>candidate</div>
    <div v-if="hasThor" class="key"><span class="ring both"><span class="dot cfar" /></span> found by both</div>
    <div v-if="hasTruth" class="key"><span class="cross">✕</span> IIP analyst label, same pass (THOR training frames)</div>
    <div class="sizes-title">Size = target extent, same classes for both</div>
    <div class="sizes">
      <div v-for="c in sizeClasses" :key="c.label" class="size">
        <span class="slot">
          <span v-if="hasThor" class="ring" :style="{ width: `${c.px}px`, height: `${c.px}px` }" />
          <span v-else class="dot cfar" :style="{ width: `${c.px}px`, height: `${c.px}px` }" />
        </span>
        <span>{{ c.label }}</span><span class="muted">{{ c.range }}</span>
      </div>
    </div>
    <div class="key" :class="{ off: !currentHasIce }"><span class="dot grey" /> within {{ legend.nearIceKm }} km of pack ice (either detector)</div>
    <div class="key" :class="{ off: !currentHasIce }"><span class="swatch ice" /> pack-ice mask</div>
    <div class="key"><span class="swatch footprint" /> scene footprint</div>
    <div v-if="hasCoverage" class="key"><span class="swatch aoi" /> AOI, dimmed where the pass didn't image it</div>
    <p class="note">Click a point for its attributes. Candidates are not verified icebergs; ships also
      appear as bright targets.</p>

    <h2>Scenes</h2>
    <ul class="scenes">
      <li
        v-for="s in scenes"
        :key="s.id"
        :class="{ active: s.id === currentId, compare: s.compare, [`split-${s.split}`]: s.split }"
        :title="s.compare ? 'THOR vs CFAR compared on this pass: click for the recall chart' : splitTitle(s)"
        @click="emit('jump', s)"
      >
        <span class="date">{{ s.date }}</span>
        <span v-if="s.compare" class="badge">THOR vs CFAR</span>
        <span v-else-if="s.split" class="badge" :class="`badge-${s.split}`">{{ SPLIT_BADGE[s.split] ?? s.split }}</span>
      </li>
    </ul>

    <h2>Layers</h2>
    <label v-if="hasThor"><input v-model="layers.thor" type="checkbox" /> THOR detections (foundation model)</label>
    <label v-if="hasThor"><input v-model="layers.cfar" type="checkbox" /> CA-CFAR baseline</label>
    <label v-if="hasTruth"><input v-model="layers.truth" type="checkbox" /> IIP labels on training frames</label>
    <label :class="{ off: !currentHasIce }">
      <input v-model="layers.nearIce" type="checkbox" :disabled="!currentHasIce" /> Near-ice candidates (likely floes)
    </label>
    <label :class="{ off: !currentHasIce }">
      <input v-model="layers.seaIce" type="checkbox" :disabled="!currentHasIce" /> Pack-ice mask
      <span v-if="!currentHasIce" class="muted">(none in this scene)</span>
    </label>
    <label><input v-model="layers.footprints" type="checkbox" /> Scene footprint</label>
    <label v-if="hasCoverage"><input v-model="layers.unobserved" type="checkbox" /> Dim area not imaged</label>
    </div>
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
.title { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.actions { display: flex; align-items: center; gap: 4px; flex: none; }
.toggle { width: 22px; padding: 0; border: none; background: none; color: #c9d4e0; font-size: 14px; cursor: pointer; }
.toggle:hover { color: #fff; }
@media (max-width: 600px) { .panel { width: calc(100% - 120px); max-width: 300px; } }
h2 { margin: 14px 0 6px; font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: #8fa3bb; }
.sub { margin: 2px 0 0; color: #8fa3bb; }
.pass { margin-top: 6px; }
.nw, .n-thor, .n-cfar { white-space: nowrap; }
.n-thor { color: rgb(0, 229, 255); font-weight: 600; }
.n-cfar { color: rgb(251, 162, 56); }
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
.scenes li.split-train, .scenes li.split-val { display: flex; justify-content: space-between; align-items: center; border-left: 3px solid #1e4fd8; }
.scenes li.split-val { border-left-style: dashed; }
.cross { width: 12px; text-align: center; font-size: 12px; font-weight: 700; color: #1e4fd8; flex: none; }
.badge { font-size: 10px; padding: 1px 6px; border-radius: 8px; color: rgb(0, 229, 255); border: 1px solid rgba(0, 229, 255, 0.6); }
.badge-train, .badge-val { color: #9db4f0; border-color: #9db4f0; }
.badge-val { border-style: dashed; }
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
.swatch.footprint { border: 1px solid rgb(255, 255, 0); }
.swatch.aoi { border: 1px solid #fff; background: rgba(8, 12, 20, 0.35); }
.ring { display: inline-block; width: 10px; height: 10px; border-radius: 50%; border: 2px solid rgb(0, 229, 255); flex: none; }
.ring.both { display: inline-flex; align-items: center; justify-content: center; width: 12px; height: 12px; }
.ring.both .dot { width: 8px; height: 8px; }
.note { margin: 10px 0 0; font-size: 11px; color: #8fa3bb; }
</style>
