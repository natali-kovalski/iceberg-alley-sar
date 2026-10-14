<script setup lang="ts">
const emit = defineEmits<{ close: [] }>();
const REPO = "https://github.com/natali-kovalski/iceberg-alley-sar";
</script>

<template>
  <div class="backdrop" @click.self="emit('close')">
    <section class="about" role="dialog" aria-labelledby="about-title">
      <header>
        <h2 id="about-title">About this viewer</h2>
        <button type="button" aria-label="Close" @click="emit('close')">×</button>
      </header>

      <p>
        Iceberg candidates detected automatically in Sentinel-1 radar images of "Iceberg Alley",
        the northeast coast of Newfoundland. Radar sees through cloud and darkness, so every pass
        gives a picture of the sea. Step through the passes with the list or the timeline, and
        click a point to see its measurements.
      </p>

      <h3>What the points are</h3>
      <ul>
        <li><b class="cfar">Orange dots</b>: bright targets found by a CA-CFAR detector.</li>
        <li><b class="thor">Cyan rings</b>: targets found by the THOR foundation model (frozen
          encoder + a small trained head).</li>
        <li><b class="grey">Grey</b>: within a few km of pack ice, so probably ice floes.</li>
        <li><b class="truth">Blue ✕</b>: icebergs marked by International Ice Patrol analysts on
          the same pass (training frames only).</li>
      </ul>
      <p class="note">
        These are <b>candidates</b>, not confirmed icebergs: ships look similar to radar, and
        without ship positions (AIS) the false-alarm rate can't be measured. On a validation pass
        the CFAR found 78% of the icebergs IIP analysts marked.
      </p>

      <h3>Method</h3>
      <p>
        Cross-polarized (HV) backscatter, land and pack ice masked. CA-CFAR compares each pixel
        with the sea around it and keeps those brighter than a 10⁻⁶ false-alarm threshold, then a
        co-polarized (HH) check removes speckle. THOR runs on the same masked scenes.
      </p>

      <h3>Data</h3>
      <ul>
        <li>Contains modified Copernicus Sentinel data (2019–2026), processed by ESA; terrain
          correction by ASF HyP3.</li>
        <li>Iceberg sightings: International Ice Patrol, via NSIDC (G00807).</li>
        <li>Basemap and terrain: Esri World Imagery and World Elevation.</li>
      </ul>

      <p class="links">
        <a :href="REPO" target="_blank" rel="noopener">Code, method and validation on GitHub →</a>
      </p>
    </section>
  </div>
</template>

<style scoped>
.backdrop {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 16px;
  background: rgba(4, 8, 14, 0.55);
  box-sizing: border-box;
  z-index: 10;
}
.about {
  width: 560px;
  max-width: 100%;
  max-height: 100%;
  overflow-y: auto;
  padding: 14px 18px 16px;
  border-radius: 8px;
  background: rgba(16, 24, 38, 0.97);
  color: #e6edf5;
  font-size: 13px;
  line-height: 1.5;
  box-sizing: border-box;
}
header { display: flex; justify-content: space-between; align-items: center; }
h2 { margin: 0; font-size: 16px; }
h3 { margin: 14px 0 4px; font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: #8fa3bb; }
p { margin: 8px 0; }
ul { margin: 4px 0; padding-left: 18px; }
li { margin: 2px 0; }
.note { color: #c9d4e0; }
.cfar { color: #fba238; }
.thor { color: rgb(0, 229, 255); }
.grey { color: #a9b3bf; }
.truth { color: #6f93f5; }   /* the map's #1e4fd8, lightened to read on the dark panel */
a { color: rgb(0, 229, 255); }
.links { margin-top: 14px; }
button { background: none; border: none; color: #8fa3bb; font-size: 22px; line-height: 1; cursor: pointer; }
button:hover { color: #fff; }
</style>
