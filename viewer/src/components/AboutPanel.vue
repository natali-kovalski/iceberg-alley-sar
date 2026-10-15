<script setup lang="ts">
const emit = defineEmits<{ close: [] }>();
const REPO = "https://github.com/natali-kovalski/iceberg-alley-sar";
const WRITEUP = `${REPO}/blob/main/docs/thor-vs-cfar.md`;
</script>

<template>
  <div class="backdrop" @click.self="emit('close')">
    <section class="about" role="dialog" aria-labelledby="about-title">
      <header>
        <h2 id="about-title">Iceberg Alley SAR</h2>
        <button type="button" aria-label="Close" @click="emit('close')">×</button>
      </header>

      <p class="lead">
        Every spring, icebergs from Greenland drift down the Labrador Current into "Iceberg
        Alley", past Newfoundland's northeast coast, its shipping lanes, fishing grounds and
        offshore platforms. Aircraft patrols are costly and often grounded by fog. Satellite
        radar (Sentinel-1) sees through cloud and darkness, so every pass gives a picture of the
        sea. This viewer shows the icebergs found automatically in those pictures.
      </p>

      <h3>The question</h3>
      <p>
        Can a foundation model, used off the shelf, find icebergs better than the classical radar
        detector? <b>THOR</b> is a model pretrained on huge amounts of satellite imagery,
        including Sentinel-1. It stays frozen here; only a tiny head of 769 numbers was trained
        on its features, using icebergs that International Ice Patrol analysts marked on 2019
        and 2020 passes. The comparison is a <b>CA-CFAR</b>, the textbook method that flags
        pixels much brighter than the sea around them.
      </p>

      <h3>The result</h3>
      <p>
        On four April 2019 passes that neither detector was tuned on, with 512 icebergs marked by
        Ice Patrol analysts, THOR found <b>84%</b> and the CFAR <b>74%</b>, with the same number of
        detections each. For large bergs it was <b>11 of 14</b> against 4. On the globe, cyan
        rings are THOR and orange dots the CFAR baseline. Click a pass marked "THOR vs CFAR" for
        its chart.
      </p>
      <p class="note">
        The fine print: it is one month of one year. THOR still needs the classical pack-ice
        mask, because on its own it mistakes pack ice for bergs. Small bergs are a tie. Ships look
        like bergs to radar, and without ship positions (AIS) neither detector's false-alarm rate
        can be measured.
      </p>

      <h3>How it's built</h3>
      <p>
        Python pipeline: scene search and terrain correction (ASF HyP3), land and pack-ice masks,
        THOR (PyTorch, TerraTorch) and the CFAR, then CZML for this Vue + CesiumJS globe. New
        passes are added with two commands and deployed by GitHub Actions. Next: ship positions to
        measure precision, and a finer THOR patch size for small bergs.
      </p>

      <p class="who">
        A portfolio project by <b>Natali Kovalski</b>, geospatial engineer (photogrammetry, 3D,
        geospatial AI), learning SAR on a very Newfoundland problem.
      </p>

      <h3>Data</h3>
      <ul>
        <li>Contains modified Copernicus Sentinel data (2019–2026), processed by ESA; terrain
          correction by ASF HyP3.</li>
        <li>THOR foundation model: FM4CS (Apache 2.0); method after Forgaard et al., IGARSS 2026.</li>
        <li>Iceberg sightings: International Ice Patrol, via NSIDC (G00807).</li>
        <li>Basemap and terrain: Esri World Imagery and World Elevation.</li>
      </ul>

      <p class="links">
        <a :href="REPO" target="_blank" rel="noopener">Code and README on GitHub →</a><br>
        <a :href="WRITEUP" target="_blank" rel="noopener">Full THOR vs CFAR write-up →</a>
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
.lead { font-size: 14px; }
.note { color: #a9b8c8; font-size: 12px; }
.who { margin-top: 14px; padding-top: 10px; border-top: 1px solid rgba(143, 163, 187, 0.25); color: #c9d4e0; }
a { color: rgb(0, 229, 255); }
.links { margin-top: 14px; }
button { background: none; border: none; color: #8fa3bb; font-size: 22px; line-height: 1; cursor: pointer; }
button:hover { color: #fff; }
</style>
