# Benchmark environment (`cfar-bench`)

Environment for the Færch et al. 2023 reference CFAR detectors
(github.com/LaustFaerch/cfar-object-detection, GPL-3.0). It is kept apart from the main
project env: the library is never imported into `src/iceberg_sar/`, only run through
`external/cfar_runner/run_cfar.py` (files in, files out).

## Reference code

| | |
|---|---|
| Repository | https://github.com/LaustFaerch/cfar-object-detection |
| Commit | `4136b4c27872eb56f2e206d920f47fe82565e3ea` (2025-02-09, `main`; last tag `v3` = `f2a3e37`) |
| Local clone | `external/cfar-object-detection/` (gitignored, unmodified) |
| Installed as | editable, `cfar-filters 0.0.0` (`setup.py` has no version and no pinned deps) |

The repo ships no requirements file. `setup.py` lists numpy, pandas, numba and scipy unpinned.
scikit-image is imported by `utils.py` but not listed. The paper used Python 3.8 and the notebook
was last run on Python 3.10.9.

## Environment

uv venv (no conda on this machine), Python **3.10.9**, at `external/cfar-bench/` (gitignored).
`benchmark.cfar_env_python` in `benchmarks/config.yaml` points at its `python.exe`.

| Package | Version |
|---|---|
| numpy | 1.26.4 |
| numba | 0.60.0 |
| scipy | 1.13.1 |
| pandas | 2.2.3 |
| scikit-image | 0.24.0 |
| rasterio | 1.4.4 |
| matplotlib | 3.9.4 |

The bounds are in `external/cfar_runner/requirements.txt` and the full freeze in
`external/cfar_runner/requirements.lock`. The editable `cfar-filters` line is left out of the
lock because it holds an absolute path.

numpy is kept below 2: the library was written against numpy 1.x (it used `np.bool` until 2024).

### Recreate (PowerShell, from the repo root)

```powershell
git clone https://github.com/LaustFaerch/cfar-object-detection.git external/cfar-object-detection
git -C external/cfar-object-detection checkout 4136b4c27872eb56f2e206d920f47fe82565e3ea
uv venv external/cfar-bench --python 3.10
uv pip install --python external/cfar-bench/Scripts/python.exe -r external/cfar_runner/requirements.lock
uv pip install --python external/cfar-bench/Scripts/python.exe --no-deps -e external/cfar-object-detection
```

## 0a check: `examples.ipynb` as-is

Run headless from the repo directory (so the notebook's relative `test-files/` paths resolve).
The executed copy goes to `data/benchmark/`, so the clone stays untouched:

```powershell
cd external/cfar-object-detection
../cfar-bench/Scripts/python -m jupyter nbconvert --to notebook --execute examples.ipynb `
  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=3600 `
  --output-dir ../../data/benchmark/faerch2023/m0 --output examples_executed.ipynb
```

Result (2026-10-21): all 15 cells ran, with no errors and no stderr output, in 75 s
(including Numba compilation).

### Fingerprint

These are outlier pixel counts per detector on the bundled `test-files/sentinel1-image.tif`
(1500 × 1500, dB) with the notebook's dual-band settings (`AND`, PFA 1e-12, ENL 10.7,
minsize 2, wi 9, wo 15). They let a rebuilt env be checked against this one.

| Detector | Outlier px |
|---|---|
| gamma | 468 |
| lognorm | 47 |
| k | 73 |
| wishart | 734 |
| nis | 605 |
| idpolrad | 77 |

Timings in this run included Numba compilation, so they are not reported. Compile on a
warm-up call before timing anything (see Stage 2 pitfalls).

## Code versions vs the paper

The paper (published 2023-12-15) predates HEAD. Changes to the detector code between tag `v2`
(2023-10-30) and HEAD:

- **NIS:** in v2, `normsum.transform` ignored `wi`/`wo` and always used its defaults
  (9 / 15 px). This was fixed 2024-02-19 (`6c57c93`). With the paper's windows (9 / 13 px at
  40 m) HEAD therefore uses a slightly smaller clutter ring than the published NIS run did.
- **K:** the LUT size moved from a parameter (default 40) to a constant 40. The paper also
  states 40. Other changes only add guards and suppress warnings.
- `np.bool` → `bool`, plus warning suppression. No change in results.

We run HEAD. If an NIS result in 1f falls outside tolerance, this difference is the first
thing to check.
