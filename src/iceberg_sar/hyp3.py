"""Order HyP3 RTC jobs and download the results."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import hyp3_sdk
from dotenv import load_dotenv

from iceberg_sar.config import Config

JOB_NAME = "iceberg-alley"


def get_client() -> hyp3_sdk.HyP3:
    """HyP3 client authenticated with Earthdata credentials from `.env`."""
    load_dotenv()
    user = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    if not user or not password:
        raise RuntimeError("Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD in .env")
    return hyp3_sdk.HyP3(username=user, password=password)


def rtc_options(cfg: Config) -> dict[str, object]:
    h = cfg.section("hyp3")
    return {
        "resolution": h["resolution"],
        "scale": h["scale"],
        "radiometry": h["radiometry"],
        "dem_matching": h["dem_matching"],
        "include_inc_map": h["include_inc_map"],
        "include_scattering_area": h["include_scattering_area"],
    }


def submit_rtc(
    client: hyp3_sdk.HyP3,
    granule: str,
    cfg: Config,
    name: str = JOB_NAME,
    overrides: dict[str, object] | None = None,
) -> hyp3_sdk.Batch:
    """Submit one RTC job, unless one with this granule, resolution and radiometry exists."""
    options = {**rtc_options(cfg), **(overrides or {})}
    existing = client.find_jobs(name=name, job_type="RTC_GAMMA")
    for job in existing:
        params = job.job_parameters
        same_res = float(params.get("resolution", 30)) == float(options["resolution"])
        same_rad = params.get("radiometry", "gamma0") == options["radiometry"]
        if granule in params.get("granules", []) and same_res and same_rad and not job.failed():
            return hyp3_sdk.Batch([job])
    return client.submit_rtc_job(granule, name=name, **options)


def download_jobs(batch: hyp3_sdk.Batch, out_dir: Path) -> list[Path]:
    """Download succeeded jobs as zips and extract them. Returns product directories."""
    out_dir.mkdir(parents=True, exist_ok=True)
    product_dirs: list[Path] = []
    for job in batch:
        if not job.succeeded():
            continue
        for zip_path in job.download_files(out_dir):
            product_dirs.append(extract(Path(zip_path)))
    return product_dirs


def extract(zip_path: Path) -> Path:
    target = zip_path.with_suffix("")
    if not target.is_dir():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(zip_path.parent)
    return target
