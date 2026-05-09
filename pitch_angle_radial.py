#!/usr/bin/env python3
"""Compute and plot radial magnetic-field pitch angles for NGC 6946.

The profile calculation follows the Monte-Carlo/median profile workflow used in
previous NGC 6946 notebooks: pitch angles are evaluated per valid pixel, radial
bins are summarized with circular medians, and bootstrap 68/95 percentiles are
written with the same ``R_s1*``/``pitch_s1*`` style columns used by the legacy
``magnetic_pitch_angle_profile_plot`` helper.

Important geometry fixes are retained from the newer standalone script:
positions may be measured from the FITS WCS, vectors are rotated into the galaxy
major/minor axes, magnetic-field vectors are deprojected in the galaxy plane,
and the final magnetic pitch angles are wrapped to the 180-degree-ambiguous
interval [-90, +90) degrees.

Example
-------
python pitch_angle_radial.py \
  --mask /home/amir/Documents/PhD/NGC6946/data/regional_mask.fits \
  --fir-i /home/amir/Documents/PhD/NGC6946/FIR-data_18arcsec/NGC6946-FIR-I-18arcsec.fits \
  --fir-q /home/amir/Documents/PhD/NGC6946/FIR-data_18arcsec/NGC6946-FIR-Q-18arcsec.fits \
  --fir-u /home/amir/Documents/PhD/NGC6946/FIR-data_18arcsec/NGC6946-FIR-U-18arcsec.fits \
  --radio-i /home/amir/Documents/PhD/NGC6946/Radio_data/NGC6946_6cm_I.fits \
  --radio-q /home/amir/Documents/PhD/NGC6946/Radio_data/NGC6946-Q-radio-18.fits \
  --radio-u /home/amir/Documents/PhD/NGC6946/Radio_data/NGC6946-U-radio-18.fits
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path



REGION_VALUES = {
    "Arm": 1,
    "Interarm": 2,
    "Center": 3,
}

# Legacy masks used in the reference notebook code.  ``True`` means the pixel is
# retained for the selected mode.  The values mirror ``apply_mask`` in the
# prompt, where bool_remove was the complement of the desired science region.
LEGACY_REGION_KEEP_VALUES = {
    "all": None,
    "arm": (0, 3, 4),
    "interarm": (1, 3),
    "north_arm": (4,),
    "south_arm": (0,),
    "nan": None,
}


DEFAULT_PATHS = {
    "mask": Path("/home/amir/Documents/PhD/NGC6946/data/regional_mask.fits"),
    "fir_i": Path("/home/amir/Documents/PhD/NGC6946/FIR-data_18arcsec/NGC6946-FIR-I-18arcsec.fits"),
    "fir_q": Path("/home/amir/Documents/PhD/NGC6946/FIR-data_18arcsec/NGC6946-FIR-Q-18arcsec.fits"),
    "fir_u": Path("/home/amir/Documents/PhD/NGC6946/FIR-data_18arcsec/NGC6946-FIR-U-18arcsec.fits"),
    "radio_i": Path("/home/amir/Documents/PhD/NGC6946/Radio_data/NGC6946_6cm_I.fits"),
    "radio_q": Path("/home/amir/Documents/PhD/NGC6946/Radio_data/NGC6946-Q-radio-18.fits"),
    "radio_u": Path("/home/amir/Documents/PhD/NGC6946/Radio_data/NGC6946-U-radio-18.fits"),
}


@dataclass(frozen=True)
class Geometry:
    """Galaxy geometry and distance parameters."""

    distance_mpc: float = 7.72
    inclination_deg: float = 38.0
    position_angle_deg: float = 242.0
    pixel_scale_arcsec: float = 1.0
    use_wcs: bool = False
    center_x: float | None = 84.0
    center_y: float | None = 71.0
    center_ra_deg: float | None = None
    center_dec_deg: float | None = None

    @property
    def arcsec_to_kpc(self) -> float:
        return self.distance_mpc * 1_000.0 / 206_265.0


@dataclass(frozen=True)
class PolarizationDataset:
    name: str
    i_path: Path
    q_path: Path
    u_path: Path
    b_rotation_deg: float
    color: str
    marker: str
    linestyle: str


def load_fits(path: Path) -> tuple[np.ndarray, fits.Header]:
    """Load a FITS image as a squeezed 2-D float array with its header."""
    import numpy as np
    from astropy.io import fits

    data, header = fits.getdata(path, header=True)
    data = np.asarray(np.squeeze(data), dtype=float)
    if data.ndim != 2:
        raise ValueError(f"{path} must contain a 2-D image after squeezing; got {data.shape}")
    return data, header


def ensure_same_shape(reference: np.ndarray, named_arrays: dict[str, np.ndarray]) -> None:
    """Fail clearly if inputs are not already on the same pixel grid."""
    ref_shape = reference.shape
    bad = {name: array.shape for name, array in named_arrays.items() if array.shape != ref_shape}
    if bad:
        details = ", ".join(f"{name}: {shape}" for name, shape in bad.items())
        raise ValueError(
            "All maps must be reprojected/convolved to the same grid before pitch-angle "
            f"analysis. Expected {ref_shape}; mismatched arrays: {details}"
        )


def center_pixel(header: fits.Header, shape: tuple[int, int], geom: Geometry) -> tuple[float, float]:
    """Return the galaxy centre in zero-indexed pixel coordinates (x, y)."""
    from astropy.wcs import WCS

    if geom.center_x is not None and geom.center_y is not None:
        return geom.center_x, geom.center_y

    wcs = WCS(header).celestial
    if geom.center_ra_deg is not None and geom.center_dec_deg is not None and wcs.has_celestial:
        x0, y0 = wcs.world_to_pixel_values(geom.center_ra_deg, geom.center_dec_deg)
        return float(x0), float(y0)

    if "CRPIX1" in header and "CRPIX2" in header:
        return float(header["CRPIX1"] - 1.0), float(header["CRPIX2"] - 1.0)

    ny, nx = shape
    return (nx - 1.0) / 2.0, (ny - 1.0) / 2.0


def sky_offsets_arcsec(header: fits.Header, shape: tuple[int, int], geom: Geometry) -> tuple[np.ndarray, np.ndarray]:
    """Return east and north offsets from the galaxy centre in arcseconds."""
    import numpy as np
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales

    ny, nx = shape
    yy, xx = np.indices(shape, dtype=float)
    x0, y0 = center_pixel(header, shape, geom)

    wcs = WCS(header).celestial
    if geom.use_wcs and wcs.has_celestial:
        ra, dec = wcs.pixel_to_world_values(xx, yy)
        ra0, dec0 = wcs.pixel_to_world_values(x0, y0)
        east = (ra - ra0) * np.cos(np.deg2rad(dec0)) * 3600.0
        north = (dec - dec0) * 3600.0
        return east, north

    pixel_scale_arcsec = geom.pixel_scale_arcsec
    if geom.use_wcs:
        if "CDELT1" in header or "CDELT2" in header:
            pixel_scale_arcsec = abs(float(header.get("CDELT2", header.get("CDELT1")))) * 3600.0
        elif any(key.startswith(("CD", "PC")) for key in header):
            try:
                scales_deg = np.asarray(proj_plane_pixel_scales(WCS(header)), dtype=float)
                scales_deg = scales_deg[np.isfinite(scales_deg) & (scales_deg != 0.0)]
                if scales_deg.size:
                    pixel_scale_arcsec = float(np.mean(np.abs(scales_deg[:2])) * 3600.0)
            except Exception:
                pixel_scale_arcsec = geom.pixel_scale_arcsec

    if not np.isfinite(pixel_scale_arcsec) or pixel_scale_arcsec <= 0.0:
        pixel_scale_arcsec = geom.pixel_scale_arcsec
    east = (xx - x0) * pixel_scale_arcsec
    north = (yy - y0) * pixel_scale_arcsec
    return east, north


def deproject_positions(
    east_arcsec: np.ndarray,
    north_arcsec: np.ndarray,
    geom: Geometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deproject sky offsets into disk-plane x/y, radius, and azimuth."""
    import numpy as np

    pa = np.deg2rad(geom.position_angle_deg)
    inc = np.deg2rad(geom.inclination_deg)

    x_major = east_arcsec * np.sin(pa) + north_arcsec * np.cos(pa)
    y_minor = -east_arcsec * np.cos(pa) + north_arcsec * np.sin(pa)
    y_disk = y_minor / np.cos(inc)

    radius_arcsec = np.hypot(x_major, y_disk)
    radius_kpc = radius_arcsec * geom.arcsec_to_kpc
    phi = np.arctan2(y_disk, x_major)
    return x_major, y_disk, radius_kpc, phi


def debiased_fractional_polarization(
    i_map: np.ndarray,
    q_map: np.ndarray,
    u_map: np.ndarray,
    sigma_i: float,
    sigma_q: float,
    sigma_u: float,
) -> np.ndarray:
    """Return the Wardle-Kronberg style debiased fractional polarization."""
    import numpy as np

    polarized = np.hypot(q_map, u_map)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = polarized / i_map
        sigma_p = np.sqrt((q_map * sigma_q) ** 2 + (u_map * sigma_u) ** 2) / polarized
        sigma_frac = np.sqrt((sigma_p / i_map) ** 2 + (polarized * sigma_i / i_map**2) ** 2)
        debiased = np.sqrt(np.clip(frac**2 - sigma_frac**2, 0.0, None))
    debiased[~np.isfinite(debiased)] = np.nan
    return debiased


def magnetic_pitch_angle(
    q_map: np.ndarray,
    u_map: np.ndarray,
    phi: np.ndarray,
    geom: Geometry,
    b_rotation_deg: float,
) -> np.ndarray:
    """Compute disk-plane magnetic pitch angle in degrees."""
    import numpy as np

    pa = np.deg2rad(geom.position_angle_deg)
    inc = np.deg2rad(geom.inclination_deg)

    b_angle = 0.5 * np.arctan2(u_map, q_map) + np.deg2rad(b_rotation_deg)

    # Unit vector components on the sky. Angles are east of north.
    v_east = np.sin(b_angle)
    v_north = np.cos(b_angle)

    # Rotate vector into projected major/minor axes, then deproject its minor-axis
    # component into the galaxy plane.
    v_major = v_east * np.sin(pa) + v_north * np.cos(pa)
    v_minor = -v_east * np.cos(pa) + v_north * np.sin(pa)
    v_minor_disk = v_minor / np.cos(inc)

    norm = np.hypot(v_major, v_minor_disk)
    with np.errstate(divide="ignore", invalid="ignore"):
        v_major = v_major / norm
        v_minor_disk = v_minor_disk / norm

    radial_component = v_major * np.cos(phi) + v_minor_disk * np.sin(phi)
    azimuthal_component = -v_major * np.sin(phi) + v_minor_disk * np.cos(phi)
    pitch = np.arctan2(radial_component, azimuthal_component)

    # Magnetic orientations are 180-degree ambiguous; report pitch in [-90, 90).
    pitch = (pitch + np.pi / 2.0) % np.pi - np.pi / 2.0
    pitch_deg = np.rad2deg(pitch)
    pitch_deg[~np.isfinite(pitch_deg)] = np.nan
    return pitch_deg


def wrap_pitch_degrees(values: np.ndarray | float) -> np.ndarray | float:
    """Wrap 180-degree-ambiguous pitch angles to [-90, 90) degrees."""
    import numpy as np

    return (np.asarray(values) + 90.0) % 180.0 - 90.0


def circular_mean_pitch_degrees(values: np.ndarray) -> float:
    """Mean pitch angle for 180-degree-ambiguous magnetic-field directions."""
    import numpy as np

    doubled = np.deg2rad(2.0 * values)
    mean = 0.5 * np.rad2deg(np.arctan2(np.nanmean(np.sin(doubled)), np.nanmean(np.cos(doubled))))
    return float(wrap_pitch_degrees(mean))


def circular_median_pitch_degrees(values: np.ndarray) -> float:
    """Circular median for 180-degree-ambiguous magnetic pitch angles."""
    import numpy as np

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan
    center = circular_mean_pitch_degrees(finite)
    return float(wrap_pitch_degrees(center + np.nanmedian(wrap_pitch_degrees(finite - center))))


def bootstrap_circular_median(
    values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, float | np.ndarray]:
    """Bootstrap the circular median and return 68/95 percentile intervals."""
    import numpy as np

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        empty = np.asarray([], dtype=float)
        return {
            "median": np.nan,
            "s1_down": np.nan,
            "s1_up": np.nan,
            "s2_down": np.nan,
            "s2_up": np.nan,
            "boot": empty,
        }

    median = circular_median_pitch_degrees(finite)
    if finite.size == 1 or n_bootstrap <= 0:
        boot = np.full(max(n_bootstrap, 1), median, dtype=float)
    else:
        indices = rng.integers(0, finite.size, size=(n_bootstrap, finite.size))
        boot = np.asarray([circular_median_pitch_degrees(sample) for sample in finite[indices]], dtype=float)

    # Percentiles are measured in a local coordinate system around the median so
    # bins that straddle -90/+90 do not get artificial huge intervals.
    deviations = wrap_pitch_degrees(boot - median)
    d16, d84 = np.nanpercentile(deviations, [16.0, 84.0])
    d025, d975 = np.nanpercentile(deviations, [2.5, 97.5])
    return {
        "median": median,
        "s1_down": float(wrap_pitch_degrees(median + d16)),
        "s1_up": float(wrap_pitch_degrees(median + d84)),
        "s2_down": float(wrap_pitch_degrees(median + d025)),
        "s2_up": float(wrap_pitch_degrees(median + d975)),
        "boot": boot,
    }


def measure_median_pitch_angle(pitch_angle_array: np.ndarray, n_bootstrap: int = 1_000, seed: int = 0) -> dict[str, float | np.ndarray]:
    """Measure the global median pitch angle from a 2-D simulation/pixel array."""
    import numpy as np

    rng = np.random.default_rng(seed)
    pitch_angle_array = np.asarray(pitch_angle_array, dtype=float)
    if pitch_angle_array.ndim == 1:
        pitch_angle_array = pitch_angle_array[None, :]

    median_pitch_angle_boot = np.zeros(pitch_angle_array.shape[0], dtype=float)
    for index in range(pitch_angle_array.shape[0]):
        median_pitch_angle_boot[index] = bootstrap_circular_median(
            pitch_angle_array[index, :],
            max(100, n_bootstrap // 10),
            rng,
        )["median"]

    summary = bootstrap_circular_median(median_pitch_angle_boot, n_bootstrap, rng)
    return {
        "median": summary["median"],
        "e95_down": summary["s2_down"],
        "e95_up": summary["s2_up"],
        "boot": median_pitch_angle_boot,
    }


def region_selection(mask: np.ndarray, mode: str | None, value: int | None = None) -> np.ndarray:
    """Return the region selection used for radial profiles."""
    import numpy as np

    if value is not None:
        return mask == value
    mode = (mode or "all").lower()
    if mode == "all":
        return np.ones(mask.shape, dtype=bool)
    if mode == "nan":
        return np.isnan(mask)
    if mode not in LEGACY_REGION_KEEP_VALUES:
        raise ValueError(f"Unknown region mode {mode!r}")
    keep_values = LEGACY_REGION_KEEP_VALUES[mode]
    if keep_values is None:
        return np.ones(mask.shape, dtype=bool)
    return np.isin(mask, keep_values)


def radial_profile(
    pitch_map: np.ndarray,
    radius_map: np.ndarray,
    region_mask: np.ndarray,
    bins: np.ndarray,
    valid_data: np.ndarray,
    errorbar_stat: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Radial pitch profile with legacy-compatible median/bootstrap columns."""
    import numpy as np

    centers = 0.5 * (bins[:-1] + bins[1:])
    pitch = np.full_like(centers, np.nan, dtype=float)
    errors = np.full_like(centers, np.nan, dtype=float)
    counts = np.zeros_like(centers, dtype=int)
    columns = {
        "R": np.full_like(centers, np.nan, dtype=float),
        "R_min": bins[:-1].astype(float),
        "R_max": bins[1:].astype(float),
        "R_s1up": np.full_like(centers, np.nan, dtype=float),
        "R_s1down": np.full_like(centers, np.nan, dtype=float),
        "R_s2up": np.full_like(centers, np.nan, dtype=float),
        "R_s2down": np.full_like(centers, np.nan, dtype=float),
        "pitch": pitch,
        "pitch_s1up": np.full_like(centers, np.nan, dtype=float),
        "pitch_s1down": np.full_like(centers, np.nan, dtype=float),
        "pitch_s2up": np.full_like(centers, np.nan, dtype=float),
        "pitch_s2down": np.full_like(centers, np.nan, dtype=float),
        "npix": counts.astype(float),
    }

    for index, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        selected = (radius_map >= lo) & (radius_map < hi) & region_mask & valid_data & np.isfinite(pitch_map)
        values = pitch_map[selected]
        radii = radius_map[selected]
        counts[index] = int(values.size)
        columns["npix"][index] = float(values.size)
        if values.size == 0:
            continue

        columns["R"][index] = float(np.nanmedian(radii))
        columns["R_s1down"][index], columns["R_s1up"][index] = np.nanpercentile(radii, [16.0, 84.0])
        columns["R_s2down"][index], columns["R_s2up"][index] = np.nanpercentile(radii, [2.5, 97.5])

        if errorbar_stat == "mean":
            center = circular_mean_pitch_degrees(values)
            deviations = wrap_pitch_degrees(values - center)
            sigma = float(np.nanstd(deviations, ddof=1)) if values.size > 1 else 0.0
            interval = {
                "median": center,
                "s1_down": center - sigma,
                "s1_up": center + sigma,
                "s2_down": center - 2.0 * sigma,
                "s2_up": center + 2.0 * sigma,
            }
        else:
            interval = bootstrap_circular_median(values, n_bootstrap, rng)
            center = float(interval["median"])
            deviations = wrap_pitch_degrees(values - center)
            sigma = float(np.nanstd(deviations, ddof=1)) if values.size > 1 else 0.0

        pitch[index] = center
        columns["pitch_s1down"][index] = float(interval["s1_down"])
        columns["pitch_s1up"][index] = float(interval["s1_up"])
        columns["pitch_s2down"][index] = float(interval["s2_down"])
        columns["pitch_s2up"][index] = float(interval["s2_up"])

        if errorbar_stat == "sem":
            errors[index] = sigma / np.sqrt(values.size) if values.size > 1 else 0.0
        elif errorbar_stat == "scatter":
            errors[index] = sigma
        elif errorbar_stat == "iqr":
            p16, p84 = np.nanpercentile(deviations, [16, 84])
            errors[index] = float(0.5 * (p84 - p16))
        elif errorbar_stat in {"bootstrap", "mean"}:
            errors[index] = float(0.5 * abs(wrap_pitch_degrees(columns["pitch_s1up"][index] - columns["pitch_s1down"][index])))
        elif errorbar_stat == "none":
            errors[index] = np.nan
        else:
            raise ValueError(f"Unknown errorbar_stat={errorbar_stat!r}")

    return pitch, errors, counts, columns


def analyse_dataset(
    dataset: PolarizationDataset,
    mask: np.ndarray,
    radius_kpc: np.ndarray,
    radius_pixel: np.ndarray,
    phi: np.ndarray,
    geom: Geometry,
    bins: np.ndarray,
    radius_unit: str,
    sigma_i: float,
    sigma_q: float,
    sigma_u: float,
    min_snr_i: float,
    min_snr_p: float,
    require_debiased_pol_cut: bool = False,
    errorbar_stat: str = "bootstrap",
    n_bootstrap: int = 1_000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    i_map, _ = load_fits(dataset.i_path)
    q_map, _ = load_fits(dataset.q_path)
    u_map, _ = load_fits(dataset.u_path)
    ensure_same_shape(mask, {f"{dataset.name} I": i_map, f"{dataset.name} Q": q_map, f"{dataset.name} U": u_map})

    pitch = magnetic_pitch_angle(q_map, u_map, phi, geom, dataset.b_rotation_deg)

    p_signal = np.hypot(q_map, u_map)
    valid = np.isfinite(q_map) & np.isfinite(u_map) & (p_signal > 0.0)

    if min_snr_i > 0.0:
        valid &= np.isfinite(i_map) & (i_map > min_snr_i * sigma_i)
    if min_snr_p > 0.0:
        valid &= p_signal > min_snr_p * np.hypot(sigma_q, sigma_u)
    if require_debiased_pol_cut:
        frac_pol = debiased_fractional_polarization(i_map, q_map, u_map, sigma_i, sigma_q, sigma_u)
        valid &= frac_pol > 0.0

    radius_map = radius_pixel if radius_unit == "pixel" else radius_kpc
    rng = np.random.default_rng(seed)
    profiles: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    profile_columns: dict[str, dict[str, np.ndarray]] = {}
    for name, value in REGION_VALUES.items():
        med, err, count, columns = radial_profile(
            pitch,
            radius_map,
            mask == value,
            bins,
            valid,
            errorbar_stat,
            n_bootstrap,
            rng,
        )
        profiles[name] = (med, err, count)
        profile_columns[name] = columns
    return pitch, valid, profiles, profile_columns


def write_profiles_csv(
    output_csv: Path,
    all_profile_columns: dict[str, dict[str, dict[str, np.ndarray]]],
) -> None:
    fieldnames = [
        "dataset",
        "region",
        "index",
        "R",
        "R_min",
        "R_max",
        "R_s1up",
        "R_s1down",
        "R_s2up",
        "R_s2down",
        "pitch",
        "pitch_s1up",
        "pitch_s1down",
        "pitch_s2up",
        "pitch_s2down",
        "npix",
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset_name, profiles in all_profile_columns.items():
            for region_name, columns in profiles.items():
                for index in range(len(columns["R"])):
                    row = {"dataset": dataset_name, "region": region_name, "index": index}
                    row.update({key: columns[key][index] for key in fieldnames if key in columns})
                    writer.writerow(row)


def print_profile_diagnostics(
    mask: np.ndarray,
    radius_map: np.ndarray,
    all_profiles: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    max_radius: float,
    radius_unit: str,
) -> None:
    """Print enough counts to explain empty/NaN radial profiles."""
    import numpy as np

    print(
        "Radius coverage: "
        f"min={np.nanmin(radius_map):.3f} {radius_unit}, "
        f"max={np.nanmax(radius_map):.3f} {radius_unit}, "
        f"pixels within requested radius={int(np.count_nonzero(radius_map < max_radius))}"
    )
    for region_name, region_value in REGION_VALUES.items():
        in_region = (mask == region_value) & (radius_map < max_radius)
        print(f"{region_name}: mask pixels within {max_radius:g} {radius_unit} = {int(np.count_nonzero(in_region))}")

    for dataset_name, profiles in all_profiles.items():
        for region_name, (_, _, counts) in profiles.items():
            total = int(np.sum(counts))
            if total == 0:
                print(
                    f"WARNING: {dataset_name} {region_name} has no valid pixels in the radial bins. "
                    "If this is unexpected, check the region-mask values, the image grid alignment, "
                    "the galaxy centre/PA/inclination, and any S/N cuts."
                )
            else:
                print(f"{dataset_name} {region_name}: valid binned pixels = {total}")


def magnetic_pitch_angle_profile_plot(
    output_png: Path,
    profile_columns: dict[str, np.ndarray],
    title: str = "Pitch angle - Monte Carlo simulations",
    xlabel: str = "R (pixels)",
) -> None:
    """Plot one median profile in the same visual style as the reference code."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 7))
    valid = np.isfinite(profile_columns["R"]) & np.isfinite(profile_columns["pitch"])
    for index in np.where(valid)[0]:
        ax.scatter(profile_columns["R"][index], profile_columns["pitch"][index], color="black", s=8, marker="s")
        ax.plot(
            [profile_columns["R_s1up"][index], profile_columns["R_s1down"][index]],
            [profile_columns["pitch"][index], profile_columns["pitch"][index]],
            color="black",
            linewidth=1.2,
        )
        ax.plot(
            [profile_columns["R"][index], profile_columns["R"][index]],
            [profile_columns["pitch_s1up"][index], profile_columns["pitch_s1down"][index]],
            color="black",
            linewidth=1.2,
        )
        ax.plot(
            [profile_columns["R_s2up"][index], profile_columns["R_s2down"][index]],
            [profile_columns["pitch"][index], profile_columns["pitch"][index]],
            color="black",
            linewidth=0.5,
        )
        ax.plot(
            [profile_columns["R"][index], profile_columns["R"][index]],
            [profile_columns["pitch_s2up"][index], profile_columns["pitch_s2down"][index]],
            color="black",
            linewidth=0.5,
        )

    ax.set_xlabel(xlabel)
    ax.axhline(y=0.0, color="black", linestyle="--", linewidth=3)
    ax.set_ylabel("Pitch angle (degrees)")
    ax.set_title(title)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_profiles(
    output_png: Path,
    bins: np.ndarray,
    datasets: list[PolarizationDataset],
    all_profiles: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    max_radius: float,
    radius_unit: str,
    errorbar_stat: str,
) -> None:
    import matplotlib.pyplot as plt

    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True, constrained_layout=True)

    for ax, region_name in zip(axes, REGION_VALUES):
        for dataset in datasets:
            median, scatter, counts = all_profiles[dataset.name][region_name]
            has_data = counts > 0
            yerr = None if errorbar_stat == "none" else scatter[has_data]
            ax.errorbar(
                centers[has_data],
                median[has_data],
                yerr=yerr,
                marker=dataset.marker,
                linestyle=dataset.linestyle,
                color=dataset.color,
                label=dataset.name,
                capsize=2,
            )
        ax.axhline(0.0, color="0.35", linewidth=0.8, linestyle=":")
        ax.set_title(region_name)
        ax.set_ylabel("Magnetic pitch angle (deg)")
        ax.set_xlim(0.0, max_radius)
        ax.set_ylim(-90.0, 90.0)
        ax.grid(True, alpha=0.3)
        ax.legend(title=f"error: {errorbar_stat}")

    axes[-1].set_xlabel(f"Deprojected galactocentric radius ({radius_unit})")
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def _running_in_ipython_kernel() -> bool:
    """Return True when this file was pasted or executed inside a Jupyter kernel."""
    return "ipykernel" in sys.modules


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None and _running_in_ipython_kernel():
        argv = []

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", type=Path, default=DEFAULT_PATHS["mask"], help="Regional mask FITS file")
    parser.add_argument("--fir-i", type=Path, default=DEFAULT_PATHS["fir_i"])
    parser.add_argument("--fir-q", type=Path, default=DEFAULT_PATHS["fir_q"])
    parser.add_argument("--fir-u", type=Path, default=DEFAULT_PATHS["fir_u"])
    parser.add_argument("--radio-i", type=Path, default=DEFAULT_PATHS["radio_i"])
    parser.add_argument("--radio-q", type=Path, default=DEFAULT_PATHS["radio_q"])
    parser.add_argument("--radio-u", type=Path, default=DEFAULT_PATHS["radio_u"])
    parser.add_argument("--distance-mpc", type=float, default=7.72)
    parser.add_argument("--inclination-deg", type=float, default=38.0)
    parser.add_argument("--position-angle-deg", type=float, default=242.0, help="Major-axis PA, degrees east of north")
    parser.add_argument("--pixel-scale-arcsec", type=float, default=1.0, help="Pixel scale used unless --use-wcs is set")
    parser.add_argument("--use-wcs", action="store_true", help="Use celestial WCS from the mask header for pixel positions")
    parser.add_argument("--center-x", type=float, default=84.0, help="Zero-indexed centre x pixel")
    parser.add_argument("--center-y", type=float, default=71.0, help="Zero-indexed centre y pixel")
    parser.add_argument("--center-ra-deg", type=float, default=None, help="Optional centre RA in degrees")
    parser.add_argument("--center-dec-deg", type=float, default=None, help="Optional centre Dec in degrees")
    parser.add_argument("--max-radius", type=float, default=50.0, help="Maximum radial profile radius in the selected --radius-unit")
    parser.add_argument("--n-bins", type=int, default=11)
    parser.add_argument("--radius-unit", choices=["pixel", "kpc"], default="pixel", help="Use pixel radii for legacy tables or kpc for physical profiles")
    parser.add_argument("--sigma-i", type=float, default=22e-6)
    parser.add_argument("--sigma-q", type=float, default=1.5e-6)
    parser.add_argument("--sigma-u", type=float, default=1.5e-6)
    parser.add_argument("--min-snr-i", type=float, default=0.0, help="Optional Stokes-I S/N cut; 0 disables it")
    parser.add_argument("--min-snr-p", type=float, default=0.0, help="Optional polarized-intensity S/N cut; 0 disables it")
    parser.add_argument(
        "--require-debiased-pol-cut",
        action="store_true",
        help="Require positive debiased fractional polarization",
    )
    parser.add_argument(
        "--errorbar-stat",
        choices=["bootstrap", "sem", "scatter", "iqr", "mean", "none"],
        default="bootstrap",
        help="Error bars to plot; bootstrap writes the legacy median percentile columns",
    )
    parser.add_argument("--n-bootstrap", type=int, default=1_000, help="Bootstrap draws per radial bin")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--fir-b-rotation-deg", type=float, default=90.0, help="Use 0 if FIR Q/U already traces B vectors")
    parser.add_argument("--radio-b-rotation-deg", type=float, default=90.0, help="Use 0 if radio Q/U already traces B vectors")
    parser.add_argument("--output-png", type=Path, default=Path("pitchangle_radial.png"))
    parser.add_argument("--output-csv", type=Path, default=Path("pitchangle_radial_profiles.csv"))
    parser.add_argument(
        "--legacy-profile-png",
        type=Path,
        default=None,
        help="Optional single-profile plot using the legacy black square/error-bar style",
    )
    parser.add_argument("--legacy-dataset", choices=["FIR", "6 cm"], default="6 cm")
    parser.add_argument("--legacy-region", choices=list(REGION_VALUES), default="Arm")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    import numpy as np

    geom = Geometry(
        distance_mpc=args.distance_mpc,
        inclination_deg=args.inclination_deg,
        position_angle_deg=args.position_angle_deg,
        pixel_scale_arcsec=args.pixel_scale_arcsec,
        use_wcs=args.use_wcs,
        center_x=args.center_x,
        center_y=args.center_y,
        center_ra_deg=args.center_ra_deg,
        center_dec_deg=args.center_dec_deg,
    )

    mask, mask_header = load_fits(args.mask)
    east, north = sky_offsets_arcsec(mask_header, mask.shape, geom)
    _, _, radius_kpc, phi = deproject_positions(east, north, geom)
    x0, y0 = center_pixel(mask_header, mask.shape, geom)
    yy, xx = np.indices(mask.shape, dtype=float)
    radius_pixel = np.hypot(xx - x0, yy - y0)
    radius_map = radius_pixel if args.radius_unit == "pixel" else radius_kpc
    bins = np.linspace(0.0, args.max_radius, args.n_bins + 1)

    datasets = [
        PolarizationDataset("FIR", args.fir_i, args.fir_q, args.fir_u, args.fir_b_rotation_deg, "tab:blue", "o", "-"),
        PolarizationDataset("6 cm", args.radio_i, args.radio_q, args.radio_u, args.radio_b_rotation_deg, "tab:red", "s", "--"),
    ]

    all_profiles: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    all_profile_columns: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for index, dataset in enumerate(datasets):
        _, _, profiles, profile_columns = analyse_dataset(
            dataset,
            mask,
            radius_kpc,
            radius_pixel,
            phi,
            geom,
            bins,
            args.radius_unit,
            args.sigma_i,
            args.sigma_q,
            args.sigma_u,
            args.min_snr_i,
            args.min_snr_p,
            args.require_debiased_pol_cut,
            args.errorbar_stat,
            args.n_bootstrap,
            args.random_seed + index,
        )
        all_profiles[dataset.name] = profiles
        all_profile_columns[dataset.name] = profile_columns

    print_profile_diagnostics(mask, radius_map, all_profiles, args.max_radius, args.radius_unit)
    write_profiles_csv(args.output_csv, all_profile_columns)
    plot_profiles(args.output_png, bins, datasets, all_profiles, args.max_radius, args.radius_unit, args.errorbar_stat)
    if args.legacy_profile_png is not None:
        magnetic_pitch_angle_profile_plot(
            args.legacy_profile_png,
            all_profile_columns[args.legacy_dataset][args.legacy_region],
            title=f"{args.legacy_dataset} {args.legacy_region} pitch angle - Monte Carlo simulations",
            xlabel=f"R ({args.radius_unit})",
        )
        print(f"Wrote {args.legacy_profile_png}")
    print(f"Wrote {args.output_png}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
