"""Source packing and model-coordinate conventions; no inferred unit changes."""
from __future__ import annotations
import numpy as np

AORC_CHANNELS = (
    "APCP_surface", "DLWRF_surface", "DSWRF_surface", "PRES_surface",
    "SPFH_2maboveground", "TMP_2maboveground", "UGRD_10maboveground", "VGRD_10maboveground",
)
# Pressure is reported in hPa; NOAA stores 10 Pa per integer code.
AORC_TO_PHYSICAL = dict(zip(AORC_CHANNELS, (.1, .1, .1, .1, .0001, .1, .1, .1)))
AORC_UNITS = dict(zip(AORC_CHANNELS, ("mm", "W m-2", "W m-2", "hPa", "kg kg-1", "K", "m s-1", "m s-1")))
NUMERIC_PROFILES = ("physical", "legacy_model", "legacy_packed")


def decode_aorc(raw, variable: str, *, profile: str = "physical", fill_value=-32767):
    """Mask integer missing codes BEFORE scaling or any spatial operation.

    legacy_model retains the historical 10*q humidity coordinate. legacy_packed
    retains source stored numbers for explicitly historical loader replay; the
    historical loader turns those numbers into 10*q, not physical specific q.
    Neither compatibility profile reproduces historical missing-value defects.
    """
    if variable not in AORC_CHANNELS or profile not in NUMERIC_PROFILES:
        raise ValueError((variable, profile))
    if hasattr(raw, "dims"):
        valid = np.isfinite(raw) & (raw != fill_value)
        values = raw.astype("float32").where(valid)
    else:
        a = np.asarray(raw)
        values = np.where(np.isfinite(a) & (a != fill_value), a, np.nan).astype("float32")
    factor = 1.0 if profile == "legacy_packed" else AORC_TO_PHYSICAL[variable]
    if profile == "legacy_model" and variable == "SPFH_2maboveground":
        factor *= 10
    return values * np.float32(factor)


def mixing_ratio_to_specific_humidity(r):
    """r=m_v/m_d -> q=m_v/(m_v+m_d); preserve missing data, reject negatives."""
    negative=(r<0).any()
    if hasattr(negative,'compute'):negative=negative.compute().item()
    if bool(negative):
        raise ValueError("Mixing ratio cannot be negative")
    return r / (1 + r)


def rotate_grid_wind(u, v, cosalpha, sinalpha):
    """WRF grid-relative -> east/north, requiring supplied rotation angles."""
    norm = np.asarray(cosalpha) ** 2 + np.asarray(sinalpha) ** 2
    if not np.allclose(norm[np.isfinite(norm)], 1.0, atol=1e-5):
        raise ValueError("Wind rotation coefficients are not unit length")
    return u * cosalpha - v * sinalpha, v * cosalpha + u * sinalpha


ERA5_SURFACE = {
    "10m_u_component_of_wind": ("u10", "U10", 1.0),
    "10m_v_component_of_wind": ("v10", "V10", 1.0),
    "2m_temperature": ("t2m", "T2", 1.0),
    "surface_pressure": ("sp", "PSFC", .01),
    "total_precipitation": ("tp", "PREC_ACC_NC", 1000.0),
    "surface_solar_radiation_downwards": ("ssrd", "SWDOWN", 1 / 3600),
    "surface_thermal_radiation_downwards": ("strd", "GLW", 1 / 3600),
}
ERA5_PRESSURE = {
    "temperature": "t", "u_component_of_wind": "u", "v_component_of_wind": "v",
    "specific_humidity": "q", "geopotential": "z", "fraction_of_cloud_cover": "cc", "relative_humidity": "r",
}
PRESSURE_LEVELS = (500, 700, 850, 925)
