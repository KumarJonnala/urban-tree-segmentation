from __future__ import annotations
"""Sun position from geographic coordinates and UTC datetime."""

import datetime as dt


def _tile_center(bbox: dict) -> tuple[float, float]:
    """Return (lat, lon) of the tile centre from a WGS84 bbox dict."""
    lat = (bbox["south"] + bbox["north"]) / 2.0
    lon = (bbox["west"] + bbox["east"]) / 2.0
    return lat, lon


def sun_position(lat: float, lon: float, dt_utc: dt.datetime) -> tuple[float, float]:
    """Return (azimuth_deg, elevation_deg) for a given location and UTC time.

    Parameters
    ----------
    lat, lon : float
        WGS84 decimal degrees.
    dt_utc : datetime.datetime
        Timezone-aware UTC datetime.

    Returns
    -------
    azimuth_deg : float
        Compass bearing of the sun (0° = North, clockwise).
    elevation_deg : float
        Altitude above horizon in degrees. Negative when sun is below horizon.
    """
    if dt_utc.tzinfo is None:
        raise ValueError("dt_utc must be timezone-aware (use datetime.timezone.utc)")

    from pysolar import solar as _solar

    elevation_deg = _solar.get_altitude(lat, lon, dt_utc)

    # pysolar.get_azimuth returns compass bearing (0°=North, clockwise) directly
    azimuth_deg = _solar.get_azimuth(lat, lon, dt_utc) % 360.0

    return azimuth_deg, elevation_deg
