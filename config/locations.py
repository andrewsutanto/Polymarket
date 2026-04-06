"""City configurations for weather market tracking.

Each city specifies coordinates for NWS API lookups, the settlement
weather station, and timezone for time-of-day calculations.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CityConfig:
    name: str
    short: str
    lat: float
    lon: float
    station_id: str
    timezone: str


LOCATIONS: dict[str, CityConfig] = {
    "NYC": CityConfig(
        name="New York City",
        short="NYC",
        lat=40.7128,
        lon=-73.9352,
        station_id="KNYC",
        timezone="America/New_York",
    ),
    "Chicago": CityConfig(
        name="Chicago",
        short="CHI",
        lat=41.8781,
        lon=-87.6298,
        station_id="KORD",
        timezone="America/Chicago",
    ),
    "Seattle": CityConfig(
        name="Seattle",
        short="SEA",
        lat=47.6062,
        lon=-122.3321,
        station_id="KSEA",
        timezone="America/Los_Angeles",
    ),
    "Atlanta": CityConfig(
        name="Atlanta",
        short="ATL",
        lat=33.7490,
        lon=-84.3880,
        station_id="KATL",
        timezone="America/New_York",
    ),
    "Dallas": CityConfig(
        name="Dallas",
        short="DAL",
        lat=32.7767,
        lon=-96.7970,
        station_id="KDFW",
        timezone="America/Chicago",
    ),
}

SETTLEMENT_STATIONS: dict[str, str] = {
    cfg.name: cfg.station_id for cfg in LOCATIONS.values()
}
