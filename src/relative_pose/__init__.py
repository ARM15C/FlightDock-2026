"""
Relative Pose Estimation Module

This package provides sensor fusion for relative positioning between
aircraft during formation flight.

Modules:
    uwb_gps_fusion: Kalman filter fusing GPS + UWB range measurements
"""

from .uwb_gps_fusion import (
    FusedState,
    GPSMeasurement,
    RelativePoseFusion,
    UWBMeasurement,
    create_gps_measurement,
    create_uwb_measurement,
)

__all__ = [
    "RelativePoseFusion",
    "GPSMeasurement",
    "UWBMeasurement",
    "FusedState",
    "create_gps_measurement",
    "create_uwb_measurement",
]
