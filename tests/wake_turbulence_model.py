#!/usr/bin/env python3
"""
Wake Turbulence Two-Vortex-Core Model for DHC-2 Beaver Lead Aircraft
=====================================================================

Models the hazardous wake vortex region behind the DHC-2 Beaver using a
physically accurate two-vortex-core model for safety validation and DRL
training realism.

Physics Model (Two-Vortex-Core):
- Two counter-rotating vortex cores at ±(WINGSPAN * 0.95 / 2) laterally
- Each vortex core has a CORE_RADIUS_M (dangerous) and PERIPHERY_RADIUS_M (warning)
- Vortices descend at 2.0 m/s until stabilization at ~200m below leader
- Dissipation follows exponential decay with time constant DECAY_TAU_S
- Wind drift moves both vortices laterally

Key Improvements Over Single-Zone Model:
1. Two distinct vortex cores (left and right of leader track)
2. in_zone = CORE only (periphery is warning, not "in zone")
3. Tunable radii via constants
4. Distance-based hazard calculation to nearest vortex core
5. Proper clearance metrics for each core

Usage:
    from wake_turbulence_model import WakeTurbulenceZone

    wake_model = WakeTurbulenceZone()

    # Check if follower is in danger zone
    in_core, details = wake_model.is_in_avoid_zone(
        rel_geom=relative_geometry,
        time_since_passage_s=5.0,
        wind_ned_ms=(2.0, 1.0, 0.0) # Optional wind drift
    )

    if in_core:
        print(f"CRITICAL: In vortex core! Intensity: {details['hazard_intensity']:.2f}")
    elif details['zone_type'] == 'periphery':
        print(f"WARNING: Near vortex periphery. Intensity: {details['hazard_intensity']:.2f}")

Date: 2026-01-21
"""

import math
from typing import Any, Dict, Optional, Tuple


class WakeTurbulenceZone:
    """
    Two-vortex-core model for DHC-2 Beaver wake turbulence.

    Models two counter-rotating vortex cores trailing behind the leader aircraft.
    Each core has a dangerous inner radius (CORE) and a warning outer radius (PERIPHERY).
    """

    # =========================================================================
    # DHC-2 Beaver Aircraft Specifications
    # =========================================================================
    WINGSPAN_M = 14.6  # de Havilland DHC-2 Beaver wingspan
    VORTEX_SPACING_M = WINGSPAN_M * 0.95  # ~13.87m between vortex cores

    # =========================================================================
    # Wake Vortex Dynamics (Based on Aviation Research)
    # =========================================================================
    DESCENT_RATE_NOMINAL_MS = 2.0  # 120 m/min (nominal vortex sink rate)
    STABILIZATION_ALT_NOMINAL_M = 200.0  # Depth where vortices stop descending
    DISSIPATION_TIME_NOMINAL_S = 240.0  # 4 minutes (nominal vortex lifespan)

    # =========================================================================
    # Tunable Radii (Conservative defaults to avoid over-triggering)
    # Half-spacing = VORTEX_SPACING_M / 2 ≈ 6.935m
    # PERIPHERY_RADIUS must be < half_spacing to avoid false positives at centerline
    # =========================================================================
    CORE_RADIUS_M = 2.0  # Dangerous zone around each vortex core
    PERIPHERY_RADIUS_M = 5.0  # Warning zone around each vortex core
    DECAY_TAU_S = 120.0  # Intensity decay time constant (seconds)

    def __init__(
        self,
        enable_wind_drift: bool = True,
        descent_rate_ms: float = None,
        stabilization_alt_m: float = None,
        dissipation_time_s: float = None,
        core_radius_m: float = None,
        periphery_radius_m: float = None,
        decay_tau_s: float = None,
    ):
        """
        Initialize two-vortex-core wake turbulence model.

        Args:
            enable_wind_drift: If True, vortices drift laterally with wind
            descent_rate_ms: Vortex descent rate (default 2.0 m/s)
            stabilization_alt_m: Depth where vortices stabilize (default 200m)
            dissipation_time_s: Time until vortices fully dissipate (default 240s)
            core_radius_m: Dangerous core radius (default 2.0m)
            periphery_radius_m: Warning periphery radius (default 5.0m)
            decay_tau_s: Intensity decay time constant (default 120s)
        """
        self.enable_wind_drift = enable_wind_drift
        self.descent_rate_ms = (
            descent_rate_ms if descent_rate_ms is not None else self.DESCENT_RATE_NOMINAL_MS
        )
        self.stabilization_alt_m = (
            stabilization_alt_m
            if stabilization_alt_m is not None
            else self.STABILIZATION_ALT_NOMINAL_M
        )
        self.dissipation_time_s = (
            dissipation_time_s
            if dissipation_time_s is not None
            else self.DISSIPATION_TIME_NOMINAL_S
        )
        self.core_radius_m = core_radius_m if core_radius_m is not None else self.CORE_RADIUS_M
        self.periphery_radius_m = (
            periphery_radius_m if periphery_radius_m is not None else self.PERIPHERY_RADIUS_M
        )
        self.decay_tau_s = decay_tau_s if decay_tau_s is not None else self.DECAY_TAU_S

    def is_in_avoid_zone(
        self,
        rel_geom: Any,
        time_since_passage_s: float,
        wind_ned_ms: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[bool, Dict]:
        """
        Check if follower is in wake turbulence avoid zone (CORE only).

        Uses a two-vortex-core model where:
        - Left vortex core at cross = -VORTEX_SPACING_M/2 + drift
        - Right vortex core at cross = +VORTEX_SPACING_M/2 + drift
        - Both cores descend at descent_rate_ms until stabilization_alt_m

        Args:
            rel_geom: Relative geometry object with behind_m, cross_m, below_m attributes
            time_since_passage_s: Time since leader passed follower's current position
            wind_ned_ms: Wind vector in NED frame (N, E, D) in m/s (optional)

        Returns:
            Tuple of:
                in_zone (bool): True ONLY if follower is in CORE zone (periphery returns False)
                details (dict): Detailed hazard information with keys:
                    - 'in_zone': bool (True only for CORE)
                    - 'hazard_intensity': float (0.0 = safe, 1.0 = extreme danger)
                    - 'vortex_descent_m': float (current vortex depth below leader)
                    - 'vortex_drift_m': float (lateral drift due to wind)
                    - 'lateral_clearance_m': float (dist_to_nearest_core - CORE_RADIUS)
                    - 'vertical_clearance_m': float (vertical component of clearance)
                    - 'time_to_dissipation_s': float (time until vortices dissipate)
                    - 'zone_type': str ('core', 'periphery', 'safe')
                    - 'nearest_core_side': str ('left' or 'right')
                    - 'dist_to_core_m': float (distance to nearest vortex core center)
        """
        # Default safe result
        safe_result = {
            "in_zone": False,
            "hazard_intensity": 0.0,
            "vortex_descent_m": 0.0,
            "vortex_drift_m": 0.0,
            "lateral_clearance_m": float("inf"),
            "vertical_clearance_m": float("inf"),
            "time_to_dissipation_s": 0.0,
            "zone_type": "safe",
            "nearest_core_side": "none",
            "dist_to_core_m": float("inf"),
        }

        # Follower ahead of leader = safe (no wake exists ahead)
        if rel_geom.behind_m <= 0:
            return False, safe_result

        # Vortices dissipated = safe
        if time_since_passage_s <= 0 or time_since_passage_s >= self.dissipation_time_s:
            safe_result["time_to_dissipation_s"] = max(
                0, self.dissipation_time_s - time_since_passage_s
            )
            return False, safe_result

        # Compute vortex positions
        vortex_descent = self._compute_vortex_descent(time_since_passage_s)
        vortex_drift = self._compute_wind_drift(time_since_passage_s, wind_ned_ms)

        # Vortex core positions (lateral, vertical relative to leader track)
        # Left vortex: cross = -SPACING/2 + drift
        # Right vortex: cross = +SPACING/2 + drift
        half_spacing = self.VORTEX_SPACING_M / 2
        left_core_cross = -half_spacing + vortex_drift
        right_core_cross = half_spacing + vortex_drift
        vortex_below = vortex_descent  # Vortex altitude below leader

        # Follower position in vortex frame
        follower_cross = rel_geom.cross_m
        follower_below = rel_geom.below_m  # positive = below leader

        # Distance to each vortex core (2D: cross + vertical)
        # Vortex core is at (core_cross, vortex_below) in (cross, below) space
        # Follower is at (follower_cross, follower_below)
        dist_to_left = math.sqrt(
            (follower_cross - left_core_cross) ** 2 + (follower_below - vortex_below) ** 2
        )
        dist_to_right = math.sqrt(
            (follower_cross - right_core_cross) ** 2 + (follower_below - vortex_below) ** 2
        )

        # Use nearest core for zone classification
        if dist_to_left <= dist_to_right:
            dist_to_nearest = dist_to_left
            nearest_side = "left"
            nearest_core_cross = left_core_cross
        else:
            dist_to_nearest = dist_to_right
            nearest_side = "right"
            nearest_core_cross = right_core_cross

        # Zone classification based on distance to nearest core
        if dist_to_nearest <= self.core_radius_m:
            zone_type = "core"
        elif dist_to_nearest <= self.periphery_radius_m:
            zone_type = "periphery"
        else:
            zone_type = "safe"

        # CRITICAL: in_zone = CORE only (periphery is warning, not "in zone")
        in_zone = zone_type == "core"

        # Compute hazard intensity
        hazard_intensity = self._compute_hazard_intensity(dist_to_nearest, time_since_passage_s)

        # Compute clearances
        lateral_clearance = dist_to_nearest - self.core_radius_m  # Negative if inside core
        vertical_clearance = follower_below - vortex_below  # Negative if above vortex

        # Time to dissipation
        time_to_dissipation = max(0, self.dissipation_time_s - time_since_passage_s)

        details = {
            "in_zone": in_zone,
            "hazard_intensity": hazard_intensity,
            "vortex_descent_m": vortex_descent,
            "vortex_drift_m": vortex_drift,
            "lateral_clearance_m": lateral_clearance,
            "vertical_clearance_m": vertical_clearance,
            "time_to_dissipation_s": time_to_dissipation,
            "zone_type": zone_type,
            "nearest_core_side": nearest_side,
            "dist_to_core_m": dist_to_nearest,
        }

        return in_zone, details

    def _compute_vortex_descent(self, time_since_passage_s: float) -> float:
        """
        Compute vortex descent depth below leader altitude.

        Vortices descend at a constant rate until reaching stabilization altitude.

        Args:
            time_since_passage_s: Time since vortex generation

        Returns:
            Descent depth in meters (positive = below leader)
        """
        if time_since_passage_s <= 0:
            return 0.0

        # Linear descent until stabilization
        descent = self.descent_rate_ms * time_since_passage_s

        # Cap at stabilization altitude
        return min(descent, self.stabilization_alt_m)

    def _compute_wind_drift(
        self, time_since_passage_s: float, wind_ned_ms: Optional[Tuple[float, float, float]]
    ) -> float:
        """
        Compute lateral drift of vortices due to wind.

        Vortices move with the prevailing wind. This computes the cross-track
        drift component (assuming leader's heading defines the along-track axis).

        Args:
            time_since_passage_s: Time since vortex generation
            wind_ned_ms: Wind vector in NED frame (N, E, D) in m/s

        Returns:
            Lateral drift in meters (positive = right of leader's track)
        """
        if not self.enable_wind_drift or wind_ned_ms is None:
            return 0.0

        if time_since_passage_s <= 0:
            return 0.0

        # Use wind_east component as lateral drift
        # Note: cross-track wind projection not in scope
        wind_east = wind_ned_ms[1]

        # Vortices drift with wind
        drift = wind_east * time_since_passage_s

        return drift

    def _compute_hazard_intensity(self, dist_m: float, time_since_passage_s: float) -> float:
        """
        Compute hazard intensity from 0.0 (safe) to 1.0 (extreme danger).

        Combines spatial proximity to vortex core with temporal decay.

        Args:
            dist_m: Distance to nearest vortex core center
            time_since_passage_s: Age of vortex

        Returns:
            Hazard intensity (0.0-1.0)
        """
        # Beyond periphery = safe
        if dist_m >= self.periphery_radius_m:
            return 0.0

        # Spatial factor: linear decay from core (1.0) to periphery edge (0.0)
        spatial = max(0.0, 1.0 - dist_m / self.periphery_radius_m)

        # Temporal factor: exponential decay with time constant
        temporal = math.exp(-time_since_passage_s / self.decay_tau_s)

        # Combined intensity
        intensity = spatial * temporal

        return min(1.0, max(0.0, intensity))

    def get_vortex_positions(
        self, time_since_passage_s: float, wind_ned_ms: Optional[Tuple[float, float, float]] = None
    ) -> Dict:
        """
        Get current vortex core positions for visualization/debugging.

        Args:
            time_since_passage_s: Time since vortex generation
            wind_ned_ms: Wind vector in NED frame (optional)

        Returns:
            Dictionary with vortex positions:
                - 'left_cross_m': Lateral position of left vortex
                - 'right_cross_m': Lateral position of right vortex
                - 'below_m': Vertical position (both vortices descend equally)
                - 'intensity': Current hazard intensity at core center
        """
        vortex_descent = self._compute_vortex_descent(time_since_passage_s)
        vortex_drift = self._compute_wind_drift(time_since_passage_s, wind_ned_ms)

        half_spacing = self.VORTEX_SPACING_M / 2
        intensity = self._compute_hazard_intensity(0.0, time_since_passage_s)

        return {
            "left_cross_m": -half_spacing + vortex_drift,
            "right_cross_m": half_spacing + vortex_drift,
            "below_m": vortex_descent,
            "intensity": intensity,
        }


# =============================================================================
# UNIT TESTS
# =============================================================================


class MockGeometry:
    """Mock relative geometry for unit tests."""

    def __init__(self, behind_m: float, cross_m: float, below_m: float):
        self.behind_m = behind_m
        self.cross_m = cross_m
        self.below_m = below_m


def test_behind_negative_returns_safe():
    """Follower ahead of leader should always be safe."""
    model = WakeTurbulenceZone()
    geom = MockGeometry(behind_m=-10, cross_m=0, below_m=5)
    in_zone, details = model.is_in_avoid_zone(geom, 1.0, (0, 0, 0))
    assert not in_zone, "Should not be in zone when ahead of leader"
    assert details["zone_type"] == "safe", f"Expected 'safe', got '{details['zone_type']}'"
    print("PASS: test_behind_negative_returns_safe")


def test_at_vortex_core_is_core():
    """Directly at a vortex core should be zone_type='core'."""
    model = WakeTurbulenceZone()
    # Right core is at cross = +VORTEX_SPACING_M/2 ≈ +6.935m
    # After 2s, vortex_below = 2.0 * 2 = 4m
    half_spacing = model.VORTEX_SPACING_M / 2  # ~6.935m
    vortex_below = 2.0 * 2.0  # 4m descent after 2s
    geom = MockGeometry(behind_m=50, cross_m=half_spacing, below_m=vortex_below)
    in_zone, details = model.is_in_avoid_zone(geom, 2.0, (0, 0, 0))
    assert in_zone, "Should be in zone when at vortex core"
    assert details["zone_type"] == "core", f"Expected 'core', got '{details['zone_type']}'"
    assert (
        details["nearest_core_side"] == "right"
    ), f"Expected 'right', got '{details['nearest_core_side']}'"
    assert (
        details["dist_to_core_m"] < 0.1
    ), f"Expected ~0m distance, got {details['dist_to_core_m']:.2f}m"
    print("PASS: test_at_vortex_core_is_core")


def test_periphery_is_not_in_zone():
    """Periphery zone should NOT be counted as 'in zone'."""
    model = WakeTurbulenceZone()
    # Position 3.5m from right core center (inside periphery, outside core)
    # With CORE_RADIUS_M=2.0 and PERIPHERY_RADIUS_M=5.0, 3.5m is clearly in periphery
    half_spacing = model.VORTEX_SPACING_M / 2
    vortex_below = 2.0 * 2.0  # 4m descent after 2s
    geom = MockGeometry(behind_m=50, cross_m=half_spacing + 3.5, below_m=vortex_below)
    in_zone, details = model.is_in_avoid_zone(geom, 2.0, (0, 0, 0))
    assert not in_zone, "Periphery should NOT be in_zone"
    assert (
        details["zone_type"] == "periphery"
    ), f"Expected 'periphery', got '{details['zone_type']}'"
    assert details["hazard_intensity"] > 0.0, "Periphery should have some hazard intensity"
    print("PASS: test_periphery_is_not_in_zone")


def test_far_from_cores_is_safe():
    """Far from vortex cores should be safe."""
    model = WakeTurbulenceZone()
    # 10m above leader (below_m = -10)
    geom = MockGeometry(behind_m=50, cross_m=0, below_m=-10)
    in_zone, details = model.is_in_avoid_zone(geom, 2.0, (0, 0, 0))
    assert not in_zone, "Should not be in zone when far from cores"
    assert details["zone_type"] == "safe", f"Expected 'safe', got '{details['zone_type']}'"
    print("PASS: test_far_from_cores_is_safe")


def test_left_core_detection():
    """Left vortex core should be detected correctly."""
    model = WakeTurbulenceZone()
    # Left core is at cross = -VORTEX_SPACING_M/2 ≈ -6.935m
    half_spacing = model.VORTEX_SPACING_M / 2
    vortex_below = 2.0 * 2.0  # 4m descent after 2s
    geom = MockGeometry(behind_m=50, cross_m=-half_spacing, below_m=vortex_below)
    in_zone, details = model.is_in_avoid_zone(geom, 2.0, (0, 0, 0))
    assert in_zone, "Should be in zone at left core"
    assert details["zone_type"] == "core"
    assert (
        details["nearest_core_side"] == "left"
    ), f"Expected 'left', got '{details['nearest_core_side']}'"
    print("PASS: test_left_core_detection")


def test_vortex_descent():
    """Vortex should descend over time and cap at stabilization altitude."""
    model = WakeTurbulenceZone()

    # At t=0, no descent
    assert model._compute_vortex_descent(0.0) == 0.0

    # At t=10s, descent = 20m
    assert abs(model._compute_vortex_descent(10.0) - 20.0) < 0.01

    # At t=200s, descent capped at 200m (stabilization)
    assert abs(model._compute_vortex_descent(200.0) - 200.0) < 0.01

    print("PASS: test_vortex_descent")


def test_hazard_intensity_decay():
    """Hazard intensity should decay with distance and time."""
    model = WakeTurbulenceZone()

    # At core center, t=0: high intensity
    intensity_0 = model._compute_hazard_intensity(0.0, 0.0)
    assert intensity_0 > 0.9, f"Expected high intensity at core, got {intensity_0}"

    # Same position, t=120s (one decay constant): intensity reduced
    intensity_120 = model._compute_hazard_intensity(0.0, 120.0)
    assert intensity_120 < intensity_0, "Intensity should decay with time"
    assert abs(intensity_120 - intensity_0 / math.e) < 0.1, "Should follow exp decay"

    # At periphery edge, any time: zero intensity
    intensity_edge = model._compute_hazard_intensity(model.periphery_radius_m, 0.0)
    assert intensity_edge < 0.01, f"Expected ~0 intensity at periphery edge, got {intensity_edge}"

    print("PASS: test_hazard_intensity_decay")


def test_wind_drift():
    """Vortices should drift with wind."""
    model = WakeTurbulenceZone()

    # No wind
    drift_no_wind = model._compute_wind_drift(10.0, (0, 0, 0))
    assert drift_no_wind == 0.0

    # East wind (2 m/s) for 10s
    drift_east = model._compute_wind_drift(10.0, (0, 2.0, 0))
    assert abs(drift_east - 20.0) < 0.01, f"Expected 20m drift, got {drift_east}"

    # West wind (-2 m/s) for 10s
    drift_west = model._compute_wind_drift(10.0, (0, -2.0, 0))
    assert abs(drift_west - (-20.0)) < 0.01, f"Expected -20m drift, got {drift_west}"

    print("PASS: test_wind_drift")


def test_dissipated_vortex_is_safe():
    """Vortex should be safe after dissipation time."""
    model = WakeTurbulenceZone()
    # Position right at what would be vortex core, but after dissipation
    half_spacing = model.VORTEX_SPACING_M / 2
    geom = MockGeometry(behind_m=50, cross_m=half_spacing, below_m=200)
    in_zone, details = model.is_in_avoid_zone(geom, 300.0, (0, 0, 0))  # 300s > 240s dissipation
    assert not in_zone, "Should not be in zone after dissipation"
    assert details["zone_type"] == "safe"
    print("PASS: test_dissipated_vortex_is_safe")


def test_corridor_precontact_is_safe():
    """Precontact position (10m behind, centerline, 10m above vortex) should be SAFE."""
    model = WakeTurbulenceZone()
    # Precontact: behind=10m, cross=0 (centerline), below=-10m (10m above leader)
    # At behind=10m, time_since_passage = 10/24.7 ≈ 0.405s, vortex_below ≈ 0.81m
    # Follower at below_m=-10 (10m above leader) => vertical distance to vortex = 10 + 0.81 ≈ 10.8m
    # Distance to nearest core = sqrt(6.935² + 10.8²) ≈ 12.8m > 5.0m (periphery)
    leader_speed = 24.7
    behind_m = 10.0
    time_since_passage = behind_m / leader_speed

    geom = MockGeometry(behind_m=behind_m, cross_m=0, below_m=-10)
    in_zone, details = model.is_in_avoid_zone(geom, time_since_passage, (0, 0, 0))

    assert not in_zone, "Precontact should NOT be in_zone"
    assert (
        details["zone_type"] == "safe"
    ), f"Precontact expected 'safe', got '{details['zone_type']}'"
    print("PASS: test_corridor_precontact_is_safe")


def test_corridor_contact_centerline_not_core():
    """Contact position (5m behind, centerline, ~1.5m below) should NOT be CORE."""
    model = WakeTurbulenceZone()
    # Contact/boom offset: behind=5m, cross=0 (centerline), below=1.5m
    # At behind=5m, time_since_passage = 5/24.7 ≈ 0.202s, vortex_below ≈ 0.40m
    # Follower at below_m=1.5 => vertical distance to vortex = 1.5 - 0.40 = 1.1m
    # Distance to nearest core = sqrt(6.935² + 1.1²) ≈ 7.0m
    # 7.0m > 5.0m (periphery) => SAFE
    # 7.0m > 2.0m (core) => NOT CORE
    leader_speed = 24.7
    behind_m = 5.0
    time_since_passage = behind_m / leader_speed

    geom = MockGeometry(behind_m=behind_m, cross_m=0, below_m=1.5)
    in_zone, details = model.is_in_avoid_zone(geom, time_since_passage, (0, 0, 0))

    assert not in_zone, "Contact centerline should NOT be in CORE zone"
    assert (
        details["zone_type"] != "core"
    ), f"Contact centerline should NOT be 'core', got '{details['zone_type']}'"
    print("PASS: test_corridor_contact_centerline_not_core")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Wake Turbulence Model Unit Tests")
    print("=" * 60)

    test_behind_negative_returns_safe()
    test_at_vortex_core_is_core()
    test_periphery_is_not_in_zone()
    test_far_from_cores_is_safe()
    test_left_core_detection()
    test_vortex_descent()
    test_hazard_intensity_decay()
    test_wind_drift()
    test_dissipated_vortex_is_safe()
    test_corridor_precontact_is_safe()
    test_corridor_contact_centerline_not_core()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
