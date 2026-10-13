import math
import time
from typing import Optional, Tuple

import formation_config as config


class SensorFusion:
    """
    Handles sensor fusion logic, specifically for UWB and GPS data.
    """

    @staticmethod
    def is_uwb_valid(
        uwb_range: Optional[float],
        uwb_timestamp: float,
        last_uwb_range: Optional[float],
        current_time: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Check if UWB measurement is valid for use."""
        if current_time is None:
            raise RuntimeError(
                "is_uwb_valid() requires current_time (sim-time). "
                "Pass self.sim_now_s() from your controller."
            )

        if uwb_range is None:
            return False, "uwb_range is None"

        if config.UWB_NAN_IS_INVALID and math.isnan(uwb_range):
            return False, "uwb_range is NaN"

        if uwb_range < config.UWB_RANGE_MIN_M:
            return False, f"uwb_range {uwb_range:.1f}m < min {config.UWB_RANGE_MIN_M}m"

        if uwb_range > config.UWB_RANGE_MAX_M:
            return False, f"uwb_range {uwb_range:.1f}m > max {config.UWB_RANGE_MAX_M}m"

        age = current_time - uwb_timestamp
        if age > config.UWB_STALE_THRESHOLD_S:
            return False, f"uwb_age {age:.2f}s > stale_threshold {config.UWB_STALE_THRESHOLD_S}s"

        if last_uwb_range is not None:
            jump = abs(uwb_range - last_uwb_range)
            if jump > config.UWB_JUMP_THRESHOLD_M:
                return False, f"uwb_jump {jump:.1f}m > threshold {config.UWB_JUMP_THRESHOLD_M}m"

        return True, "valid"

    @staticmethod
    def compute_uwb_dominant_weight(
        range_m: float, threshold_m: float, very_close_m: float
    ) -> float:
        """Compute UWB weight for uwb_dominant mode."""
        if range_m >= threshold_m:
            return config.UWB_DOMINANT_WEIGHT_FAR

        if range_m <= very_close_m:
            return config.UWB_DOMINANT_WEIGHT_NEAR

        t = (range_m - very_close_m) / (threshold_m - very_close_m)
        return config.UWB_DOMINANT_WEIGHT_NEAR + t * (
            config.UWB_DOMINANT_WEIGHT_FAR - config.UWB_DOMINANT_WEIGHT_NEAR
        )
