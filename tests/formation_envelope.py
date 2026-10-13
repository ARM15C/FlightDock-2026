from typing import Tuple

import formation_config as config
from geometry import RelativeGeometry


class EnvelopeChecker:
    """
    Handles logic for checking if the aircraft is within various
    pre-defined geometric envelopes.

    E4 FIX: All envelope checks now use geom.behind_m, geom.cross_m, geom.below_m
    consistently. Previously, below_error was computed from MSL altitudes which
    caused a mismatch (behind_err could be ~0 while range_tip stayed ~15m because
    below used a different coordinate source than behind/cross).
    """

    @staticmethod
    def get_envelope_errors(
        geom: RelativeGeometry,
        lead_alt_msl: float,
        follower_alt_msl: float,
        target_behind: float,
        target_cross: float,
        target_below: float,
    ) -> Tuple[float, float, float]:
        """Get errors relative to a target body-frame position.

        E4: All errors now use geometry values (boom-tip-referenced) for consistency.
        The lead_alt_msl and follower_alt_msl parameters are retained for API
        compatibility but are no longer used for below_error computation.
        """
        behind_error = geom.behind_m - target_behind
        cross_error = geom.cross_m - target_cross
        # E4 FIX: Use geom.below_m (boom-tip-referenced, consistent with behind/cross)
        # instead of altitude-based computation which used different coordinate source
        below_error = geom.below_m - target_below
        return (behind_error, cross_error, below_error)

    @staticmethod
    def is_in_body_envelope(
        geom: RelativeGeometry,
        lead_alt_msl: float,
        follower_alt_msl: float,
        target_behind: float,
        target_cross: float,
        target_below: float,
        tol_behind: float,
        tol_cross: float,
        tol_below: float,
    ) -> bool:
        """Generic body-frame envelope check.

        E4: All errors now use geometry values (boom-tip-referenced) for consistency.
        """
        behind_error = abs(geom.behind_m - target_behind)
        cross_error = abs(geom.cross_m - target_cross)
        # E4 FIX: Use geom.below_m for consistency with behind/cross
        below_error = abs(geom.below_m - target_below)

        return behind_error <= tol_behind and cross_error <= tol_cross and below_error <= tol_below

    def is_in_precontact_outer_envelope(self, geom, lead_alt, fol_alt) -> bool:
        return self.is_in_body_envelope(
            geom,
            lead_alt,
            fol_alt,
            config.PRECONTACT_BEHIND_M,
            config.PRECONTACT_CROSS_M,
            config.PRECONTACT_BELOW_M,
            config.PRECONTACT_BEHIND_OUTER_TOL_M,
            config.PRECONTACT_CROSS_OUTER_TOL_M,
            config.PRECONTACT_BELOW_OUTER_TOL_M,
        )

    def is_in_precontact_inner_envelope(self, geom, lead_alt, fol_alt) -> bool:
        return self.is_in_body_envelope(
            geom,
            lead_alt,
            fol_alt,
            config.PRECONTACT_BEHIND_M,
            config.PRECONTACT_CROSS_M,
            config.PRECONTACT_BELOW_M,
            config.PRECONTACT_BEHIND_INNER_TOL_M,
            config.PRECONTACT_CROSS_INNER_TOL_M,
            config.PRECONTACT_BELOW_INNER_TOL_M,
        )

    def is_in_precontact_entry_envelope(self, geom, lead_alt, fol_alt) -> bool:
        """Wider envelope for entering HOLD_PRECONTACT (capture-only, P1 fix)."""
        return self.is_in_body_envelope(
            geom,
            lead_alt,
            fol_alt,
            config.PRECONTACT_BEHIND_M,
            config.PRECONTACT_CROSS_M,
            config.PRECONTACT_BELOW_M,
            config.PRECONTACT_BEHIND_ENTRY_TOL_M,
            config.PRECONTACT_CROSS_ENTRY_TOL_M,
            config.PRECONTACT_BELOW_ENTRY_TOL_M,
        )

    def is_in_contact_outer_envelope(self, geom, lead_alt, fol_alt) -> bool:
        return self.is_in_body_envelope(
            geom,
            lead_alt,
            fol_alt,
            config.CONTACT_BEHIND_M,
            config.CONTACT_CROSS_M,
            config.CONTACT_BELOW_M,
            config.CONTACT_BEHIND_OUTER_TOL_M,
            config.CONTACT_CROSS_OUTER_TOL_M,
            config.CONTACT_BELOW_OUTER_TOL_M,
        )

    def is_in_contact_inner_envelope(self, geom, lead_alt, fol_alt) -> bool:
        return self.is_in_body_envelope(
            geom,
            lead_alt,
            fol_alt,
            config.CONTACT_BEHIND_M,
            config.CONTACT_CROSS_M,
            config.CONTACT_BELOW_M,
            config.CONTACT_BEHIND_INNER_TOL_M,
            config.CONTACT_CROSS_INNER_TOL_M,
            config.CONTACT_BELOW_INNER_TOL_M,
        )
