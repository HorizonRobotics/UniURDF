# Project uniurdf
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Tests for the axis-to-axis rotation primitive.

``rotation_from_ez_to`` is what makes axis normalization safe: the whole
compensation scheme assumes ``S @ ez == axis`` *exactly*, because ``S`` is
applied forward to the joint origin and as ``S.T`` to everything below it. If
that identity fails, the two no longer cancel and the joint rotates about the
wrong line -- for every angle except zero.

The condition is easy to break near the poles. ``ez . target`` is quadratic in
the angle there, so a ``1e-9`` threshold on the dot product really admits
``sqrt(2e-9) ~ 4.5e-5`` rad. Inside that band the degenerate branch fires on an
axis that is *not* actually degenerate and returns an ``S`` that misses by up
to 0.005 degrees. Testing on ``sin`` -- i.e. ``||ez x target||`` -- is what
makes the threshold mean what it reads as.

No shipped URDF sits in that band (every actuated axis is either exactly on a
pole or far from one), so these tests guard a latent trap rather than a live
bug: the first hand-edited or CAD-exported axis lands there without warning.
"""

import numpy as np
import pytest

from uniurdf.transform._linalg import (
    _EPS,
    rotation_from_axis_angle,
    rotation_from_ez_to,
)

EZ = np.array([0.0, 0.0, 1.0])

#: Machine-precision ceiling for ``||S @ ez - target||``. The primitive is a
#: handful of flops on unit inputs, so anything above this is a branch bug and
#: not accumulated round-off.
ROUNDTRIP_ATOL = 1e-14


def _roundtrip_tolerance(alpha: float) -> float:
    """Allowed ``||S @ ez - target||`` for an axis ``alpha`` rad from a pole.

    Inside the deadzone the primitive deliberately snaps to the pole, so the
    residual is ``sin(alpha) <= _EPS`` -- that is the contract, not a defect.
    Outside it, only floating-point noise is acceptable.
    """
    return max(ROUNDTRIP_ATOL, alpha) if alpha < _EPS else ROUNDTRIP_ATOL


def _rotation_gap(first: np.ndarray, second: np.ndarray) -> float:
    """Frobenius distance between two rotations.

    Deliberately *not* ``arccos((trace - 1) / 2)``: that geodesic angle is
    ill-conditioned exactly where these tests look, since ``arccos(1 - x/2)``
    grows like ``sqrt(x)``, turning a 1e-16 matrix difference into an apparent
    1e-6 degree error. Measuring the defect with a metric that has the same
    defect hides it. Frobenius is smooth at identity.
    """
    return float(np.linalg.norm(first - second))


#: Angles away from a pole, in radians. Straddles the old dot-product band
#: (``4.47e-5``) on both sides so a regression to ``arccos``/dot thresholds
#: fails here rather than in the field.
POLE_OFFSETS = [
    0.0,
    1e-12,
    1e-9,
    1e-7,
    1e-6,
    1e-5,
    4.4e-5,  # old code: degenerate branch, residual 4.4e-5
    4.5e-5,  # old code: Rodrigues branch, residual 6e-13
    1e-4,
    1e-3,
    1e-1,
]


def _axis_near_pole(alpha: float, sign: float, phi: float = 0.0) -> np.ndarray:
    """A unit vector ``alpha`` radians from ``sign * ez``, azimuth ``phi``."""
    return np.array(
        [
            np.sin(alpha) * np.cos(phi),
            np.sin(alpha) * np.sin(phi),
            sign * np.cos(alpha),
        ]
    )


class TestRoundTrip:
    """``S @ ez == target`` -- the identity the compensation relies on."""

    @pytest.mark.parametrize("alpha", POLE_OFFSETS)
    @pytest.mark.parametrize("sign", [-1.0, 1.0], ids=["minus_z", "plus_z"])
    @pytest.mark.parametrize(
        "phi", [0.0, np.pi / 3, np.pi], ids=lambda p: f"phi{p:.2f}"
    )
    def test_maps_ez_onto_target_near_poles(
        self, alpha: float, sign: float, phi: float
    ) -> None:
        target = _axis_near_pole(alpha, sign, phi)
        result = rotation_from_ez_to(target) @ EZ
        # At exactly alpha=0 the target *is* a pole and the degenerate branch
        # is correct by construction; everywhere else this is the real gate.
        np.testing.assert_allclose(
            result,
            target,
            atol=_roundtrip_tolerance(alpha),
            err_msg=(
                f"S @ ez missed an axis {alpha:.1e} rad from "
                f"{'+' if sign > 0 else '-'}Z: the joint would rotate about "
                f"the wrong line"
            ),
        )

    def test_maps_ez_onto_random_directions(self) -> None:
        """Full-sphere sweep, so the poles are not the only coverage."""
        rng = np.random.default_rng(20260811)
        directions = rng.normal(size=(4096, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        worst = max(
            float(np.linalg.norm(rotation_from_ez_to(t) @ EZ - t)) for t in directions
        )
        assert worst < ROUNDTRIP_ATOL, f"worst round-trip residual {worst:.3e}"

    def test_accepts_unnormalized_input(self) -> None:
        """Callers pass raw ``axis xyz`` strings; length must not matter."""
        target = np.array([0.0, 3.0, -4.0])
        result = rotation_from_ez_to(target) @ EZ
        np.testing.assert_allclose(
            result, target / np.linalg.norm(target), atol=ROUNDTRIP_ATOL
        )


class TestPoleBandIsClosed:
    """The specific regression: a threshold on cosine instead of sine.

    Pinned as its own test because the parametrized round-trip above would
    also fail, but would not say *why*. This one names the mechanism.
    """

    @pytest.mark.parametrize("sign", [-1.0, 1.0], ids=["minus_z", "plus_z"])
    def test_no_residual_cliff_across_the_old_threshold(self, sign: float) -> None:
        # sqrt(2 * 1e-9) is where a dot-product threshold of 1e-9 switches
        # branches. Residual must not jump there.
        edge = float(np.sqrt(2e-9))
        below = _axis_near_pole(edge * 0.98, sign)
        above = _axis_near_pole(edge * 1.02, sign)
        residual_below = float(np.linalg.norm(rotation_from_ez_to(below) @ EZ - below))
        residual_above = float(np.linalg.norm(rotation_from_ez_to(above) @ EZ - above))
        assert residual_below < ROUNDTRIP_ATOL, (
            f"axis {edge * 0.98:.2e} rad off the pole took a degenerate "
            f"branch it should not have: residual {residual_below:.3e}"
        )
        assert residual_above < ROUNDTRIP_ATOL


class TestExactPoles:
    """Both poles are handled by closed forms, and both must be exact."""

    def test_positive_z_is_identity(self) -> None:
        result = rotation_from_ez_to([0, 0, 1])
        np.testing.assert_array_equal(result, np.eye(3))

    def test_negative_z_is_an_exact_half_turn(self) -> None:
        result = rotation_from_ez_to([0, 0, -1])
        np.testing.assert_array_equal(result, np.diag([1.0, -1.0, -1.0]))
        # Exactly antipodal is *not* ill-conditioned: the arc is degenerate
        # but the answer is not. 75 shipped axes are written `0 0 -1`.
        np.testing.assert_array_equal(result @ EZ, np.array([0.0, 0.0, -1.0]))

    @pytest.mark.parametrize("sign", [-1.0, 1.0], ids=["minus_z", "plus_z"])
    def test_is_a_proper_rotation_at_the_poles(self, sign: float) -> None:
        result = rotation_from_ez_to([0.0, 0.0, sign])
        np.testing.assert_allclose(result.T @ result, np.eye(3), atol=1e-15)
        assert float(np.linalg.det(result)) == pytest.approx(1.0)


class TestIsAProperRotation:
    """Orthogonality and ``det == +1`` over the whole sphere.

    A reflection would also map ``ez`` onto the target while silently mirroring
    every mesh hanging below the joint.
    """

    def test_orthonormal_with_unit_determinant(self) -> None:
        rng = np.random.default_rng(11062026)
        directions = rng.normal(size=(1024, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        for target in directions:
            result = rotation_from_ez_to(target)
            np.testing.assert_allclose(result.T @ result, np.eye(3), atol=1e-14)
            assert float(np.linalg.det(result)) == pytest.approx(1.0)


class TestMotionAxisIsPreserved:
    """The end-to-end consequence, stated in the form the caller needs.

    ``normalize_actuated_axes`` rewrites a joint to spin about ``+Z`` and wraps
    the subtree in ``S`` / ``S.T``. For that to be a no-op physically, rotating
    by ``theta`` about the *original* axis must equal ``S @ Rz(theta) @ S.T``.
    This is the property the pole band actually broke.
    """

    @pytest.mark.parametrize("alpha", [1e-9, 1e-7, 1e-5, 4.4e-5, 1e-3, 0.5])
    @pytest.mark.parametrize("sign", [-1.0, 1.0], ids=["minus_z", "plus_z"])
    def test_conjugated_z_rotation_matches_the_original_axis(
        self, alpha: float, sign: float
    ) -> None:
        axis = _axis_near_pole(alpha, sign)
        s_matrix = rotation_from_ez_to(axis)
        worst = 0.0
        for theta in np.linspace(0.0, 2.0 * np.pi, 37):
            expected = rotation_from_axis_angle(axis, theta)
            spin = rotation_from_axis_angle(EZ, theta)
            actual = s_matrix @ spin @ s_matrix.T
            worst = max(worst, _rotation_gap(expected, actual))
        assert worst < 1e-12, (
            f"axis {alpha:.1e} rad from "
            f"{'+' if sign > 0 else '-'}Z: normalized joint rotates about a "
            f"different line (Frobenius gap {worst:.3e})"
        )

    def test_metric_would_catch_a_real_axis_error(self) -> None:
        """Guard the gate above against passing vacuously.

        ``_rotation_gap`` is a loose-looking 1e-12 threshold on a quantity
        whose scale is not obvious. This pins that a wrong axis -- the exact
        failure the old pole band produced -- clears it by many orders of
        magnitude, so the test above is measuring something.
        """
        axis = _axis_near_pole(4.4e-5, -1.0)
        # What the old degenerate branch returned for this axis.
        wrong = np.diag([1.0, -1.0, -1.0])
        worst = 0.0
        for theta in np.linspace(0.0, 2.0 * np.pi, 37):
            expected = rotation_from_axis_angle(axis, theta)
            actual = wrong @ rotation_from_axis_angle(EZ, theta) @ wrong.T
            worst = max(worst, _rotation_gap(expected, actual))
        assert worst > 1e-6, (
            f"the old antipodal branch should be visibly wrong here, but the "
            f"metric reports only {worst:.3e}"
        )
