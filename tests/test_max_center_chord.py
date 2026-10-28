"""Contracts for optional maximum centre-crossing OD chord measurements.

The independent reference evaluates the union of the periodic radial profile's
knots and its antipodal knots. It deliberately does not use tensor rolls.
Real Torch is required for these tests, including autograd checks.
"""

from __future__ import annotations

import math
import random

import pytest

torch = pytest.importorskip("torch")

from src.quantification.polar_readout import PolarReadout, _max_center_chord


def _linear_periodic(profile, position):
    """Interpolate at a position measured in angular-bin units."""
    position %= len(profile)
    left = math.floor(position)
    fraction = position - left
    return (
        (1.0 - fraction) * profile[left]
        + fraction * profile[(left + 1) % len(profile)]
    )


def _union_knot_maximum(profile):
    count = len(profile)
    half_turn = count / 2.0
    knots = {float(index) for index in range(count)}
    knots.update((index - half_turn) % count for index in range(count))
    return max(
        _linear_periodic(profile, angle)
        + _linear_periodic(profile, angle + half_turn)
        for angle in knots
    )


def _readout(angle_bins, *, mode="max_center_chord", max_radius=32.0, **kwargs):
    return PolarReadout(
        valid_grid=torch.ones(angle_bins, 32, dtype=torch.bool),
        max_radius=max_radius,
        dv_mode=mode,
        **kwargs,
    )


def _prefix_mask(radii, ppa_width=3):
    mask = torch.zeros(1, len(radii), 32, dtype=torch.long)
    for angle, radius in enumerate(radii):
        mask[0, angle, : radius + 1] = 1
        mask[0, angle, radius + 1 : radius + ppa_width + 1] = 2
    return mask


def test_maximum_chord_is_not_the_axis_of_the_longest_radius():
    profile = torch.tensor([[10.0, 9.0, 6.0, 4.0, 1.0, 8.0, 6.0, 4.0]])
    # The longest ray is 10, whose opposite ray is only 1. Another axis has
    # radii 9 and 8, hence the desired diameter is 17 rather than 11 or 20.
    assert _max_center_chord(profile).item() == 17.0
    result = _readout(8)._summarize(profile, torch.zeros_like(profile))
    assert result["dv"].item() == 17.0


def test_even_asymmetric_radii_are_added_without_assuming_a_centred_disc():
    profile = torch.tensor([[10.0, 3.0, 1.0, 2.0]])
    assert _max_center_chord(profile).item() == 11.0
    assert _max_center_chord(profile).item() != 2 * profile.max().item()


def test_odd_bins_interpolate_the_actual_antipode():
    profile = torch.tensor([[1.0, 2.0, 10.0, 3.0, 4.0]])
    # At the 10-valued ray the antipode is halfway between radii 4 and 1.
    # An integer two-bin shift would instead give a false maximum of 14.
    assert _max_center_chord(profile).item() == 12.5


@pytest.mark.parametrize("angle_bins", [4, 5, 6, 7, 9, 11])
def test_matches_independent_union_of_interpolation_knots(angle_bins):
    generator = random.Random(137 + angle_bins)
    profiles = [
        [generator.uniform(0.0, 20.0) for _ in range(angle_bins)]
        for _ in range(3)
    ]
    expected = [_union_knot_maximum(profile) for profile in profiles]
    actual = _max_center_chord(torch.tensor(profiles, dtype=torch.float64))
    torch.testing.assert_close(
        actual, torch.tensor(expected, dtype=torch.float64), rtol=0, atol=1e-12,
    )


def test_batch_members_have_independent_maxima():
    profiles = torch.tensor([
        [10.0, 9.0, 6.0, 4.0, 1.0, 8.0, 6.0, 4.0],
        [6.0] * 8,
        [1.0] * 8,
    ])
    result = _max_center_chord(profiles)
    assert tuple(result.shape) == (3,)
    torch.testing.assert_close(result, torch.tensor([17.0, 12.0, 2.0]))


@pytest.mark.parametrize("angle_bins", [5, 6, 9, 10])
def test_circular_rotation_preserves_maximum_chord(angle_bins):
    generator = random.Random(31 + angle_bins)
    profiles = torch.tensor([
        [generator.uniform(0.0, 20.0) for _ in range(angle_bins)]
        for _ in range(2)
    ], dtype=torch.float64)
    expected = _max_center_chord(profiles)
    for shift in (1, angle_bins // 2, angle_bins - 1):
        actual = _max_center_chord(torch.roll(profiles, shift, dims=-1))
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)


def test_default_and_explicit_fixed_vertical_keep_the_legacy_axis():
    profile = torch.tensor([[10.0, 3.0, 1.0, 2.0]])
    width = torch.zeros_like(profile)
    default = PolarReadout(torch.ones(4, 32, dtype=torch.bool), max_radius=32.0)
    fixed = _readout(4, mode="fixed_vertical")
    assert default.dv_mode == "fixed_vertical"
    assert default._summarize(profile, width)["dv"].item() == 5.0
    assert fixed._summarize(profile, width)["dv"].item() == 5.0
    assert _readout(4)._summarize(profile, width)["dv"].item() == 11.0


@pytest.mark.parametrize("angle_deg", [0.0, 45.0, 101.25, 179.0])
def test_max_mode_does_not_use_the_legacy_vertical_axis(angle_deg):
    profile = torch.tensor([[10.0, 3.0, 1.0, 2.0]])
    result = _readout(4, vertical_angle_deg=angle_deg)._summarize(
        profile, torch.zeros_like(profile),
    )
    assert result["dv"].item() == 11.0


@pytest.mark.parametrize("bad_mode", ["", "max_radius", "unknown"])
def test_unknown_diameter_mode_is_rejected(bad_mode):
    with pytest.raises(ValueError, match="dv_mode"):
        _readout(8, mode=bad_mode)


@pytest.mark.parametrize("max_radius, expected_dv", [(32.0, 12.5), (16.0, 6.25)])
def test_odd_hard_and_one_hot_soft_readouts_agree_in_cartesian_pixel_units(
    max_radius, expected_dv,
):
    mask = _prefix_mask([1, 2, 10, 3, 4])
    readout = _readout(5, max_radius=max_radius)
    hard = readout.measure_mask(mask)
    one_hot = torch.nn.functional.one_hot(mask, 3).permute(0, 3, 1, 2).float()
    soft = readout.read_probabilities(one_hot)
    assert hard["reference_valid"].item()
    assert hard["aux_valid"].item()
    assert hard["dv"].item() == expected_dv
    assert hard["pmw"].item() == 3.0 * max_radius / 32.0
    for key in ("od_profile", "width_profile", "dv", "pmw", "pvdr"):
        torch.testing.assert_close(soft[key], hard[key], rtol=0, atol=0)


@pytest.mark.parametrize("profile", [
    [10.0, 9.0, 6.0, 4.0, 1.0, 8.0, 6.0, 4.0],
    [1.0, 2.0, 10.0, 3.0, 4.0],
])
def test_non_tied_maximum_chord_passes_autograd_gradcheck(profile):
    values = torch.tensor([profile], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        _max_center_chord, (values,), eps=1e-6, atol=1e-5, rtol=1e-4,
    )


def test_odd_interpolation_gradients_reach_both_antipodal_neighbours():
    profile = torch.tensor(
        [[1.0, 2.0, 10.0, 3.0, 4.0]], dtype=torch.float64, requires_grad=True,
    )
    _max_center_chord(profile).sum().backward()
    torch.testing.assert_close(
        profile.grad,
        torch.tensor([[0.5, 0.0, 1.0, 0.0, 0.5]], dtype=torch.float64),
        rtol=0, atol=0,
    )


@pytest.mark.parametrize("angle_bins", [4, 5, 8, 9])
def test_all_tied_axes_have_finite_symmetric_subgradients(angle_bins):
    profile = torch.full(
        (2, angle_bins), 6.0, dtype=torch.float64, requires_grad=True,
    )
    _max_center_chord(profile).sum().backward()
    assert torch.isfinite(profile.grad).all()
    torch.testing.assert_close(
        profile.grad, torch.full_like(profile, 2.0 / angle_bins),
        rtol=0, atol=1e-12,
    )


def test_invalid_mask_flags_still_protect_max_mode_measurements():
    masks = _prefix_mask([1, 2, 10, 3, 4]).repeat(6, 1, 1)
    masks[1] = 0  # Empty OD.
    masks[2, 2, 3] = 0  # Broken OD radial prefix.
    masks[3, 2, 12] = 0  # PPA hole: hard boundary valid, occupancy incompatible.
    masks[4, 2, 27] = 2  # Foreground reaches the last valid radial sample.
    masks[5, 2, 29] = 2  # Foreground exists outside the image-valid grid.
    valid_grid = torch.ones(5, 32, dtype=torch.bool)
    valid_grid[:, 28:] = False
    readout = PolarReadout(valid_grid, max_radius=32.0, dv_mode="max_center_chord")
    hard = readout.measure_mask(masks)
    assert hard["reference_valid"].tolist() == [True, False, False, True, False, False]
    assert hard["aux_valid"].tolist() == [True, False, False, False, False, False]
    assert hard["dv"][0].item() == 12.5
    assert hard["dv"][2].item() == 12.5  # Numeric output alone does not establish validity.
    assert torch.isfinite(hard["dv"]).all()


def test_denominator_floor_protects_ratio_without_changing_the_diameter():
    readout = _readout(5, denominator_floor_px=16.0)
    hard = readout.measure_mask(_prefix_mask([1, 2, 10, 3, 4]))
    assert hard["dv"].item() == 12.5
    assert hard["pvdr"].item() == 3.0 / 16.0
    assert not hard["reference_valid"].item()
