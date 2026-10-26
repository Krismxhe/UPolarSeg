import torch
import torch.nn.functional as F

from src.losses.polar_quantification import (
    PolarQuantificationLoss,
)


q = PolarQuantificationLoss(
    valid_grid=torch.ones(
        8,
        32,
        dtype=torch.bool,
    ),
    max_radius=32.0,
    vertical_angle_deg=90.0,
)

# Shape: [B, angle, radius].
y = torch.zeros(
    1,
    8,
    32,
    dtype=torch.long,
)

# Last foreground sample radius at each angle.
# Vertical directions are angle indices 2 and 6:
# Dv = 6 + 12 = 18.
radii = [15, 9, 6, 8, 13, 10, 12, 7]

for angle, radius in enumerate(radii):
    y[:, angle, :radius + 1] = 1

# PPA extends from OD radius 15 to outer radius 20.
# PMW = 20 - 15 = 5.
y[:, 0, 16:21] = 2

hard = q.measure_mask(y)

assert hard['reference_valid'].item()
assert hard['pmw'].item() == 5.0
assert hard['dv'].item() == 18.0

assert torch.allclose(
    hard['pvdr'],
    torch.tensor([5.0 / 18.0]),
)

# On interval-compatible one-hot masks, soft and hard readouts agree.
p = F.one_hot(
    y,
    num_classes=3,
).permute(0, 3, 1, 2).float()

soft = q.read_probabilities(p)

for key in ('pmw', 'dv', 'pvdr'):
    assert torch.allclose(
        soft[key],
        hard[key],
    )

# A broken OD radial interval is not accepted as a normal measurement.
y_gap = y.clone()
y_gap[:, 2, 3] = 0

assert not q.measure_mask(
    y_gap
)['reference_valid'].item()

# Verify gradients flow back to segmentation logits.
z = torch.randn(
    1,
    3,
    8,
    32,
    requires_grad=True,
)

loss, _ = q(z, y)
loss.backward()

assert torch.isfinite(loss)
assert torch.isfinite(z.grad).all()
assert z.grad.abs().sum() > 0

print(
    f"PMW={hard['pmw'].item():.4f} px, "
    f"Dv={hard['dv'].item():.4f} px, "
    f"PVDR={hard['pvdr'].item():.6f}"
)

print(
    "Soft/hard identity, invalid OD "
    "and gradient checks passed."
)