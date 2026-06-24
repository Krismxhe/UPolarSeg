"""Shared base for research models.

Research models follow the project output contract: ``forward`` returns either
a logits tensor ``B×C×H×W`` or a dict containing at least ``"seg_logits"``
(see :mod:`src.models.outputs`). They consume the shared dataset / loss /
evaluator pipeline and must not define their own training loop.

``ResearchModel`` is an optional convenience base. Subclassing it is **not**
required — any ``nn.Module`` honouring the output contract is a valid research
model. It exists so future research methods can share small conveniences
without coupling to baseline code.
"""

from __future__ import annotations

from torch import nn


class ResearchModel(nn.Module):
    """Optional base class for research models.

    Subclasses implement :meth:`forward` and must honour the output contract
    enforced by :func:`src.models.outputs.normalize_model_output`.
    """

    def forward(self, x):  # pragma: no cover - abstract-style hook
        raise NotImplementedError
