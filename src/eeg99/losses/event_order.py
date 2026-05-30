"""EventOrderConsistencyLoss — original algorithm #4.

The six Grasp-and-Lift events always appear in the fixed temporal order:
  HandStart(0) → FirstDigitTouch(1) → BothStartLoadPhase(2) →
  LiftOff(3) → Replace(4) → BothReleased(5)

Within a trial, if event j occurs before event i (i < j) then p_i ≥ p_j at
any given timestep is expected near event boundaries.  This loss penalises
"out-of-order" probability mass in the prediction sequence.

Formulation
-----------
For consecutive event pair (i, i+1) and a window of T predictions:
  violation_t = max(0, p_{i+1,t} - p_{i,t})   when t is in early phase

We use a smooth hinge: ReLU(p_{i+1} - p_i - margin) summed over timesteps
and events.  This penalises predictions where a later event appears more
probable than an earlier one at the same time, which is causally impossible
before event i has occurred.

Usage
-----
::

    criterion = EventOrderConsistencyLoss(weight=0.1)
    task_loss = bce_loss(logits, targets)
    order_loss = criterion(torch.sigmoid(logits))
    total = task_loss + order_loss
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from eeg99.utils.constants import EVENT_ORDER

__all__ = ["EventOrderConsistencyLoss", "EventOrderConfig"]


@dataclass(frozen=True)
class EventOrderConfig:
    weight: float = 0.1          # multiplier applied to the order penalty
    margin: float = 0.05         # tolerance before penalising


class EventOrderConsistencyLoss(nn.Module):
    """Penalise predictions that violate the fixed event temporal order.

    Parameters
    ----------
    cfg : EventOrderConfig
    """

    def __init__(self, cfg: EventOrderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or EventOrderConfig()
        # Consecutive pairs in the canonical order
        self._pairs: list[tuple[int, int]] = [
            (EVENT_ORDER[i], EVENT_ORDER[i + 1])
            for i in range(len(EVENT_ORDER) - 1)
        ]

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute order consistency penalty.

        Parameters
        ----------
        probs : torch.Tensor
            Either ``(B, n_events)`` for per-window predictions, or
            ``(B, T, n_events)`` for sequence predictions.

        Returns
        -------
        torch.Tensor  scalar loss (already multiplied by cfg.weight)
        """
        if probs.dim() == 2:
            return self._window_loss(probs)
        return self._sequence_loss(probs)

    def _window_loss(self, probs: torch.Tensor) -> torch.Tensor:
        """(B, E) → scalar.
        Penalise p_{i+1} >> p_i across the batch.
        """
        penalty = torch.tensor(0.0, device=probs.device)
        for i, j in self._pairs:
            # j is a later event; it should not greatly exceed i
            violation = torch.relu(probs[:, j] - probs[:, i] - self.cfg.margin)
            penalty = penalty + violation.mean()
        return self.cfg.weight * penalty / len(self._pairs)

    def _sequence_loss(self, probs: torch.Tensor) -> torch.Tensor:
        """(B, T, E) → scalar.
        At each timestep, enforce the same pairwise constraint.
        """
        penalty = torch.tensor(0.0, device=probs.device)
        for i, j in self._pairs:
            violation = torch.relu(probs[:, :, j] - probs[:, :, i] - self.cfg.margin)
            penalty = penalty + violation.mean()
        return self.cfg.weight * penalty / len(self._pairs)
