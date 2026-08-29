"""Shared, permutation-invariant target-conditioned portal binding head."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn


class BindingState(str, Enum):
    NONE = "NONE"
    SET_VALUED = "SET_VALUED"
    COMMIT = "COMMIT"


@dataclass(frozen=True)
class HeadConfig:
    hidden_size: int = 32
    embedding_size: int = 16
    learning_rate: float = 0.003
    weight_decay: float = 0.0001
    maximum_epochs: int = 240
    early_stopping_patience: int = 35
    minimum_delta: float = 0.0001
    candidate_set_logit_gap: float = 0.35
    seed: int = 20260829


@dataclass(frozen=True)
class Episode:
    key: str
    features: np.ndarray
    positive_mask: np.ndarray
    expected_none: bool


@dataclass(frozen=True)
class BindingDecision:
    state: BindingState
    selected_index: int | None
    candidate_indices: tuple[int, ...]
    candidate_logits: tuple[float, ...]
    none_logit: float


class PortalBindingHead(nn.Module):
    """Score candidates independently, then derive NONE from a symmetric set summary."""

    def __init__(self, input_size: int, config: HeadConfig = HeadConfig()) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.config = config
        self.shared = nn.Sequential(
            nn.Linear(input_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.embedding_size),
            nn.ReLU(),
        )
        self.candidate = nn.Linear(config.embedding_size, 1)
        summary_size = 3 * config.embedding_size + 4
        self.none = nn.Sequential(
            nn.Linear(summary_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or features.shape[0] <= 0:
            raise ValueError("PORTAL_FEATURES_MUST_BE_NONEMPTY_2D")
        hidden = self.shared(features)
        candidate_logits = self.candidate(hidden).squeeze(-1)
        summary = torch.cat(
            (
                hidden.mean(dim=0),
                hidden.max(dim=0).values,
                hidden.min(dim=0).values,
                candidate_logits.mean().reshape(1),
                candidate_logits.max().reshape(1),
                candidate_logits.std(unbiased=False).reshape(1),
                torch.log1p(torch.tensor(float(features.shape[0]), device=features.device)).reshape(1),
            )
        )
        none_logit = self.none(summary).squeeze()
        return candidate_logits, none_logit


def listwise_loss(
    candidate_logits: torch.Tensor,
    none_logit: torch.Tensor,
    positive_mask: torch.Tensor,
    expected_none: bool,
) -> torch.Tensor:
    logits = torch.cat((candidate_logits, none_logit.reshape(1)))
    denominator = torch.logsumexp(logits, dim=0)
    if expected_none:
        numerator = none_logit
    else:
        if positive_mask.shape != candidate_logits.shape or not bool(positive_mask.any()):
            raise ValueError("POSITIVE_EPISODE_REQUIRES_POSITIVE_CANDIDATE")
        numerator = torch.logsumexp(candidate_logits[positive_mask], dim=0)
    return denominator - numerator


def reduce_logits(
    candidate_logits: Sequence[float],
    none_logit: float,
    candidate_set_logit_gap: float,
) -> BindingDecision:
    values = np.asarray(candidate_logits, dtype=np.float32)
    if values.ndim != 1 or values.size <= 0:
        raise ValueError("CANDIDATE_LOGITS_MUST_BE_NONEMPTY")
    order = np.argsort(-values, kind="stable")
    best = int(order[0])
    if float(none_logit) >= float(values[best]):
        return BindingDecision(
            BindingState.NONE,
            None,
            (),
            tuple(float(value) for value in values),
            float(none_logit),
        )
    selected = tuple(
        int(index)
        for index in order
        if float(values[best] - values[int(index)]) <= candidate_set_logit_gap
    )
    if len(selected) > 1:
        return BindingDecision(
            BindingState.SET_VALUED,
            None,
            selected,
            tuple(float(value) for value in values),
            float(none_logit),
        )
    return BindingDecision(
        BindingState.COMMIT,
        best,
        (best,),
        tuple(float(value) for value in values),
        float(none_logit),
    )


def _normalized_episode(episode: Episode, mean: np.ndarray, scale: np.ndarray) -> Episode:
    return Episode(
        episode.key,
        ((episode.features - mean) / scale).astype(np.float32),
        episode.positive_mask,
        episode.expected_none,
    )


def fit_head(
    train_episodes: Sequence[Episode],
    development_episodes: Sequence[Episode],
    config: HeadConfig = HeadConfig(),
) -> tuple[PortalBindingHead, dict[str, object], np.ndarray, np.ndarray]:
    if not train_episodes or not development_episodes:
        raise ValueError("TRAIN_AND_DEVELOPMENT_EPISODES_REQUIRED")
    feature_count = int(train_episodes[0].features.shape[1])
    if any(row.features.shape[1] != feature_count for row in train_episodes + development_episodes):
        raise ValueError("FEATURE_WIDTH_MISMATCH")
    stacked = np.concatenate([row.features for row in train_episodes], axis=0).astype(np.float32)
    mean = stacked.mean(axis=0)
    scale = stacked.std(axis=0)
    scale[scale < 1e-6] = 1.0
    train = [_normalized_episode(row, mean, scale) for row in train_episodes]
    development = [_normalized_episode(row, mean, scale) for row in development_episodes]
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model = PortalBindingHead(feature_count, config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = np.random.default_rng(config.seed)
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    patience = 0
    history = []
    for epoch in range(1, config.maximum_epochs + 1):
        model.train()
        train_losses = []
        for index in generator.permutation(len(train)):
            row = train[int(index)]
            features = torch.from_numpy(row.features)
            mask = torch.from_numpy(row.positive_mask.astype(np.bool_))
            candidate_logits, none_logit = model(features)
            loss = listwise_loss(candidate_logits, none_logit, mask, row.expected_none)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            development_losses = []
            for row in development:
                candidate_logits, none_logit = model(torch.from_numpy(row.features))
                development_losses.append(
                    float(
                        listwise_loss(
                            candidate_logits,
                            none_logit,
                            torch.from_numpy(row.positive_mask.astype(np.bool_)),
                            row.expected_none,
                        )
                    )
                )
        development_loss = float(np.mean(development_losses))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "development_loss": development_loss,
            }
        )
        if development_loss < best_loss - config.minimum_delta:
            best_loss = development_loss
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("NO_HEAD_CHECKPOINT_SELECTED")
    model.load_state_dict(best_state)
    receipt = {
        "training_device": "cpu",
        "selection_reason": "TASK_NOT_GPU_SUITABLE",
        "best_epoch": best_epoch,
        "best_development_loss": best_loss,
        "epochs_executed": len(history),
        "history": history,
    }
    return model, receipt, mean, scale


def predict_head(
    model: PortalBindingHead,
    episode: Episode,
    mean: np.ndarray,
    scale: np.ndarray,
) -> BindingDecision:
    normalized = _normalized_episode(episode, mean, scale)
    model.eval()
    with torch.inference_mode():
        candidate, none = model(torch.from_numpy(normalized.features))
    return reduce_logits(
        candidate.detach().cpu().numpy(),
        float(none.detach().cpu()),
        model.config.candidate_set_logit_gap,
    )


def self_test() -> dict[str, object]:
    config = HeadConfig(hidden_size=8, embedding_size=4, maximum_epochs=2)
    torch.manual_seed(7)
    model = PortalBindingHead(3, config)
    features = torch.tensor([[0.2, 0.4, 0.6], [0.7, 0.1, 0.3], [0.5, 0.9, 0.2]])
    permutation = torch.tensor([2, 0, 1])
    logits, none = model(features)
    permuted_logits, permuted_none = model(features[permutation])
    if not torch.allclose(permuted_none, none, atol=1e-6):
        raise AssertionError("NONE_HEAD_NOT_PERMUTATION_INVARIANT")
    if not torch.allclose(permuted_logits, logits[permutation], atol=1e-6):
        raise AssertionError("CANDIDATE_HEAD_NOT_PERMUTATION_EQUIVARIANT")
    commit = reduce_logits([1.0, 0.0], -1.0, 0.35)
    ambiguous = reduce_logits([1.0, 0.8], -1.0, 0.35)
    none_decision = reduce_logits([0.0, -0.2], 0.1, 0.35)
    if commit.state != BindingState.COMMIT or ambiguous.state != BindingState.SET_VALUED:
        raise AssertionError("BINDING_REDUCER_STATE_MISMATCH")
    if none_decision.state != BindingState.NONE:
        raise AssertionError("EXPLICIT_NONE_NOT_SELECTED")
    return {
        "schema": "l10-target-conditioned-portal-binding-self-test-v1",
        "status": "PASS",
        "candidate_equivariance": True,
        "set_summary_invariance": True,
        "states": [commit.state.value, ambiguous.state.value, none_decision.state.value],
    }


if __name__ == "__main__":
    print(self_test())
