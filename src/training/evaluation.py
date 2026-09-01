from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.amp import autocast


@dataclass(frozen=True)
class EvaluationMetrics:
    rank_loss: float
    masked_mse: float
    smallest_label_accuracy: float | None
    correct_smallest: int
    evaluated_samples: int
    top_2_accuracy: float | None = None
    top_3_accuracy: float | None = None
    mean_reciprocal_rank: float | None = None
    mean_spearman: float | None = None

    def to_dict(self) -> dict:
        return {
            "rank_loss": self.rank_loss,
            "masked_mse": self.masked_mse,
            "top_1_accuracy": self.smallest_label_accuracy,
            "top_2_accuracy": self.top_2_accuracy,
            "top_3_accuracy": self.top_3_accuracy,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "mean_spearman": self.mean_spearman,
            "correct_smallest": self.correct_smallest,
            "evaluated_samples": self.evaluated_samples,
        }


@dataclass(frozen=True)
class EvaluationResult:
    predictions: torch.Tensor
    labels: torch.Tensor
    label_masks: torch.Tensor
    metrics: EvaluationMetrics
    batch_sizes: tuple[int, ...] = ()


def ranknet_loss_components(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    label_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return weighted RankNet numerator, denominator, and differentiable loss."""
    y_inf = labels.clone()
    y_inf[~label_masks] = float("inf")
    true_ranks = torch.argsort(torch.argsort(y_inf, dim=1), dim=1).float()

    p1, p2 = predictions.unsqueeze(2), predictions.unsqueeze(1)
    y1, y2 = labels.unsqueeze(2), labels.unsqueeze(1)
    m1, m2 = label_masks.unsqueeze(2), label_masks.unsqueeze(1)
    order_mask = m1 & m2 & (y1 < y2)
    average_rank = (true_ranks.unsqueeze(2) + true_ranks.unsqueeze(1)) / 2.0
    rank_weights = 1.0 / torch.log2(average_rank + 2.0)
    weighted_pairs = F.softplus(p1 - p2) * rank_weights * order_mask.float()
    numerator = weighted_pairs.sum()
    denominator = (rank_weights * order_mask.float()).sum().clamp(min=1.0)
    return numerator, denominator, numerator / denominator


def _average_tie_ranks(values: torch.Tensor) -> torch.Tensor:
    """Return zero-based average ranks, with smaller values ranked first."""
    order = torch.argsort(values)
    ranks = torch.empty(values.numel(), dtype=torch.float64)
    sorted_values = values[order]
    start = 0
    while start < values.numel():
        end = start + 1
        while end < values.numel() and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(labels: torch.Tensor, predictions: torch.Tensor) -> float | None:
    if labels.numel() < 2:
        return None
    true_ranks = _average_tie_ranks(labels.detach().cpu().double())
    pred_ranks = _average_tie_ranks(predictions.detach().cpu().double())
    true_centered = true_ranks - true_ranks.mean()
    pred_centered = pred_ranks - pred_ranks.mean()
    denominator = torch.linalg.vector_norm(true_centered) * torch.linalg.vector_norm(pred_centered)
    if denominator.item() == 0.0:
        return None
    return float(torch.dot(true_centered, pred_centered) / denominator)


def compute_evaluation_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    label_masks: torch.Tensor,
) -> EvaluationMetrics:
    """Compute ranking and regression metrics from one aligned prediction set."""
    if predictions.shape != labels.shape or labels.shape != label_masks.shape:
        raise ValueError("Predictions, labels, and label masks must have identical shapes.")
    if predictions.ndim != 2:
        raise ValueError("Evaluation tensors must have shape [examples, labels].")

    mask_f = label_masks.float()
    mask_count = mask_f.sum().item()
    masked_mse = float((((predictions - labels) ** 2) * mask_f).sum().item() / (mask_count + 1e-8))

    rank_numerator, rank_denominator, _ = ranknet_loss_components(
        predictions, labels, label_masks
    )
    rank_loss = float(rank_numerator.item() / rank_denominator.item())

    correct = {1: 0, 2: 0, 3: 0}
    reciprocal_ranks = []
    correlations = []
    evaluated = 0
    for pred_row, label_row, mask_row in zip(predictions, labels, label_masks):
        valid = torch.nonzero(mask_row, as_tuple=False).flatten()
        if valid.numel() == 0:
            continue
        evaluated += 1
        valid_labels = label_row[valid]
        true_minimum = valid[valid_labels == valid_labels.min()]
        predicted_order = valid[torch.argsort(pred_row[valid])]
        matches = torch.isin(predicted_order, true_minimum)
        first_match = int(torch.nonzero(matches, as_tuple=False)[0].item()) + 1
        reciprocal_ranks.append(1.0 / first_match)
        for k in correct:
            if matches[:k].any().item():
                correct[k] += 1
        correlation = _spearman(valid_labels, pred_row[valid])
        if correlation is not None:
            correlations.append(correlation)

    def percentage(count):
        return 100.0 * count / evaluated if evaluated else None

    return EvaluationMetrics(
        rank_loss=rank_loss,
        masked_mse=masked_mse,
        smallest_label_accuracy=percentage(correct[1]),
        correct_smallest=correct[1],
        evaluated_samples=evaluated,
        top_2_accuracy=percentage(correct[2]),
        top_3_accuracy=percentage(correct[3]),
        mean_reciprocal_rank=(sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else None),
        mean_spearman=(sum(correlations) / len(correlations) if correlations else None),
    )


def evaluate_model(model, loader, device, batch_forward) -> EvaluationResult:
    """Run one deterministic loader pass and retain aligned predictions/targets."""
    model = model.to(device).eval()
    predictions = []
    labels = []
    masks = []
    with torch.no_grad():
        for batch in loader:
            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                result = batch_forward(model, batch, device, apply_label_mask=False)
            predictions.append(result.predictions.detach().cpu())
            labels.append(result.labels.detach().cpu())
            masks.append(result.label_mask.detach().cpu())
    if not predictions:
        raise ValueError("Cannot evaluate an empty dataloader.")
    predictions_tensor = torch.cat(predictions, dim=0)
    labels_tensor = torch.cat(labels, dim=0)
    masks_tensor = torch.cat(masks, dim=0).bool()
    metrics = compute_evaluation_metrics(predictions_tensor, labels_tensor, masks_tensor)
    return EvaluationResult(
        predictions_tensor,
        labels_tensor,
        masks_tensor,
        metrics,
        tuple(len(batch_predictions) for batch_predictions in predictions),
    )

def is_min_predicted(true_labels, pred_labels):
    """
    true_labels : list of floats, length L, where missing labels are -1
    pred_labels : list of floats, same length, model predictions over all L slots

    Returns True if the model’s lowest‐scoring valid label
    matches one of the true‐minimum indices; else False.
    """
    # Indices where we actually have a ground-truth label
    valid_idx = [i for i, y in enumerate(true_labels) if y != -1]
    if not valid_idx:
        # no valid labels → can’t predict; return False
        return False

    # Find the true min and all indices that achieve it
    true_vals     = [true_labels[i] for i in valid_idx]
    min_val       = min(true_vals)
    true_min_idxs = {i for i in valid_idx if true_labels[i] == min_val}

    # Find which of those valid indices the model scored smallest
    pred_min_idx = min(valid_idx, key=lambda i: pred_labels[i])

    return pred_min_idx in true_min_idxs

def test_model(
    model,
    test_loader,
    device,
    criterion,  # e.g. nn.MSELoss(reduction='none')
    batch_forward,
    return_metrics=False,
):
    """
    Evaluate the model and print the percentage of examples where
    the model correctly predicts the smallest true label.

    Returns:
      - outputs_all:   list of all prediction lists (cpu)
      - total_loss:    average pairwise RankNet loss
    """
    result = evaluate_model(model, test_loader, device, batch_forward)
    metrics = result.metrics
    outputs_all = list(result.predictions.split(result.batch_sizes))
    loss = metrics.rank_loss

    # print top-1 smallest-label accuracy
    if metrics.evaluated_samples > 0:
        print(
            "Test smallest-label accuracy: "
            f"{metrics.smallest_label_accuracy:.3f}% "
            f"({metrics.correct_smallest}/{metrics.evaluated_samples})"
        )
    else:
        print("Test smallest-label accuracy: no valid samples to evaluate.")

    if return_metrics:
        return outputs_all, loss, metrics
    return outputs_all, loss
