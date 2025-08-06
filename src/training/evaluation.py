import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

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
):
    """
    Evaluate the model and print the percentage of examples where
    the model correctly predicts the smallest true label.

    Returns:
      - outputs_all:   list of all prediction lists (cpu)
      - total_loss:    weighted_mse + avg_rank_loss
      - avg_rank_loss: average pairwise RankNet loss
    """
    model = model.to(device).eval()
    outputs_all     = []
    rank_sum        = 0.0
    rank_weight_sum = 0.0

    correct_min   = 0
    total_samples = 0

    weighted_mse_sum = 0.0
    mask_count       = 0.0

    with torch.no_grad():
        for inputs, pos_enc, token_mask, labels, label_masks in test_loader:
            inputs      = inputs.to(device)
            pos_enc     = pos_enc.to(device)
            token_mask  = token_mask.to(device)
            labels      = labels.to(device).float()   # [B, L]
            label_masks = label_masks.to(device)      # [B, L]

            # Forward pass under autocast
            with autocast():
                preds = model(inputs, pos_enc, token_mask).squeeze()  # [B, L]

            outputs_all.extend(preds.cpu().tolist())

            # — weighted masked MSE —
            loss_tensor = criterion(preds, labels)  # [B, L]
            mask_f      = label_masks.float()       # [B, L]
            weighted_mse_sum += (loss_tensor * mask_f).sum().item()
            mask_count       += mask_f.sum().item()

            # — compute true ranks —
            y_inf = labels.clone()
            y_inf[~label_masks] = float('inf')
            true_ranks = torch.argsort(torch.argsort(y_inf, dim=1), dim=1).float()  # [B, L]

            # — pairwise RankNet with ordering mask —
            p1 = preds.unsqueeze(2)    # [B, L, 1]
            p2 = preds.unsqueeze(1)    # [B, 1, L]
            y1 = labels.unsqueeze(2)   # [B, L, 1]
            y2 = labels.unsqueeze(1)   # [B, 1, L]
            m1 = label_masks.unsqueeze(2)
            m2 = label_masks.unsqueeze(1)

            order_mask = m1 & m2 & (y1 < y2)  # [B, L, L]
            pair_losses = F.softplus(p1 - p2) * order_mask.float()

            # — rank‐based weighting —
            r1 = true_ranks.unsqueeze(2)
            r2 = true_ranks.unsqueeze(1)
            avg_rank = (r1 + r2) / 2.0
            rank_weights = 1.0 / torch.log2(avg_rank + 2.0)
            weighted_pair = pair_losses * (rank_weights * order_mask.float())

            total_pair_loss = weighted_pair.sum().item()
            sum_weights     = (rank_weights * order_mask.float()).sum().clamp(min=1.0).item()
            rank_sum        += total_pair_loss
            rank_weight_sum += sum_weights

            # — compute top-1 smallest-label accuracy —
            preds_list  = preds.cpu().tolist()
            labels_list = labels.cpu().tolist()
            masks_list  = label_masks.cpu().tolist()
            for true_row, pred_row, mask_row in zip(labels_list, preds_list, masks_list):
                if any(mask_row):
                    if is_min_predicted(true_row, pred_row):
                        correct_min += 1
                    total_samples += 1

    # finalize metrics
    avg_rank_loss = rank_sum / (rank_weight_sum + 1e-8)
    total_loss    = avg_rank_loss

    # print top-1 smallest-label accuracy
    if total_samples > 0:
        acc = 100.0 * correct_min / total_samples
        print(f"Test smallest-label accuracy: {acc:.3f}% ({correct_min}/{total_samples})")
    else:
        print("Test smallest-label accuracy: no valid samples to evaluate.")

    return outputs_all, total_loss, avg_rank_loss