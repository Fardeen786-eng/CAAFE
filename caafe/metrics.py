from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    f1_score,
)
import torch
import numpy as np

higher_is_better = {
    "auc": True,
    "accuracy": True,
    "balanced_accuracy": True,
    "average_precision": True,
    "f1": True,
    "r2": True,
    "mse": False,
    "mae": False,
    "rmse": False,
    "rae": False,
    "1-rae": True,  # 1 - rae is higher is better
}

def _to_torch(target, pred):
    """Helper to convert inputs to torch tensors."""
    if not torch.is_tensor(target):
        target = torch.tensor(target)
    if not torch.is_tensor(pred):
        pred = torch.tensor(pred)
    return target, pred

def auc_metric(target, pred, multi_class="ovo", numpy=False):
    lib = np if numpy else torch
    try:
        if not numpy:
            target, pred = _to_torch(target, pred)
        if len(lib.unique(target)) > 2:
            score = roc_auc_score(target, pred, multi_class=multi_class)
        else:
            if pred.ndim == 2:
                pred = pred[:, 1]
            score = roc_auc_score(target, pred)
        return lib.tensor(score) if not numpy else score
    except ValueError:
        return lib.tensor(np.nan) if not numpy else np.nan

def accuracy_metric(target, pred):
    target, pred = _to_torch(target, pred)
    if len(torch.unique(target)) > 2:
        preds = torch.argmax(pred, dim=-1)
    else:
        if pred.ndim == 2:
            preds = (pred[:, 1] > 0.5).long()
        else:
            # Assume pred is already class labels or probabilities
            preds = (pred > 0.5).long() if pred.dtype.is_floating_point else pred.long()    
    return torch.tensor(accuracy_score(target.cpu(), preds.cpu()))

def balanced_accuracy_metric(target, pred):
    target, pred = _to_torch(target, pred)
    if len(torch.unique(target)) > 2:
        preds = torch.argmax(pred, dim=-1)
    else:
        preds = (pred[:, 1] > 0.5).long()
    return torch.tensor(balanced_accuracy_score(target, preds))

def average_precision_metric(target, pred):
    target, pred = _to_torch(target, pred)
    scores = pred[:, 1] if pred.ndim == 2 else pred
    return torch.tensor(average_precision_score(target, scores))

def mse_metric(target, pred):
    target_np = np.asarray(target)
    pred_np = np.asarray(pred)
    return torch.tensor(mean_squared_error(target_np, pred_np))

def mae_metric(target, pred):
    target_np = np.asarray(target)
    pred_np = np.asarray(pred)
    return torch.tensor(mean_absolute_error(target_np, pred_np))

def r2_metric(target, pred):
    target_np = np.asarray(target)
    pred_np = np.asarray(pred)
    return torch.tensor(r2_score(target_np, pred_np))

def f1_metric(target, pred, average="macro"):
    """
    F1 Score metric.
    For binary classification: average="binary"
    For multiclass: average can be "macro", "micro", or "weighted"
    """
    if len(np.unique(pred)) > 2:
        return f1_score(target, pred, average=average)
    return f1_score(target, pred)
    # binary or logits
    # if pred.dim() == 2 and pred.size(1) == 2:
    #     preds = (pred[:, 1] > 0.5).long()
    # else:
    #     preds = (pred > 0.5).long() if pred.dtype == torch.float else pred.long()
    # return torch.tensor(f1_score(target.cpu().numpy().ravel(), preds.cpu().numpy().ravel(), average=average))

def rmse_metric(target, pred):
    target_np = np.asarray(target)
    pred_np = np.asarray(pred)
    mse = mean_squared_error(target_np, pred_np)
    return torch.tensor(np.sqrt(mse))

def rae_metric(target, pred):
    target_np = np.asarray(target)
    pred_np = np.asarray(pred)
    numerator = np.sum(np.abs(target_np - pred_np))
    denominator = np.sum(np.abs(target_np - np.mean(target_np)))
    if denominator == 0:
        return torch.tensor(np.nan)  # Avoid division by zero
    return torch.tensor(numerator / denominator)

def rae_metric_1(target, pred):
    """
    1 - RAE, where higher is better.
    """
    target_np = np.asarray(target)
    pred_np = np.asarray(pred)
    numerator = np.sum(np.abs(target_np - pred_np))
    denominator = np.sum(np.abs(target_np - np.mean(target_np)))
    if denominator == 0:
        return torch.tensor(np.nan)  # Avoid division by zero
    return torch.tensor(1 - (numerator / denominator))

# map string names to functions
METRIC_FUNCTIONS = {
    "auc": auc_metric,
    "accuracy": accuracy_metric,
    "balanced_accuracy": balanced_accuracy_metric,
    "average_precision": average_precision_metric,
    "mse": mse_metric,
    "mae": mae_metric,
    "r2": r2_metric,
    "f1": f1_metric,
    "rmse": rmse_metric,
    "rae": rae_metric,
    "1-rae": rae_metric_1,
}

def evaluate_metric(metric_name: str, y_true, y_pred, **kwargs):
    """
    Compute the given metric.
    Parameters
    ----------
    metric_name : str
        One of the keys of METRIC_FUNCTIONS, e.g. "auc", "mse", "f1", etc.
    y_true, y_pred
        Ground-truth and predictions.
    **kwargs
        Additional arguments to pass to the metric (e.g., average="macro", numpy=True).
    """
    if metric_name not in METRIC_FUNCTIONS:
        raise ValueError(f"Unsupported metric: {metric_name}")
    fn = METRIC_FUNCTIONS[metric_name]
    return fn(y_true, y_pred, **kwargs)
