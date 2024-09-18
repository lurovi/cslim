"""
This module provides various error metrics functions for evaluating machine learning models.
"""

import torch


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Compute Root Mean Squared Error (RMSE).

    Args:
        y_true (torch.Tensor): True values.
        y_pred (torch.Tensor): Predicted values.

    Returns:
        float: RMSE value.
    """
    return float(torch.sqrt(
        torch.mean(torch.square(torch.sub(y_true, y_pred)), dim=len(y_pred.shape) - 1)
    ).item())


def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Compute Mean Squared Error (MSE).

    Args:
        y_true (torch.Tensor): True values.
        y_pred (torch.Tensor): Predicted values.

    Returns:
        float: MSE value.
    """
    return float(torch.mean(
        torch.square(torch.sub(y_true, y_pred)), dim=len(y_pred.shape) - 1
    ).item())


def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Compute Mean Absolute Error (MAE).

    Args:
        y_true (torch.Tensor): True values.
        y_pred (torch.Tensor): Predicted values.

    Returns:
        float: MAE value.
    """
    return float(torch.mean(torch.abs(torch.sub(y_true, y_pred)), dim=len(y_pred.shape) - 1).item())


def mae_int(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Compute Mean Absolute Error (MAE) for integer values.

    Args:
        y_true (torch.Tensor): True values.
        y_pred (torch.Tensor): Predicted values.

    Returns:
        float: MAE value for integer predictions.
    """
    return float(torch.mean(
        torch.abs(torch.sub(y_true, torch.round(y_pred))), dim=len(y_pred.shape) - 1
    ).item())


def signed_errors(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute signed errors between true and predicted values.

    Args:
        y_true (torch.Tensor): True values.
        y_pred (torch.Tensor): Predicted values.

    Returns:
        torch.Tensor: Signed error values.
    """
    return torch.sub(y_true, y_pred)
