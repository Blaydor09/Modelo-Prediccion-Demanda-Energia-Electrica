"""Demand forecasting package."""

from .model import RidgeForecaster, load_model, save_model

__all__ = ["RidgeForecaster", "load_model", "save_model"]

