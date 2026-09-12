import math
from pathlib import Path

import joblib
import numpy as np


class FusionError(RuntimeError):
    pass


class Fusion:
    def __init__(self, mode="single_model", weights=None, meta_model_path="models/fusion_model.joblib"):
        if mode not in {"single_model", "weighted", "meta_model"}:
            raise FusionError(f"FUSION_MODE inválido: {mode}")
        self.mode = mode
        self.weights = weights or {}
        self.meta_model = None
        if mode == "weighted":
            try:
                numeric_weights = [float(v) for v in self.weights.values()]
            except (TypeError, ValueError):
                numeric_weights = []
            if not self.weights or not numeric_weights or any(not math.isfinite(v) or v < 0 for v in numeric_weights) or sum(numeric_weights) <= 0:
                raise FusionError("Los pesos de fusion deben ser finitos, no negativos y sumar más de cero")
        if mode == "meta_model":
            if not Path(meta_model_path).is_file():
                raise FusionError("El modelo de fusion no está disponible")
            try:
                self.meta_model = joblib.load(meta_model_path)
            except Exception as exc:
                raise FusionError("No se pudo cargar el modelo de fusion") from exc

    def predict(self, scores=None, single_model=None):
        if self.mode == "single_model":
            if single_model is None: raise FusionError("Falta la predicción del modelo principal")
            return self._probability(single_model)
        scores = scores or {}
        available = [(name, float(score), float(self.weights.get(name, 0))) for name, score in scores.items() if score is not None and name in self.weights]
        if self.mode == "weighted":
            total = sum(weight for _, score, weight in available)
            if not available or total <= 0: raise FusionError("No hay scores disponibles para weighted fusion")
            return self._probability(sum(score * weight for _, score, weight in available) / total)
        vector = np.asarray([[scores.get(name) for name in ("behavior", "acoustic", "semantic")]], dtype=float)
        if not np.isfinite(vector).all(): raise FusionError("El meta-modelo requiere scores de todas las modalidades")
        try:
            value = self.meta_model.predict_proba(vector)[0, 1] if hasattr(self.meta_model, "predict_proba") else self.meta_model.predict(vector)[0]
        except Exception as exc: raise FusionError("Error durante la inferencia del modelo de fusion") from exc
        return self._probability(value)

    @staticmethod
    def _probability(value):
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1: raise FusionError("Fusion produjo una probabilidad fuera de rango")
        return value
