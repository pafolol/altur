import logging
from pathlib import Path

import joblib

from .feature_extractor import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, features_to_vector

logger = logging.getLogger(__name__)


class DetectorUnavailable(RuntimeError): pass
class DetectorIncompatible(RuntimeError): pass


class Detector:
    def __init__(self, mode="mock", model_path="models/detector.joblib", threshold=0.5):
        self.mode, self.model, self.threshold, self.available = mode, None, float(threshold), mode in {"mock", "fusion"}
        self.warmed_up = False
        if mode == "mock": logger.warning("DETECTOR_MODE=mock: resultados simulados, no usar en producción")
        elif mode == "fusion": logger.info("DETECTOR_MODE=fusion: la inferencia la resuelve src/fusion.py")
        elif mode == "model": self._load(model_path)
        else: raise DetectorUnavailable(f"Modo de detector inválido: {mode}")

    def _load(self, path):
        if not Path(path).is_file(): raise DetectorUnavailable("El modelo no está disponible")
        try: bundle = joblib.load(path)
        except Exception as exc: raise DetectorUnavailable("No se pudo cargar el modelo") from exc
        if not isinstance(bundle, dict) or "model" not in bundle or bundle["model"] is None: raise DetectorIncompatible("El bundle debe contener un 'model' válido")
        expected = bundle.get("feature_names", FEATURE_NAMES)
        if not isinstance(expected, (list, tuple)) or len(expected) != len(FEATURE_NAMES) or list(expected) != FEATURE_NAMES: raise DetectorIncompatible("Las feature_names del modelo no son compatibles en cantidad u orden")
        if bundle.get("feature_schema_version", FEATURE_SCHEMA_VERSION) != FEATURE_SCHEMA_VERSION: raise DetectorIncompatible("Feature schema version incompatible")
        threshold = float(bundle.get("threshold", self.threshold))
        if not 0 <= threshold <= 1: raise DetectorIncompatible("Threshold fuera de rango")
        modalities = bundle.get("modalities", ["behavior", "acoustic"])
        if not isinstance(modalities, list) or any(m not in {"behavior", "acoustic", "semantic"} for m in modalities): raise DetectorIncompatible("Modalidades inválidas en el bundle")
        self.model = bundle["model"]; self.threshold = threshold; self.available = True
        self.model_version = bundle.get("model_version", "unknown")
        self.warmed_up = False

    def warm_up(self):
        if self.mode == "model" and not self.warmed_up:
            self._warm_up()
            self.warmed_up = True

    def _warm_up(self):
        try:
            vector = features_to_vector({name: 0.0 for name in FEATURE_NAMES})
            if hasattr(self.model, "predict_proba"): self.model.predict_proba(vector)
            else: self.model.predict(vector)
            logger.info("Detector model warm-up completed")
        except Exception as exc:
            raise DetectorIncompatible("El modelo no acepta el schema de features durante warm-up") from exc

    def predict_synthetic_probability(self, features):
        if not self.available: raise DetectorUnavailable("Detector no disponible")
        if self.mode == "mock": return 0.5
        # In fusion mode the verdict comes from the whole call, not from this backend's feature vector,
        # so analyze() never reaches here - see app/fusion_bridge.py.
        if self.mode == "fusion": raise DetectorUnavailable("En modo fusion la inferencia la resuelve el bridge")
        vector = features_to_vector(features)
        try:
            if hasattr(self.model, "predict_proba"): probability = float(self.model.predict_proba(vector)[0, 1])
            else: probability = float(self.model.predict(vector)[0])
        except Exception as exc: raise RuntimeError("Error durante la inferencia") from exc
        if not 0.0 <= probability <= 1.0: raise RuntimeError("El modelo produjo una probabilidad fuera de rango")
        return probability
