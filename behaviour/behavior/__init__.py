"""Conversational timing evidence, independent of Acoustic and Semantic modules."""

__version__ = "1.0.0"


def __getattr__(name):
    if name == "BehaviorDetector":
        from .inference import BehaviorDetector

        return BehaviorDetector
    raise AttributeError(name)
