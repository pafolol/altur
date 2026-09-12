SEMANTIC_ENABLED = False


def extract_semantic_features(caller, agent, sample_rate):
    """Reserved for a future semantic model. Transcription/LLMs are intentionally disabled."""
    if not SEMANTIC_ENABLED:
        return None
    raise NotImplementedError("Semantic features are not enabled")
