from .feature_extractor import FEATURE_NAMES, extract_features

BEHAVIOR_FEATURE_NAMES = [name for name in FEATURE_NAMES if name in {
    "duration", "caller_speech_seconds", "agent_speech_seconds", "caller_speech_ratio", "agent_speech_ratio",
    "caller_turns", "agent_turns", "caller_turn_duration_mean", "caller_turn_duration_std", "caller_turn_duration_min",
    "caller_turn_duration_max", "caller_turn_duration_median", "agent_turn_duration_mean", "agent_turn_duration_std",
    "agent_turn_duration_min", "agent_turn_duration_max", "agent_turn_duration_median", "caller_pause_mean",
    "caller_pause_std", "caller_pause_min", "caller_pause_max", "caller_pause_median", "pause_cv", "response_count",
    "response_latency_mean", "response_latency_std", "response_latency_min", "response_latency_max", "response_latency_median",
    "response_latency_cv", "quick_response_ratio", "interruptions", "interruption_rate", "overlap_seconds", "overlap_ratio"
}]


def extract_behavior_features(caller, agent, sample_rate):
    all_features = extract_features(caller, agent, sample_rate)
    return {name: all_features[name] for name in BEHAVIOR_FEATURE_NAMES}
