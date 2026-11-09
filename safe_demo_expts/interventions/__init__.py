__all__ = [
    "RecoveryOracle",
    "oracle_recovery_action",
    "InterventionController",
    "build_controller",
    "default_variant_kwargs",
    "GateMLP",
    "stuck_features",
    "train_gate",
    "decide_query",
]


def __getattr__(name: str):
    if name in ("RecoveryOracle", "oracle_recovery_action"):
        from .oracle import RecoveryOracle, oracle_recovery_action

        return RecoveryOracle if name == "RecoveryOracle" else oracle_recovery_action
    if name in ("InterventionController", "build_controller", "default_variant_kwargs"):
        from . import variants

        return getattr(variants, name)
    if name in (
        "GateMLP",
        "stuck_features",
        "train_gate",
        "decide_query",
        "collect_escape_fail_labels",
        "FEATURE_DIM",
        "GATE_MODES",
    ):
        from . import query_escape_gate

        return getattr(query_escape_gate, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
